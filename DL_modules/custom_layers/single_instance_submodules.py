import numpy as np
import torch
from torch import nn

from DL_modules.custom_layers.two_level_normal_normal_sub_modules import Q


def update_covariances_and_grand_mean(proj_embeddings: torch.Tensor, proj_means: torch.Tensor):
    """
    Calculate within and between covariances and grand mean.

    :param proj_embeddings: projected embeddings of shape (n_sources, n_rep, n_features).
    :param proj_means: projected means of shape (n_sources, n_features).
    """
    # Recompute pooled within-source variance
    within_cov_matrices = []

    # Loop over each slice to compute the covariance
    for slice in proj_embeddings:
        # slice of shape (n_rep, n_axes)
        # Compute covariance of the slice, by treating the rows as variables and columns as observations
        within_cov_matrix = torch.cov(slice.T, correction=0)  # covariance of shape (18, 18)

        within_cov_matrices.append(within_cov_matrix)

    # Convert list of covariance matrices to a tensor
    within_cov_tensor = torch.stack(within_cov_matrices)  # shape (batch_size, n_axes, n_axes)

    # Compute the mean of all the covariance matrices (across the first dimension)
    pooled_within_cov = within_cov_tensor.mean(dim=0)  # shape (n_axes, n_axes)

    # Compute grand mean and between-source variance
    proj_grand_mean = proj_means.mean(dim=0).unsqueeze(0)  # shape (1, n_axes)
    between_cov = proj_means.T.cov(correction=0)  # correction=0 ->  division by n_sources, shape(n_axes, n_axes)

    return pooled_within_cov, between_cov, proj_grand_mean


class OrthoProjector(nn.Module):
    """
    Project data by an orthonormal projection.

    """

    def __init__(self, n_axes: int):
        super().__init__()
        self.q = Q(n_axes)

    def forward(self, data):
        q_matrix = self.q()
        # data: (batch_size, num_repeats, n_axes)
        return torch.matmul(data, q_matrix)  # shape: (batch_size, num_repeats, n_axes)


class WhiteThanOrthoProjector(nn.Module):
    """
    Project data by an orthonormal projection.

    """

    def __init__(self, n_axes):
        super().__init__()
        self.q_w = Q(n_axes)
        self.register_buffer("inv_sd_w", -1 * torch.eye(n_axes, dtype=torch.float64))  # initiate with dummy values
        self.q_b = Q(n_axes)

    def forward(self, data):
        q_matrix_w = self.q_w()
        q_matrix_b = self.q_b()
        transformation_matrix = torch.matmul(torch.matmul(q_matrix_w, self.inv_sd_w), q_matrix_b)

        # data: (batch_size, num_repeats, n_axes) or (if means, (batch_size, n_axes))
        return torch.matmul(data, transformation_matrix)

    def update_var_w(self, embeddings, means):
        with torch.no_grad():  # Use torch.no_grad since we just want to update the buffer.
            q_matrix_w = self.q_w()
            proj_embeddings = torch.matmul(embeddings, q_matrix_w)
            proj_means = torch.matmul(means, q_matrix_w)
            pooled_within_cov, _, _ = update_covariances_and_grand_mean(proj_embeddings, proj_means)
            n_per_source = embeddings.shape[1]
            pooled_within_cov = pooled_within_cov * n_per_source / (n_per_source - 1)
            self.inv_sd_w.copy_(
                torch.diag(1 / torch.sqrt(torch.diag(pooled_within_cov)))
            )  # Creates a matrix with off-diagonal zeros and same 1/within_sd in the diagonal


def get_subset_indices(all_labels_seen: list[int], minibatch_labels: torch.Tensor) -> torch.Tensor:
    """
    Get indices of all_labels_seen that match minibatch_labels.

    For batch label element i, index l(i) is the identical label in the ordered label_set.
    :param all_labels_seen: the list of source_idx that the model has seen up till now
        (this includes the current minibatch_labels)
    :param minibatch_labels: the labels in the minibatch.
    :return: the indices of all_labels_seen that match minibatch_labels.
        Row i in minibatch_labels and row indices[i] of all_labels_seen have the same label for all i.
    """
    all_labels_tensor = torch.tensor(all_labels_seen)

    # Create a mapping from label → index in the ordered set
    label_to_index = {label.item(): idx for idx, label in enumerate(all_labels_tensor)}

    # Map each batch label to its index in the ordered set
    indices = torch.tensor([label_to_index[label.item()] for label in minibatch_labels])
    return indices


def make_ss_ds_pairs(
    minibatch_data: torch.Tensor, embeddings: torch.Tensor, embedding_label_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Collect all same_source pairs in minibatch_data and all different_source combinations in minibatch_data and
    self.embeddings.

    :param minibatch_data: minibatch_data of shape (n_sources_minibatch, n_rep, n_features).
    :param embeddings: embeddings of shape (n_sources, n_rep, n_features).
    :param minibatch_label_indices: indices connecting minibatch_row (i) to embedding source
        (embedding_label_indices[i])
    :return cr_data, susp_data, ss_ds_labels: row by row combination of cr_instance, susp_instance and
        label (1 = same_source, 0 = different_source)
    """
    n_sources_in_minibatch, n_rep, n_features = minibatch_data.shape
    # crime_related same_source
    cr_ss_data = torch.empty(0, n_features)
    # crime_related different_source
    cr_ds_data = torch.empty(0, n_features)
    susp_ss_data = torch.empty(0, n_features)
    susp_ds_data = torch.empty(0, n_features)
    for i, minibatch_at_i in enumerate(minibatch_data):
        cr_ss_data_at_i, susp_ss_data_at_i = get_ss_pairs_at_i(minibatch_at_i)
        cr_ss_data = torch.cat((cr_ss_data, cr_ss_data_at_i), dim=0)
        susp_ss_data = torch.cat((susp_ss_data, susp_ss_data_at_i), dim=0)

        embedding_label_indices_cloned = embedding_label_indices.clone()
        ss_index_removed = torch.cat((embedding_label_indices_cloned[:i], embedding_label_indices_cloned[i + 1 :]))
        ds_embeddings = embeddings[ss_index_removed]
        cr_ds_data_at_i, susp_ds_data_at_i = make_pairs_of_source_i_with_embeddings(minibatch_at_i, ds_embeddings)
        cr_ds_data = torch.cat((cr_ds_data, cr_ds_data_at_i), dim=0)
        susp_ds_data = torch.cat((susp_ds_data, susp_ds_data_at_i), dim=0)

    cr_data = torch.cat((cr_ss_data, cr_ds_data), dim=0)
    susp_data = torch.cat((susp_ss_data, susp_ds_data), dim=0)

    # same_source_different_source_lebels
    ss_ds_labels = np.zeros(cr_data.shape[0], dtype=int)
    ss_ds_labels[: cr_ss_data.shape[0]] = 1

    # crime_related_data[i], suspect_data[i]. same_source_different_source_labels[i] form one data row
    return cr_data, susp_data, ss_ds_labels


def get_ss_pairs_at_i(minibatch_at_i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Get all possible pair-wise combinations.

    :param minibatch_at_i: minibatch_at_i of shape (n_rep, n_features).
    :return: two torch.Tensors, each of shape (n_rep * (n_rep -1)/2, n_features),
        holding all possible n_rep * (n_rep -1)/2 pairs
    """
    n_rep = minibatch_at_i.shape[0]
    ids = torch.arange(n_rep)
    # default: pairwise without replacement, exactly what we need
    pair_idx = torch.combinations(ids)

    return minibatch_at_i[pair_idx[:, 0]], minibatch_at_i[pair_idx[:, 1]]


def make_pairs_of_source_i_with_embeddings(
    minibatch_at_i: torch.Tensor, embeddings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Make all pairs combining rows of minibatch_at_i (shape (n_rep, n_features)) with embeddings
    (shape (n_sources, n_rep, n_features)).

    :param minibatch_at_i: minibatch_at_i of shape (n_rep, n_features).
    :param embeddings: embeddings of shape (n_sources, n_rep, n_features).
    :return: two torch Tensors holding all possible n_rep * n_rep * n_sources pairs.
        Shape of each: (n_rep^2 * n_sources, n_features)
    """
    # assumption: n_rep for minibatch is the same as for embeddings
    n_minibatch, n_rep, n_features = embeddings.shape
    # Step 1: expand A to match combinations
    minibatch_at_i_expanded = minibatch_at_i.unsqueeze(0).unsqueeze(1)  # shape (1, 1, n_rep, n_features)
    minibatch_at_i_expanded = minibatch_at_i_expanded.expand(n_minibatch, n_rep, n_rep, n_features)

    # Step 2: expand B so each minibatch element is paired with every A
    embeddings_expanded = embeddings.unsqueeze(2)  # (n_mini, n_rep, 1, n_features)
    embeddings_expanded = embeddings_expanded.expand(n_minibatch, n_rep, n_rep, n_features)

    # Step 3: flatten minibatch & rep dimensions
    minibatch_at_i_out = minibatch_at_i_expanded.reshape(-1, n_features)
    embeddings_out = embeddings_expanded.reshape(-1, n_features)
    return minibatch_at_i_out, embeddings_out
