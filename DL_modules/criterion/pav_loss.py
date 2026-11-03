import torch
from sklearn.isotonic import IsotonicRegression
from torch import nn

from ..custom_layers.top_layers import ResultTLM
from .cllr import CLLR


class PAVLossConcentrated(torch.nn.Module):
    def __init__(
        self, fr_hard_negs: float, cllr_threshold: float = 0.95, n_axes: str | int = "all", mode: str = "train"
    ):
        super().__init__()
        self.n_axes = n_axes
        self.fr_hard_negs = fr_hard_negs
        self.cllr_threshold = cllr_threshold
        self.cllr_evaluate = CLLR(fr_hard_negs, 'from_llr', 'evaluate')
        self.ir = IsotonicRegression()
        if mode not in ['train', 'evaluate']:
            raise ValueError("mode must be 'train' or 'evaluate'")
        self.mode = mode

    """
    Select axes where cllr is smaller than cllr_threshold (the 'focal' axes)
    For these axes and the sum_llrs, return mean of pav mean square error loss to the identity line per axes.
    :param fr_hard_negs: fr_negatives that are included in the DataLoader (the hard negatives among all negatives)
    :param cllr_threshold: maximum cllr for an axis to be included when calculating PAVloss.
        This prevents immediate focus on all axes since when combining this loss with cllr_concentrated such a broad
        focus is unwanted.
    :param n_axes: axes 0:n_axes are considered.
    :param mode: 'train' or 'evaluate'. WHen train, e mean is returned.
        When 'evaluate', a loss per axis (and for the summed_llrs) is returned.
    """

    def forward(self, predictions: ResultTLM, labels: torch.Tensor) -> torch.Tensor | list[list[torch.tensor]]:
        prediction_llrs = predictions.log_norm['llr_norm']
        cllrs = self.cllr_evaluate(prediction_llrs, labels)[:-1]
        indices_focus = torch.nonzero(cllrs <= self.cllr_threshold, as_tuple=True)[0]

        if len(indices_focus) == 0:
            return 0
        else:
            n_devpavs = len(indices_focus) + 1
            llrs_sum = torch.sum(prediction_llrs[:, indices_focus], dim=1)
            prediction_llrs_focus = torch.cat((prediction_llrs[:, indices_focus], llrs_sum), dim=1)
            pav_losses = torch.empty((0,))
            for i in range(n_devpavs):
                torch.cat((pav_losses, self.pav_loss(prediction_llrs_focus[:, i], labels)), dim=0)

        if self.mode == "train":
            return torch.sum(pav_losses) / n_devpavs
        if self.mode == "evaluate":
            return pav_losses

    def pav_loss(self, llrs: torch.tensor, labels: torch.tensor) -> torch.tensor:
        # Create a tensor with a value of negative infinity
        neg_inf_tensor = torch.tensor(float('-inf'))
        pos_inf_tensor = torch.tensor(float('inf'))

        # check for pathological cases
        condition = (torch.isinf(llrs)) & (llrs > 0) & (labels == 0)
        llrs_hard_fp = llrs[condition]
        condition = (torch.isinf(llrs)) & (llrs < 0) & (labels == 1)
        llrs_hard_fn = llrs[condition]

        if len(llrs_hard_fp) > 0 or len(llrs_hard_fn) > 0:
            raise ValueError('Cannot compute pav_loss for H1 LLRs that are neg_inf or H0 LLRs that are pos_inf')

        condition = ~torch.isinf(llrs)  # checks for pos and negative inf
        n_inf = torch.sum(llrs == pos_inf_tensor)
        n_neginf = torch.sum(llrs == neg_inf_tensor)

        n_pos_labels = torch.sum(labels)
        n_neg_labels = len(labels) - n_pos_labels
        prior_odds = (n_pos_labels - n_inf) / (n_neg_labels - n_neginf)

        llrs_stripped = llrs[condition]
        labels_stripped = labels[condition]

        llrs_np = llrs_stripped.detach().numpy()
        labels_np = labels_stripped.detach().numpy()
        # Apply isotonic regression to the llrs
        # Ensuring the targets comply with isotonic constraints
        targets_iso = self.ir.fit_transform(llrs_np, labels_np)
        posterior_odds_iso = targets_iso / (1 - targets_iso)

        posterior_odds_iso_torch = torch.tensor(posterior_odds_iso)
        lrs_iso_torch = posterior_odds_iso_torch / prior_odds
        llrs_iso_torch = torch.log10(lrs_iso_torch)
        not_inf_mask = ~torch.isinf(llrs_iso_torch)  # checks for pos and negative inf
        llrs_iso_torch_stripped = llrs_iso_torch[not_inf_mask]
        llrs_stripped = llrs_stripped[not_inf_mask]

        # Calculate mean squared error
        loss = nn.MSELoss()(llrs_stripped, llrs_iso_torch_stripped)
        return loss
