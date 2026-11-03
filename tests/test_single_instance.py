import torch

from DL_modules.custom_layers.single_instance_submodules import (
    make_pairs_of_source_i_with_embeddings,
    update_covariances_and_grand_mean,
)


def test_make_pairs_of_source_i_with_embeddings():
    A = torch.tensor([[1.0], [2.0]])
    B = torch.tensor([[[10.0], [11.0]], [[20.0], [21.0]], [[30.0], [31.0]]])
    n_mini, n_rep, n_features = B.shape
    assert B.shape == (3, 2, 1)
    A_out, B_out = make_pairs_of_source_i_with_embeddings(A, B)
    assert A_out.shape == B_out.shape == (n_mini * n_rep * n_rep, 1)
    assert torch.equal(
        A_out, torch.tensor([[1.0], [2.0], [1.0], [2.0], [1.0], [2.0], [1.0], [2.0], [1.0], [2.0], [1.0], [2.0]])
    )
    assert torch.equal(
        B_out,
        torch.tensor([[10.0], [10.0], [11.0], [11.0], [20.0], [20.0], [21.0], [21.0], [30.0], [30.0], [31.0], [31.0]]),
    )


def test_source_stats():
    # Embeddings shape (n_sources, n_rep, n_features) = (3,2,4)
    embeddings = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
            [[2.0, 3.0, 4.0, 5.0], [2.5, 3.6, 4.7, 5.8]],
            [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]],
        ]
    )
    means = torch.mean(embeddings, dim=1)
    output = update_covariances_and_grand_mean(embeddings, means)
    means_exp = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.25, 3.3, 4.35, 5.4], [1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(means, means_exp)
    between_cov = torch.cov(means.T, correction=0)
    grand_mean = torch.mean(means, dim=0).unsqueeze(0)
    pooled_within_cov = torch.zeros(means.shape[1], means.shape[1])
    n_instances = embeddings.shape[0]
    for i in range(n_instances):
        embedding = embeddings[i]
        pooled_within_cov += torch.cov(embedding.T, correction=0)
    pooled_within_cov = pooled_within_cov / n_instances
    assert len(output) == 3
    assert torch.equal(output[0], pooled_within_cov)
    assert torch.equal(output[1], between_cov)
    assert torch.equal(output[2], grand_mean)
