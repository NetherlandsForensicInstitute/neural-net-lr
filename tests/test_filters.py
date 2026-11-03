import torch

from DL_modules.post_hoc import CoherenceFilter


def test_coherence_filter_for_coherent_llrs():
    sample_tensor = torch.tensor([[0.2, -0.2, 0.3, -0.1, 0.9], [0.1, 0.2, -0.3, -0.1, 0.8]])
    # bias,   bias, discr, bias, bias
    sample_cllrs: torch.Tensor = torch.tensor([1, 0.5, -1, 0.5, 0.25])
    res = sample_tensor.clone()
    filter = CoherenceFilter(threshold_bias_axis=0.05)
    res = filter.apply(res, sample_cllrs)

    # filter should not do anything, since only one discriminating axis.
    assert sample_tensor.shape == (2, 5)
    assert torch.equal(res, sample_tensor)


def test_coherence_filter_threshold():
    sample_tensor = torch.tensor([[0.3, 0, 0.3, -0.3, 0.9], [0.6, 0.3, -0.3, -0.3, 0.9]])
    sample_cllrs: torch.Tensor = torch.tensor([1, 0.5, 1.5, 0.5, 0.25])

    filter = CoherenceFilter(threshold_bias_axis=0.95)
    res = filter.apply(sample_tensor, sample_cllrs)
    assert res.shape == (2, 5)
    assert torch.all(torch.isclose(res, torch.tensor([[0.1, 0, 0.1, 0, 0], [0.2, 0.3, -0.1, 0, 0]])))


def test_coherence_filter_extreme():
    sample_tensor = torch.tensor([[0.3, -12, 0.3, -7, 0.9], [0.3, -12, -0.3, -0.4, 0.9], [-12, -12, -0.3, -0.4, 0.9]])
    sample_cllrs: torch.Tensor = torch.tensor([1, 0.5, 1.5, 0.5, 0.25])

    filter = CoherenceFilter(threshold_bias_axis=0.95, exclude_llrs_smaller_than=-6)
    res = filter.apply(sample_tensor, sample_cllrs)
    assert res.shape == (3, 5)
    assert torch.all(
        torch.isclose(
            res, torch.tensor([[0.3, -12, 0.3, -7, 0.9], [0.1, -12, -0.1, 0.0, 0.0], [-4, -12, -0.1, 0.0, 0.0]])
        )
    )
