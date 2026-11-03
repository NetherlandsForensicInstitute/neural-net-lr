import numpy as np
import pytest
import torch


def test_correlation_coefficient(correlation_penalty_instance, some_llrs_per_axis):
    result = correlation_penalty_instance.compute_mean_correlation(some_llrs_per_axis)

    r01 = ((2 - 3) * (7 - 2) + (4 - 3) * (-3 - 2) + (5 - 3) * (3 - 2) + (1 - 3) * (1 - 2)) / np.sqrt(
        ((2 - 3) ** 2 + (4 - 3) ** 2 + (5 - 3) ** 2 + (1 - 3) ** 2)
        * ((7 - 2) ** 2 + (-3 - 2) ** 2 + (3 - 2) ** 2 + (1 - 2) ** 2)
    )
    r02 = ((2 - 3) * (-2) + (4 - 3) * (1) + (5 - 3) * (1) + (1 - 3) * (0)) / np.sqrt(
        ((2 - 3) ** 2 + (4 - 3) ** 2 + (5 - 3) ** 2 + (1 - 3) ** 2) * ((-2) ** 2 + (1) ** 2 + (1) ** 2 + (0) ** 2)
    )
    r12 = ((7 - 2) * (-2) + (-3 - 2) * (1) + (3 - 2) * (1) + (1 - 2) * (0)) / np.sqrt(
        ((7 - 2) ** 2 + (-3 - 2) ** 2 + (3 - 2) ** 2 + (1 - 2) ** 2) * ((-2) ** 2 + (1) ** 2 + (1) ** 2 + (0) ** 2)
    )

    expected_mean_corr = torch.mean(torch.abs(torch.tensor([r01, r02, r12])))

    assert result == pytest.approx(expected_mean_corr, abs=0.000005)
