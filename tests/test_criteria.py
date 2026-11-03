import numpy as np
import pytest
import torch
from torch.distributions import Normal, kl_divergence

from DL_modules.criterion import KLDivergenceLossOneDim
from tests.conftest import to_tlm_result


def test_cllr_ignorance(cllr_from_p_instance, ignorant_ps):
    preds, labels = ignorant_ps
    loss = cllr_from_p_instance(preds, labels)
    assert all(loss == torch.Tensor([1, 1]))


def test_cllr_ignorance_2(cllr_instance_evaluate, ignorant_llrs):
    preds, labels = ignorant_llrs
    loss, _ = cllr_instance_evaluate(preds, labels)
    assert torch.allclose(loss, torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32), rtol=1e-7)


def test_quarter_ps(cllr_from_p_instance, quarter_ps):
    preds, labels = quarter_ps
    loss = cllr_from_p_instance(preds, labels)
    assert all(loss == torch.tensor([2.0, 2.0], dtype=torch.float32))


def test_0_1_ps(cllr_from_p_instance, all_1_or_0_ps):
    preds, labels = all_1_or_0_ps
    loss = cllr_from_p_instance(preds, labels)
    assert loss == np.inf


def test_some_llrs(cllr_instance_evaluate, some_llrs):
    preds, labels = some_llrs
    loss, _ = cllr_instance_evaluate(preds, labels)
    assert loss[0] == pytest.approx(
        -1 * 0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(7 / 8))), 1e-6
    )
    assert loss[2] == pytest.approx(
        -1 * 0.5 * (0.5 * (np.log2(9 / 10) + np.log2(49 / 50)) + 0.5 * (np.log2(225 / 226) + np.log2(49 / 50))), 1e-6
    )


def test_certainty(cllr_from_p_instance, certainty_ps):
    preds, labels = certainty_ps
    loss = cllr_from_p_instance(preds, labels)
    assert loss == 0


def test_cllr_instance_evaluate(cllr_instance_evaluate, some_llrs):
    preds, labels = some_llrs
    loss, _ = cllr_instance_evaluate(preds, labels)
    assert len(loss) == 4


def test_cllr_instance_train_evenly(cllr_instance_train_evenly, some_llrs_3_columns_tlm_result):
    tlm_result, labels = some_llrs_3_columns_tlm_result
    loss = cllr_instance_train_evenly(tlm_result, labels)
    assert loss == pytest.approx(
        -1
        * 0.5
        * 1.0
        / 4.0
        * (
            3 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(7 / 8)))
            + 1
            * (
                0.5 * (np.log2(27 / 28) + np.log2(49 * 7 / (49 * 7 + 1)))
                + 0.5 * (np.log2(225 * 15 / (225 * 15 + 1)) + np.log2(49 * 7 / (49 * 7 + 1)))
            )
        ),
        1e-7,
    )


def test_cllr_instance_train_focus_spread_sum(cllr_instance_train_focus_spread_sum, some_llrs_diff_axes):
    preds, labels = some_llrs_diff_axes
    loss = cllr_instance_train_focus_spread_sum(preds, labels)

    cllr_sum_llr = -0.5 * (
        0.5 * (np.log2(9 / 10) + np.log2(49 / 50)) + 0.5 * (np.log2(225 / 226) + np.log2((7 * 14) / (7 * 14 + 1)))
    )
    cllr_llr_axes_1 = -0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(7 / 8)))
    cllr_llr_axes_2 = -0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(14 / 15)))
    expected = cllr_sum_llr + (cllr_llr_axes_1 + cllr_llr_axes_2)

    assert loss == pytest.approx(expected, 1e-7)


def test_cllr_instance_train_focus_spread_mean(cllr_instance_train_focus_spread_mean_n_axes_2, some_llrs_diff_axes):
    preds, labels = some_llrs_diff_axes
    loss = cllr_instance_train_focus_spread_mean_n_axes_2(preds, labels)

    cllr_sum_llr = -0.5 * (
        0.5 * (np.log2(9 / 10) + np.log2(49 / 50)) + 0.5 * (np.log2(225 / 226) + np.log2((7 * 14) / (7 * 14 + 1)))
    )
    cllr_llr_axes_1 = -0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(7 / 8)))
    cllr_llr_axes_2 = -0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(14 / 15)))
    cllr_llr_axes_avg = (cllr_llr_axes_1 + cllr_llr_axes_2) / 2
    expected = cllr_sum_llr + cllr_llr_axes_avg

    assert loss == pytest.approx(expected, 1e-7)


def test_cllr_instance_train_concentrated(cllr_instance_train_concentrated, some_llrs):
    preds, labels = some_llrs
    loss = cllr_instance_train_concentrated(to_tlm_result(preds), labels)
    expected_loss_axis_1 = (
        -1 * 0.5 * (0.5 * (np.log2(3 / 4) + np.log2(7 / 8)) + 0.5 * (np.log2(15 / 16) + np.log2(7 / 8)))
    )
    expected_loss_axis_2 = (
        -1 * 0.5 * (0.5 * (np.log2(9 / 10) + np.log2(49 / 50)) + 0.5 * (np.log2(225 / 226) + np.log2(49 / 50)))
    )
    assert loss == pytest.approx(np.mean([expected_loss_axis_1, expected_loss_axis_2]), 1e-7)


# Test if kl_div for sample from true distribution approaches zero.
def test_kl_div_to_sample_standard_normal_large_sample():
    torch.manual_seed(250323)
    n = 20000
    data = torch.randn(n)
    kl_div_instance = KLDivergenceLossOneDim(1, sample_size=20000)
    result = kl_div_instance(data)
    expected = torch.tensor([0.00066])

    assert result == pytest.approx(expected, abs=0.000005)


# Test if kl_div for sample from true distribution has low value for realistic sample sizes.
def test_kl_div_to_sample_standard_normal_realistic_sample():
    torch.manual_seed(250323)
    n = 20
    data = torch.randn(n)
    kl_div_instance = KLDivergenceLossOneDim(1, sample_size=200)
    result = kl_div_instance(data)
    expected = torch.tensor([0.21259])

    assert result == pytest.approx(expected, abs=0.000005)


# Test if kl-divergence of a sample from a normal converges to the theoretical kl_div.
def test_compare_kl_divergence_to_standard_normal():
    torch.manual_seed(250323)
    sd = 10
    n = 20000
    data = torch.randn(n) * sd
    kl_div_instance = KLDivergenceLossOneDim(1, sample_size=20000)
    result = kl_div_instance(data)
    expected = kl_divergence(Normal(0, sd), Normal(0, 1)).unsqueeze(0)
    difference = torch.tensor([1.64833])

    assert result == pytest.approx(expected + difference, abs=0.000005)


# Test that error in kl_divergence is largely due to sampling
# (so that when comparing two kl_divergences based on samples, the error is much smaller)
def test_compare_kl_divergence_to_normal():
    torch.manual_seed(250323)
    sd = 10
    n = 20
    data = torch.randn(n) * sd

    # calculate kl_div based on kde-probs for the input data
    kl_div_instance = KLDivergenceLossOneDim(1)
    result = kl_div_instance(data)

    # do the same, but then based on normally distributed probs on the input data
    normal_dist = torch.distributions.Normal(0, sd)
    ln_probs_norm = normal_dist.log_prob(data)
    standard_normal_dist = torch.distributions.Normal(0, 1)
    ln_probs_st_norm = standard_normal_dist.log_prob(data)
    expected_kl_div = torch.mean(ln_probs_norm - ln_probs_st_norm).unsqueeze(0)
    difference = torch.tensor([0.34041])

    assert result == pytest.approx(expected_kl_div + difference, abs=0.000005)  # smaller error then for test above,
