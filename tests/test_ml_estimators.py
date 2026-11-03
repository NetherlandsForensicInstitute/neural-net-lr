import pytest
import torch

from estimators.inverse_gamma_parameters_scipy import (
    fit_alpha_beta_map_scipy_theta,
    fit_alpha_beta_ml_scipy_theta,
    fit_alpha_beta_personal_scipy,
    plot_inversegammas,
)
from estimators.inverse_gamma_parameters_torch import (
    fit_alpha_beta_MAP,
    fit_alpha_beta_ML,
    fit_alpha_beta_personal,
)


def get_mode_variance(alpha, beta):
    mode = beta / (alpha + 1)
    variance = beta**2 / ((alpha - 1) ** 2 * (alpha - 2))

    return mode, variance


def test_fit_alpha_beta():
    # synthetic test
    torch.manual_seed(0)

    n_sources = 351  # n_sources
    n_rep = 2

    # true hierarchical variances for testing. Mean inversegamma: beta/(alpha-1) mode: beta/(alpha+1)
    true_alpha = 8.0
    true_beta = 1.0

    # sample variances from inverse-gamma distribution
    # σ^2 ~ InvGamma(α,β)
    sig2 = torch.distributions.InverseGamma(true_alpha, true_beta).sample((n_sources,))

    data = torch.stack([torch.normal(0, sig2[i].sqrt(), size=(n_rep,)) for i in range(n_sources)])  # (n_sources, n_rep)

    alpha_hat, beta_hat, _ = fit_alpha_beta_ML(data)
    alpha_ml_scipy, beta_ml_scipy, _ = fit_alpha_beta_ml_scipy_theta(data.numpy())

    print("Estimated alpha ML:", alpha_hat)
    print("Estimated beta ML: ", beta_hat)

    assert alpha_hat == pytest.approx(alpha_ml_scipy, rel=1e-3)
    assert beta_hat == pytest.approx(beta_ml_scipy, rel=1e-3)

    alpha_map, beta_map, _ = fit_alpha_beta_MAP(data, alpha_hat, beta_hat)

    print("Estimated alpha MAP:", alpha_map)
    print("Estimated beta MAP: ", beta_map)

    assert beta_map < beta_hat

    alpha_map_scipy, beta_map_scipy, _ = fit_alpha_beta_map_scipy_theta(data.numpy())

    assert alpha_map == pytest.approx(alpha_map_scipy, rel=1e-3)
    assert beta_map == pytest.approx(beta_map_scipy, rel=1e-3)

    alpha_personal, beta_personal, _ = fit_alpha_beta_personal(data)
    alpha_personal_scipy, beta_personal_scipy, mean_within_variance = fit_alpha_beta_personal_scipy(data.numpy())

    plot_inversegammas(
        alpha_ml_scipy,
        beta_ml_scipy,
        alpha_personal_scipy,
        beta_personal_scipy,
        true_alpha,
        true_beta,
        mean_within_variance,
    )

    assert alpha_personal == pytest.approx(alpha_personal_scipy, rel=1e-3)
    assert beta_personal == pytest.approx(beta_personal_scipy, rel=1e-3)

    # check that mode is the same as ML estimates and that variance has increased.
    mode_ml, variance_ml = get_mode_variance(alpha_hat, beta_hat)
    mode_personal, variance_personal = get_mode_variance(alpha_personal, beta_personal)

    assert mode_ml == pytest.approx(mode_personal, rel=1e-6)
    assert variance_ml <= variance_personal
