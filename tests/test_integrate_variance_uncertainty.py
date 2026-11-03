import torch

from DL_modules.custom_layers import GaussianMixtureModel, IntegrateVarianceUncertainty, sample_sigma2_w
from DL_modules.KDE import get_kde_bandwidth
from estimators.inverse_gamma_parameters_scipy import fit_alpha_beta_personal_scipy


def prepare_for_ss_int_var(Y_batch, gmm_model):
    # Calculate sufficient statistics for Y_batch: n_y, means and sum_of_squares
    n_y = torch.tensor(Y_batch.shape[1])
    means_y, ssq = gmm_model.get_mean_ssq(Y_batch)

    return means_y, n_y, ssq


def test_batch_log_pxy():
    B = 2

    Y_batch = torch.tensor([[0.8, 1.0, 0.7], [1.0, 0.9, 0.8]])
    X_batch = torch.tensor([[1.2, 1.1], [0.9, 1.0]])

    kde_locations = torch.tensor([0.0, 0.5, 1.0])  # scalar means
    weights = torch.tensor([1.0, 1.0, 1.0])
    alpha, beta = torch.tensor([1.0]), torch.tensor([1.0])
    sigma2_e = torch.tensor([0.5])
    model = IntegrateVarianceUncertainty(kde_locations, weights)
    sample_sigma2_within = sample_sigma2_w(alpha, beta, 2000)
    llr_numerators = model.get_cs_llr_numerators(X_batch, Y_batch, sigma2_e, sample_sigma2_within)
    assert llr_numerators.shape == (B,)


def test_batch_logy():
    B = 3

    X_batch = torch.tensor([[0.8, 1.0, 0.7, 0.9], [1.1, 1.0, 1.2, 0.8], [0.9, 0.7, 1.0, 0.8]])

    kde_locations = torch.tensor([0.0, 0.5, 1.0])
    weights = torch.tensor([1.0, 1.0, 1.0])
    alpha, beta = torch.tensor([1.0]), torch.tensor([1.0])
    sigma2_e = torch.tensor([0.5])
    model = IntegrateVarianceUncertainty(kde_locations, weights)
    sample_sigma2_within = sample_sigma2_w(alpha, beta, 2000)

    llr_numerators = model.get_llr_denominators(X_batch, sigma2_e, sample_sigma2_within)
    assert llr_numerators.shape == (B,)


def test_numerical_output_one_sigma2_w_n_rep_1():
    torch.manual_seed(0)
    Y_batch = torch.tensor([[0.8], [1.0]])
    X_batch = torch.tensor([[1.2], [0.9]])

    kde_locations = torch.tensor([0.0, 0.5, 1.0])  # scalar means
    weights = torch.tensor([1.0, 1.0, 1.0])
    alpha, beta = torch.tensor([1.0]), torch.tensor([1.0])
    sigma2_e = torch.tensor([0.5])
    model = IntegrateVarianceUncertainty(kde_locations, weights)

    # get one sigma2_w
    sigma2_w = sample_sigma2_w(alpha, beta, sample_size=1)
    llr_numerators = model.get_cs_llr_numerators(X_batch, Y_batch, sigma2_e, sigma2_w)
    llr_denominators = model.get_llr_denominators(X_batch, sigma2_e, sigma2_w)

    # calculate specific source llrs by integrating variance method
    mean_y, n_y, ssq = prepare_for_ss_int_var(Y_batch, model)

    # assert that when n_y = 1, ssq == 0 and alpha_post == alpha and beta_post == beta
    alphas_post, betas_post = model.get_posterior_alpha_beta(ssq, n_y, alpha, beta)
    assert torch.allclose(ssq, torch.tensor([0.0, 0.0]), atol=1e-8)
    assert all(alpha_post == alpha for alpha_post in alphas_post)
    assert all(beta_post == beta for beta_post in betas_post)

    ss_log10_numerators = []
    log10_denominators = []
    for X, m_y in zip(X_batch, mean_y):
        ss_log10_numerators.append(
            model._ln_ss_prob(X.to("cuda"), m_y.to("cuda"), n_y, sigma2_e.to("cuda"), sigma2_w, torch.device("cuda"))
            / torch.log(torch.tensor(10.0))
        )
        log10_denominators.append(model.get_llr_denominators(X.unsqueeze(0).to("cuda"), sigma2_e, sigma2_w))
    intvar_ss_llrs = [n - d for n, d in zip(ss_log10_numerators, log10_denominators)]
    intvar_ss_llrs = torch.cat(intvar_ss_llrs, dim=0)

    # compute expected output using GMM and this sigma2_w
    kde_variance = get_kde_bandwidth(kde_locations) ** 2
    gmm = GaussianMixtureModel(
        locations=kde_locations.T.unsqueeze(1), variances=kde_variance.unsqueeze(0), weights=weights
    )
    log10_post_predictive = gmm.compute_log_posterior_predictive(
        measurements=X_batch.unsqueeze(2),
        given=Y_batch.unsqueeze(2),
        within_variances=torch.tensor([sigma2_w]),
        measurement_variances_extra=torch.tensor([sigma2_e]),
    )
    log10_prior_predictive = gmm.compute_log_prior_predictive(
        measurements=X_batch.unsqueeze(2),
        measurement_variances=torch.tensor([sigma2_w]),
        variances_extra=torch.tensor([sigma2_e]),
    )

    log10_post_predictive_ss = gmm.compute_log_posterior_predictive_specific_source(
        measurements=X_batch.unsqueeze(2),
        given=Y_batch.unsqueeze(2),
        within_variances=torch.tensor([sigma2_w]),
        measurement_variances_extra=torch.tensor([sigma2_e]),
    )

    # specific source llrs by 'fixed sigma_w' method
    llrs_ss = log10_post_predictive_ss - log10_prior_predictive

    assert torch.allclose(llr_numerators.to(device='cpu').unsqueeze(1), log10_post_predictive, rtol=1e-6)
    assert torch.allclose(llr_denominators.to(device='cpu').unsqueeze(1), log10_prior_predictive, rtol=1e-6)
    assert torch.allclose(intvar_ss_llrs.to(device='cpu').unsqueeze(1), llrs_ss, rtol=1e-6)


def test_numerical_output_cs_and_ss_numerators():
    torch.manual_seed(0)
    Y_batch = torch.tensor([[0.8, 2.0], [1.0, 0.0]])
    X_batch = torch.tensor([[1.2, 2.4, 2.0], [0.9, 1.8, 1.3]])

    kde_locations = torch.tensor([0.0, 2.0, 4.0])  # scalar means
    weights = torch.tensor([1.0, 1.0, 1.0])
    sigma2_e = torch.tensor([0.5])
    # initiate model with very large kde_variance, so that effectively the prior is flat so that SS_numerator and
    # CS_numerator should be nearly identical.
    model = IntegrateVarianceUncertainty(kde_locations, weights, kde_variance=torch.tensor(1e5))

    # initialize sigma2_ws
    n_sources = 351  # n_sources
    n_rep = 2

    # true hierarchical variances for testing. Mean inversegamma: beta/(alpha-1) mode: beta/(alpha+1)
    true_alpha = 4.0
    true_beta = 1.0

    # sample variances from inverse-gamma distribution
    # σ^2 ~ InvGamma(α,β)
    sig2 = torch.distributions.InverseGamma(true_alpha, true_beta).sample((n_sources,))

    data = torch.stack([torch.normal(0, sig2[i].sqrt(), size=(n_rep,)) for i in range(n_sources)])  # (n_sources, n_rep)

    alpha_est, beta_est, _ = fit_alpha_beta_personal_scipy(data.numpy())
    sigma2_w_sample = sample_sigma2_w(torch.tensor(alpha_est), torch.tensor(beta_est), sample_size=2000)

    # calculate both type of numerators
    llr_ss_numerators = model.get_ss_llr_numerators(
        X_batch, Y_batch, sigma2_e, torch.tensor(alpha_est), torch.tensor(beta_est), sample_size=2000
    )
    llr_cs_numerators = model.get_cs_llr_numerators(X_batch, Y_batch, sigma2_e, sigma2_w_sample)

    assert torch.allclose(llr_ss_numerators, llr_cs_numerators, rtol=5e-3)
