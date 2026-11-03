import torch
from torch import Tensor, tensor
from torch.distributions import Exponential
from torch.optim import LBFGS

# ------------------------------------------------------------
# Compute sum of squares S_i from raw data
# ------------------------------------------------------------


def compute_sum_of_squares(data):
    """
    Compute sum of squares.

    data: tensor of shape (n_sources, m_reps)
    Returns tensor S of shape (n_sources,)
    """
    means = data.mean(dim=1, keepdim=True)
    S = ((data - means) ** 2).sum(dim=1)
    return S


# ------------------------------------------------------------
# Beta-prime log-pdf in Torch
#
# T ~ BetaPrime(p, q)
# pdf(t) = t^(p-1) (1+t)^(-(p+q)) / Beta(p,q)
# ------------------------------------------------------------


def betaprime_logpdf(t: Tensor, p: Tensor, q: Tensor) -> Tensor:
    """
    Log pdf of BetaPrime(p, q) at t>0 (Torch).
    Supports broadcasting.

    :param t: S/(2*beta), shape (n_sources,)
    :param p: (n_rep - 1) / 2, shape ()
    :param q: tensor of shape () (alpha of InverseGamma distribution)
    :return: logpdfs, shape (n_sources,)
    """
    return (p - 1) * torch.log(t) - (p + q) * torch.log1p(t) - (torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q))


# ------------------------------------------------------------
# Marginal log-pdf of S_i
#
# S = 2 * beta * T, where T ~ BetaPrime(nu/2, alpha)
# log p(S) = log p(T) - log(2*beta)
# ------------------------------------------------------------


def logpdf_S(S: Tensor, alpha: Tensor, beta: Tensor, nu: Tensor):
    """
    Compute marginal log-pdf of S_i: P(S_I|alpha, beta).

    S = 2 * beta * T, where T ~ BetaPrime(nu/2, alpha)
    log p(S) = log p(T) - log(2*beta)

    :param S: tensor of shape (n_sources,)
    :param alpha: alpha parameters of InverseGamma distribution, shape ()
    :param beta: beta parameter of InverseGamma distribution, shape ()
    :param nu: # repetitions -1, shape ()

    """
    t = S / (2 * beta)
    p = nu / 2

    return betaprime_logpdf(t, p, alpha) - torch.log(2 * beta)


# ------------------------------------------------------------
# Maximum Likelihood estimation
# ------------------------------------------------------------


def fit_alpha_beta_ML(
    data: Tensor, alpha0: float = 2.0, beta0: float | None = None, max_iter: int = 2000
) -> tuple[float, float, float]:
    """
    Fit ML estimate of alpha and beta.

    :param data: tensor of shape (n_sources, m_reps)
    :param alpha0: initial value for alpha of InverseGamma distribution
    :param beta0: initial value for beta of InverseGamma distribution
    :param max_iter: maximum number of iterations

    Return alpha_hat, beta_hat and mean within_variance
    """
    # compute S_i
    S = compute_sum_of_squares(data)
    nu = data.shape[1] - 1  # degrees of freedom

    S = S.to(torch.float64)
    nu = tensor(nu, dtype=torch.float64)

    # initialize parameters
    if beta0 is None:
        beta0 = S.mean() / 2.0

    alpha = torch.tensor(alpha0, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(beta0, dtype=torch.float64, requires_grad=True)

    # optimizer
    opt = LBFGS([alpha, beta], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        # enforce positivity via clamp transform inside loss
        alpha_pos = torch.clamp(alpha, min=1e-6)
        beta_pos = torch.clamp(beta, min=1e-6)

        ll = logpdf_S(S, alpha_pos, beta_pos, nu).sum()
        loss = -ll
        loss.backward()
        return loss

    opt.step(closure)

    # return positive reparameterized values
    return alpha.item(), beta.item(), float(torch.mean(S) / nu)


def fit_alpha_beta_MAP(
    data: Tensor, alpha0: float = 2.0, beta0: float | None = None, max_iter: int = 2000
) -> tuple[float, float, float]:
    """
    Fit MAP estimate of alpha and beta assuming exponential distribution for prior on beta.

    :param data: tensor of shape (n_sources, m_reps)
    :param alpha0: initial value for alpha of InverseGamma distribution
    :param beta0: initial value for beta of InverseGamma distribution
    :param max_iter: maximum number of iterations

    Return alpha_hat, beta_hat and mean within_variance
    """
    # compute S_i
    S = compute_sum_of_squares(data)
    nu = data.shape[1] - 1  # degrees of freedom

    S = S.to(torch.float64)
    nu = tensor(nu, dtype=torch.float64)

    # initialize parameters
    if beta0 is None:
        beta0 = S.mean() / 2.0

    _, beta_ml, _ = fit_alpha_beta_ML(data, alpha0, beta0, max_iter)

    alpha = torch.tensor(alpha0, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(beta0, dtype=torch.float64, requires_grad=True)

    # optimizer
    opt = LBFGS([alpha, beta], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        # enforce positivity via clamp transform inside loss
        alpha_pos = torch.clamp(alpha, min=1e-6)
        beta_pos = torch.clamp(beta, min=1e-6)

        ll = logpdf_S(S, alpha_pos, beta_pos, nu).sum()

        # prior distribution for beta: Exponential with rate = 0.5 * beta_ml.
        # This emphasizes smaller beta. Later alpha is scaled to the same mode as ML_estimates, so a larger variance is
        # obtained.
        rate = torch.tensor(0.5 * beta_ml).double()

        # prior distribution for alpha: Exponential with rate = 2. This emphasizes smaller alpha.
        prior_exponential = Exponential(rate)
        lpdf_prior_beta = prior_exponential.log_prob(beta_pos)

        loss = -(ll + lpdf_prior_beta)
        loss.backward()
        return loss

    opt.step(closure)

    return alpha.item(), beta.item(), float(torch.mean(S) / nu)


def fit_alpha_beta_personal(
    data: Tensor, alpha0: float = 2.0, beta0: float | bool = None, max_iter: int = 2000
) -> tuple[float, float, float]:
    """
    Estimate alpha and beta parameters of an InverseGamma distribution.

    Use a combination of ML and MAP estimates and fix the ML mode.

    P.S. This leads to heavier right tail, emphasizing larger within variance.
    :param data: tensor of shape (n_sources, m_reps)
    :param alpha0: initial value for alpha of InverseGamma distribution
    :param beta0: initial value for beta of InverseGamma distribution
    :param max_iter: maximum number of iterations

    Return alpha, beta and mean within_variance
    """
    alpha_ml, beta_ml, _ = fit_alpha_beta_ML(data, alpha0, beta0, max_iter)
    _, beta_map, mean_variance = fit_alpha_beta_MAP(data, alpha0, beta0, max_iter)

    # map alpha_est to distribution InverseGamma(alpha_est, beta_map) with same mode but larger spread as
    # InverseGamma(alpha_ml, beta_ml).
    alpha_est = (alpha_ml + 1) * beta_map / beta_ml - 1

    if alpha_est < 2 + 1e-6:
        # then alpha is too small, therefore distribution too wide. Clamp alpha to 2, but keep ML mode.
        print(
            f"Warning, alpha estimated at {alpha_est} which is smaller than 2 + 1e-6. Therefore InverseGamma "
            "distribution too wide. Clamping alpha to 2 and keeping mode."
        )
        beta_map = beta_ml / (alpha_ml + 1) * (3 + 1e-6)
        alpha_est = 2 + 1e-6
    return (alpha_est, beta_map, mean_variance)
