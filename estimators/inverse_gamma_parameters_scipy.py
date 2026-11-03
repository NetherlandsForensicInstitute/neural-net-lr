"""
Fit alpha and beta parameters of InverseGamma distribution.
Data: sum of squares from repeated measurements of n sources with n_rep repetitions.

Model:
Sum of squares ssq_i of a source given true variance is Chisquare distributed.
True variances of sources are InverseGamma distributed.

According to this model, the marginal likelihood of ssq_i given alpha and beta is a scaled BetaPrime distribution,
    with scale 2 * Beta..
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import betaprime, expon, invgamma


def compute_sum_of_squares(data: np.ndarray):
    """
    Compute the sum of squares.

    data: shape (n_sources, m_reps)
    Returns ssq_i = sum_j (y_ij - mean_i)^2
    """
    means = np.mean(data, axis=1, keepdims=True)
    ssq = np.sum((data - means) ** 2, axis=1)
    return ssq


# ------------------------------------------------------------
# Marginal log-pdf of ssq_i
#
# ssq = 2 * beta * T, where T ~ BetaPrime(nu/2, alpha)
# log p(ssq) = log p(T) - log(2*beta)
# ------------------------------------------------------------


def logpdf_ssq(s: np.ndarray, alpha: float, beta: float, nu: int):
    """
    Log marginal pdf of ssq given InvGamma(alpha,beta) prior on variance.
    ssq_i > 0.

    ssq ~ 2*beta * BetaPrime( nu/2, alpha )

    :param s: sum of squares per source. shape (n_sources, )
    :param alpha: alpha parameter of InverseGamma distribution.
    :param beta: beta parameter of InverseGamma distribution.
    :param nu: number of repetitions per source -1

    Return: log probability of scaled betaprime distribution (scale: 2 * beta)
    """
    t = s / (2 * beta)
    p = nu / 2
    q = alpha

    # log p_ssq(s) = log p_T(t) - log(2*beta)
    return betaprime.logpdf(t, p, q) - np.log(2 * beta)


def convert_to_alpha_beta(theta_alpha: float, theta_beta: float) -> tuple[float, float]:
    """Convert thetas back to alpha and beta.

    :param theta_alpha: reparameterized alpha parameter of InverseGamma distribution
    :param theta_beta: reparameterized beta parameter of InverseGamma distribution
    :return alpha, beta: parameters of InverseGamma distribution
    """
    # using this, we constrain alpha to values larger than 2, enforcing finite variance of the InverseGamma distribution
    alpha = 2 + np.exp(theta_alpha)
    # using this, we constrain beta to positive values
    beta = np.exp(theta_beta)

    return alpha, beta


def neg_log_likelihood_theta(theta: tuple[float, float], ssq: np.ndarray, nu: int):
    """
    Compute negative of log likelihood.

    To get a Maximum A Posteriori estimate, you need to optimize this quantity for alpha and beta.
    Explanation: Bayes Rule: Log Posterior = Log likelihood (ssq | alpha, beta) + log Prior (alpha, beta) - Log P (ssq).
    Since log P (ssq) is constant you do not need to compute it.

    :param theta: tuple of (theta_alpha, theta_beta), alpha beta are the parameters of the InverseGamma distribution
        i'm trying to fit. Both floats.
    :param ssq: shape (n_sources, )
    :param nu: n_rep - 1
    :return: neg log likelihood, float
    """
    theta_alpha, theta_beta = theta
    alpha, beta = convert_to_alpha_beta(theta_alpha, theta_beta)

    ll = np.sum(logpdf_ssq(ssq, alpha, beta, nu))

    return -ll


def neg_log_likelihood_map_theta(theta: tuple[float, float], ssq: np.ndarray, nu: int, beta_ml: float):
    """Compute negative of sum of log likelihood and log prior probability for beta.

    To get a Maximum A Posteriori estimate, you need to optimize this quantity for alpha and beta.
    Explanation: Bayes Rule: Log Posterior = Log likelihood (ssq | alpha, beta) + log Prior (alpha, beta) - Log P (ssq).
    Since log P (ssq) is constant you do not need to compute it.

    :param theta: tuple of (theta_alpha, theta_beta), alpha beta are the parameters of the InverseGamma distribution
        i'm trying to fit. Both floats.
    :param ssq: shape (n_sources, )
    :param nu: n_rep - 1
    """
    theta_alpha, theta_beta = theta
    alpha, beta = convert_to_alpha_beta(theta_alpha, theta_beta)

    ll = np.sum(logpdf_ssq(ssq, alpha, beta, nu))

    # prior for beta: exponential with rate 0.5 * beta_ml
    rate = 0.5 * beta_ml
    lpdf_prior_beta = expon.logpdf(beta, scale=1 / rate)

    return -(ll + lpdf_prior_beta)


def alpha_beta_to_thetas(alpha: float, beta: float) -> tuple[float, float]:
    """Convert alpha and beta to theta_alpha and theta_beta.
    Note that in the function convert_to_alpha_beta alpha is constrained to > 2 and beta to > 0.

    :param alpha: alpha parameter of InverseGamma distribution
    :param beta: beta parameter of InverseGamma distribution
    :return tuple: theta_alpha, theta_beta
    """
    theta = (np.log(alpha - 2), np.log(beta))
    return theta


def fit_alpha_beta_ml_scipy_theta(data: np.ndarray) -> tuple[float, float, float]:
    """
    Fit ML estimate using theta reparameterization.
    """
    ssq = compute_sum_of_squares(data)
    nu = data.shape[1] - 1

    # initialize initial attempt
    attempt_number = 0
    max_attempts = 10
    res = None
    while attempt_number < max_attempts and (res is None or not res.success):
        if attempt_number == max_attempts - 1:
            raise RuntimeError(f"MAP optimization failed for max of {max_attempts} attempts.")

        # initial guess in theta space for attempt
        alpha0 = 4.0 + attempt_number * 0.5
        beta0 = np.mean(ssq) / (nu * (alpha0 - 1))
        theta0 = alpha_beta_to_thetas(alpha0, beta0)

        res = minimize(neg_log_likelihood_theta, theta0, args=(ssq, nu), method='L-BFGS-B')

        if not res.success:
            print(
                "MLE optimization failed: "
                + res.message
                + f"at {attempt_number + 1}th of {max_attempts} attempts. Try again"
            )

        attempt_number += 1

    alpha_hat, beta_hat = convert_to_alpha_beta(res.x[0], res.x[1])

    return float(alpha_hat), float(beta_hat), float(np.mean(ssq) / nu)


def fit_alpha_beta_map_scipy_theta(data: np.ndarray) -> tuple[float, float, float]:
    """
    Fit MAP estimate using theta reparameterization.
    """
    ssq = compute_sum_of_squares(data)
    nu = data.shape[1] - 1

    # first, ML estimate for beta (needed for MAP prior)
    _, beta_ml, _ = fit_alpha_beta_ml_scipy_theta(data)

    # initialize initial attempt
    attempt_number = 0
    max_attempts = 10
    res = None
    while attempt_number < max_attempts and (res is None or not res.success):
        if attempt_number == max_attempts - 1:
            raise RuntimeError(f"MAP optimization failed for max of {max_attempts} attempts.")

        # initial guess in theta space for attempt
        alpha0 = 4.0 + attempt_number * 0.5
        beta0 = np.mean(ssq) / (nu * (alpha0 - 1))
        theta0 = alpha_beta_to_thetas(alpha0, beta0)

        res = minimize(neg_log_likelihood_map_theta, theta0, args=(ssq, nu, beta_ml), method='L-BFGS-B')

        if not res.success:
            print(
                "MAP optimization failed: "
                + res.message
                + f"at {attempt_number + 1}th of {max_attempts} attempts.  Try again"
            )

        attempt_number += 1

    alpha_hat, beta_hat = convert_to_alpha_beta(res.x[0], res.x[1])

    return float(alpha_hat), float(beta_hat), float(np.mean(ssq) / nu)


def fit_alpha_beta_personal_scipy(data: np.ndarray) -> tuple[float, float, float]:
    """
    Fit ML estimate of alpha and beta.

    :param data: np.array (n_sources, m_reps)

    Return alpha_personal, beta_personal and mean within_variance, all floats
    """
    alpha_ml, beta_ml, _ = fit_alpha_beta_ml_scipy_theta(data)
    _, beta_map, mean_within_variance = fit_alpha_beta_map_scipy_theta(data)

    # map alpha_est to distribution InverseGamma(alpha_est, beta_map) with same mode but larger spread as
    # InverseGamma(alpha_ml, beta_ml).
    alpha_est = (alpha_ml + 1) * beta_map / beta_ml - 1
    if alpha_est < 2 + 1e-6:
        # then alpha is too small, therefore distribution too wide. Clamp alpha to 2 + 1e-6, but keep ML mode.
        print(
            f"Warning, alpha estimated at {alpha_est} which is smaller than 2 + 1e-6. Therefore InverseGamma "
            "distribution too wide. Clamping alpha to 2 + 1e-6 and keeping mode."
        )
        beta_map = beta_ml / (alpha_ml + 1) * (3 + 1e-6)
        alpha_est = 2 + 1e-6

    return alpha_est, beta_map, mean_within_variance


def plot_inversegammas(
    alpha_ml: float,
    beta_ml: float,
    alpha_personal: float,
    beta_personal: float,
    alpha_true: float,
    beta_true: float,
    mean_within_variance: float,
):
    """
    Plot InverseGamma distributions with given alpha and beta parameters.

    :param alpha_ml: alpha maximum likelhood estimate
    :param beta_ml: beta likelhood estimate
    :param alpha_personal: alpha personal estimate (thicker tails)
    :param beta_personal: beta personal estimate (thicker tails)
    :param alpha_true: true alpha
    :param beta_true: true beta
    :param mean_within_variance: mean within variance
    """
    # --- x-range for plotting ---
    x = np.linspace(0, 2, 500)

    # --- PDF values ---
    pdf1 = invgamma.pdf(x, a=alpha_true, scale=beta_true)
    pdf2 = invgamma.pdf(x, a=alpha_personal, scale=beta_personal)
    pdf3 = invgamma.pdf(x, a=alpha_ml, scale=beta_ml)

    # --- Plot ---
    plt.figure(figsize=(7, 4))
    plt.plot(x, pdf1, label=f"InverseGammaTrue(alpha={alpha_true}, beta={beta_true})", linewidth=2, color='black')
    plt.plot(
        x,
        pdf2,
        label=f"InverseGammaPersonal(alpha={round(alpha_personal, 2)}, beta={round(beta_personal, 2)})",
        linewidth=2,
        color='red',
    )
    plt.plot(
        x,
        pdf3,
        label=f"InverseGammaML(alpha={round(alpha_ml, 2)}, beta={round(beta_ml, 2)})",
        linewidth=2,
        color='green',
    )
    plt.vlines(
        mean_within_variance,
        ymin=0,
        ymax=4,
        color='black',
        linestyle='dashed',
        label=f"mean within_variance at {round(mean_within_variance, 2)}",
    )
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Three Inverse-Gamma Distributions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
