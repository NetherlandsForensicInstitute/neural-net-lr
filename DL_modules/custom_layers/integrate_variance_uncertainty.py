"""
Below is a PyTorch batch implementation of BF for integrating out sigm2_w uncertainty, for one feature.

It assumes:

X_batch and Y_batch are torch tensors with repeated measurements for each source.

L KDE components (locations + weights), sigma2_w prior, kde_variance, sigma2_extra: posthoc correction of variance
for X_mean, for numerator and for denominator.

The output is batch of posterior predictive densities: p_X_given_Y[b] for each source b and p_X[b] for each source b,
and its Bayes factor.

Input shapes:

X_batch (B, n_x)
Y_batch (B, n_y)

Output:

ln_p_X_given_Y[b] is the ln posterior predictive density for the b-th source.
ln_p_X[b] is the ln prior predictive density for the b-th source

ln_p_X_given_Y[b] is its ln-density (numerically more stable).

Memory:

Arrays are (M,B,L) — for large B or L, you may need to process in blocks over B.
"""

import torch
from torch.distributions import InverseGamma, MultivariateNormal

from DL_modules.KDE import get_kde_bandwidth


class IntegrateVarianceUncertainty:
    def __init__(self, kde_locations: torch.Tensor, weights: torch.Tensor, kde_variance: torch.Tensor | None = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.kde_locations = kde_locations  # (L,)
        self.ln_weights_normalized = self._set_ln_weights(weights)  # (L,)
        self.kde_variance = kde_variance if kde_variance else get_kde_bandwidth(kde_locations) ** 2  # (1,)

        """
        :param locations: (L,)
        :param weights: (L,)
        :param kde_variance: (1,)
        """

    ### method only used in init
    def _set_ln_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Init class by setting ln weights."""
        if weights.dim() != 1:
            raise ValueError("weights must have 1 dimension")
        if len(weights) != len(self.kde_locations):
            raise ValueError("the number of weights and the number of locations must be equal")
        weights_normalized = weights / torch.sum(weights)
        return torch.log(weights_normalized).to(self.device)

    def get_cs_llr_numerators(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        sigma2_extra: torch.Tensor,
        sigma2_w_sample: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Calculate common source posterior predictive P(X|Y, sigma2_extra, h^2).

        i.e. Calculate log10 P(X_batch[b] given Y_batch[b]) for each b in batch. Integrate out uncertainty in
        self.sigma2_w (by numerical means; by sampling from the prior distribution of sigma2_w) and uncertainty in
        mean_object (in closed form).

        :param X_batch: (B, n_x)
        :param Y_batch: (B, n_y)
        :param sigma2_extra: (1,), posthoc correction of variance of X_mean
        :param sigma2_w_sample: (sample_size,), sample of sigma_2w from InverseGamma distribution
        """
        if sigma2_w_sample is None:
            raise ValueError("sigma2_w_sample not present. Use method: 'sample_sigma2_w' first")

        # calculate joint ln probabilities, X gets sigma2_extra variance term added to the variance of its mean
        ln_pXYs = self._ln_joint_probabilities(X_batch, Y_batch, sigma2_extra, sigma2_w_sample)  # (B, M, L)

        # calculate marginal ln probabilities of Y, we do not want the posthoc correction with sigma2_extra here since
        # we do not want Bayesian updates for this parameter
        ln_pYs = self._ln_marginal_probabilities(Y_batch, sigma2_w_sample)  # (B, M, L)

        # get sum_ts (take average over KDE locations)
        ln_sum_T_XY = self.get_sum_ts(ln_pXYs)  # (B, M)
        ln_sum_T_Y = self.get_sum_ts(ln_pYs)  # (B, M)

        # average over sigma_w draws (integrate out uncertainty in Sigma2_w)
        ln_P_tilde_XY = self.average_sigma_w(ln_sum_T_XY)  # (B,)
        ln_P_tilde_Y = self.average_sigma_w(ln_sum_T_Y)  # (B,)

        return (ln_P_tilde_XY - ln_P_tilde_Y) / torch.log(torch.tensor(10.0))  # (B,) numerator of CS BF, log10 scale

    def get_ss_llr_numerators(
        self,
        X_batch: torch.Tensor,
        Y_batch: torch.Tensor,
        sigma2_extra: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        sample_size: int,
    ) -> torch.Tensor:
        """
        Calculate specific source posterior predictive P(X|Y, sigma2_extra).

        i.e. Calculate log10 P(X_batch[b] given Y_batch[b]) for each b in batch. Integrate out uncertainty in
        self.sigma2_w (by numerical means; by sampling from the posterior distribution of sigma2_w) and uncertainty in
        mean_object (in closed form, and we assume a flat prior).

        :param X_batch: (B, n_x)
        :param Y_batch: (B, n_y)
        :param sigma2_extra: (1,), posthoc correction of variance of X_mean
        :param alpha: (1,) alpha parameter of InverseGamma distribution
        :param beta: (1,) beta parameter of InverseGamma distribution
        :param sample_size: (1,) the number of sigma2_w drawn

        return: ss_llr_numerators: (B,)
        """
        device = self.device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        sigma2_extra = sigma2_extra.to(device)

        # Calculate sufficient statistics for Y_batch: n_y, means and sum_of_squares
        n_y = torch.tensor(Y_batch.shape[1])
        means_y, ssqs_y = self.get_mean_ssq(Y_batch)

        # get posterior InverseGamma parameters
        prior_alpha = alpha
        prior_beta = beta
        posterior_alphas, posterior_betas = self.get_posterior_alpha_beta(ssqs_y, n_y, prior_alpha, prior_beta)

        # calculate llr_numerator per row
        ss_llr_numerators = []
        for X_instance, mean_y, posterior_alpha, posterior_beta in zip(
            X_batch, means_y, posterior_alphas, posterior_betas
        ):
            sigma2_w_sample = sample_sigma2_w(posterior_alpha, posterior_beta, sample_size)
            ss_ln_numerator = self._ln_ss_prob(X_instance, mean_y, n_y, sigma2_extra, sigma2_w_sample, self.device)
            ss_llr_numerators.append(ss_ln_numerator)

        return torch.tensor(ss_llr_numerators).to(device) / torch.log(
            torch.tensor(10.0)
        )  # log10 numerator of SS BF, shape (B,)

    def get_llr_denominators(
        self, X_batch: torch.Tensor, sigma2_extra: torch.Tensor, sigma2_w_sample: torch.Tensor = None
    ):
        """
        Calculate log10 P(X_batch[b]) for each b in batch. Integrate out uncertainty in
        self.sigma2_w (by numerical means; by sampling from the prior distribution of sigma2_w) and uncertainty in
        mean_object (in closed form).

        :param X_batch: (B, n_x)
        :param sigma2_extra (1,), posthoc correction of variance of X_mean
        :param sigma2_w_sample: (,), posthoc correction of sigma2_w

        return llr_denominators: (B,)
        """
        if sigma2_w_sample is None:
            raise ValueError("sigma2_w_sample not present. Use method: 'sample_sigma2_w' first")

        ln_pXs = self._ln_marginal_probabilities(X_batch, sigma2_w_sample, sigma2_extra)  # (B, M, L)

        # get sum_ts (take average over KDE locations)
        ln_sum_T_X = self.get_sum_ts(ln_pXs)  # (B, M)

        # average over sigma_w draws (integrate out uncertainty in Sigma2_w)
        ln_P_tilde_X = self.average_sigma_w(ln_sum_T_X)  # (B,)

        return ln_P_tilde_X / torch.log(torch.tensor(10.0))  # (B, ), denominator of BF, log10 scale

    def get_sum_ts(self, ln_vals):
        """
        Sum over (in this use-case) n KDE-locations and weight by 1/n.
        """
        location_dim = 2
        ln_weights_normalized = self.ln_weights_normalized[None, None, :]
        return torch.logsumexp(ln_weights_normalized + ln_vals, dim=location_dim)

    @staticmethod
    def get_mean_ssq(Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate means and sum of squares per row of Y.

        :param Y: (B, n_y)

        return: mean: (B,), sum_of_squares: (B,)
        """
        means = Y.mean(dim=1, keepdim=True)
        S = ((Y - means) ** 2).sum(dim=1)
        return means.squeeze(1), S

    @staticmethod
    def average_sigma_w(ln_vals):
        """
        Integrate over (in this use-case) sigma2_w^2 by averaging in ln-space.
        """
        sigma_w_dim = 1
        return ln_mean_exp(ln_vals, dim=sigma_w_dim)

    @staticmethod
    def get_posterior_alpha_beta(
        ssq_y: torch.Tensor, n_y: torch.Tensor, prior_alphas: torch.Tensor, prior_betas: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assuming inversegamma prior and normal likelihood with flat prior on mean, update parameters for conjugate
        posterior InverseGamma.

        param: ssq_y: (B,)
        param: n_y: (1,)

        return: posterior_alphas (B,), posterior_betas (B,)
        """
        B = len(ssq_y)
        if torch.sum(prior_alphas) == 0:  # then also not self.prior_beta
            raise ValueError("prior_alpha and prior_beta not set. Run function fit_alpha_beta_personal_scipy")

        posterior_alphas, posterior_betas = prior_alphas + (n_y - 1) / 2, prior_betas + ssq_y / 2
        return posterior_alphas.repeat(B), posterior_betas

    @staticmethod
    def _ln_ss_prob(
        X_instance: torch.Tensor,
        mean_y: torch.Tensor,
        n_y: torch.Tensor,
        sigma2_extra: torch.Tensor,
        sigma2_w_sample: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Calculate a ln_ss_prob for X_instance given mean_y, averaged over sigma2_w_sample.

        :param X_instance: (n_x,)
        :param mean_y: (1,)
        :param n_y: (1,)
        :param sigma2_extra: (1,)
        :param sigma2_w_sample: (M,)
        :param device: device

        :return: ln_ss_prob: (1,)
        """
        n_x = X_instance.size(0)
        M = sigma2_w_sample.shape[0]

        # set up MVN covariance structure
        eye = torch.eye(n_x, device=device)
        J = torch.ones((n_x, n_x), device=device)
        sigma2_w_exp = sigma2_w_sample.view(M, 1, 1)
        Sigma = (sigma2_w_exp + n_x * sigma2_extra) * eye + sigma2_w_exp / n_y * J  # (M, n_x, n_x)

        # Broadcast shapes:
        # X_instance to: (1, 1, n_x)
        # Sigma: (M, 1, n_x, n_x)
        # mean_y_expanded: (1, 1, n_x) -> becomes loc = mean_y * ones(n_x)
        X_instance_expanded = X_instance.view(1, 1, n_x)
        mean_y_expanded = mean_y.view(1, 1, 1) * torch.ones(n_x, device=device)  # (1, 1, n_x)
        sigma2_exp = Sigma[:, None, :, :]  # (M, 1, n_x, n_x)

        # --- MultivariateNormal with broadcasting ---
        mvn_distribution = MultivariateNormal(loc=mean_y_expanded, covariance_matrix=sigma2_exp)

        # Log probabilities for all M
        ln_probs = mvn_distribution.log_prob(X_instance_expanded)  # shape (M)

        return ln_mean_exp(ln_probs, dim=0)  # shape (1,)

    def _ln_joint_probabilities(
        self, X_batch: torch.Tensor, Y_batch: torch.Tensor, sigma2_extra: torch.Tensor, sigma2_w_sample: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ln p(X_batch[b], Y_batch[b] | self.sigma2_w[m], self.kde_locations[l]) for all combinations b, m and l,
        assuming X_batch[b] and Y_batch[b] are measurements from the same object.

        Args:
            X_batch: (B, n_x)
            Y_batch: (B, n_y)
            sigma2_e: torch.Tensor[float] (1,)
            sigma2_w_sample: (M,)
            self.kde_locations: (L,)  scalar mean values
            self.kde_variance: scalar shared covariance term. torch.Tensor[float] (1,)

        Returns:
            ln_probs: (B, M, L)
        """
        kde_locations = self.kde_locations
        kde_variance = self.kde_variance
        device = self.device
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        sigma2_extra = sigma2_extra.to(device)

        B, n_x = X_batch.shape
        _, n_y = Y_batch.shape
        M = sigma2_w_sample.shape[0]
        L = kde_locations.shape[0]
        D = n_y + n_x

        # --- Build covariance structure ---
        eye_y = torch.eye(n_y, device=device)
        eye_x = torch.eye(n_x, device=device)
        J_y = torch.ones((n_y, n_y), device=device)
        J_x = torch.ones((n_x, n_x), device=device)
        J_yx = torch.ones((n_y, n_x), device=device)

        sigma2_w_exp = sigma2_w_sample.view(M, 1, 1)

        # Covariance submatrices
        Sigma_YY = sigma2_w_exp * eye_y + kde_variance * J_y
        Sigma_XX = (sigma2_w_exp + n_x * sigma2_extra) * eye_x + kde_variance * J_x
        Sigma_XY = kde_variance * J_yx.expand(M, n_y, n_x)

        # Combine blocks → (M, D, D)
        # cat colums
        top = torch.cat([Sigma_YY, Sigma_XY], dim=2)
        # torch.transpose: swap dimensions 2 and 1 -> 2 becomes 1 and 1 becomes 2.
        bottom = torch.cat([Sigma_XY.transpose(2, 1), Sigma_XX], dim=2)
        # cat rows
        Sigma = torch.cat([top, bottom], dim=1)

        # --- Build data and mean ---
        # cat repetitions
        Z = torch.cat([Y_batch, X_batch], dim=1)  # (B, D)
        kde_locations = kde_locations.to(device)

        # Broadcast shapes:
        # Z: (B, 1, 1, D)
        # Sigma: (1, M, 1, D, D)
        # mu_batch: (1, 1, L, 1) -> becomes loc = mu * ones(D)
        kde_locations_expanded = kde_locations.view(1, 1, L, 1) * torch.ones(D, device=device)  # (1, 1, L, D)

        Z_exp = Z[:, None, None, :]  # (B, 1, 1, D)
        sigma2_exp = Sigma[None, :, None, :, :]  # (1, M, 1, D, D)

        # --- MultivariateNormal with broadcasting ---
        mvn_distribution = MultivariateNormal(loc=kde_locations_expanded, covariance_matrix=sigma2_exp)

        # Log probabilities for all (B, M, L)
        ln_probs = mvn_distribution.log_prob(Z_exp)
        return ln_probs  # shape (B, M, L)

    def _ln_marginal_probabilities(
        self, meas_batch: torch.Tensor, sigma2_w_sample: torch.Tensor, sigma2_extra: torch.Tensor = torch.tensor(0)
    ) -> torch.Tensor:
        """
        Compute ln p(meas_batch[b] | sigma2_w[m], kde_locations[l]) for all combinations of b, m, l.

        Args:
            meas_batch: (B, n_rep) meas_batch[b] are n_rep repeated measurements on the same objects
            sigma2_w_sample: (M,)  sample of within variances
            sigma2_extra: (1,)
            self.kde_locations: (L,)
            self.kde_variance: scalar shared covariance term. torch.Tensor[float] (1,)



        Returns:
            ln_probs: (B, M, L)
        """
        kde_locations = self.kde_locations
        kde_variance = self.kde_variance
        device = self.device
        meas_batch = meas_batch.to(device)  # (B, n_rep)
        kde_locations = kde_locations.to(device)
        sigma2_extra = sigma2_extra.to(device)

        B, n_rep = meas_batch.shape
        M = sigma2_w_sample.shape[0]
        L = kde_locations.shape[0]

        # --- Covariance matrices ---
        eye = torch.eye(n_rep, device=device)
        J = torch.ones((n_rep, n_rep), device=device)

        sigma2_w_exp = sigma2_w_sample.view(M, 1, 1)
        Sigma = (sigma2_w_exp + n_rep * sigma2_extra) * eye + kde_variance * J  # (M, n_rep, n_rep)

        # Broadcast shapes:
        # meas_batch_exp: (B, 1, 1, n_rep)
        # Sigma2_exp: (1, M, 1, n_rep, n_rep)
        # kde_locations_exp: (1, 1, L, 1) -> will broadcast to (1, 1, L, n_rep)
        kde_locations_exp = kde_locations.view(1, 1, L, 1) * torch.ones(n_rep, device=device)

        meas_batch_exp = meas_batch[:, None, None, :]  # (B, 1, 1, n_rep)
        sigma2_exp = Sigma[None, :, None, :, :]  # (1, M, 1, n_rep, n_rep)

        # --- Distribution with full broadcasting ---
        mvn_distribution = MultivariateNormal(loc=kde_locations_exp, covariance_matrix=sigma2_exp)

        ln_probs = mvn_distribution.log_prob(meas_batch_exp)  # (B, M, L)
        return ln_probs


def ln_mean_exp(ln_vals, dim):
    n = torch.tensor(ln_vals.size(dim=dim))
    return torch.logsumexp(ln_vals, dim=dim) - torch.log(n)


### precondition for LLR calculation
def sample_sigma2_w(
    alpha_w: torch.Tensor,
    beta_w: torch.Tensor,
    sample_size: int,
):
    """
    Get Monte Carlo sample for sigma_w^2.

    :param alpha_w: (1,) alpha parameter of InverseGamma distribution
    :param beta_w: (1,) beta parameter of InverseGamma distribution
    :param sample_size: (1,) the number of sigma2_w drawn
    :param return_sample: (bool) whether to return samples or else store in self

    :return: (sample_size, ) sample of sigma2_w from InverseGamma distribution
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alpha_w = alpha_w.to(device)
    beta_w = beta_w.to(device)
    inv_gamma_w = InverseGamma(alpha_w, beta_w)
    sample = inv_gamma_w.sample((sample_size,)).to(device)

    return sample
