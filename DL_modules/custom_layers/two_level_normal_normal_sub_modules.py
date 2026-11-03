import torch
from torch import Tensor, nn

from DL_modules.train_validate_loops.utils import EmptyResultTLM
from DL_modules.utils import ResultTLM


class Variance(nn.Module):
    def __init__(self, n_axes: int, init_mode: float | str = 'randn'):
        super().__init__()
        if init_mode not in ["30", "1", "randn"] and not isinstance(init_mode, float):
            raise ValueError("Variance: init_mode must be either '30', '1', 'randn', or a float. ")
        self.init_mode = init_mode
        self.n_axes = n_axes
        self.log_var = self.initialize_log_var()
        """
        :param n_axes: number of variables
        :param init_mode: mode to initialize the variances with, either a number or a prespecified string.
        """

    def initialize_log_var(self):
        match self.init_mode:
            case "30":
                return nn.Parameter(
                    torch.full((self.n_axes,), float(torch.log10(torch.tensor(30))), dtype=torch.float64)
                )
            case "1":
                return nn.Parameter(torch.zeros(self.n_axes, dtype=torch.float64))
            case "randn":
                return nn.Parameter(torch.randn(self.n_axes, dtype=torch.float64))
            case _:
                return nn.Parameter(
                    torch.full((self.n_axes,), float(torch.log10(torch.tensor(self.init_mode))), dtype=torch.float64)
                )

    def forward(self):
        variance = torch.pow(10, self.log_var)
        return variance


class Mu(nn.Module):
    def __init__(self, n_axes: int, trainable_or_fixed_at_0: str):
        super().__init__()
        self.trainable_or_fixed_at_0 = trainable_or_fixed_at_0
        if self.trainable_or_fixed_at_0 not in ["trainable", "fixed_at_0"]:
            raise ValueError("Mu: trainable_or_fixed_at_0 must be either 'trainable' or 'fixed_at_0'.")

        self.n_axes = n_axes
        self.mu = self.initialize_mu()
        """
        :param n_axes: number of variables
        :param trainable_or_fixed_at_0: whether to use trainable mode or fix mu's at 0.
        """

    def initialize_mu(self):
        if self.trainable_or_fixed_at_0 == "trainable":
            return nn.Parameter(torch.randn(self.n_axes, dtype=torch.float64))
        if self.trainable_or_fixed_at_0 == "fixed_at_0":
            return torch.zeros(self.n_axes, dtype=torch.float64)

    def forward(self):
        return self.mu


class Q(nn.Module):
    """
    Get orthonormal matrix.
    """

    def __init__(self, n_axes: int):
        super().__init__()
        self.n_axes = n_axes
        self.random_square = self.initialize_q()
        """
        :param n_axes: number of variables
        """

    def initialize_q(self):
        square_matrix = nn.Parameter(torch.randn(self.n_axes, self.n_axes, dtype=torch.float64))
        return square_matrix

    def forward(self):
        q, _ = torch.linalg.qr(self.random_square)
        return q


class LLHNormal(nn.Module):
    """Calculate normal likelihood given vector of z's and variances."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(z: Tensor, variance: Tensor) -> Tensor:
        sd = torch.sqrt(variance)
        normal_dist = torch.distributions.Normal(0, sd)
        x = z * sd
        ln_pdf = normal_dist.log_prob(x)
        return ln_pdf / torch.log(torch.tensor(10))


class LLHT(nn.Module):
    """
    Calculate t-distributional likelihood given vector of z's and variances.
    See https://en.wikipedia.org/wiki/Conjugate_prior and look for a normal likelihood with unknown parameters mu and
    sigma.
    """

    def __init__(
        self,
        n_for_estimation: int,
        mu_prior: float = 0.0,
        nu: float = 0.0,
        alpha: float = 2.0,
        scaled_beta: float = 3.0,
    ):
        super().__init__()
        self.n_for_estimation = torch.tensor(n_for_estimation)
        self.mu_prior = torch.tensor(mu_prior)
        self.nu = torch.tensor(nu)
        self.alpha = torch.tensor(alpha)
        self.scaled_beta = torch.tensor(scaled_beta)

    """
    :param n_for_estimation: degrees of freedom of the t-distribution. Usually the number of SS training samples in
        the train set.
    :param mu_prior: the prior mean of the t-distribution
    :param nu: the prior sample size of the t-distribution
    :param alpha: 0.5 times the prior number of observations for the standard deviation of the t-distribution
    :param scaled_beta: the prior for the scaled mean square deviation (as compared to a mean square deviation of 1)

    Nb: I recommend to use alpha = 2 and beta = 3, since this is leads to a slightly conservative likelihood value for
    small sample sizes, since the mean of the gamma function with alpha=2 and beta=3 is equal to 3
    (so the scaled variance is larger than the empirical scaled variance of 1) and the mode is equal to 1
    (so for larger sample sizes it converges to the emperical scaled variance).
    """

    def forward(self, z: Tensor, variance: Tensor) -> Tensor:
        x_mean_est = 0
        # posterior :parameters of normal_inverse gamma distribution
        mu_post = (self.nu * self.mu_prior + self.n_for_estimation * x_mean_est) / (self.nu + self.n_for_estimation)
        nu_post = self.nu + self.n_for_estimation
        alpha_post = self.alpha + self.n_for_estimation / 2.0
        beta_post = (
            self.scaled_beta + self.n_for_estimation / 2.0
        ) * variance + self.nu * self.n_for_estimation / nu_post * torch.square(x_mean_est - self.mu_prior) / 2

        # posterior :parameters of T-distributed variable
        est_sd_for_t_distr = torch.sqrt(beta_post / alpha_post * (nu_post + 1) / nu_post)
        t_post = (z * torch.sqrt(variance) - mu_post) / est_sd_for_t_distr

        t_dist = torch.distributions.StudentT(2 * alpha_post)

        # don't forget to scale the t-distributed pdf with the variance, since t-distribution is defined for
        # variance of 1
        ln_t = t_dist.log_prob(t_post) - torch.log(est_sd_for_t_distr)

        return ln_t / torch.log(torch.tensor(10))


def merge_results_tlm(first: ResultTLM, second: ResultTLM) -> ResultTLM:
    # concatenate tlm predictions over batches.
    if first == EmptyResultTLM:
        return second
    else:
        for i, dictionary in enumerate(first):
            if i != 3:  # variances do not need to be merged
                for key in dictionary:
                    first[i][key] = torch.cat((dictionary[key], second[i][key]), dim=0)
    return first


def add_variances(*variances: torch.Tensor) -> torch.Tensor:
    # we assume independence of the error sources
    return torch.sum(torch.stack(variances), dim=0)
