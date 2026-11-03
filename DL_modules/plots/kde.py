import matplotlib.pyplot as plt
import torch

from DL_modules.KDE import OneDimKDE


def plot_kde_distribution_from_normal_data(
    loc_normal: float, sd_normal: float, sample_size: int | None = None, data: torch.Tensor | None = None
):
    """Plot KDE distribution from normally distributed data (or data sampled from normal distribution if
    sample_size is specified).

    :param loc_normal: location of normal distribution
    :param sd_normal: sd of normal distribution
    :param sample_size: sample size of sample drawn from normal distribution with above parameters
    :param data: sample from normal distribution with above parameters
    """
    # Check that at least one parameter is specified
    if (sample_size is None and data is None) or (sample_size is not None and data is not None):
        raise ValueError("Either 'sample_size' or 'data' must be specified.")

    # Now you can work with param1 and param2
    if sample_size is not None:
        data = torch.randn(sample_size) * sd_normal + loc_normal

    sorted_indices = torch.argsort(data)
    data = data[sorted_indices]
    normal_dist = torch.distributions.Normal(loc_normal, sd_normal)
    ln_prob_norm = normal_dist.log_prob(data)

    kde = OneDimKDE(data)
    # calculate kde_log_probs on same data
    ln_prob_kde = kde.ln_prob(data)

    plt.plot(data, torch.exp(ln_prob_kde), label='KDE')
    plt.plot(data, torch.exp(ln_prob_norm), label='Normal distribution')
    plt.legend()
    plt.show()
