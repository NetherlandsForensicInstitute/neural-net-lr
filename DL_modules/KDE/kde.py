import torch


def get_kde_bandwidth(data: torch.Tensor) -> torch.Tensor:
    sd = torch.std(data)
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)
    iqr = q3 - q1
    if sd == 0.0:
        raise ValueError('KDE: Silverman bandwidth cannot be calculated if standard deviation is 0')

    h = 0.9 * torch.pow(torch.tensor(len(data)), -0.2) * torch.min(sd, iqr / 1.34)
    return h


class OneDimKDE(torch.nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data.unsqueeze(1)
        self.bandwidth = get_kde_bandwidth(self.data)
        self.normal_dist = torch.distributions.Normal(0, self.bandwidth)

    """
        Kernel Density Estimation (KDE) for one-dimensional data. Uses bandwidth according to Silverman,page XXX

        :param data torch tensor with shape (n,)

    """

    def ln_prob(self, new_data: torch.Tensor) -> torch.Tensor:
        """Compute natural log transformed probabilities for new data.

        :param new_data. Torch tensor with shape (n)
        """
        new_data = new_data.unsqueeze(0)
        offsets = new_data - self.data
        ln_ps = self.normal_dist.log_prob(offsets)
        return torch.logsumexp(ln_ps, dim=0) - torch.log(torch.tensor(self.data.size(0)))

    def sample(self, n: int) -> torch.Tensor:
        """
        Draw a random sample of length n from the KDE distribution.

        """
        # Generate n random indices with replacement
        indices = torch.randint(0, self.data.size(0), (n,))

        # Sample from the original tensor using these indices
        means = self.data[indices]
        deviations = torch.randn(n) * self.bandwidth
        sample = means.squeeze(1) + deviations
        return sample
