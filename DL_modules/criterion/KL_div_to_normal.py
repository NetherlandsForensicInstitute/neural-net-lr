import torch

from DL_modules.KDE import OneDimKDE

from ..custom_layers.top_layers import ResultTLM
from .cllr import CLLR, CLLREvaluate, CLLRTrainDynamicallyConcentrated
from .pav_loss import PAVLossConcentrated


class KLDivergenceLossOneDim(torch.nn.Module):
    def __init__(self, sd_normal_dist: float = 1.0, sample_size: str | int | None = "use_data"):
        super().__init__()
        self.sd_normal_dist = sd_normal_dist
        self.sample_size = sample_size

    """
    Class to compute a KL-divergence loss of a sample of points when compared to a normal distribution
        (the reference distribution).
    The population distribution of the sample of points is empirically approxiated by a KDE distribution.

    :param sd_normal_dist: standard deviation of the reference distribution.
    :param sample_size: if sample_size=="use_data", in the forward pass the sample data is used twice: to build the KDE
        as well as to compute the KL-divergence. If sample_size == n, The data is used to build the KDE, and the
        KL-divergence is calculated for a random sample of sample_size n from the KDE.
    """

    def forward(self, data: torch.tensor) -> torch.tensor:
        one_dim_kde_instance = OneDimKDE(data)
        sample = one_dim_kde_instance.sample(self.sample_size) if self.sample_size != "use_data" else data.detach()
        ln_probs_kde = one_dim_kde_instance.ln_prob(sample)
        normal_dist = torch.distributions.Normal(loc=0.0, scale=self.sd_normal_dist)
        ln_probs_st_norm = normal_dist.log_prob(sample)
        dif = ln_probs_kde - ln_probs_st_norm

        # negative values are allowed since we evaluate KL divergence based on a sample
        return torch.mean(dif).unsqueeze(0)


class KLDivergenceConcentrated(torch.nn.Module):
    """
    Select axes where cllr is smaller than cllr_threshold (the 'focal' axes).

    Training mode: calculate the KL-divergence under H0 and H1 for the focal axes and the sum_z of the focal axes and
    return the mean in training mode
    Evaluation mode: return KL-divergence per axis, for sum_z of all axes and sum_z of focal axes
    """

    def __init__(
        self,
        fr_hard_negs: float,
        cllr_threshold: float = 0.95,
        n_axes: str | int = "all",
        sd_normal_dist: float = 1.0,
        sample_size: str | int | None = "use_data",
        mode: str | None = "train",
    ):
        super().__init__()
        self.n_axes = n_axes
        self.cllr_threshold = cllr_threshold
        self.cllr_evaluate = CLLREvaluate(fr_hard_negs)
        self.kldivergence_one_column = KLDivergenceLossOneDim(sd_normal_dist, sample_size)
        if mode not in ['train', 'evaluate']:
            raise ValueError("mode must be 'train' or 'evaluate'")
        self.mode = mode

    def forward(self, predictions: ResultTLM, labels: torch.Tensor) -> torch.Tensor | list[list[torch.tensor]]:
        # Select SS samples for which normality assumptions should hold when using a TLM model.
        z_h0 = predictions.z["z_0"][labels == 1]
        z_h1 = predictions.z["z_1"][labels == 1]

        # Add colsum/sqrt(n_columns), in order to aim for uncorrelated (independent) errors for all focal axes
        sum_z0_all_axes = self.sum_z(z_h0)
        sum_z1_all_axes = self.sum_z(z_h1)

        # Select focal axes based on cllr
        prediction_llrs = predictions.log_norm['llr_norm']
        cllrs = self.cllr_evaluate(prediction_llrs, labels)[:-1]
        indices_focus = torch.nonzero(cllrs <= self.cllr_threshold, as_tuple=True)[0]
        if len(indices_focus) == 0:
            # then, there were no axes to focus on and we calculate the sum_z_focus(z_hx) for all axes.
            indices_focus = torch.arange(0, 18)

        z_focus_h0 = z_h0[:, indices_focus]
        z_focus_h1 = z_h1[:, indices_focus]

        # Add colsum/sqrt(n_columns), in order to aim for uncorrelated (independent) errors for all focal axes
        z_h0_concat = torch.cat((z_h0, sum_z0_all_axes, self.sum_z(z_focus_h0)), dim=1)
        # same for h1
        z_h1_concat = torch.cat((z_h1, sum_z1_all_axes, self.sum_z(z_focus_h1)), dim=1)
        kl_div_0 = torch.empty((0,))
        kl_div_1 = torch.empty((0,))
        for i in range(z_h0.shape[1] + 2):
            kl_div_0 = torch.cat((kl_div_0, self.kldivergence_one_column(z_h0_concat[:, i])), dim=0)
            kl_div_1 = torch.cat((kl_div_1, self.kldivergence_one_column(z_h1_concat[:, i])), dim=0)
        if self.mode == "train":
            index_to_remove = -2  # remove KL-div for sum_z0_all_axes and sum_z1_all_axes
            kl_div_0 = torch.cat((kl_div_0[:index_to_remove], kl_div_0[index_to_remove + 1 :]))
            kl_div_1 = torch.cat((kl_div_1[:index_to_remove], kl_div_1[index_to_remove + 1 :]))
            ind_focus_and_sum = torch.cat((indices_focus, torch.tensor([len(kl_div_0) - 1])))
            sum_kl_div = torch.sum(kl_div_0[ind_focus_and_sum] + kl_div_1[ind_focus_and_sum])
            kl_div = sum_kl_div / (torch.tensor(2 * (len(indices_focus) + 1)))

            return kl_div
        if self.mode == "evaluate":
            return torch.stack((kl_div_0, kl_div_1), dim=0), indices_focus

    @staticmethod
    def sum_z(z: torch.Tensor) -> torch.Tensor:
        # Compute col-wise sum for specified number of axes
        col_sum = torch.sum(z, dim=1)
        col_sum = col_sum / torch.sqrt(torch.tensor(z.shape[1]))
        return col_sum.unsqueeze(1)


class KLDivergenceAndCLLR(torch.nn.Module):
    def __init__(
        self,
        kldiv_instance: KLDivergenceConcentrated,
        cllr_instance: CLLR | CLLRTrainDynamicallyConcentrated,
        kldiv_weight: float,
    ):
        super().__init__()
        self.kldiv_instance = kldiv_instance
        self.cllr_instance = cllr_instance
        self.kldiv_weight = kldiv_weight

    """
        Weighted average of KL_divergence and Cllr losses.

        :param kldiv_instance: KLDivergenceConcentrated instance
        :param cllr_instance: CLLR instance in a training mode or CLLRTrainDynamicallyConcentrated instance
        :param kldiv_weight: weight of kl_div loss with respect to cllr (= 1).
    """

    def forward(self, predictions: ResultTLM, labels: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        llrs = predictions.log_norm["llr_norm"]
        cllr_loss = self.cllr_instance(llrs, labels)
        kldiv = self.kldiv_instance(predictions, labels)
        return cllr_loss + self.kldiv_weight * kldiv


class KLDivergenceAndCLLRAndPAVLoss(torch.nn.Module):
    def __init__(
        self,
        kldiv_instance: KLDivergenceConcentrated,
        cllr_instance: CLLR | CLLRTrainDynamicallyConcentrated,
        pav_loss_instance: PAVLossConcentrated,
        kldiv_weight: float = 0.1,
        pav_weight: float = 0.001,
    ):
        super().__init__()
        self.kldiv_instance = kldiv_instance
        self.cllr_instance = cllr_instance
        self.pav_loss_instance = pav_loss_instance
        self.kldiv_weight = kldiv_weight
        self.pav_weight = pav_weight

    """
    Weighted average of KL_divergence, Cllr and PAV losses.

    :param kldiv_instance: KLDivergenceConcentrated instance
    :param cllr_instance: CLLR instance in a training mode or CLLRTrainDynamicallyConcentrated instance
    :param pav_loss_instance: PAVLossConcentred instance
    :param kldiv_weight: weight of kl_div loss with respect to cllr (= 1).
    :param pav_weight: weight of pav loss with respect to cllr (= 1).
    """

    def forward(self, predictions: ResultTLM, labels: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        llrs = predictions.log_norm["llr_norm"]
        cllr_loss = self.cllr_instance(llrs, labels)
        kldiv = self.kldiv_instance(predictions, labels)
        devpav = self.pav_loss_instance(predictions, labels)
        return cllr_loss + self.kldiv_weight * kldiv + self.pav_weight * devpav
