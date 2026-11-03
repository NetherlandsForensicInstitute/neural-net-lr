import itertools
from enum import Enum

import numpy as np
import torch
from torch import Tensor, nn

from DL_modules.custom_layers.contract_axes import ContractAxes
from DL_modules.custom_layers.gaussian_mixture import GaussianMixtureModel
from DL_modules.custom_layers.integrate_variance_uncertainty import IntegrateVarianceUncertainty, sample_sigma2_w
from DL_modules.custom_layers.single_instance_submodules import (
    OrthoProjector,
    WhiteThanOrthoProjector,
    get_subset_indices,
    make_ss_ds_pairs,
    update_covariances_and_grand_mean,
)
from DL_modules.custom_layers.two_level_normal_normal_sub_modules import (
    LLHT,
    LLHNormal,
    Mu,
    Q,
    Variance,
    add_variances,
)
from DL_modules.post_hoc import CoherenceFilter
from DL_modules.post_hoc.utils import CoherenceClipConfig
from DL_modules.utils import ResultTLM
from estimators.inverse_gamma_parameters_scipy import fit_alpha_beta_personal_scipy


class LRSystemPredictionType(Enum):
    COMMON = "common"
    SPECIFIC = "specific"


class TopLayerSubtractLogreg(nn.Module):
    """
    Subtract (and absolute) two input vectors,
    apply a dense layer that reduces the dimensionality to 1 and do sigmoid activation.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.sigmoid = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())

    def forward(self, susp_emb: Tensor, cr_emb: Tensor) -> Tensor:
        combined = torch.abs(susp_emb - cr_emb)
        output = self.sigmoid(combined)
        return output


class TopLayerSubtractDenseSigmoid(nn.Module):
    """
    Subtract (and absolute) two input vectors, apply a dense layer with ELU activation,
    apply a dense layer that reduces the dimensionality to 1 and do sigmoid activation.
    """

    def __init__(self, dense_input_length: int):
        super().__init__()
        self.dense_sigmoid = nn.Sequential(
            nn.Linear(dense_input_length, dense_input_length), nn.ELU(), nn.Linear(dense_input_length, 1), nn.Sigmoid()
        )

    def forward(self, susp_emb: Tensor, cr_emb: Tensor) -> Tensor:
        combined = torch.abs(susp_emb - cr_emb)
        output = self.dense_sigmoid(combined)
        return output


class TopLayerConcatDenseSigmoid(nn.Module):
    """
    Concat two input vectors, apply a dense layer with ELU activation,
    apply a dense layer that reduces the dimensionality to 1 and do sigmoid activation.
    """

    def __init__(self, dense_input_length: int):
        super().__init__()
        self.dense_sigmoid = nn.Sequential(
            nn.Linear(2 * dense_input_length, 2 * dense_input_length),
            nn.ELU(),
            nn.Linear(2 * dense_input_length, 1),
            nn.Sigmoid(),
        )

    def forward(self, susp_emb: Tensor, cr_emb: Tensor) -> Tensor:
        combined = torch.cat((susp_emb, cr_emb), dim=-1)
        output = self.dense_sigmoid(combined)
        return output


class TwoLevelNormalNormalNProj(nn.Module):
    """
    Trainable Two Level Normal Normal LR-model as proposed by Colin Aitken & David Lucy.

    Model: X_susp_i| mu_i, var_within ~ N(mu_within, var_within)
            X_cr_i| mu_i, var_within ~ N(mu_within, var_within)
            mu_susp_i| mu_between, var_between ~ N(mu_between, var_between)
    The implementation of the LR model is: P(X_susp|X_cr, H1)/P(X_susp|H0) since:
    P(X_susp, X_cr|H1)/P(X_susp, X_cr|H0) = P(X_susp|X_cr, H1)*P(X_cr|H1)/(P(X_susp|X_cr, H0)*P(X_cr|H0))
    = P(X_susp|X_cr, H1)/P(X_susp|H0)  using that P(X_cr|H0) = P(X_cr|H1) and P(X_susp|X_cr, H0) =approx P(X_susp|H0)

    And it is implemented in athe following way:
    H0: X_susp_i ~ N(mu_0, var_0)
    H1: X_susp_i - X_cr_i ~ N(mu_1, var_1),
    with mu's and var's parameters of the layer.

    We assume independence of LRs from all axes by multiplying them.

    Finally, I implemented the covariance matrix as a factorization of an orthonormal projection matrix and a diagonal
    variance matrix, allowing loss criteria and layer operations to focus on specific projection axes.

    I use a either different (TwoLevelNormalNormalTwoProj) or the same (TwoLevelNormalNormalOneProj) rotation matrix
    to rotate H0 and H1 axes.

    Additional
    Where the TwoLevelNormalNormal model operates in linear space, this class also allows to contract one or both
    hypothes spaces making distributions closer to the normal approximation
    when they were overdispersed as compared to the normal distribution for the original axes.
    """

    def __init__(
        self,
        n_axes: int,
        n_for_t_distr: int,
        variance_susp: str | float = "1",
        variance_cr: str | float = "30",
        variance_between: str | float = "30",
        mu_0: str = "fixed_at_0",
        mu_1: str = "fixed_at_0",
        contract_h0_axes: ContractAxes | None = None,
        contract_h1_axes: ContractAxes | None = None,
        n_axes_to_keep: int | None = None,
        n_clip: float | None = None,
        n_ss_train: int = 0,
    ):
        """
        :param n_axes: the number of variables to consider
        :param n_for_t_distr: the sample size used to calculate the LLR based on t-distributions
            (treating the fitted variances in a Bayesian way as informative to the posterior variance distribution).
        :param variance_h0: initial within variance of h0 projected variable, same for all variables.
        :param variance_h1: initial between variance of a h1 projected variable, same for all variables.
        :param mu_0: either 'fixed_at_0' or 'trainable' (and initialized as randn). mu's of projections.
        :param mu_1: either 'fixed_at_0' or 'trainable' (and initialized as randn). mu's of susp minus cr projection
        :param contract_h0_axes. ContractAxes instance. See docstrings of ContractAxes.
        :param contract_h1_axes. ContractAxes instance. See docstrings of ContractAxes.
        :param n_axes_to_keep: number of axes to keep during fit.
        :param contract_h0_axes: ContractAxes instance. See docstrings of ContractAxes.
        :param contract_h1_axes: ContractAxes instance. See docstrings of ContractAxes.
        :param n_clip: number of expected points to be clipped.
        :param n_ss_train: number of same source pairs in the dataset. Used to calculate z_clip.
        """
        super().__init__()
        options_mu = ["trainable", "fixed_at_0"]
        if mu_0 not in options_mu or mu_1 not in options_mu:
            raise ValueError("TwoLevelNormalNormal toplayer: options for mu's are 'trainable' or 'fixed_at_0'")

        self.n_axes = n_axes
        self.variance_susp_instance = Variance(n_axes, variance_susp)
        self.variance_cr_instance = Variance(n_axes, variance_cr)
        self.variance_between_instance = Variance(n_axes, variance_between)
        self.mu_0_instance = Mu(n_axes, mu_0)
        self.mu_1_instance = Mu(n_axes, mu_1)
        self.q_0 = Q(n_axes)
        self.q_1 = Q(n_axes)
        self.llh_norm_instance = LLHNormal()
        self.llh_t_instance = LLHT(n_for_t_distr)
        self.contract_h0_axes = contract_h0_axes
        self.contract_h1_axes = contract_h1_axes
        self.n_axes_to_keep = n_axes_to_keep
        # "if clip" is False if clip is None or 0.0
        self.z_clip = self.get_clip_value(n_clip, n_ss_train) if n_clip else None

    def z(self, projection: torch.Tensor, variance: torch.Tensor, use_z_clip: bool = False) -> torch.Tensor:
        inv_diag_stdv = torch.diag(1 / torch.sqrt(variance))
        z = torch.matmul(projection, inv_diag_stdv)

        if use_z_clip and self.z_clip is not None:
            z = torch.clamp(z, self.z_clip, -self.z_clip)

        return z

    @staticmethod
    def get_clip_value(n_clip: float, n_ss: int) -> float:
        """Calculate the z value such that 'n_clip' out of 'n_ss' values are expected to be c < -z or c > z."""
        normal_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
        clip_value_z = normal_dist.icdf(torch.tensor(n_clip / n_ss))
        return float(clip_value_z)

    @staticmethod
    def offset(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return left - right

    @staticmethod
    def project(emb: Tensor, q: Tensor) -> Tensor:
        projected = torch.matmul(emb, q)
        return projected


class TwoLevelNormalNormalOneProj(TwoLevelNormalNormalNProj):
    """Implementation of TwoLevelNormalNormalNRot where both covariance matrices (H0 and H1)
    have the same projection matrix (q_0).
    """

    def forward(self, susp_emb: Tensor, cr_emb: Tensor, use_z_clip: bool = False) -> ResultTLM:
        """Run forward pass.

        return: a ResultTLM object containing for all axes:
            LLR(for normal approximation),
            LLR(for t distribution approximation),
        """
        susp_emb = susp_emb.to(torch.float64)
        cr_emb = cr_emb.to(torch.float64)
        variance_susp = self.variance_susp_instance()
        variance_cr = self.variance_cr_instance()
        variance_between = self.variance_between_instance()
        variance_0 = add_variances(variance_susp, variance_between)
        variance_1 = add_variances(variance_susp, variance_cr)
        mu_0 = self.mu_0_instance()
        mu_1 = self.mu_1_instance()
        q = self.q_0()

        # first transform h0 data
        # first subtract mean since contract assumes 0-centered data.
        offset_susp = self.offset(susp_emb, mu_0)

        if self.contract_h0_axes:
            offset_susp = self.contract_h0_axes(offset_susp)
        project_susp = self.project(offset_susp, q)

        # then transform h1 data
        # again data is 0-centered.
        offset_h1 = self.offset(susp_emb - cr_emb, mu_1)

        if self.contract_h1_axes:
            offset_h1 = self.contract_h1_axes(offset_h1)
        project_h1 = self.project(offset_h1, q)

        z_0 = self.z(project_susp, variance_0, use_z_clip)
        z_1 = self.z(project_h1, variance_1)
        llh_norm_0 = self.llh_norm_instance(z_0, variance_0)
        llh_norm_1 = self.llh_norm_instance(z_1, variance_1)
        llh_t_0 = self.llh_t_instance(z_0, variance_0)
        llh_t_1 = self.llh_t_instance(z_1, variance_1)

        if self.n_axes_to_keep:
            llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
                self.n_axes_to_keep, llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1
            )

        llh_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llh_norm_1 - llh_norm_0}
        llh_t = {"llh_t_0": llh_t_0, "llh_t_1": llh_t_1, "llr_t": llh_t_1 - llh_t_0}
        z = {"z_0": z_0, "z_1": z_1}
        variances = {"variances_1": variance_1, "variances_0": variance_0}

        return ResultTLM(llh_norm, llh_t, z, variances)


class TwoLevelNormalNormalTwoProj(TwoLevelNormalNormalNProj):
    """Implementation of TwoLevelNormalNormalNProj where both covariance matrices (H0 and H1) have
    their own projection matrix (q_0, q_1).
    """

    def forward(self, susp_emb: Tensor, cr_emb: Tensor, use_z_clip: bool = False) -> ResultTLM:
        """Run forward pass.

        return: a ResultTLM object containing for all axes:
                    LLR(for normal approximation),
                    LLR(for t distribution approximation),
                    z
        """
        susp_emb = susp_emb.to(torch.float64)
        cr_emb = cr_emb.to(torch.float64)

        # we are using two variances only, since variance_susp and variance_between do not share the same rotation so
        # their variances cannot be added like in OneProj,
        # Ignore the confusing names 'variance_between' en 'variance_cr'. The point is that we model two
        # covariances matrices, one for H_0 and the other for H_1.
        variance_0 = self.variance_cr_instance()
        variance_1 = self.variance_between_instance()

        mu_0 = self.mu_0_instance()
        mu_1 = self.mu_1_instance()
        q_0 = self.q_0()
        q_1 = self.q_1()

        # first suspect
        # first subtract mean since contract assumes 0-centered data.
        offset_susp = self.offset(susp_emb, mu_0)

        if self.contract_h0_axes:
            offset_susp = self.contract_h0_axes(offset_susp)
        project_susp = self.project(offset_susp, q_0)
        z_0 = self.z(project_susp, variance_0, use_z_clip=use_z_clip)
        # and rotate one more time to keep axes in the same direction as h1
        z_0 = self.project(z_0, q_1)

        # next susp - cr
        offset_h1 = self.offset(susp_emb - cr_emb, mu_1)

        # again data is 0-centered and we can contract if we want.
        if self.contract_h1_axes:
            offset_h1 = self.contract_h1_axes(offset_h1)

        # and project according to the same procedure as suspect
        project_h1 = self.project(offset_h1, q_0)
        scaled_h1 = self.z(project_h1, variance_0)
        projected_twice = self.project(scaled_h1, q_1)
        # but additionally scale one more time
        z_1 = self.z(projected_twice, variance_1)

        # now, z_0 comes from a distribution with variances of 1 by definition due to the simultaneous scaling under
        # h0 and h1 by variance_0
        llh_norm_0 = self.llh_norm_instance(z_0, torch.ones(self.n_axes, dtype=torch.float64))
        llh_norm_1 = self.llh_norm_instance(z_1, variance_1)
        llh_t_0 = self.llh_t_instance(z_0, torch.ones(self.n_axes, dtype=torch.float64))
        llh_t_1 = self.llh_t_instance(z_1, variance_1)

        # set variance_0 to eye_matrix, since it is this value by definition of the whitening operation above
        variance_0 = torch.ones(self.n_axes, dtype=torch.float64)

        if self.n_axes_to_keep:
            llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
                self.n_axes_to_keep, llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1
            )

        llh_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llh_norm_1 - llh_norm_0}
        llh_t = {"llh_t_0": llh_t_0, "llh_t_1": llh_t_1, "llr_t": llh_t_1 - llh_t_0}
        z = {"z_0": z_0, "z_1": z_1}
        variances = {"variances_1": variance_1, "variances_0": variance_0}

        return ResultTLM(llh_norm, llh_t, z, variances)


class SingleInstanceTwoLevelNormalNormalNProj(nn.Module):
    """
    Base class for single_instance TLM NN NormalNormal classes.
    """

    def __init__(
        self,
        n_axes: int,
        n_sources_train_data: int,
        numerator_variance_extra: str | int | float,
        denominator_variance_extra: str | int | float,
        sample_size_sigma2_w: int | None = None,
        empirical_init: bool = False,
        n_axes_to_keep: int | None = None,
        train_coherence_filters: bool = False,
        coherence_filter_threshold_bias_axis: list[float | None] | None = None,
        # this is the threshold for variance ratio H1/H0. If H1 > 1.5 * H0 then axes is bias_axes.
        coherence_filter_max_iterations: list[int | None] | None = None,
        coherence_filter_exclude_llrs_smaller_than: list[float | None] | None = None,
        train_clip_denominators: bool = False,
        clip_denominator_percentage: list[float] | None = None,
    ):
        """
        :param n_axes: the number of axes of the TLM. This equals the number of features that it is trained on.
        :param n_sources_train_data: the number of sources training data.
        :param numerator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H1.
        :param denominator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H0.
        :param sample_size_sigma2_w: when integrating out the variance per axis, we draw sigma^2s from a fitted
            inverse_gamma_distrution. with this sample size
        :param empirical_init: whether to initialize variances and projection matrices by maximum likelihood estimation,
            works only for two-proj.
        :param n_axes_to_keep: the number of axes to calculate LLRs for, <= n_axes
        :param train_coherence_filters: whether to apply a coherence filter the LLRs for all axes for each comparison
        :param coherence_filter_threshold_bias_axis: the threshold used by the coherence filter in order to distinguish
            between 'discriminating' and 'bias' axes.
        :param coherence_filter_max_iterations: the maximum number of iterations to use by the coherence filter.
        :param coherence_filter_exclude_llrs_smaller_than: exclude LLRs per axis smaller than this value from filtering.
        :param train_clip_denominators: whether to train a denominator clip.
        :param clip_denominator_percentage: percentage of train data to clip.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_axes = n_axes
        self.n_sources_train_data = n_sources_train_data
        self.embeddings = {}  # source_id -> embeddings
        self.means = {}  # source_id → mean vector

        # use register_buffer since these are not trainable parameters and therefore will not be saved by
        # best_model = deepcopy(model), which only saves trainable parameters
        self.register_buffer("pooled_within_var", torch.empty(self.n_axes))
        self.register_buffer("between_var", torch.empty(self.n_axes))
        self.numerator_variance_extra = Variance(n_axes, numerator_variance_extra)
        self.denominator_variance_extra = Variance(n_axes, denominator_variance_extra)
        self.llh_norm_instance = LLHNormal()
        self.sample_size_sigma2_w = sample_size_sigma2_w if sample_size_sigma2_w else 20000
        self.empirical_init = empirical_init
        self.all_data_to_integrate_out_var = []
        self.alpha = torch.empty(self.n_axes, dtype=torch.float64).to(self.device)
        self.beta = torch.empty(self.n_axes, dtype=torch.float64).to(self.device)
        self.sigma2_w_prior_sample = torch.empty((self.n_axes, self.sample_size_sigma2_w), dtype=torch.float64).to(
            self.device
        )
        self.n_axes_to_keep = n_axes_to_keep if n_axes_to_keep else n_axes
        self.train_coherence_filters = train_coherence_filters
        self.coherence_filter_threshold_bias_axis = coherence_filter_threshold_bias_axis
        self.coherence_filter_max_iterations = coherence_filter_max_iterations
        self.coherence_filter_exclude_llrs_smaller_than = coherence_filter_exclude_llrs_smaller_than
        self.train_clip_denominators = train_clip_denominators
        self.clip_denominator_percentage = clip_denominator_percentage
        self.predict_result_tlm = False
        self.coherence_filter = None
        self.clip_denominator = None
        self.coherence_clip_configurations = self.build_coherence_clip_config_dict()

    def update_self_embeddings_means(self, x: Tensor, source_labels: Tensor):
        n_sources_minibatch, num_repeats, num_features = x.shape

        # use a detached version of x for the parameters below, for which we do not want to calculate gradients.
        x_detached = x.detach()
        source_means = x_detached.mean(dim=1)  # (n_sources_in_minibatch, num_features)
        for i in range(n_sources_minibatch):
            sid = source_labels[i].item()
            self.means[sid] = source_means[i]  # always update with latest means and embeddings,
            # since in the future some preprocessing neural net may have changed them.
            self.embeddings[sid] = x_detached[i]

    @staticmethod
    def z(offsets: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
        """Calculate z_scores for zero_centered data.

        :param offsets: data with mean zero (n-sources, n_axes)
        :param variance: variances of data (n_axes,)
        """
        inv_stdv = 1 / torch.sqrt(variance)
        z = offsets * inv_stdv
        return z

    def validate_sigma2_w_sample(self):
        if (self.sigma2_w_prior_sample <= 0).any():
            raise ValueError(
                "sigma2_w_sample contains negative values. "
                "Did you run 'integrate_variance_uncertainty.sample_sigma2_w()'?"
            )

    def build_coherence_clip_config_dict(self):
        """
        Build a Cartesian product of CoherenceFilter and ClipDenominator parameters in order to be able to iterate
        over it.

        :param self.train_coherence_filters: whether to train a coherence filter
        :param self.threshold_bias_axis: threshold for determining bias verseus discriminative axis for coherence
            filter. Axis with property > threshold -> bias axis.
        :param self.max_iterations: maximum number of iterations in coherence filter
        :param self.exclude_llrs_smaller_than: exclude llrs smaller than this value in coherence filter
        :param self.train_clip_denominators: whether to train a denominator clip
        :param self.clip_denominator_percentage: percentage of train data denomiator value at which to clip
        :returns a dict with post_hoc_config_folder: CoherenceClipConfig
        """
        results = {}

        # Normalize inputs
        if self.train_coherence_filters:
            tba_list = self.coherence_filter_threshold_bias_axis
            mi_list = self.coherence_filter_max_iterations
            excl_list = self.coherence_filter_exclude_llrs_smaller_than
        else:
            tba_list = [None]
            mi_list = [None]
            excl_list = [None]

        clip_list = self.clip_denominator_percentage if self.train_clip_denominators else [None]

        for tba, mi, excl, clip in itertools.product(tba_list, mi_list, excl_list, clip_list):
            cfg = CoherenceClipConfig(self.train_coherence_filters, tba, mi, excl, self.train_clip_denominators, clip)
            results[cfg.key()] = cfg

        return results


class SingleInstanceTwoLevelNormalNormalOneProj(SingleInstanceTwoLevelNormalNormalNProj):
    def __init__(
        self,
        n_axes: int,
        n_sources_train_data: int,
        numerator_variance_extra: str | int | float,
        denominator_variance_extra: str | int | float,
        sample_size_sigma2_w: int,
        empirical_init: bool = False,
        n_axes_to_keep: int | None = None,
        train_coherence_filters: bool = False,
        coherence_filter_threshold_bias_axis: list[float | None] | None = None,
        # this is the threshold for variance ratio H1/H0. If H1 > 1.5 * H0 then axes is bias_axes.
        coherence_filter_max_iterations: list[int | None] | None = None,
        coherence_filter_exclude_llrs_smaller_than: list[float | None] | None = None,
        train_clip_denominators: bool = False,
        clip_denominator_percentage: list[float] | None = None,
    ):
        """
        :param n_axes: the number of axes of the TLM. This equals the number of features that it is trained on.
        :param n_sources_train_data: the number of sources training data.
        :param numerator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H1.
        :param denominator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H0.
        :param sample_size_sigma2_w: when integrating out the variance per axis, we draw sigma^2s from a fitted
            inverse_gamma_distrution. with this sample size
        :param empirical_init: whether to initialize variances and projection matrices by maximum likelihood estimation,
            works only for two-proj.
        :param n_axes_to_keep: the number of axes to calculate LLRs for, <= n_axes
        :param train_coherence_filters: whether to apply a coherence filter the LLRs for all axes for each comparison
        :param coherence_filter_threshold_bias_axis: the threshold used by the coherence filter in order to distinguish
            between 'discriminating' and 'bias' axes.
        :param coherence_filter_max_iterations: the maximum number of iterations to use by the coherence filter.
        :param coherence_filter_exclude_llrs_smaller_than: exclude LLRs per axis smaller than this value from filtering.
        :param train_clip_denominators: whether to train a denominator clip.
        :param clip_denominator_percentage: percentage of train data to clip.
        """
        super().__init__(
            n_axes,
            n_sources_train_data,
            numerator_variance_extra,
            denominator_variance_extra,
            sample_size_sigma2_w,
            empirical_init,
            n_axes_to_keep=n_axes_to_keep,
            train_coherence_filters=train_coherence_filters,
            coherence_filter_threshold_bias_axis=coherence_filter_threshold_bias_axis,
            coherence_filter_max_iterations=coherence_filter_max_iterations,
            coherence_filter_exclude_llrs_smaller_than=coherence_filter_exclude_llrs_smaller_than,
            train_clip_denominators=train_clip_denominators,
            clip_denominator_percentage=clip_denominator_percentage,
        )
        self.register_buffer("proj_means", torch.empty(n_sources_train_data, self.n_axes))
        self.register_buffer("proj_grand_mean", torch.empty(1, self.n_axes))
        self.projector = OrthoProjector(n_axes)

    def forward(self, x: torch.Tensor, source_labels: torch.Tensor) -> tuple[ResultTLM, np.ndarray]:
        """Calculate Common Source TLM LLRs assuming normally distributed within and between distributions.

        The model, after projection, per axis is:
        {cr, susp} ~ N(mu_i, sigma2_w)
        mu_i ~ N(mu, sigma2_b + sigma2_e_h),
        where h = 0, 1 depending on whether numerator or denominator of the LLR is calculated.

        :param x: minibatch_data of shape (n_sources, n_repetitions, n_features)
        :param source_labels: source labels of shape (n_sources)
        :return: ResultTLM
        """
        # Update per-source means and embeddings
        # x: (n_sources_in_minibatch, num_repeats, num_features)
        x = x.type(torch.float64)
        self.update_self_embeddings_means(x, source_labels)

        # Calculate projections and update self parameters.
        all_source_ids_seen = list(self.means)
        proj_x = self.projector(x)
        proj_embeddings = self.projector(torch.stack([self.embeddings[sid] for sid in all_source_ids_seen]))
        self.proj_means = self.projector(torch.stack([self.means[sid] for sid in all_source_ids_seen]))

        pooled_within_cov, between_cov, self.proj_grand_mean = update_covariances_and_grand_mean(
            proj_embeddings, self.proj_means
        )

        # take diagonal of matrix, since we are not interested in correlations
        pooled_within_var = torch.diag(pooled_within_cov)
        between_var = torch.diag(between_cov)

        # Correct for the fact that means are estimated on the same data
        n_per_source = x.shape[1]
        self.pooled_within_var = pooled_within_var * n_per_source / (n_per_source - 1)
        self.between_var = (
            between_var * len(all_source_ids_seen) / (len(all_source_ids_seen) - 1)
            - self.pooled_within_var / n_per_source
        )

        # Get ss pairs from proj_x and ds pairs connecting proj_x as cr_data to proj_embeddings as susp_data
        mini_batch_label_indices = get_subset_indices(all_source_ids_seen, source_labels)
        # crime_related_data, suspect_data
        cr_data, susp_data, ss_ds_labels = make_ss_ds_pairs(proj_x, proj_embeddings, mini_batch_label_indices)

        # Get the right shapes for using GaussianMixtureModel for llh calculations
        cr_data = cr_data.unsqueeze(-1)
        susp_data = susp_data.unsqueeze(-1)

        # Initiate extra variance terms that allow TLM to self-regularize by making them large
        numerator_variance_extra = self.numerator_variance_extra()
        denominator_variance_extra = self.denominator_variance_extra()

        # Initiate GMM and get llr numerators and denominators
        gmm = GaussianMixtureModel(self.proj_grand_mean, self.between_var, torch.tensor([1]))

        # Numerators
        log_posterior_predictive = gmm.compute_log_posterior_predictive(
            cr_data, susp_data, self.pooled_within_var, numerator_variance_extra
        )

        # Denominators
        log_prior_predictive = gmm.compute_log_prior_predictive(
            cr_data, self.pooled_within_var, denominator_variance_extra
        )

        # Get an approximate estimate for the variances (ignoring Bayesian updating of posterior mean)
        # approximate variances, based on SpecificSource normal-normal model
        variance_h1 = add_variances(
            (1 / cr_data.shape[2] + 1 / susp_data.shape[2]) * self.pooled_within_var, numerator_variance_extra
        )
        variance_h0 = add_variances(
            1 / cr_data.shape[2] * self.pooled_within_var, self.between_var, denominator_variance_extra
        )

        # create zeros Tensor to fill t_predictions
        zeros = log_posterior_predictive - log_posterior_predictive

        z_0 = self.z(cr_data.mean(dim=2), variance_h0)
        z_1 = self.z(cr_data.mean(dim=2) - susp_data.mean(dim=2), variance_h1)

        # drop axes if wanted
        n_axes_to_keep = self.n_axes_to_keep if self.n_axes_to_keep else self.n_axes
        llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
            n_axes_to_keep,
            log_prior_predictive,
            log_posterior_predictive,
            zeros,
            zeros,
            z_0,
            z_1,
            variance_h0,
            variance_h1,
        )

        llh_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llh_norm_1 - llh_norm_0}
        llh_t = {"llh_t_0": llh_t_0, "llh_t_1": llh_t_1, "llr_t": llh_t_1 - llh_t_0}
        z = {"z_0": z_0, "z_1": z_1}
        variances = {"variances_1": variance_h1, "variances_0": variance_h0}

        return ResultTLM(llh_norm, llh_t, z, variances), ss_ds_labels

    def predict_cs_llr(
        self, susp_emb: Tensor, cr_emb: Tensor, also_integrate_variance: bool = False
    ) -> dict[str, ResultTLM] | ResultTLM:
        """Predict common source TLM LLRs for normalnormal model.
        Same model as in forward pass.

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the variance
        :return: ResultTLM if self.predict_result_tlm = True else return dict[str, ResultTLM]
        """
        return self.predict_llr(susp_emb, cr_emb, LRSystemPredictionType.COMMON, also_integrate_variance)

    def predict_ss_llr(
        self, susp_emb: Tensor, cr_emb: Tensor, also_integrate_variance: bool = False
    ) -> dict[str, ResultTLM] | ResultTLM:
        """Predict specific source TLM LLRs for normalnormal model.
        The model, after projection, per axis is:
        {cr, susp} ~ N(mu_i, sigma2_w)
        mu_i ~ N(mu, sigma2_b_h + sigma2_e_h),
        where h = 0, 1 depending on whether numerator or denominator of the LLR is calculated,
        and sigma2_b_1 is infinite when h = 1 (i.e. a flat prior for the numerator).

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the variance
        :return: ResultTLM if self.predict_result_tlm = True else return dict[str, ResultTLM]
        """
        return self.predict_llr(susp_emb, cr_emb, LRSystemPredictionType.SPECIFIC, also_integrate_variance)

    def predict_llr(
        self,
        susp_emb: Tensor,
        cr_emb: Tensor,
        common_or_specific: LRSystemPredictionType,
        also_integrate_variance: bool,
    ) -> dict[str, ResultTLM] | ResultTLM:  # return ResultTLM if self.predict_result_tlm = True
        """Predict  TLM LLRs for normalnormal model, either specific or common source.

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param common_or_specific: Enum, whether to calculate a common or specific source LLRs
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the variance
        :return: ResultTLM
        """
        cr_emb = cr_emb.to(torch.float64)
        susp_emb = susp_emb.to(torch.float64)
        proj_cr = self.projector(cr_emb).unsqueeze(-1)
        proj_susp = self.projector(susp_emb).unsqueeze(-1)
        denominator_variance_extra = self.denominator_variance_extra()
        numerator_variance_extra = self.numerator_variance_extra()

        gmm = GaussianMixtureModel(self.proj_grand_mean, self.between_var, torch.tensor([1]))

        # Numerators
        match common_or_specific:
            case common_or_specific.SPECIFIC:
                log_posterior_predictive = gmm.compute_log_posterior_predictive_specific_source(
                    proj_cr, proj_susp, self.pooled_within_var, numerator_variance_extra
                )
            case common_or_specific.COMMON:
                log_posterior_predictive = gmm.compute_log_posterior_predictive(
                    proj_cr, proj_susp, self.pooled_within_var, numerator_variance_extra
                )

        # Denominators
        log_prior_predictive = gmm.compute_log_prior_predictive(
            proj_cr, self.pooled_within_var, denominator_variance_extra
        )

        # Get an approximate estimate for the variances (ignoring Bayesian updating of posterior mean)
        # approximate variances, based on SpecificSource normal-normal model
        variance_h0 = add_variances(
            1 / proj_cr.shape[2] * self.pooled_within_var, self.between_var, denominator_variance_extra
        )
        variance_h1 = add_variances(
            (1 / proj_cr.shape[2] + 1 / proj_susp.shape[2]) * self.pooled_within_var, numerator_variance_extra
        )

        # approximate z-scores of means (of reps), using approximate variances
        z_0 = self.z(proj_cr.mean(dim=2), variance_h0)
        z_1 = self.z(proj_cr.mean(dim=2) - proj_susp.mean(dim=2), variance_h1)

        if also_integrate_variance:
            self.validate_sigma2_w_sample()
            log_posterior_predictive_int_var = []
            log_prior_predictive_int_var = []
            for i in range(self.n_axes):
                model_int_var = IntegrateVarianceUncertainty(
                    self.proj_grand_mean[:, i], weights=torch.tensor([1]), kde_variance=self.between_var[i]
                )
                # Numerators with variance integration
                match common_or_specific:
                    case common_or_specific.SPECIFIC:
                        log_posterior_predictive_int_var_ax = model_int_var.get_ss_llr_numerators(
                            proj_cr[:, i],
                            proj_susp[:, i],
                            numerator_variance_extra[i],
                            self.alpha[i],
                            self.beta[i],
                            sample_size=self.sample_size_sigma2_w,
                        )
                    case common_or_specific.COMMON:
                        log_posterior_predictive_int_var_ax = model_int_var.get_cs_llr_numerators(
                            proj_cr[:, i],
                            proj_susp[:, i],
                            numerator_variance_extra[i],
                            self.sigma2_w_prior_sample[i],
                        )

                # Denominators with variance integration
                log_prior_predictive_int_var_ax = model_int_var.get_llr_denominators(
                    proj_cr[:, i], denominator_variance_extra[i], self.sigma2_w_prior_sample[i]
                )
                log_posterior_predictive_int_var.append(log_posterior_predictive_int_var_ax)
                log_prior_predictive_int_var.append(log_prior_predictive_int_var_ax)

            log_posterior_predictive_int_var = torch.stack(log_posterior_predictive_int_var).t().to('cpu')
            log_prior_predictive_int_var = torch.stack(log_prior_predictive_int_var).t().to('cpu')

        else:
            # set llh_t to zero's
            zeros = log_posterior_predictive - log_posterior_predictive
            log_prior_predictive_int_var = zeros
            log_posterior_predictive_int_var = zeros

        # drop axes if wanted
        n_axes_to_keep = self.n_axes_to_keep if self.n_axes_to_keep else self.n_axes
        llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
            n_axes_to_keep,
            log_prior_predictive,
            log_posterior_predictive,
            log_prior_predictive_int_var,
            log_posterior_predictive_int_var,
            z_0,
            z_1,
            variance_h0,
            variance_h1,
        )

        variances = {
            "variances_1": variance_1,
            "variances_0": variance_0,
        }
        large_are_bias_axes = variance_1 / variance_0

        z = {
            "z_0": z_0,
            "z_1": z_1,
        }

        if self.predict_result_tlm:
            if self.clip_denominator:
                llh_norm_0 = self.clip_deominator(llh_norm_0)
                llh_t_0 = self.clip_deominator(llh_t_0)

            llr_norm = llh_norm_1 - llh_norm_0
            llr_t = llh_t_1 - llh_t_0

            if self.coherence_filter:
                llr_norm = self.coherence_filter.apply(llr_norm, large_are_bias_axes)
                llr_t = self.coherence_filter.apply(llr_t, large_are_bias_axes)

            log_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llr_norm}
            log_t = {
                'llh_t_0': llh_t_0,
                'llh_t_1': llh_t_1,
                'llr_t': llr_t,
            }

            return ResultTLM(log_norm=log_norm, log_t=log_t, z=z, variances=variances)

        llr_norm = llh_norm_1 - llh_norm_0
        llr_t = llh_t_1 - llh_t_0
        log_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llr_norm}
        log_t = {
            'llh_t_0': llh_t_0,
            'llh_t_1': llh_t_1,
            'llr_t': llr_t,
        }

        result = {}

        for key, post_hoc_config in self.coherence_clip_configurations.items():
            log_norm_updated = log_norm.copy()
            log_t_updated = log_t.copy()
            if post_hoc_config.train_clip_denominators:
                pass

            if post_hoc_config.train_coherence_filters:
                filter = CoherenceFilter(
                    threshold_bias_axis=post_hoc_config.threshold_bias_axis,
                    max_iterations=post_hoc_config.max_iterations,
                    exclude_llrs_smaller_than=post_hoc_config.exclude_llrs_smaller_than,
                )
                llr_norm_updated = filter.apply(llr_norm, large_are_bias_axes)
                llr_t_updated = filter.apply(llr_t, large_are_bias_axes)
                log_norm_updated["llr_norm"] = llr_norm_updated
                log_t_updated["llr_t"] = llr_t_updated
            result[key] = ResultTLM(log_norm_updated, log_t_updated, z, variances)

        return result

    def initialize_integrate_out_variance(self, x: torch.Tensor, dataset_length: int):
        """Initialize whitening parameters and second variance.

        :param x: minibatch_data of shape (n_sources, n_repetitions, n_features)
        :param source_labels: source labels of shape (n_sources)
        """
        # Collect all data in global variable all_data_to_integrate_out_var
        x = x.type(torch.float64)
        # Preprocess
        x = self.projector(x)
        self.all_data_to_integrate_out_var.append(x)
        total_len = sum(tensor.shape[0] for tensor in self.all_data_to_integrate_out_var)

        if total_len == dataset_length:
            # We have all the data, lets initialize parameters to integrate out variance
            all_data_to_integrate_out_var = torch.cat(self.all_data_to_integrate_out_var)

            with torch.no_grad():
                for ax, data_ax in enumerate(torch.permute(all_data_to_integrate_out_var, (2, 0, 1))):
                    alpha, beta, _ = fit_alpha_beta_personal_scipy(data_ax.numpy())
                    self.alpha[ax], self.beta[ax] = torch.tensor(alpha), torch.tensor(beta)
                    self.sigma2_w_prior_sample[ax] = sample_sigma2_w(
                        self.alpha[ax], self.beta[ax], sample_size=self.sample_size_sigma2_w
                    )
                self.all_data_to_integrate_out_var = []


class SingleInstanceTwoLevelNormalNormalTwoProj(SingleInstanceTwoLevelNormalNormalNProj):
    def __init__(
        self,
        n_axes: int,
        n_sources_train_data: int,
        numerator_variance_extra: str | int | float,
        denominator_variance_extra: str | int | float,
        sample_size_sigma2_w: int,
        empirical_init: bool = False,
        n_axes_to_keep: int | None = None,
        train_coherence_filters: bool = False,
        coherence_filter_threshold_bias_axis: list[float | None] | None = None,
        # this is the threshold for variance ratio H1/H0. If H1 > 1.5 * H0 then axes is bias_axes.
        coherence_filter_max_iterations: list[int | None] | None = None,
        coherence_filter_exclude_llrs_smaller_than: list[float | None] | None = None,
        train_clip_denominators: bool = False,
        clip_denominator_percentage: list[float] | None = None,
    ):
        """
        :param n_axes: the number of axes of the TLM. This equals the number of features that it is trained on.
        :param n_sources_train_data: the number of sources training data.
        :param numerator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H1.
        :param denominator_variance_extra: this parameter is used to initialize a trainable Variance object.
            The variance from this trainable variance object are added to the maximum-likelihood estimated variances
            under H0.
        :param sample_size_sigma2_w: when integrating out the variance per axis, we draw sigma^2s from a fitted
            inverse_gamma_distrution. with this sample size
        :param empirical_init: whether to initialize variances and projection matrices by maximum likelihood estimation,
            works only for two-proj.
        :param n_axes_to_keep: the number of axes to calculate LLRs for, <= n_axes
        :param train_coherence_filters: whether to apply a coherence filter the LLRs for all axes for each comparison
        :param coherence_filter_threshold_bias_axis: the threshold used by the coherence filter in order to distinguish
            between 'discriminating' and 'bias' axes.
        :param coherence_filter_max_iterations: the maximum number of iterations to use by the coherence filter.
        :param coherence_filter_exclude_llrs_smaller_than: exclude LLRs per axis smaller than this value from filtering.
        :param train_clip_denominators: whether to train a denominator clip.
        :param clip_denominator_percentage: percentage of train data to clip.
        """
        super().__init__(
            n_axes,
            n_sources_train_data,
            numerator_variance_extra,
            denominator_variance_extra,
            sample_size_sigma2_w,
            empirical_init,
            n_axes_to_keep=n_axes_to_keep,
            train_coherence_filters=train_coherence_filters,
            coherence_filter_threshold_bias_axis=coherence_filter_threshold_bias_axis,
            coherence_filter_max_iterations=coherence_filter_max_iterations,
            coherence_filter_exclude_llrs_smaller_than=coherence_filter_exclude_llrs_smaller_than,
            train_clip_denominators=train_clip_denominators,
            clip_denominator_percentage=clip_denominator_percentage,
        )

        self.n_sources_train_data = n_sources_train_data
        self.embeddings = {}  # source_id -> embeddings
        self.means = {}  # source_id → mean vector

        # use register_buffer since these are not trainable parameters and therefore will not be saved by
        # best_model = deepcopy(model), which only saves trainable parameters
        self.register_buffer("wh_and_proj_means", torch.empty(n_sources_train_data, n_axes))
        self.register_buffer("wh_and_proj_grand_mean", torch.empty(1, n_axes))
        self.white_and_projector = WhiteThanOrthoProjector(self.n_axes)

    def forward(self, x: torch.Tensor, source_labels: torch.Tensor) -> tuple[ResultTLM, np.ndarray]:
        """Calculate common source TLM LLRs assuming normally distributed within and between distributions.

         The model, after projection, per axis is:
        {cr, susp} ~ N(mu_i, sigma2_w)
        mu_i ~ N(mu, sigma2_b + sigma2_e_h),
        where h = 0, 1 depending on whether numerator or denominator of the LLR is calculated.

        :param x: minibatch_data of shape (n_sources, n_repetitions, n_features)
        :param source_labels: source labels of shape (n_sources)
        :return: ResultTLM
        """
        # Update per-source means and embeddings
        # x: (n_sources_in_minibatch, num_repeats, num_features)
        x = x.type(torch.float64)
        self.update_self_embeddings_means(x, source_labels)

        # Calculate whitening-projections and update self parameters
        all_source_ids_seen = [sid for sid in self.means.keys()]

        stacked_embeddings = torch.stack([self.embeddings[sid] for sid in all_source_ids_seen])
        stacked_means = torch.stack([self.means[sid] for sid in all_source_ids_seen])
        self.white_and_projector.update_var_w(stacked_embeddings, stacked_means)

        wh_and_proj_x = self.white_and_projector(x)
        wh_and_proj_embeddings = self.white_and_projector(stacked_embeddings)
        self.wh_and_proj_means = self.white_and_projector(stacked_means)

        _, between_cov, self.wh_and_proj_grand_mean = update_covariances_and_grand_mean(
            wh_and_proj_embeddings, self.wh_and_proj_means
        )

        # Take diagonal of between matrix, since we are not interested in correlations
        between_var = torch.diag(between_cov)

        # self.pooled_within_var is whitened, so we know it should be torch.ones(x.shape[-1])
        self.pooled_within_var = torch.ones(x.shape[-1])

        # Correct empirical between_var for the fact that means are estimated on the same data
        n_per_source = x.shape[1]
        self.between_var = (
            between_var * len(all_source_ids_seen) / (len(all_source_ids_seen) - 1)
            - self.pooled_within_var / n_per_source
        )

        # Get ss pairs from proj_x and ds pairs connecting proj_x as cr_data to proj_embeddings as susp_data
        mini_batch_label_indices = get_subset_indices(all_source_ids_seen, source_labels)
        cr_data, susp_data, ss_ds_labels = make_ss_ds_pairs(
            wh_and_proj_x, wh_and_proj_embeddings, mini_batch_label_indices
        )

        # Get the right shapes for using GaussianMixtureModel for llh calculations
        cr_data = cr_data.unsqueeze(-1)
        susp_data = susp_data.unsqueeze(-1)

        # Initiate GMM and get llr numerators and denominators
        gmm = GaussianMixtureModel(self.wh_and_proj_grand_mean, self.between_var, torch.tensor([1]))

        # Initiate extra variance terms that allow TLM to self-regularize by making them large
        numerator_variance_extra = self.numerator_variance_extra()
        denominator_variance_extra = self.denominator_variance_extra()

        # Numerators
        log_posterior_predictive = gmm.compute_log_posterior_predictive(
            cr_data, susp_data, self.pooled_within_var, numerator_variance_extra
        )

        # Denominators
        log_prior_predictive = gmm.compute_log_prior_predictive(
            cr_data, self.pooled_within_var, denominator_variance_extra
        )

        # Get an approximate estimate for the variances (ignoring Bayesian updating of posterior mean)
        variance_h1 = add_variances(
            (1 / cr_data.shape[2] + 1 / susp_data.shape[2]) * self.pooled_within_var, numerator_variance_extra
        )
        variance_h0 = add_variances(
            1 / cr_data.shape[2] * self.pooled_within_var, denominator_variance_extra, self.between_var
        )

        # create zeros Tensor to fill t_predictions
        zeros = log_posterior_predictive - log_posterior_predictive

        # approximate z-scores of means (of reps), using approximate variances
        z_0 = self.z(cr_data.mean(dim=2), variance_h0)
        z_1 = self.z(cr_data.mean(dim=2) - susp_data.mean(dim=2), variance_h1)

        # drop axes if wanted
        n_axes_to_keep = self.n_axes_to_keep if self.n_axes_to_keep else self.n_axes
        llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
            n_axes_to_keep,
            log_prior_predictive,
            log_posterior_predictive,
            zeros,
            zeros,
            z_0,
            z_1,
            variance_h0,
            variance_h1,
        )

        llh_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llh_norm_1 - llh_norm_0}
        llh_t = {"llh_t_0": llh_t_0, "llh_t_1": llh_t_1, "llr_t": llh_t_1 - llh_t_0}
        z = {"z_0": z_0, "z_1": z_1}
        variances = {"variances_1": variance_h1, "variances_0": variance_h0}

        return ResultTLM(llh_norm, llh_t, z, variances), ss_ds_labels

    def predict_cs_llr(
        self, susp_emb: Tensor, cr_emb: Tensor, also_integrate_variance: bool = False
    ) -> ResultTLM | dict[str, ResultTLM]:
        """Predict common source TLM LLRs for normalnormal model.

        Same model as in forward pass.

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the variance
        :return: ResultTLM
        """
        return self.predict_llr(susp_emb, cr_emb, LRSystemPredictionType.COMMON, also_integrate_variance)

    def predict_ss_llr(
        self, susp_emb: Tensor, cr_emb: Tensor, also_integrate_variance: bool = False
    ) -> ResultTLM | dict[str, ResultTLM]:
        """Predict specific source TLM LLRs for normalnormal model.

        The model, after projection, per axis is:
        {cr, susp} ~ N(mu_i, sigma2_w)
        mu_i ~ N(mu, sigma2_b_h + sigma2_e_h),
        where h = 0, 1 depending on whether numerator or denominator of the LLR is calculated,
        and sigma2_b_1 is infinite when h = 1 (i.e. a flat prior for the numerator).

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the variance
        :return: ResultTLM
        """
        return self.predict_llr(susp_emb, cr_emb, LRSystemPredictionType.SPECIFIC, also_integrate_variance)

    def predict_llr(
        self,
        susp_emb: Tensor,
        cr_emb: Tensor,
        common_or_specific_source: LRSystemPredictionType,
        also_integrate_variance: bool = False,
    ) -> ResultTLM | dict[str, ResultTLM]:
        """Predict common source TLM LLRs for normalnormal model.

        :param susp_emb: suspect_embedding, tensor of shape n_sources, n_features
        :param cr_emb: crime_related_embedding, tensor of shape n_sources, n_features
        :param common_or_specific_source: Enum, whether to predict common or specific source LLRs
        :param also_integrate_variance: boolean indicating whether also calculate LLRs with integrating out the
            variance.
        :return: ResultTLM
        """
        cr_emb = cr_emb.to(torch.float64)
        susp_emb = susp_emb.to(torch.float64)
        wh_and_proj_cr = self.white_and_projector(cr_emb).unsqueeze(-1)
        wh_and_proj_susp = self.white_and_projector(susp_emb).unsqueeze(-1)

        gmm = GaussianMixtureModel(self.wh_and_proj_grand_mean, self.between_var, torch.tensor([1]))

        # Get extra variances
        numerator_variance_extra = self.numerator_variance_extra()
        denominator_variance_extra = self.denominator_variance_extra()

        # Numerators
        match common_or_specific_source:
            case common_or_specific_source.COMMON:
                log_posterior_predictive = gmm.compute_log_posterior_predictive(
                    wh_and_proj_cr, wh_and_proj_susp, self.pooled_within_var, numerator_variance_extra
                )
            case common_or_specific_source.SPECIFIC:
                log_posterior_predictive = gmm.compute_log_posterior_predictive_specific_source(
                    wh_and_proj_cr, wh_and_proj_susp, self.pooled_within_var, numerator_variance_extra
                )

        # Denominators
        log_prior_predictive = gmm.compute_log_prior_predictive(
            wh_and_proj_cr, self.pooled_within_var, denominator_variance_extra
        )

        # Get an approximate estimate for the variances (ignoring Bayesian updating of posterior mean)
        # approximate variances, based on SpecificSource normal-normal model
        variance_1 = add_variances(
            (1 / wh_and_proj_cr.shape[2] + 1 / wh_and_proj_susp.shape[2]) * self.pooled_within_var,
            numerator_variance_extra,
        )
        variance_0 = add_variances(
            1 / wh_and_proj_cr.shape[2] * self.pooled_within_var, self.between_var, denominator_variance_extra
        )

        # approximate z-scores of means (of reps), using approximate variances
        z_0 = self.z(wh_and_proj_cr.mean(dim=2), variance_0)
        z_1 = self.z(wh_and_proj_cr.mean(dim=2) - wh_and_proj_susp.mean(dim=2), variance_1)

        if also_integrate_variance:
            self.validate_sigma2_w_sample()
            log_posterior_predictive_int_var = []
            log_prior_predictive_int_var = []
            for i in range(self.n_axes):
                model_int_var = IntegrateVarianceUncertainty(
                    self.wh_and_proj_grand_mean[:, i], weights=torch.tensor([1]), kde_variance=self.between_var[i]
                )
                # Numerators with variance integration
                match common_or_specific_source:
                    case common_or_specific_source.COMMON:
                        log_posterior_predictive_int_var_ax = model_int_var.get_cs_llr_numerators(
                            wh_and_proj_cr[:, i],
                            wh_and_proj_susp[:, i],
                            numerator_variance_extra[i],
                            self.sigma2_w_prior_sample[i],
                        )
                    case common_or_specific_source.SPECIFIC:
                        log_posterior_predictive_int_var_ax = model_int_var.get_ss_llr_numerators(
                            wh_and_proj_cr[:, i],
                            wh_and_proj_susp[:, i],
                            numerator_variance_extra[i],
                            self.alpha[i],
                            self.beta[i],
                            sample_size=self.sample_size_sigma2_w,
                        )

                # Denominators with variance integration
                log_prior_predictive_int_var_ax = model_int_var.get_llr_denominators(
                    wh_and_proj_cr[:, i], denominator_variance_extra[i], self.sigma2_w_prior_sample[i]
                )
                log_posterior_predictive_int_var.append(log_posterior_predictive_int_var_ax)
                log_prior_predictive_int_var.append(log_prior_predictive_int_var_ax)

            log_posterior_predictive_int_var = torch.stack(log_posterior_predictive_int_var).t().to('cpu')
            log_prior_predictive_int_var = torch.stack(log_prior_predictive_int_var).t().to('cpu')

        else:
            # set llh_t to zero's
            zeros = log_posterior_predictive - log_posterior_predictive
            log_prior_predictive_int_var = zeros
            log_posterior_predictive_int_var = zeros

        # drop axes if wanted
        n_axes_to_keep = self.n_axes_to_keep if self.n_axes_to_keep else self.n_axes
        llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1 = drop_axes(
            n_axes_to_keep,
            log_prior_predictive,
            log_posterior_predictive,
            log_prior_predictive_int_var,
            log_posterior_predictive_int_var,
            z_0,
            z_1,
            variance_0,
            variance_1,
        )

        variances = {
            "variances_1": variance_1,
            "variances_0": variance_0,
        }
        large_are_bias_axes = variance_1 / variance_0

        z = {
            "z_0": z_0,
            "z_1": z_1,
        }

        if self.predict_result_tlm:
            if self.clip_denominator:
                llh_norm_0 = self.clip_deominator(llh_norm_0)
                llh_t_0 = self.clip_deominator(llh_t_0)

            llr_norm = llh_norm_1 - llh_norm_0
            llr_t = llh_t_1 - llh_t_0

            if self.coherence_filter:
                llr_norm = self.coherence_filter.apply(llr_norm, large_are_bias_axes)
                llr_t = self.coherence_filter.apply(llr_t, large_are_bias_axes)

            log_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llr_norm}
            log_t = {
                'llh_t_0': llh_t_0,
                'llh_t_1': llh_t_1,
                'llr_t': llr_t,
            }

            return ResultTLM(log_norm=log_norm, log_t=log_t, z=z, variances=variances)

        llr_norm = llh_norm_1 - llh_norm_0
        llr_t = llh_t_1 - llh_t_0
        log_norm = {"llh_norm_0": llh_norm_0, "llh_norm_1": llh_norm_1, "llr_norm": llr_norm}
        log_t = {
            'llh_t_0': llh_t_0,
            'llh_t_1': llh_t_1,
            'llr_t': llr_t,
        }

        result = {}

        for key, post_hoc_config in self.coherence_clip_configurations.items():
            log_norm_updated = log_norm.copy()
            log_t_updated = log_t.copy()
            if post_hoc_config.train_clip_denominators:
                pass

            if post_hoc_config.train_coherence_filters:
                filter = CoherenceFilter(
                    threshold_bias_axis=post_hoc_config.threshold_bias_axis,
                    max_iterations=post_hoc_config.max_iterations,
                    exclude_llrs_smaller_than=post_hoc_config.exclude_llrs_smaller_than,
                )
                llr_norm_updated = filter.apply(llr_norm, large_are_bias_axes)
                llr_t_updated = filter.apply(llr_t, large_are_bias_axes)
                log_norm_updated["llr_norm"] = llr_norm_updated
                log_t_updated["llr_t"] = llr_t_updated
            result[key] = ResultTLM(log_norm_updated, log_t_updated, z, variances)

        return result

    def z(self, projection: torch.Tensor, variance: torch.Tensor, use_z_clip: bool = False) -> torch.Tensor:
        inv_stdv = 1 / torch.sqrt(variance)
        z = projection * inv_stdv

        if use_z_clip and self.z_clip is not None:
            z = torch.clamp(z, self.z_clip, -self.z_clip)

        return z

    def initialize_white_and_projector(self, x: torch.tensor, source_labels: torch.tensor):
        """Initialize whitening parameters and second variance.

        :param x: minibatch_data of shape (n_sources, n_repetitions, n_features)
        :param source_labels: source labels of shape (n_sources)
        """
        # Update per-source means and embeddings
        # x: (batch_size, num_repeats, num_features)
        x = x.type(torch.float64)
        self.update_self_embeddings_means(x, source_labels)

        if len(self.means) == self.n_sources_train_data:
            # We have all the data, lets initialize

            all_source_ids_seen = [sid for sid in self.means.keys()]
            stacked_means = torch.stack([self.means[sid] for sid in all_source_ids_seen])
            stacked_embeddings = torch.stack([self.embeddings[sid] for sid in all_source_ids_seen])

            within_cov, _, _ = update_covariances_and_grand_mean(stacked_embeddings, stacked_means)

            wh_variances, wh_eigenvectors_left = torch.linalg.eigh(within_cov)

            proj_scaled_embeddings = self.project_than_scale(stacked_embeddings, wh_eigenvectors_left, wh_variances)
            proj_scaled_means = self.project_than_scale(stacked_means, wh_eigenvectors_left, wh_variances)
            _, between_cov, _ = update_covariances_and_grand_mean(proj_scaled_embeddings, proj_scaled_means)

            _, wh_eigenvectors_right = torch.linalg.eigh(between_cov)
            upper_triangle = torch.triu(torch.ones(self.n_axes, self.n_axes)).double()

            q_w_init = torch.matmul(wh_eigenvectors_left, upper_triangle)
            q_b_init = torch.matmul(wh_eigenvectors_right, upper_triangle)

            with torch.no_grad():
                self.white_and_projector.q_w.random_square.copy_(q_w_init)
                self.white_and_projector.q_b.random_square.copy_(q_b_init)

    def initialize_integrate_out_variance(self, x: torch.Tensor, n_sources_dataset: int):
        """Initialize whitening parameters and second variance.

        :param x: minibatch_data of shape (n_sources, n_repetitions, n_features)
        :param n_sources_dataset: number of sources in dataset
        """
        # Collect all data in global variable all_data_to_integrate_out_var
        x = x.type(torch.float64)
        # Preprocess
        x = self.white_and_projector(x)
        self.all_data_to_integrate_out_var.append(x)
        total_len = sum(tensor.shape[0] for tensor in self.all_data_to_integrate_out_var)

        if total_len == n_sources_dataset:
            # We have all the data, lets initialize parameters to integrate out variance
            all_data_to_integrate_out_var = torch.cat(self.all_data_to_integrate_out_var)

            with torch.no_grad():
                for ax, data_ax in enumerate(torch.permute(all_data_to_integrate_out_var, (2, 0, 1))):
                    alpha, beta, _ = fit_alpha_beta_personal_scipy(data_ax.numpy())
                    self.alpha[ax], self.beta[ax] = torch.tensor(alpha), torch.tensor(beta)
                    self.sigma2_w_prior_sample[ax] = sample_sigma2_w(
                        self.alpha[ax], self.beta[ax], sample_size=self.sample_size_sigma2_w
                    )
                self.all_data_to_integrate_out_var = []

    @staticmethod
    def project_than_scale(data, q, var):
        inv_sd = 1 / torch.sqrt(var).unsqueeze(0)
        transformation_matrix = q * inv_sd
        # data: (batch_size, num_repeats, n_axes)
        return torch.matmul(data, transformation_matrix)  # shape: (batch_size, num_repeats, n_axes)


def drop_axes(n_axes_to_keep, llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1):
    llh_norm_0 = llh_norm_0[:, :n_axes_to_keep]
    llh_norm_1 = llh_norm_1[:, :n_axes_to_keep]
    llh_t_0 = llh_t_0[:, :n_axes_to_keep]
    llh_t_1 = llh_t_1[:, :n_axes_to_keep]
    z_0 = z_0[:, :n_axes_to_keep]
    z_1 = z_1[:, :n_axes_to_keep]
    variance_0 = variance_0[:n_axes_to_keep]
    variance_1 = variance_1[:n_axes_to_keep]

    return llh_norm_0, llh_norm_1, llh_t_0, llh_t_1, z_0, z_1, variance_0, variance_1


tlm_pair_versions = {'single_projection': TwoLevelNormalNormalOneProj, 'double_projection': TwoLevelNormalNormalTwoProj}
tlm_single_instance_versions = {
    'single_projection_normal': SingleInstanceTwoLevelNormalNormalOneProj,
    'double_projection_normal': SingleInstanceTwoLevelNormalNormalTwoProj,
}
log_reg_versions = {
    'basic_subtract': TopLayerSubtractLogreg,
    'basic_concat': TopLayerConcatDenseSigmoid,
    'subtract_dense': TopLayerSubtractDenseSigmoid,
}
