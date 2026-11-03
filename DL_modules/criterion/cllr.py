from enum import Enum

import torch
from torch import Tensor
from torch.nn.functional import softplus

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.custom_layers import ResultTLM

LOG_10 = torch.log(torch.tensor(10.0))
# conversion_factor for ln's to base 2 logarithms
LN_TO_BASE_TWO = 1 / torch.log(torch.tensor(2.0))


class CllrTrainMode(Enum):
    CONCENTRATED = "train_concentrated"
    EVENLY = "train_evenly"
    RANDOM_PROJECTION = "train_random_projection"
    FOCUS_SPREAD_SUM = "train_focus_spread_sum"
    FOCUS_SPREAD_MEAN = "train_focus_spread_mean"


class CLLR(TlmLossFunction):
    """
    Cllr loss.

    :param fr_hard_negs. When negatives are filtered to not provide easy negatives to the DataLoader,
        Cllr loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
    :param mode. The mode for which to compute the Cllr when predictions are llrs.
        There are 4 modes when training in llr mode:
            concentrated: LLRs are calculated by the cumulative sum of axes 0:n for n = 0:n_axes.
                Cllr is computed per n, and the mean is returned.
                The result of this procedure is that after training, the CLlr of an axis with smaller n is
                smaller than a Cllr of an axis with larger n. This may reduce overfitting?
            evenly: Cllrs are computed per axis, for axes 0:n_axes. The mean is returned.
                The result of this procedure is that after training, scores for 0:n_axes will be approximately
                equal.
            random_projection: Compute a Cllrfor the sum of the LLRs, summed over 0:n_axis.
                The result of this procedure is that after the TLM model is trained without a preference for which
                axis to optimize.
            focus_spread_sum: Based on the hypothesis that axes with exterme LLRs are less common and thus
                are suseptiable to overfitting, this options limits the axes with larges cllr's. In practice, this
                means that Loss = Cllr_tot * sum([cllr_i for i when cllr_i < focus_threshold]) is used.
            focus_spread_mean: Similar to Train_focus_spread_sum,
                but with Loss = Cllr_tot * mean([cllr_i for i when cllr_i < focus_threshold]),
    :param focus_threshold: cutoff value for train_focus_spread_sum and train_focus_spread_mean.
    :param focus_reg_power: Set the factor that is used in train_focus_spread_sum and train_focus_spread_mean.
                            Higher numbers mean less regularization

    """

    def __init__(
        self,
        fr_hard_negs: float,
        mode: CllrTrainMode | str | None = None,
        focus_threshold: float = 0.95,
        focus_weight: float = 1.0,
    ):
        super().__init__()
        self.fr_hard_negs = fr_hard_negs
        self.mode = CllrTrainMode[mode] if isinstance(mode, str) else mode
        self.focus_threshold = focus_threshold
        self.focus_weight = focus_weight

        # ensure the cllr_train_mode is properly specified
        if self.mode is None:
            valid_modes = ", ".join(f"Mode.{member.name}" for member in CllrTrainMode)
            raise ValueError(f"CLLR criterion: mode must be either {valid_modes}")

    def forward(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        if not result_tlm.log_norm["llr_norm"].shape[0] == targets.shape[0]:
            raise ValueError("CLLR criterion: the number of predictions is not equal to the number of targets!")
        if not result_tlm.log_norm["llr_norm"].ndim == 2:
            raise ValueError("CLLR criterion: predictions should have dimensions [batch size, len(prediction_vector)]")
        if not targets.ndim == 1:
            raise ValueError("CLLR criterion: targets should have dimension (batch size)")

        match self.mode:
            case CllrTrainMode.CONCENTRATED:
                predictions = result_tlm.log_norm["llr_norm"].double()
                predictions = self.get_cumsum(predictions)
                cllr_per_axis = compute_cllr_from_llr_per_axis(predictions, targets, self.fr_hard_negs)
                return torch.mean(cllr_per_axis)
            case CllrTrainMode.EVENLY:
                predictions = result_tlm.log_norm["llr_norm"].double()
                predictions = concat_with_sum(predictions)
                cllr_per_axis = compute_cllr_from_llr_per_axis(predictions, targets, self.fr_hard_negs)
                return torch.mean(cllr_per_axis)
            case CllrTrainMode.RANDOM_PROJECTION:
                predictions = result_tlm.log_norm["llr_norm"].double()
                predictions = torch.sum(predictions, dim=1).unsqueeze(1)
                return compute_cllr_from_llr_per_axis(predictions, targets, self.fr_hard_negs)
            case CllrTrainMode.FOCUS_SPREAD_SUM:
                predictions = result_tlm.log_norm["llr_norm"].double()
                cllr_per_axis = compute_cllr_from_llr_per_axis(predictions, targets, self.fr_hard_negs)
                cllr_focus = cllr_per_axis[cllr_per_axis < self.focus_threshold].sum()

                # calculate final CLLR over the sum of predictions (single axis)
                cllr_tot = compute_cllr_from_llr_per_axis(
                    torch.sum(predictions, dim=1).unsqueeze(1), targets, self.fr_hard_negs
                )
                return cllr_tot + self.focus_weight * cllr_focus
            case CllrTrainMode.FOCUS_SPREAD_MEAN:
                predictions = result_tlm.log_norm["llr_norm"].double()
                cllr_per_axis = compute_cllr_from_llr_per_axis(predictions, targets, self.fr_hard_negs)
                cllr_focus = cllr_per_axis[cllr_per_axis < self.focus_threshold].mean()

                # calculate final CLLR over the sum of predictions (single axis)
                cllr_tot = compute_cllr_from_llr_per_axis(
                    torch.sum(predictions, dim=1).unsqueeze(1), targets, self.fr_hard_negs
                )
                return cllr_tot + self.focus_weight * cllr_focus

            case _:
                raise ValueError(f"Unknown mode: {self.mode}")

    @staticmethod
    def get_cumsum(predictions):
        return torch.cumsum(predictions, dim=1)


class CLLRTrainDynamicallyConcentrated(torch.nn.Module):
    """Dynamically minimize the best performing axes when training.

    This possibly reduces overfitting?

    Procedure:
    Select axes where cllr is smaller than cllr_threshold (the 'focal' axes). Say the number of such axes is n_focus.
    For these axes, compute a 'train_concentrated' llrs, ordering axes according to small cllr.
    For other axes (as long as the number of axes to compute the loss for is smaller/eq than n_axes), per axis,
    add the llr of the axis to the summed llr of the focal axes. Say, the numberof such axes is n_rest.
    Now, we have llrs for n_focus + n_rest axes. Compute cllr per axis for al these axes and return the mean.

    :param fr_hard_negs. When negatives are filtered to not provide easy negatives to the DataLoader,
            Cllr loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
    :param cllr_threshold. The threshold cllr to start focusing axes.
    """

    def __init__(self, fr_hard_negs: float, cllr_threshold: float = 0.95):
        super().__init__()
        self.fr_hard_negs = fr_hard_negs
        self.cllr_threshold = cllr_threshold
        self.cllr_evaluate = CLLR(self.fr_hard_negs)

    def forward(self, prediction_llrs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cllrs = self.cllr_evaluate(prediction_llrs, labels)[:-1]

        indices_focus = torch.nonzero(cllrs <= self.cllr_threshold, as_tuple=True)[0]
        if len(indices_focus) == 0:
            input_llrs = prediction_llrs
        else:
            smaller_cllrs = cllrs[indices_focus]
            sorted_indices_focus = indices_focus[torch.argsort(smaller_cllrs)]
            prediction_llrs_focus = prediction_llrs[:, sorted_indices_focus]
            prediction_llrs_focus_cumsum = self.cllr_evaluate.get_cumsum(prediction_llrs_focus)

            indices_evenly = torch.nonzero(cllrs > self.cllr_threshold, as_tuple=True)[0]
            if len(indices_evenly) == 0:
                input_llrs = prediction_llrs_focus_cumsum
            else:
                prediction_llrs_evenly = prediction_llrs[:, indices_evenly]
                prediction_llrs_evenly = prediction_llrs_evenly + prediction_llrs_focus_cumsum[:, -1].unsqueeze(-1)
                input_llrs = torch.cat((prediction_llrs_focus_cumsum, prediction_llrs_evenly), dim=1)

        # add the sum of llrs for all axes, to ensure that focus is also on decreasing cllr of combined llrs for all
        # axes
        sum_llrs = torch.sum(prediction_llrs, dim=1).unsqueeze(1)
        input_llrs = torch.cat((input_llrs, sum_llrs), dim=1)
        cllr = self.cllr_evaluate.compute_cllr_from_llr(
            input_llrs, labels, self.fr_hard_negs, CllrTrainMode.CONCENTRATED
        )
        return cllr


class CLLREvaluate:
    """
    This loss is used when evaluating a model trained by a dynamically concentrated Cllr (see directly above).

    :param fr_hard_negs. When negatives are filtered to not provide easy negatives to the DataLoader,
            Cllr loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
    :returns in forward: cllrs per axis, for all axes together and for all focal axes together.
    """

    def __init__(self, fr_hard_negs: float):
        self.fr_hard_negs = fr_hard_negs

    def __call__(self, prediction_llrs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction_llrs.dtype != torch.float32:
            prediction_llrs = prediction_llrs.to(torch.float32)
        n_axes = prediction_llrs.shape[1]
        prediction_llrs = concat_with_sum(prediction_llrs)
        last_col = prediction_llrs[:, -1:]
        prediction_llrs = torch.cat((prediction_llrs, last_col), dim=1)
        cllr = compute_cllr_from_llr_per_axis(prediction_llrs, labels, self.fr_hard_negs)
        return cllr, torch.arange(0, n_axes)


class CLLREvaluateDynamicallyConcentrated:
    """
    This loss is used when evaluating a model trained by a dynamically concentrated Cllr (see directly above).

    Procedure:
    Select axes where cllr is smaller than cllr_threshold (the 'focal' axes)
    For these axes, sum the llrs and calculate CLLR
    Also calculate the CLLR per axis for all 0:n_axis axes.
    Return all CLLRs and also return the Cllr for the 'focal' axes. This 'focal_cllr' will be the last item in the list
    of cllrs returned.

    :param fr_hard_negs. When negatives are filtered to not provide easy negatives to the DataLoader,
            Cllr loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
    :param cllr_threshold. The threshold cllr to start focusing axes.
    :returns in forward: cllrs per axis, for all axes together and for all focal axes together.
    """

    def __init__(self, fr_hard_negs: float, cllr_threshold: float):
        self.fr_hard_negs = fr_hard_negs
        self.cllr_threshold = cllr_threshold

    def __call__(self, prediction_llrs: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices_focus = torch.arange(0, prediction_llrs.shape[1])
        # Sum llrs of focal axes.
        # Remove last entry, since this is the cllr for the sum_llrs for all axes. We do not need that.
        cllrs_axes = compute_cllr_from_llr_per_axis(prediction_llrs, labels, self.fr_hard_negs)[:-1]

        indices_not_focus = torch.nonzero(cllrs_axes > self.cllr_threshold, as_tuple=True)[0]
        if len(indices_not_focus) < prediction_llrs.shape[1]:
            mask = ~torch.isin(indices_focus, indices_not_focus)
            indices_focus = indices_focus[mask]

        prediction_llrs_focus = prediction_llrs[:, indices_focus]

        sum_llrs_focus = torch.sum(prediction_llrs_focus, dim=1).unsqueeze(1)
        sum_llrs_all = torch.sum(prediction_llrs, dim=1).unsqueeze(1)
        input_llrs = torch.cat((prediction_llrs, sum_llrs_all, sum_llrs_focus), dim=1)
        cllr = compute_cllr_from_llr_per_axis(input_llrs, labels, self.fr_hard_negs)
        return cllr, indices_focus


class CLLRFromP(torch.nn.Module):
    def __init__(self, fr_hard_negs: float):
        """
        Cllr loss for probabilities as input.

        :param fr_hard_negs: When negatives are filtered to not provide easy negatives to the DataLoader,
            Cllr loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
        """
        super().__init__()
        self.fr_hard_negs = fr_hard_negs

    def forward(self, predictions_p: Tensor, targets: Tensor) -> Tensor:
        # Create masks based on binary labels
        mask_positive = targets == 1
        mask_negative = targets == 0

        # Compute column-wise sums based on masks
        cllr_1 = torch.mean(torch.log2(predictions_p[mask_positive, :]), dim=0)
        cllr_0 = torch.mean(torch.log2(1 - predictions_p[mask_negative, :]), dim=0)

        # correct negative cllr for negatives that were so easy that they were not included in data
        cllr_0 = self.fr_hard_negs * cllr_0
        cllr = (cllr_1 + cllr_0) / -2.0
        cllr = torch.mean(cllr)

        return cllr


def compute_cllr_from_llr_per_axis(predictions_llr: Tensor, targets: Tensor, fr_hard_negs: float = 1.0) -> Tensor:
    ln_lrs = predictions_llr * LOG_10

    mask_positive = targets == 1
    mask_negative = targets == 0

    # Slice first → softplus runs on ~50% of rows each instead of 100%
    ln_lrs_pos = ln_lrs[mask_positive]
    ln_lrs_neg = ln_lrs[mask_negative]

    pos_count = mask_positive.sum()
    neg_count = mask_negative.sum()

    # ln_lrs - softplus(ln_lrs)  ≡  -softplus(-ln_lrs)  → one fewer tensor
    cllr_1 = LN_TO_BASE_TWO * (-softplus(-ln_lrs_pos).sum(dim=0)) / pos_count
    cllr_0 = LN_TO_BASE_TWO * (-softplus(ln_lrs_neg).sum(dim=0)) / neg_count

    cllr_0 = fr_hard_negs * cllr_0
    return (cllr_1 + cllr_0) / -2.0


def concat_with_sum(predictions: torch.Tensor) -> torch.Tensor:
    # Compute col-wise sum for specified number of axes
    col_sum = torch.sum(predictions, dim=1)

    # Append col_sum as a new row to the original tensor
    output = torch.cat((predictions, col_sum.unsqueeze(1)), dim=1)
    return output
