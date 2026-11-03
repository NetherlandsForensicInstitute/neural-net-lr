import torch

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.criterion.cllr import compute_cllr_from_llr_per_axis
from DL_modules.custom_layers import ResultTLM


class VarianceLoss(TlmLossFunction):
    """
    Loss function that calculates the ratio of the variances (var(H0)/var(H1)) for the focus axes or all axis.
    """

    def __init__(self, focus_threshold: float | None):
        super().__init__()
        self.focus_threshold = focus_threshold

    def forward(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        predictions = result_tlm.log_norm["llr_norm"]
        variances_1 = result_tlm.variances["variances_1"]
        variances_0 = result_tlm.variances["variances_0"]

        # calculate cllr per axis, since we only want to apply the variance loss to the axes with cllr lower than a
        # certain threshold (focus axes)
        if self.focus_threshold is not None:
            cllr_by_axis = compute_cllr_from_llr_per_axis(predictions, targets)
            mask = cllr_by_axis < self.focus_threshold
            variance_ratios = torch.var(variances_0[mask] / variances_1[mask])
        else:
            variance_ratios = torch.var(variances_0 / variances_1)

        return variance_ratios
