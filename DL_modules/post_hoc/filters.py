import numpy as np
import torch


class CoherenceFilter:
    """A CoherenceFilter is a post-hoc filter on llrs_per_axis based on the observations that overfitting leads to
    incoherent LLRs over axes.
    """

    def __init__(
        self,
        threshold_bias_axis: float | None = None,
        max_iterations: int | None = None,
        exclude_llrs_smaller_than: float | None = None,
    ):
        """
        :param threshold: threshold value to distinguish between focus and bias axes. larger_are_focus_axes >= threshold
            are bias axes.
        :param max_iterations: maximum number of iterations to perform in while loop.
        :param exclude_llrs_smaller_than: llrs in focus axes smaller than this value are excluded.
        """
        self.threshold_bias_axis = threshold_bias_axis if threshold_bias_axis else 0.95
        self.max_iterations = max_iterations
        self.exclude_llrs_smaller_than = exclude_llrs_smaller_than if exclude_llrs_smaller_than else -np.inf

    def apply(
        self,
        llrs: torch.Tensor,
        large_are_bias_axes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a coherence filter. In general, this removes the extreme focus axes as long as these are not
        coherent.

        It does the following: first, it groups axes in 'bias' and 'discrimination' (i.e. 'focus') axes.
        Then, for each comparison, it checks for the discrimination axes, if LLRs are incoherent
        (e.g. if positive and negative LLRs exists)
        If so, keep substituting the largest and smallest LLR by 0 until the remaining LLRs are coherent.
        Say n out of m focus_LLRs have been substituted by this procedure.
        Then, multiply the value of each bias axes by the fraction (m - n) / m.

        :param llrs: llrs to edit based on their coherence.
        :param large_are_bias_axes: tensor of floats, used to determine focus/bias axes. E.g. cllrs or variance ratios
            per axes.


        """
        if llrs.dtype != torch.float32:
            llrs = llrs.to(torch.float32)

        threshold = self.threshold_bias_axis
        large_are_bias_axes = large_are_bias_axes.flatten()

        focus_mask = large_are_bias_axes < threshold

        if torch.all(~focus_mask):
            # then there were no axes with cllr < threshold. Return without any filtering.
            return llrs

        n_rows, n_cols = llrs.shape

        # get llrs for focus axes, set rest to 0
        focus_llrs = focus_mask * llrs

        # find mask for llrs in focus axes with values larger than exclude_smaller_than
        mask_llr_gt_threshold = focus_llrs > self.exclude_llrs_smaller_than

        # get llrs for focus axes that are excluded, rest 0
        llrs_sm_threshold = focus_llrs * ~mask_llr_gt_threshold

        # adjust focus_llrs to include 0's for excluded llrs
        focus_llrs = focus_llrs * mask_llr_gt_threshold

        # get n_zeros per row for focus_llrs
        n_zeros_at_start_per_row = (focus_llrs == 0).sum(dim=1)
        max_incoherent_pairs_to_drop = torch.tensor(self.max_iterations if self.max_iterations else n_cols // 2)

        pos = (focus_llrs > 0).sum(dim=1)
        neg = (focus_llrs < 0).sum(dim=1)
        imbalance = torch.abs(pos - neg)

        n_incoherent_pairs_to_drop = torch.minimum(
            (n_cols - imbalance - n_zeros_at_start_per_row) / 2,
            max_incoherent_pairs_to_drop,
        ).long()  # make sure it's integer

        sorted_descending = torch.argsort(focus_llrs, dim=1, descending=True)

        # Build a boolean mask in sorted-space, then scatter it back to
        # original column-space. Avoids materialising row_idx / col_idx.
        col_range = torch.arange(
            n_cols,
        ).unsqueeze(0)  # (1, n_cols)
        k = n_incoherent_pairs_to_drop.unsqueeze(1)  # (n_rows, 1)
        sorted_space_mask = (col_range < k) | (col_range >= n_cols - k)

        drop_mask = torch.zeros(n_rows, n_cols, dtype=torch.bool)
        drop_mask.scatter_(1, sorted_descending, sorted_space_mask)
        focus_llrs[drop_mask] = 0.0

        focus_llrs = focus_llrs + llrs_sm_threshold
        # get bias_llrs, rest 0
        bias_axes = ~focus_mask * llrs

        # calculate fraction to scale bias with, it is the relative number of focus axes that have been replaced by
        # zero's, rowwise.
        fraction_kept = 1 - (n_incoherent_pairs_to_drop * 2 / torch.sum(focus_mask))

        # insert scaled bias llrs
        llrs = focus_llrs + bias_axes * fraction_kept.unsqueeze(1)

        return llrs
