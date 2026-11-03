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
        threshold = self.threshold_bias_axis
        large_are_bias_axes = large_are_bias_axes.flatten()

        focus_mask = large_are_bias_axes < threshold

        if torch.all(~focus_mask):
            # then there were no axes with cllr < threshold. Return without any filtering.
            return llrs

        # get llrs for focus axes, set rest to 0
        focus_llrs = focus_mask * llrs

        # find mask for llrs in focus axes with values larger than exclude_smaller_than
        mask_llr_gt_threshold = focus_llrs > self.exclude_llrs_smaller_than

        # get llrs for focus axes that are excluded, rest 0
        llrs_sm_threshold = focus_llrs * ~mask_llr_gt_threshold

        # adjust focus_llrs to include 0's for excluded llrs
        focus_llrs = focus_llrs * mask_llr_gt_threshold

        # get mask that sets 1's for llrs in llrs > exclude_smaller_than
        mask_all_llrs_predictions_gt_threshold = llrs > self.exclude_llrs_smaller_than

        # apply this mask
        llrs_gt_threshold = llrs * mask_all_llrs_predictions_gt_threshold

        # get mask for incoherent rows. These are True
        incoherent_row_mask = self._mask_incoherent_rows(focus_mask, llrs_gt_threshold)

        # get n_zeros per row at start
        n_zeros_at_start_per_row = torch.sum(focus_mask) - torch.sum(focus_mask * llrs_gt_threshold != 0, dim=1)

        i = 0
        while torch.any(incoherent_row_mask) and (i < (self.max_iterations if self.max_iterations else llrs.shape[1])):
            i += 1
            # get min and max llr_indices per row
            max_indices = torch.argmax(focus_llrs, dim=1)
            min_indices = torch.argmin(focus_llrs, dim=1)

            # get rows that are incoherent
            indices = torch.nonzero(incoherent_row_mask).squeeze()

            # apply filter
            llrs_gt_threshold[indices, max_indices[indices]] = 0
            llrs_gt_threshold[indices, min_indices[indices]] = 0

            # update mask
            incoherent_row_mask = self._mask_incoherent_rows(focus_mask, llrs_gt_threshold)

            # update focus_llrs
            focus_llrs = focus_mask * llrs_gt_threshold
        # insert llrs_sm_threshold
        focus_llrs = focus_llrs + llrs_sm_threshold
        # get bias_llrs, rest 0
        bias_axes = ~focus_mask * llrs

        # get n_zeros per row at end, the difference between at_end and at_start is the number of focus
        #   (i.e. discriminating) axes that have been replaced by zeros.
        n_zeros_at_end_per_row = torch.sum(focus_mask) - torch.sum(focus_mask * llrs_gt_threshold != 0, dim=1)

        # calculate fraction to scale bias with, it is the relative number of focus axes that have been replaced by
        # zero's, rowwise.
        fraction_kept = 1 - ((n_zeros_at_end_per_row - n_zeros_at_start_per_row) / torch.sum(focus_mask))

        # insert scaled bias llrs
        llrs = focus_llrs + bias_axes * fraction_kept.unsqueeze(1)

        return llrs

    @staticmethod
    def _mask_incoherent_rows(focus_mask, llrs):
        """Create a mask that for all rows have both a negative and a positive value."""
        focus_mask_applied = focus_mask * llrs

        positive_present = torch.any(focus_mask_applied > 0, dim=1)
        negative_present = torch.any(focus_mask_applied < 0, dim=1)
        return positive_present & negative_present
