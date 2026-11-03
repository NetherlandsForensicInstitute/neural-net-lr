import torch
from torch import Tensor


class Brier(torch.nn.Module):
    """
    Brier loss with equal class weights.

    :param fr_hard_negs. When negatives are filtered to not provide easy negatives to the DataLoader,
        Brier loss corrects the class weights by using the fr_hard_negatives in the original negatives data.
    :param from_p_or_llr. Whether predictions of the neural network are probabilities or log LRs.
    :param mode. The mode for which to compute the Brier score when predictins are llrs.
        There are 4 modes when training in llr mode:
            Train_concentrated: LLRs are calculated by the cumulative sum of axes 0:n for n = 0:n_axes.
                A Brier score is computed per n, and the mean is returned.
                The result of this procedure is that after training, the Brier score of an axis with smaller n is
                smaller than a Brier score of an axis with larger n. This may reduce overfitting?
            Train_evenly: Brier scores are computed per axis, for axes 0:n_axes. The mean is returned.
                The result of this procedure is that after training, scores for 0:n_axes will be approximately equal.
            Train_random_projection: Compute a Brier score for the sum of the LLRs, summed over 0:n_axis.
                The result of this procedure is that after the TLM model is trained without a preference for which
                axis to optimize.
            Evaluate: compute a Brier score per axis and for all axes together (by summing LLRs over all axes).
                Return all Brier scores.
            When prediction are 'from_p', the mean of the Brier score per axis is returned, without a summed_llr.
    """

    def __init__(self, fr_hard_negs: float, from_p_or_llr: str, mode: str | None = None):
        super().__init__()
        self.fr_hard_negs = fr_hard_negs
        self.from_p_or_llr = from_p_or_llr
        self.mode = mode

        # ensure from_p_or_llr is properly specified
        if self.from_p_or_llr not in ("from_p", "from_llr"):
            raise ValueError("Brier criterion: from_p_or_llr must be 'from_p' or 'from_llr'")

        if self.from_p_or_llr == "from_llr":
            # then ensure mode is properly specified
            if self.mode not in ["evaluate", "train_concentrated", "train_evenly", "train_random_projection"]:
                raise ValueError(
                    "Brier criterion: mode must be either 'train_concentrated', 'train_evenly',"
                    " 'train_random_projection' or 'evaluate'"
                )
            if self.mode in ["train_evenly", "train_random_projection", "evaluate"]:
                # then ensure n_axes is properly specified
                if not isinstance(self.n_axes, int) and not self.n_axes == "all":
                    raise ValueError("Brier criterion: please specify a valid n_axes by providing an integer")

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.double()
        if not predictions.shape[0] == targets.shape[0]:
            raise AssertionError("Brier criterion: the number of predictions is not equal to the number of targets!")
        if not predictions.ndim == 2:
            raise AssertionError(
                "Brier criterion: predictions should have dimensions [batch size, len(prediction_vector)]"
            )
        if not targets.ndim == 1:
            raise AssertionError("Brier criterion: targets should have dimension (batch size)")

        if self.from_p_or_llr == "from_llr":
            if self.mode == "train_concentrated":
                predictions = self.get_cumsum(predictions)
            if self.mode == "train_evenly":
                predictions = self.concat_with_sum(predictions)
            if self.mode == "train_random_projection":
                predictions = torch.sum(predictions, dim=1).unsqueeze(1)
            if self.mode == "evaluate":
                predictions = self.concat_with_sum(predictions)
            predictions = self.to_p(predictions)
        brier = self.compute_brier(predictions, targets, self.fr_hard_negs, self.mode, self.from_p_or_llr)

        return brier

    @staticmethod
    def concat_with_sum(predictions: torch.Tensor, n_axes) -> torch.Tensor:
        col_sum = torch.sum(predictions, dim=1)

        # Append col_sum as a new row to the original tensor
        output = torch.cat((predictions, col_sum.unsqueeze(1)), dim=1)
        return output

    @staticmethod
    def get_cumsum(predictions):
        return torch.cumsum(predictions, dim=1)

    @staticmethod
    def to_p(llrs: torch.Tensor) -> torch.Tensor:
        llrs = llrs.double()
        lrs = torch.pow(10, llrs)
        ps = lrs / (lrs + 1)
        return ps

    @staticmethod
    def compute_brier(
        predictions: Tensor, targets: Tensor, fr_hard_negs: float, mode: None | int, from_p_or_llr
    ) -> Tensor:
        # Create masks based on binary labels
        mask_positive = targets == 1
        mask_negative = targets == 0

        # Compute column-wise sums based on masks
        brier_1 = torch.mean(torch.square(1 - predictions[mask_positive, :]), dim=0)
        brier_0 = torch.mean(torch.square(-predictions[mask_negative, :]), dim=0)

        # correct negative brier for negatives that were so easy that they were not included in data
        brier_0 = fr_hard_negs * brier_0
        brier = (brier_1 + brier_0) / 2.0

        if mode in ("train_evenly", "train_concentrated") or from_p_or_llr == "from_p":
            brier = torch.mean(brier)

        return brier
