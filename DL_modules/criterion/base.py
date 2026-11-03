import warnings
from abc import ABC, abstractmethod

import torch

from DL_modules.utils import ResultTLM


class TlmLossFunction(torch.nn.Module, ABC):
    """
    Abstract base class for loss functions for use in a two-level-model torch network.
    """

    @abstractmethod
    def forward(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class CombinedLoss(TlmLossFunction, ABC):
    """
    Abstract base class for loss functions that takes other loss functions, and combines their losses.
    """

    def __init__(self, loss_functions: list[TlmLossFunction], weights: list[float] | None = None):
        super().__init__()
        self.loss_functions = loss_functions
        self.weights = torch.Tensor(weights or [1] * len(loss_functions))

    def _calculate_losses(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        losses = [loss_function(result_tlm, targets) for loss_function in self.loss_functions]

        # check for sanity: the array should be 0-dimensional
        for i, loss in enumerate(losses):
            if len(loss.size()) > 0:
                warnings.warn(f"loss {i} has excessive dimensions: {loss.size()}")

        # make sure all tensors have the same size; this succeeds as long as the tensor has a single value
        losses = [torch.reshape(loss, ()) for loss in losses]

        return torch.stack(losses)


class CumulativeLoss(CombinedLoss):
    """
    Loss function that takes other loss functions as initialization arguments.

    The returned loss is the sum of the losses returned by the delegate loss functions.
    """

    def forward(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        losses = self._calculate_losses(result_tlm, targets)
        return torch.sum(losses * self.weights)


class ProductLoss(CombinedLoss):
    """
    Loss function that takes other loss functions as initialization arguments.

    The returned loss is the product of the losses returned by the delegate loss functions.
    """

    def forward(self, result_tlm: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        losses = self._calculate_losses(result_tlm, targets)
        return torch.prod(torch.pow(losses, self.weights))
