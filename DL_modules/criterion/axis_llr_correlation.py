import logging

import torch

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.utils import ResultTLM

LOG = logging.getLogger(__name__)


class AxisLlrCorrelationLoss(TlmLossFunction):
    """
    Loss function that penalizes a correlation between the LLR along different axes.
    """

    def __init__(self, add_intercept: bool = False):
        super().__init__()
        self.add_intercept = add_intercept

    def forward(self, tlm_result: ResultTLM, targets: torch.Tensor) -> torch.Tensor:
        predictions = tlm_result.log_norm["llr_norm"].double()

        # get same source LLRs
        same_source_llrs = predictions[targets == 1]

        # return the scaled mean correlation
        correlation_penalty = self.compute_mean_correlation(same_source_llrs)
        LOG.debug(f'Correlation penalty: {correlation_penalty}')

        if self.add_intercept:
            intercept_penalty = self.compute_intercept_penalty(same_source_llrs)
            LOG.debug(f'Intercept penalty: {intercept_penalty}')
            correlation_penalty += intercept_penalty

        return correlation_penalty

    def compute_intercept_penalty(self, values) -> torch.Tensor:
        intercepts_matrix = self.compute_intercepts(values)
        n_axis = intercepts_matrix.shape[0]
        # calculate average intercepts of the absolute values
        average_intercepts = torch.sum(torch.abs(intercepts_matrix)) / (n_axis * (n_axis - 1))
        return average_intercepts

    @staticmethod
    def compute_mean_correlation(values: torch.Tensor) -> torch.Tensor:
        # compute the correlations between the variables
        pcc_matrix = torch.corrcoef(values.T)
        n_row, n_col = pcc_matrix.shape
        # get the absolute values of the upper triangular matrix (excluding the main diagonal)
        triu_indices = torch.triu_indices(n_row, n_col, offset=1)
        correlations = torch.abs(pcc_matrix[triu_indices[0], triu_indices[1]])
        return torch.mean(correlations)

    @staticmethod
    def compute_intercepts(values: torch.Tensor) -> torch.Tensor:
        n_axis = values.shape[1]
        intercepts_matrix = torch.zeros((n_axis, n_axis))  # Matrix voor intercepts

        for i in range(n_axis):
            for j in range(n_axis):
                if i != j:
                    # Set the axis to perform linear regression
                    Y_i = values[:, i].view(-1, 1)  # dependent variable
                    X_j = values[:, j].view(-1, 1)  # independent variable

                    # Add column of ones for intercept
                    X_j_with_intercept = torch.cat((torch.ones(X_j.shape[0], 1), X_j), dim=1)

                    try:
                        coefficients = (
                            torch.inverse(X_j_with_intercept.t().mm(X_j_with_intercept))
                            .mm(X_j_with_intercept.t())
                            .mm(Y_i)
                        )
                        intercepts_matrix[i, j] = coefficients[0].item()
                    except RuntimeError as e:
                        LOG.debug(f"Could not calculate inverse between axis {i} and axis {j}. Error {e}")

        return intercepts_matrix
