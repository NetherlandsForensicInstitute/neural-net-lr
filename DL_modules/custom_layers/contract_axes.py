import torch
from torch import Tensor, nn


class ContractAxes(nn.Module):
    """Contract axes mapping x to contract(x) according to following constraints.

    (approximately, since contact(x) is approximated, see function contract):
    x >= contract(x)
    contract(x) = - contract(-x)
    when x_i < x_j and x_i >= 0:
        contract(x_i) <= contract(x_j)
        x_j - contract(x_j) <= x_i - contract(x_i)
        contract(x_j) >= contract(x_i)


    This is realized by defining the derivative of the contract function to be
    - a continuously decreasing function in the domain 0 >= x >= inf where derivative(0) == 1 and derivative(inf) == 0.
    - a continuously increasing function in the domain  -inf >= x >= 0 where derivative(-inf) = 0 and
        derivative(0) == 1.
    The latter function is the antisymmetric function of the former function.

    Function definition of derivative: derivative(x) = 1/(exp(a(x-b)) + 1), with a and b > 0.
    This is the logistic function with y=1 at -Inf.

    The function allows for the training of a weighted mean of a number of such functions
    (the number is specified by n_components). The multipliers are the unnormalized weights.
    """

    def __init__(self, input_size: int, n_components: int, a_factor: float = 1.0, b_factor: float = 1.0):
        super().__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.log10_a = nn.Parameter(self.get_log10_parameters(self.input_size, self.n_components))
        self.log10_b = nn.Parameter(self.get_log10_parameters(self.input_size, self.n_components))
        self.log10_multipl = nn.Parameter(self.get_log10_multipliers(self.input_size, self.n_components))
        self.a_factor = a_factor
        self.b_factor = b_factor

    """
    :param input_size the length of an input vector. e.g.: if a TLM of n features is used, this value should be n.
    :param n_components: the number of contract functions to take a normalizing sum over per axes.
    :param a_factor: in a(x - b) above, all a trainable :parameters are multiplied by the a_factor.
    :param b_factor: in a(x - b) above, all b trainable :parameters are multiplied by the b_factor.
    """

    def forward(self, x: Tensor) -> Tensor:
        a = torch.pow(10, self.log10_a) * self.a_factor
        b = torch.pow(10, self.log10_b) * self.b_factor
        multipliers = torch.pow(10, self.log10_multipl)
        contraction = self.contract(x, a, b, multipliers)
        return contraction

    @staticmethod
    def get_log10_multipliers(input_size: int, n_components: int) -> torch.tensor:
        """
        Multipliers are used to multiply the n_component contracted values and then divide them by the sum of the
        multipliers (resulting in a weighted average).
        10^^ exponentiation leads to :parameters that are initialized to 1 (first component) or values that show small
        deviations from 0.01 (other components).

        """
        log10_multipliers = torch.cat(
            (torch.zeros(input_size).unsqueeze(1), -2 + 1 / 100 * torch.randn(input_size, n_components - 1)), dim=1
        )
        return log10_multipliers

    @staticmethod
    def get_log10_parameters(input_size: int, n_components: int) -> torch.tensor:
        """
        Parameters are initialized as log10 to get trainable :parameters on the real line.
        10^^ exponentiation leads to :parameters that are initialized to 1 (first component) or values that show small
        deviations from 1 (other components).
        """
        log10_parameters = torch.cat(
            (torch.zeros(input_size).unsqueeze(1), 1 / 100 * torch.randn(input_size, n_components - 1)), dim=1
        )
        return log10_parameters

    @staticmethod
    def contract(x: Tensor, a: Tensor, b: Tensor, multipliers: Tensor) -> Tensor:
        x_abs = torch.abs(x)
        x_expanded = x_abs.unsqueeze(-1)
        ax = x_expanded * a
        ab = a * b
        constants = torch.full_like(ax, 0)
        constants_ab = torch.full_like(ab, 0)
        stacked = torch.stack([ab - ax, constants])
        stacked_ab = torch.stack([ab, constants_ab])
        normalizer = torch.sum(multipliers, dim=1, keepdim=True)
        weights = multipliers / normalizer
        ln_terms = (-1 * torch.logsumexp(stacked, dim=0) + torch.logsumexp(stacked_ab, dim=0)) * weights / a
        ln_terms = torch.sum(ln_terms, dim=-1)
        result = torch.sign(x) * ln_terms

        return result


class TwoContractAxes:
    """
    Initialize two ContracAxes layers with specified initial weights.
    """

    def __init__(
        self,
        input_size: int,
        n_components: int,
        a_factor_1st: float = 1.0,
        b_factor_1st: float = 1.0,
        a_factor_2nd: float = 1.0,
        b_factor_2nd: float = 1.0,
    ):
        self.n_components = n_components
        self.input_size = input_size
        self.a_factor_1st = a_factor_1st
        self.b_factor_1st = b_factor_1st
        self.a_factor_2nd = a_factor_2nd
        self.b_factor_2nd = b_factor_2nd

    @property
    def layers(self) -> tuple[nn.Module, nn.Module]:
        contract_axes_1 = ContractAxes(self.input_size, self.n_components, self.a_factor_1st, self.b_factor_1st)
        contract_axes_2 = ContractAxes(self.input_size, self.n_components, self.a_factor_2nd, self.b_factor_2nd)
        return contract_axes_1, contract_axes_2
