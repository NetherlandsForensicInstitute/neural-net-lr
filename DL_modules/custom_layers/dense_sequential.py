import torch
from torch import Tensor, nn


class OneLinearELU(nn.Module):
    """
    Dense layer with ELU activation. ELU is used instead of common RELU, since RELU would result in unwanted 0's for
    linear transformations of large negative values. Large negative values are associated with DS,
    while values around 0 are associated with SS.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_size, output_size), nn.ELU())

    """
    :param input_size: size of input vector (or length of -1 dimension of input batch)
    :param output size: size of output vector (or length of -1 dimension of output batch)
    """

    def forward(self, x: list[Tensor]) -> Tensor:
        return self.model(x)


class SandwichedELU(nn.Module):
    """
    Two Dense layers with ELU activation after the first layes.
    ELU is used instead of common RELU, since RELU would result in unwanted 0's for
    linear transformations of large negative values. Large negative values are associated with DS,
    while values around 0 are associated with SS.
    Output is on the real line, and may be used as input if TLM layer.
    """

    def __init__(self, input_size: int, intermediate_size: int, output_size: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, intermediate_size), nn.ELU(), nn.Linear(intermediate_size, output_size)
        )

    def forward(self, x: list[Tensor]) -> Tensor:
        return self.model(x)


class MediumDenseNet(nn.Module):
    """
    Dense net with ELU activation after all layers.
    ELU is used instead of common RELU, since RELU would result in unwanted 0's for
        linear transformations of large negative values. Large negative values are associated with DS,
        while values around 0 are associated with SS.
    """

    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ELU())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.ELU())
        self.model = nn.Sequential(*layers)
        """
        :param input_size:
        """

    def forward(self, x: list[Tensor]) -> Tensor:
        return self.model(x)


class TwoLinearSameInitializations(nn.Module):
    """
    Initialize two linear layers with the same initial weights and biases.

    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)

    def copy_weights_biases(self):
        with torch.no_grad():
            self.linear2.weight.copy_(self.linear1.weight.clone())
            self.linear2.bias.copy_(self.linear1.bias.clone())

    @property
    def get_layers(self):
        self.copy_weights_biases()
        return self.linear1, self.linear2
