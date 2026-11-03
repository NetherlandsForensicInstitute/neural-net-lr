from enum import Enum
from pathlib import Path
from typing import Any

from lrbenchmark import registry
from lrbenchmark.config.base import GenericConfigParser, YamlParseError, config_parser, pop_field

from DL_modules.criterion.base import CumulativeLoss, ProductLoss, TlmLossFunction


def parse_loss_function(
    module_config: dict[str, Any] | str,
    config_context_path: list[str],
    output_dir: Path,
) -> TlmLossFunction:
    """
    Construct a loss function from a string or configuration section.

    The configuration section must have the field `method`, which is an object that name is looked up the registry. All
    other fields are initialization arguments. If no arguments are required, the input can be just the object name
    instead.
    """
    if isinstance(module_config, str):
        class_name = module_config
        args = {}
    else:
        args = dict(module_config)
        class_name = pop_field(config_context_path, args, "method")

    return registry.get(class_name, GenericConfigParser, search_path=["torch_modules"]).parse(
        args, config_context_path, output_dir
    )


def as_list(arg: Any) -> list:
    if isinstance(arg, list):
        return arg
    elif isinstance(arg, dict):
        return list(arg.values())
    else:
        raise ValueError(f"not a list: {arg}")


class Operation(Enum):
    SUM = "sum"
    PRODUCT = "product"


@config_parser
def combined_loss(config: dict[str, Any], config_context_path: list[str], output_dir: Path) -> TlmLossFunction:
    """
    Config parser for a loss function that uses values from other loss functions.
    """
    loss_functions_cfg = pop_field(config_context_path, config, "functions", validate=as_list)
    if len(loss_functions_cfg) == 0:
        raise YamlParseError(config_context_path + ["functions"], "at least one loss function requried")
    loss_functions = []
    weights = []
    for cfg in loss_functions_cfg:
        weights.append(pop_field(config_context_path, cfg, "weight", validate=float, default=1.0))
        loss_functions.append(parse_loss_function(cfg, config_context_path, output_dir))

    operation = pop_field(config_context_path, config, "operation", validate=Operation, default=Operation.SUM)

    match operation:
        case Operation.SUM:
            return CumulativeLoss(loss_functions, weights)
        case Operation.PRODUCT:
            return ProductLoss(loss_functions, weights)
        case _:
            raise YamlParseError(config_context_path, "invalid operation")
