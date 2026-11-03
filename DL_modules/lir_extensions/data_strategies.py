from pathlib import Path
from typing import Any

from lrbenchmark.config.base import config_parser, get_parser_arguments_for_field
from lrbenchmark.config.data_providers import parse_data_provider
from lrbenchmark.data.models import DataStrategy


class IdentityDataStrategy(DataStrategy):
    """
    A trivial DataStrategy that just yields the entire dataset once.
    Useful for cases where you just want to provide the dataset without splits.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        # Yield the full dataset as a single "split" (the second split is None)
        yield (self.dataset.get_instances()), (self.dataset.get_instances())


def _provide_data(config: dict[str, Any], config_context_path: list[str], output_path: Path) -> DataStrategy:
    dataset = parse_data_provider(*get_parser_arguments_for_field(config, config_context_path, output_path, "source"))
    return IdentityDataStrategy(dataset)


@config_parser
def just_provide_data(config: dict[str, Any], config_context_path: list[str], output_path: Path) -> DataStrategy:
    return _provide_data(config, config_context_path, output_path)
