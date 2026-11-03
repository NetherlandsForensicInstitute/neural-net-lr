from pathlib import Path
from typing import Any

import numpy as np
import torch
from confidence import Configuration
from lrbenchmark.config.base import (
    config_parser,
    get_parser_arguments_for_field,
    parse_pairing_config,
    pop_field,
)
from lrbenchmark.config.lrsystem_architectures import parse_pipeline
from lrbenchmark.lrsystems.lrsystems import LRSystem
from lrbenchmark.pairing import PairingMethod
from sklearn.pipeline import Pipeline

from DL_modules.config.criterion import parse_loss_function
from DL_modules.criterion.base import TlmLossFunction
from DL_modules.custom_layers.top_layers import log_reg_versions
from DL_modules.lir_extensions.two_level_nn import LRSystemNN
from DL_modules.train_validate_loops.sigmoid_equal_classweights import (
    train_and_validate_sigmoid_equal_classweights_head,
)


@config_parser
def logreg_nn(config: dict[str, Any], config_context_path: list[str], output_dir: Path) -> LRSystem:
    """
    Parse the configuration for a TwoLevelSystemPairNN.

    :param config: a (hierarchical) dictionary of configuration values in this context
    :param config_context_path: the path to the current config in the full configuration files
    :param output_dir: the directory where to write output files
    """
    name = config.pop("name") if "name" in config else output_dir.name

    pairing_function = parse_pairing_config(
        *get_parser_arguments_for_field(config, config_context_path, output_dir, "pairing")
    )

    preprocessing_pipeline = parse_pipeline(
        *get_parser_arguments_for_field(config, config_context_path, output_dir, "preprocessing")
    )

    loss_function_cfg = pop_field(config_context_path, config, "loss")
    loss_function = parse_loss_function(loss_function_cfg, config_context_path + ["loss"], output_dir)

    return LogRegNN(
        name,
        preprocessing_pipeline,
        pairing_function,
        loss_function,
        **config,
        output_dir=output_dir,
    )


class LogRegNN(LRSystemNN):
    """
    LR system using a neural-network based on logistic regression.

    This class derives from LRSystem and can be used in LRBenchmark experiments.
    """

    def __init__(
        self,
        name: str,
        preprocessing_pipeline: Pipeline | None,
        pairing_function: PairingMethod,
        loss_function: TlmLossFunction,
        ss_batch_size_train: int,
        ss_batch_size_validate: int,
        log_reg_version: str,
        learning_rate: float,
        num_epochs: int,
        output_dir: Path | None,
        show_plots: bool = True,
        seed: int | None = None,
        threshold_for_focal_evaluate: float | None = None,
        n_ss_train: int = 0,
    ):
        """
        :param pairing_function: a pairing function that generates one-to-one instance pairs
        :param ss_batch_size_train: the number of same-source pairs in a batch of training data
        :param ss_batch_size_validate: the number of same-source pairs in a batch of validation/test data
        :param log_reg_version: version of logistic regression nn
        :param learning_rate: the learning rate
        :param num_epochs: the number of epochs during training
        :param output_dir: a path of a folder where to write output, or None to write no output
        :param show_plots: a boolean to indicate whether to show plots on screen
        :param seed: the seed for random number generator
        :param threshold_for_focal_evaluate: the threshold for focal evaluation
            value
        :param n_ss_train: the number of same-source pairs in the training data
        """
        super().__init__(name)
        self.preprocessing_pipeline = preprocessing_pipeline
        self.pairing_function = pairing_function
        self.criterion_train = loss_function
        self.criterion_validate = loss_function
        self.ss_batch_size_train = ss_batch_size_train
        self.ss_batch_size_validate = ss_batch_size_validate
        self.log_reg_version = log_reg_version
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._model = None
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.seed = seed
        self.threshold_for_focal_evaluate = threshold_for_focal_evaluate
        self.n_ss_train = n_ss_train
        self.n_ss_train = n_ss_train

    def fit(self, features: Any, labels: Any, meta: Any) -> "LRSystem":
        if self.seed:
            torch.manual_seed(self.seed)

        train_index, validate_index, _ = self._split_in_three(features, labels, meta)

        if self.preprocessing_pipeline:
            data_to_fit_preprocessing = features[train_index]
            self.preprocessing_pipeline.fit(data_to_fit_preprocessing)
            features = self.preprocessing_pipeline.transform(features)

        # get pair data in dataloaders
        data_train_pairs = self._get_pair_data_loader(
            features, labels, meta, train_index, self.ss_batch_size_train
        ).dataloader
        data_validate_pairs = self._get_pair_data_loader(
            features, labels, meta, validate_index, self.ss_batch_size_validate
        ).dataloader

        # Define model
        self._model = log_reg_versions[self.log_reg_version](
            features.shape[1],
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        cfg = Configuration(
            {
                "train": {
                    "ss_batch_size": self.ss_batch_size_train,
                },
                "validate": {
                    "ss_batch_size": self.ss_batch_size_validate,
                },
                "cllr_crit_eval": {
                    "threshold_for_focal_evaluate": self.threshold_for_focal_evaluate,
                },
                "saving": {"save": True, "output_parent": self.output_dir},
            }
        )

        self._model = train_and_validate_sigmoid_equal_classweights_head(
            self._model,
            self.name,
            self.criterion_train,
            self.criterion_validate,
            self.num_epochs,
            optimizer,
            data_train_pairs,
            data_validate_pairs,
            cfg,
            output_dir_lr_benchmark=self.output_dir,
        )

        return self

    def apply(self, features: Any, labels: np.ndarray | None, meta: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocessing_pipeline:
            features = self.preprocessing_pipeline.transform(features)

        _, _, test_index = self._split_in_three(features, labels, meta)

        # get pair data in dataloaders
        data_test_pairs = self._get_pair_data_loader(
            features, labels, meta, test_index, self.ss_batch_size_validate
        ).dataloader

        # Compute validate loss
        all_predictions = torch.empty(0, 1)
        all_labels = torch.empty(0)
        with torch.no_grad():
            for batch in data_test_pairs:
                susp, cr, batch_labels = batch
                predictions = self._model(susp, cr)
                all_predictions = torch.cat((all_predictions, predictions))
                all_labels = torch.cat((all_labels, batch_labels))

        # define an empty meta np.array for now
        meta = np.array([None] * len(all_predictions))

        lrs = (all_predictions[:, 0] / (1 - all_predictions[:, 0])).numpy()

        return np.log10(lrs), all_labels.numpy(), meta
