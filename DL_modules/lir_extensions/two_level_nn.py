from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from confidence import Configuration
from lrbenchmark.config.base import config_parser, get_parser_arguments_for_field, parse_pairing_config, pop_field
from lrbenchmark.config.lrsystem_architectures import parse_pipeline
from lrbenchmark.lrsystems.lrsystems import LRSystem
from lrbenchmark.pairing import PairingMethod
from sklearn.pipeline import Pipeline

from DL_modules.config.criterion import parse_loss_function
from DL_modules.criterion import CLLREvaluate
from DL_modules.criterion.base import TlmLossFunction
from DL_modules.custom_layers import ResultTLM, merge_results_tlm, tlm_pair_versions, tlm_single_instance_versions
from DL_modules.dataloader.data_loader import CustomDataLoader, ForensicSingleSampler, SourceDataset
from DL_modules.lir_extensions.save_tlm_nn_test_results import process_specific_source_results, save_tlm_nn_test_results
from DL_modules.lir_extensions.utils import LRSystemNN
from DL_modules.plots.integrated_variance import plot_integrated_vs_fixed
from DL_modules.plots.specific_source import make_ss_cs_plots
from DL_modules.train_validate_loops.pair_wise_llr_head import (
    Track,
    train_and_validate_llr_head,
)
from DL_modules.train_validate_loops.single_instance import train_and_validate_llr_head_single_instance
from DL_modules.train_validate_loops.utils import EmptyResultTLM
from DL_modules.utils import reshape_data_labels


@config_parser
def two_level_pair_nn(config: dict[str, Any], config_context_path: list[str], output_dir: Path) -> LRSystem:
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

    return TwoLevelSystemPairNN(
        name,
        preprocessing_pipeline,
        pairing_function,
        loss_function,
        **config,
        output_dir=output_dir,
    )


@config_parser
def two_level_single_instance_nn(config: dict[str, Any], config_context_path: list[str], output_dir: Path) -> LRSystem:
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

    return TwoLevelSystemSingleInstanceNN(
        name,
        preprocessing_pipeline,
        pairing_function,
        loss_function,
        **config,
        output_dir=output_dir,
    )


class TwoLevelSystemPairNN(LRSystemNN):
    """
    LR system using a neural-network based two-level model.

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
        tlm_pair_version: str,
        tlm_size: int,
        learning_rate: float,
        num_epochs: int,
        output_dir: Path | None,
        show_plots: bool = True,
        axes_to_monitor: str = 'all',
        seed: int | None = None,
        n_axes_to_keep: int = 10,
        threshold_for_focal_evaluate: float | None = None,
        n_ss_train: int = 0,
        n_ss_clip: int = 0,
        save_epoch_past_best_epoch: int | None = None,
        threshold_for_mean_cllr: float | None = None,
        coherence_filter: bool = False,
        coherence_filter_threshold_bias_axis: float | None = None,
        coherence_filter_max_iterations: int | None = None,
        coherence_filter_exclude_llrs_smaller_than: float | None = None,
    ):
        """
        :param pairing_function: a pairing function that generates one-to-one instance pairs
        :param ss_batch_size_train: the number of same-source pairs in a batch of training data
        :param ss_batch_size_validate: the number of same-source pairs in a batch of validation/test data
        :param learning_rate: the learning rate
        :param num_epochs: the number of epochs during training
        :param output_dir: a path of a folder where to write output, or None to write no output
        :param show_plots: a boolean to indicate whether to show plots on screen
        :param axes_to_monitor: a string to indicate which axes should be monitored
        :param seed: the seed for random number generator
        :param n_axes_to_keep: the number of axes that should be kept
        :param threshold_for_focal_evaluate: the threshold for focal evaluation
        :param n_ss_train: the number of same-source pairs in the training data
        :param n_ss_clip: the number of training data outside the clip_value
        :param save_epoch_past_best_epoch: the number of epochs to save stats again after the best epoch
        :param threshold_for_mean_cllr: # when calculating the mean of cllr_train over axes, skip axes with cllr > this
            value
        :param tlm_pair_version: 'one_proj' or 'two_proj' (which tlm version)
        :param coherence_filter: a boolean to indicate whether to train a coherencefilter applied to llrs in evaluation
            mode
        :param coherence_filter_threshold_bias_axis: axes with cllr_train < threshold are treated as discriminating
            axes, others as bias axes
        :param coherence_filter_max_iterations: the maximum number of iterations to use for coherence filtering
        :param coherence_filter_exclude_llrs_smaller_than: exclude llrs in discriminating axes that are smaller than
            this number

        """
        super().__init__(name)
        self.preprocessing_pipeline = preprocessing_pipeline
        self.pairing_function = pairing_function
        self.criterion_train = loss_function
        self.criterion_validate = CLLREvaluate(1)
        self.ss_batch_size_train = ss_batch_size_train
        self.ss_batch_size_validate = ss_batch_size_validate
        self.tlm_size = tlm_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._model = None
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.axes_to_monitor = axes_to_monitor
        self.seed = seed
        self.n_axes_to_keep = n_axes_to_keep
        self.threshold_for_focal_evaluate = threshold_for_focal_evaluate
        self.n_ss_train = n_ss_train
        self.n_ss_clip = n_ss_clip
        self.save_epoch_past_best_epoch = save_epoch_past_best_epoch
        self.threshold_for_mean_cllr = threshold_for_mean_cllr
        self.tlm_pair_version = tlm_pair_version
        self.coherence_filter = coherence_filter
        self.coherence_filter_threshold_bias_axis = coherence_filter_threshold_bias_axis
        self.coherence_filter_max_iterations = coherence_filter_max_iterations
        self.coherence_filter_exclude_llrs_smaller_than = coherence_filter_exclude_llrs_smaller_than

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
        self._model = tlm_pair_versions[self.tlm_pair_version](
            n_axes=self.tlm_size,
            n_for_t_distr=sum(n * (n - 1) // 2 for n in Counter(labels[train_index]).values()),  # number of same_source
            # comparisons in train data
            variance_susp="randn",
            variance_cr="randn",
            variance_between="randn",
            n_axes_to_keep=self.n_axes_to_keep,
            n_clip=self.n_ss_clip,
            n_ss_train=self.n_ss_train,
        )

        # Define loss criterion and optimization parameters
        criterion_train_evaluate = CLLREvaluate(1)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        input_for_criterion_evaluate = Track('norm')

        cfg = Configuration(
            {
                "train": {
                    "ss_batch_size": self.ss_batch_size_train,
                    "n_axes_to_keep": self.n_axes_to_keep,
                },
                "validate": {
                    "ss_batch_size": self.ss_batch_size_validate,
                    "n_ss_clip": self.n_ss_clip,
                },
                "cllr_crit_eval": {
                    "axes_to_monitor": self.axes_to_monitor,
                    "threshold_for_focal_evaluate": self.threshold_for_focal_evaluate,
                },
                "save_epoch_past_best_epoch": self.save_epoch_past_best_epoch,
                "threshold_for_mean_cllr": self.threshold_for_mean_cllr,
            }
        )

        self._model = train_and_validate_llr_head(
            self._model,
            self.name,
            self.criterion_train,
            self.criterion_validate,
            criterion_train_evaluate,
            self.num_epochs,
            optimizer,
            data_train_pairs,
            data_validate_pairs,
            cfg,
            self.tlm_size,
            input_for_criterion_evaluate,
            show_plots=self.show_plots,
            output_dir_lr_benchmark=self.output_dir,
            coherence_filter=self.coherence_filter,
            coherence_filter_threshold_bias_axis=self.coherence_filter_threshold_bias_axis,
            coherence_filter_max_iterations=self.coherence_filter_max_iterations,
            coherence_filter_exclude_llrs_smaller_than=self.coherence_filter_exclude_llrs_smaller_than,
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

        all_predictions = ResultTLM({}, {}, {}, {})
        all_labels = torch.empty(0)
        with torch.no_grad():
            for batch in data_test_pairs:
                susp, cr, batch_labels = batch
                batch_predictions = self._model(susp, cr)
                all_predictions = merge_results_tlm(all_predictions, batch_predictions)
                all_labels = torch.cat((all_labels, batch_labels))

        # TODO: also get predictions and metrics for ["llr_norm"]
        llrs_t = all_predictions.log_t["llr_t"].numpy()
        llrs_t_sum = np.sum(llrs_t, axis=1)

        save_tlm_nn_test_results(all_predictions, all_labels, self.criterion_validate, self.output_dir)

        # define an empty meta np.array for now
        meta = np.array([None] * len(llrs_t_sum))

        return llrs_t_sum, all_labels.numpy(), meta


class TwoLevelSystemSingleInstanceNN(LRSystemNN):
    """
    LR system using a neural-network based two-level model.
    For training, this class takes mini-batches of sources (with all repetitions per source).

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
        tlm_single_instance_version: str,
        tlm_size: int,
        learning_rate: float,
        num_epochs: int,
        output_dir: Path | None,
        show_plots: bool = True,
        axes_to_monitor: str = 'all',
        seed: int | None = None,
        n_axes_to_keep: int = 10,
        threshold_for_focal_evaluate: float | None = None,
        empirical_init: bool = False,
        save_epoch_past_best_epoch: int | None = None,
        threshold_for_mean_cllr: float | None = None,
        sample_size_sigma2_w: float | None = None,
        train_coherence_filters: bool = False,
        coherence_filter_threshold_bias_axis: list[float | None] | None = None,
        coherence_filter_max_iterations: list[int | None] | None = None,
        coherence_filter_exclude_llrs_smaller_than: list[float | None] | None = None,
        train_clip_denominators: bool = False,
        clip_denominator_percentage: list[float] | None = None,
        also_predict_specific_source: bool = False,
    ):
        """
        :param pairing_function: a pairing function that generates one-to-one instance pairs
        :param ss_batch_size_train: the number of same-source pairs in a batch of training data
        :param ss_batch_size_validate: the number of same-source pairs in a batch of validation/test data
        :param learning_rate: the learning rate
        :param num_epochs: the number of epochs during training
        :param output_dir: a path of a folder where to write output, or None to write no output
        :param show_plots: a boolean to indicate whether to show plots on screen
        :param axes_to_monitor: a string to indicate which axes should be monitored
        :param seed: the seed for random number generator
        :param n_axes_to_keep: the number of axes that should be kept
        :param threshold_for_focal_evaluate: the threshold for focal evaluation
        :param empirical_init: a boolean to indicate whether to use empirical maximim likelihood initialization of
            parameters
        :param save_epoch_past_best_epoch: the number of epochs to save stats again after the best epoch
        :param threshold_for_mean_cllr: # when calculating the mean of cllr_train over axes, skip axes with cllr > this
            value
        :param tlm_single_instance_version: 'one_proj' or 'two_proj' (which tlm version)
        :param sample_size_sigma2_w: the number of sigma2_w samples drawn when integrating out the variance uncertainty
        :param train_coherence_filters: a boolean to indicate whether to use coherencefilter over llrs in evaluation
            mode
        :param coherence_filter_threshold_bias_axis: axes with cllr_train < threshold are treated as discriminating
            axes, others as bias axes
        :param coherence_filter_max_iterations: the maximum number of iterations to use for coherence filtering
        :param coherence_filter_exclude_llrs_smaller_than: exclude llrs in discriminating axes that are smaller than
            this number
        :param train_clip_denominators: whether to train a clip on denominator probabilities
        :param clip_denominator_percentage: percentage of training data at which to clip
        :param also_predict_specific_source: whether also predict specific source llrs for best_epoch

        """
        super().__init__(name)
        self.preprocessing_pipeline = preprocessing_pipeline
        self.pairing_function = pairing_function
        self.criterion_train = loss_function
        self.criterion_validate = CLLREvaluate(1)
        self.ss_batch_size_train = ss_batch_size_train
        self.ss_batch_size_validate = ss_batch_size_validate
        self.tlm_size = tlm_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self._model = None
        self.output_dir = output_dir
        self.show_plots = show_plots
        self.axes_to_monitor = axes_to_monitor
        self.seed = seed
        self.n_axes_to_keep = n_axes_to_keep
        self.threshold_for_focal_evaluate = threshold_for_focal_evaluate
        self.empirical_init = empirical_init
        self.save_epoch_past_best_epoch = save_epoch_past_best_epoch
        self.threshold_for_mean_cllr = threshold_for_mean_cllr if threshold_for_mean_cllr else 0.95
        self.tlm_single_instance_version = tlm_single_instance_version
        self.sample_size_sigma2_w = sample_size_sigma2_w
        self.train_coherence_filters = train_coherence_filters
        self.coherence_filter_threshold_bias_axis = coherence_filter_threshold_bias_axis
        self.coherence_filter_max_iterations = coherence_filter_max_iterations
        self.coherence_filter_exclude_llrs_smaller_than = coherence_filter_exclude_llrs_smaller_than
        self.also_predict_specific_source = also_predict_specific_source
        self.train_clip_denominators = train_clip_denominators
        self.clip_denominator_percentage = clip_denominator_percentage

    def fit(self, features: Any, labels: Any, meta: Any) -> "LRSystem":
        if self.seed:
            torch.manual_seed(self.seed)

        train_index, validate_index, _ = self._split_in_three(features, labels, meta)

        if self.preprocessing_pipeline:
            data_to_fit_preprocessing = features[train_index]
            self.preprocessing_pipeline.fit(data_to_fit_preprocessing)
            features = self.preprocessing_pipeline.transform(features)

        train_data_single_sources = features[train_index]
        train_labels_single_sources = labels[train_index]
        validate_data_single_sources = features[validate_index]
        validate_labels_single_sources = labels[validate_index]

        # get single train data in dataloader
        train_data_single_sources, train_labels_single_sources = reshape_data_labels(
            train_data_single_sources, train_labels_single_sources
        )
        n_ss_train = len(train_labels_single_sources)
        source_dataset_train = SourceDataset(train_data_single_sources, train_labels_single_sources)
        single_source_dataloader_train = CustomDataLoader(
            source_dataset_train, ForensicSingleSampler(self.ss_batch_size_train, labels=train_labels_single_sources)
        )

        # get single validation data in dataloader
        validate_data_single_sources, validate_labels_single_sources = reshape_data_labels(
            validate_data_single_sources, validate_labels_single_sources
        )
        source_dataset_validate = SourceDataset(validate_data_single_sources, validate_labels_single_sources)
        single_source_dataloader_validate = CustomDataLoader(
            source_dataset_validate,
            ForensicSingleSampler(self.ss_batch_size_train, labels=validate_labels_single_sources),
        )

        # get pair data in dataloaders
        data_train_pairs = self._get_pair_data_loader(
            features, labels, meta, train_index, self.ss_batch_size_train
        ).dataloader
        data_validate_pairs = self._get_pair_data_loader(
            features, labels, meta, validate_index, self.ss_batch_size_validate
        ).dataloader

        # Define model
        self._model = tlm_single_instance_versions[self.tlm_single_instance_version](
            self.tlm_size,
            n_sources_train_data=n_ss_train,
            numerator_variance_extra="randn",
            denominator_variance_extra="randn",
            sample_size_sigma2_w=self.sample_size_sigma2_w,
            n_axes_to_keep=self.n_axes_to_keep,
            empirical_init=self.empirical_init,
            train_coherence_filters=self.train_coherence_filters,
            coherence_filter_threshold_bias_axis=self.coherence_filter_threshold_bias_axis,
            coherence_filter_max_iterations=self.coherence_filter_max_iterations,
            coherence_filter_exclude_llrs_smaller_than=self.coherence_filter_exclude_llrs_smaller_than,
            train_clip_denominators=self.train_clip_denominators,
            clip_denominator_percentage=self.clip_denominator_percentage,
        )

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)

        self._model = train_and_validate_llr_head_single_instance(
            self._model,
            self.name,
            self.criterion_train,
            self.criterion_validate,
            self.num_epochs,
            optimizer,
            single_source_dataloader_train,
            single_source_dataloader_validate,
            data_train_pairs,
            data_validate_pairs,
            self.threshold_for_mean_cllr,
            output_dir_lr_benchmark=self.output_dir,
            show_plots=self.show_plots,
            save_epoch_past_best_epoch=self.save_epoch_past_best_epoch,
            save=True,
            also_predict_specific_source=self.also_predict_specific_source,
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
        for post_hoc_config_folder, model in self._model.items():
            output_dir = Path(self.output_dir) / post_hoc_config_folder
            all_predictions_cs = EmptyResultTLM
            all_predictions_ss = EmptyResultTLM
            all_labels = torch.empty(0)
            with torch.no_grad():
                for batch in data_test_pairs:
                    susp, cr, batch_labels = batch
                    batch_predictions_cs = model.predict_cs_llr(susp, cr, also_integrate_variance=True)
                    all_predictions_cs = merge_results_tlm(all_predictions_cs, batch_predictions_cs)
                    all_labels = torch.cat((all_labels, batch_labels))
                    if self.also_predict_specific_source:
                        batch_predictions_ss = model.predict_ss_llr(susp, cr, also_integrate_variance=True)
                        all_predictions_ss = merge_results_tlm(all_predictions_ss, batch_predictions_ss)

            save_tlm_nn_test_results(all_predictions_cs, all_labels, self.criterion_validate, output_dir)
            plot_integrated_vs_fixed(all_predictions_cs, all_labels.numpy(), output_dir, suffix='test_cs')

            if self.also_predict_specific_source:
                process_specific_source_results(all_predictions_ss, all_labels, self.criterion_validate, output_dir)
                make_ss_cs_plots(all_predictions_ss, all_predictions_cs, all_labels, output_dir, 'test')
                plot_integrated_vs_fixed(all_predictions_ss, all_labels.numpy(), output_dir, suffix='test_ss')

        # TODO: also get predictions and metrics for ["llr_norm"]
        llrs_t = all_predictions_cs.log_t["llr_t"].numpy()
        llrs_t_sum = np.sum(llrs_t, axis=1)

        # define an empty meta np.array for now
        meta = np.array([None] * len(llrs_t_sum))

        # save results of last model.
        return llrs_t_sum, all_labels.numpy(), meta
