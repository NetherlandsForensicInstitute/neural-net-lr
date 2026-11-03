import os
from pathlib import Path

import confidence
import numpy as np
import torch
from confidence import Configuration
from lrbenchmark.pairing import SourcePairing
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from DL_modules.criterion import CLLR, CLLREvaluate
from DL_modules.criterion.cllr import CllrTrainMode
from DL_modules.custom_layers.top_layers import SingleInstanceTwoLevelNormalNormalTwoProj
from DL_modules.dataloader import SourceDataset
from DL_modules.dataloader.data_loader import CustomDataLoader, ForensicSingleSampler
from DL_modules.lir_extensions.utils import _Data
from DL_modules.train_validate_loops.single_instance import train_and_validate_llr_head_single_instance
from DL_modules.train_validate_loops.utils import get_next_entry, get_output_path
from DL_modules.utils import reshape_data_labels, save_configuration

model_name = "two_proj_single_instance"


def _get_data_loader(
    pairing_function,
    features: np.ndarray,
    labels: np.ndarray,
    meta: np.ndarray,
    indexes: np.ndarray | None,
    ss_batch_size: int,
) -> _Data:
    if indexes is not None:
        features, labels, meta = features[indexes], labels[indexes], meta[indexes]

    pair_features, pair_labels, pair_meta = pairing_function.pair(features, labels, meta)
    return _Data(pair_features, pair_labels, pair_meta, ss_batch_size)


def _split_data(features: np.ndarray, labels: np.ndarray, meta: np.ndarray, second_size: float) -> tuple[_Data, _Data]:
    splitter = GroupShuffleSplit(test_size=second_size, n_splits=1)
    ((train_index, validate_index),) = splitter.split(features, labels, labels)
    return train_index, validate_index


def train_tlm_dl_model(cfg: Configuration):
    # get seed from config or from output folder
    seed = cfg.train.seed if cfg.train.seed else get_next_entry(cfg.saving.output_parent)
    # set seed for torch globally
    torch.manual_seed(seed)
    # get data
    features = np.loadtxt(Path("DL_modules/resources/LR_benchmark/glass_duplos/data.csv"), delimiter=';')
    labels = np.loadtxt(Path("DL_modules/resources/LR_benchmark/glass_duplos/labels.csv")).astype(int)
    meta = labels.copy()
    train_validate_index, test_index = _split_data(features, labels, meta, 1 / 3)
    train_validate_data = features[train_validate_index]
    train_validate_labels = labels[train_validate_index]

    train_index, validate_index = _split_data(train_validate_data, train_validate_labels, meta, 0.5)
    train_data_single_sources = train_validate_data[train_index]
    train_labels_single_sources = train_validate_labels[train_index]
    validate_data_single_sources = train_validate_data[validate_index]
    validate_labels_single_sources = train_validate_labels[validate_index]

    # preprocess
    standard_scaler = StandardScaler()
    standard_scaler.fit(train_data_single_sources)
    train_data_single_sources = standard_scaler.transform(train_data_single_sources)
    validate_data_single_sources = standard_scaler.transform(validate_data_single_sources)
    train_validate_data = standard_scaler.transform(train_validate_data)
    features = standard_scaler.transform(features)

    # get single train data in dataloader
    train_data_single_sources, train_labels_single_sources = reshape_data_labels(
        train_data_single_sources, train_labels_single_sources
    )
    source_dataset_train = SourceDataset(train_data_single_sources, train_labels_single_sources)
    single_source_dataloader_train = CustomDataLoader(
        source_dataset_train, ForensicSingleSampler(batch_size=50, labels=train_labels_single_sources)
    )

    # get single validation data in dataloader
    validate_data_single_sources, validate_labels_single_sources = reshape_data_labels(
        validate_data_single_sources, validate_labels_single_sources
    )
    source_dataset_validate = SourceDataset(validate_data_single_sources, validate_labels_single_sources)
    single_source_dataloader_validate = CustomDataLoader(
        source_dataset_validate, ForensicSingleSampler(batch_size=50, labels=validate_labels_single_sources)
    )

    # get pair data in dataloaders
    pairing_function = SourcePairing()
    data_train_pairs = _get_data_loader(
        pairing_function, train_validate_data, train_validate_labels, meta, train_index, cfg.train.ss_batch_size
    ).dataloader
    data_validate_pairs = _get_data_loader(
        pairing_function, train_validate_data, train_validate_labels, meta, validate_index, cfg.validate.ss_batch_size
    ).dataloader
    # data_test_pairs = _get_data_loader(pairing_function, features, labels, meta, test_index,
    # cfg.train.ss_batch_size_test).dataloader

    tlm_size = cfg.train.tlm_size

    # Define model
    top_layer = SingleInstanceTwoLevelNormalNormalTwoProj(
        tlm_size,
        train_data_single_sources.shape[0],
        numerator_variance_extra="randn",
        denominator_variance_extra="randn",
        sample_size_sigma2_w=cfg.train.sample_size_sigma2_w,
        empirical_init=cfg.train.empirical_init,
        n_axes_to_keep=cfg.train.n_axes_to_keep,
        train_coherence_filters=cfg.validate.train_coherence_filters,
        coherence_filter_threshold_bias_axis=cfg.validate.coherence_filter_threshold_bias_axis,
        coherence_filter_max_iterations=cfg.validate.coherence_filter_max_iterations,
        coherence_filter_exclude_llrs_smaller_than=cfg.validate.coherence_filter_exclude_llrs_smaller_than,
        train_clip_denominators=cfg.train_clip_denominators,
        clip_denominator_percentage=cfg.clip_denominator_percentage,
    )

    # model = PairToOutput(susp_net, cr_net, top_layer)
    model = top_layer

    # # Get the model parameters
    # parameters = list(model.parameters())
    #
    # # Print the parameters
    # for param in parameters:
    #     print(param)

    # Define loss criterion and optimization parameters
    criterion_train = CLLR(
        1,
        CllrTrainMode(cfg.cllr_crit_train.mode),
        **cfg.cllr_crit_train.mode_options,
    )
    criterion_validate = CLLREvaluate(1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    num_epochs = cfg.train.num_epochs

    train_and_validate_llr_head_single_instance(
        model,
        model_name,
        criterion_train,
        criterion_validate,
        num_epochs,
        optimizer,
        single_source_dataloader_train,
        single_source_dataloader_validate,
        data_train_pairs,
        data_validate_pairs,
        cfg.threshold_for_mean_cllr,
        output_parent=cfg.saving.output_parent,
        save=cfg.saving.save,
        show_plots=cfg.show_plots,
        also_predict_specific_source=cfg.validate.also_predict_specific_source,
        save_epoch_past_best_epoch=cfg.save_epoch_past_best_epoch,
    )


if __name__ == '__main__':
    cfg = confidence.load_name(
        'dl-data-params-glass-1-rep-train-validate-test',
        'train_tlm_two_proj_single_instance_glass_1_rep_initrandn_naive_train_validate_test',
    )

    if cfg.saving.save and cfg.saving.output__parent:
        output_path = get_output_path(cfg.saving.output_parent)
        # save configuration
        os.makedirs(Path(output_path), exist_ok=True)
        save_configuration(cfg, Path(output_path) / "configuration.yaml")

    num_runs = cfg.trian.num_runs if cfg.train.num_runs else 1

    for _ in range(num_runs):
        train_tlm_dl_model(cfg)
