import os
from pathlib import Path

import confidence
import numpy as np
import pytest
import torch
from confidence import Configuration
from lrbenchmark.pairing import SourcePairing
from sklearn.preprocessing import StandardScaler

from DL_modules.criterion import CLLR
from DL_modules.criterion.cllr import CLLREvaluate, CLLREvaluateDynamicallyConcentrated, CLLRFromP, CllrTrainMode
from DL_modules.custom_layers import (
    SingleInstanceTwoLevelNormalNormalOneProj,
    TopLayerSubtractLogreg,
    TwoLevelNormalNormalOneProj,
)
from DL_modules.dataloader import SourceDataset, get_glass_1_rep_dataloaders_ttv
from DL_modules.dataloader.data_loader import CustomDataLoader, ForensicSingleSampler
from DL_modules.train_validate_loops.pair_wise_llr_head import (
    Track,
    train_and_validate_llr_head,
)
from DL_modules.train_validate_loops.sigmoid_equal_classweights import (
    train_and_validate_sigmoid_equal_classweights_head,
)
from DL_modules.train_validate_loops.single_instance import train_and_validate_llr_head_single_instance
from DL_modules.train_validate_loops.utils import get_next_entry
from DL_modules.utils import reshape_data_labels
from tests.conftest import RESOURCES_DIR
from train_tlm_one_proj_single_instance_glass_1_rep_initrandn_naive_train_validate_test import (
    _get_data_loader,
    _split_data,
)


def test_end_to_end_sigmoid_equal_classweight(tmp_path):
    model_name = "toplayersigmoidsubtract"

    def train_tlm_dl_model(cfg: Configuration):
        # get seed from config or from output folder
        seed = cfg.train.seed if cfg.train.seed else get_next_entry(cfg.saving.output_parent)
        # set seed for torch globally
        torch.manual_seed(seed)

        # Get dataloaders
        dataloader_train, dataloader_validate, dataloader_test = get_glass_1_rep_dataloaders_ttv(cfg)

        # Define model
        top_layer = TopLayerSubtractLogreg(18)

        # model = PairToOutput(susp_net, cr_net, top_layer)
        model = top_layer

        # Define loss criterion and optimization parameters
        fr_hard_negs = cfg.training_data.fr_hard_negs
        criterion_train = CLLRFromP(
            fr_hard_negs,
        )
        criterion_validate = CLLRFromP(1)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        num_epochs = cfg.train.num_epochs

        train_and_validate_sigmoid_equal_classweights_head(
            model,
            model_name,
            criterion_train,
            criterion_validate,
            num_epochs,
            optimizer,
            dataloader_train,
            dataloader_validate,
            cfg,
        )

    test_resources_mapping = {'RESOURCES_DIR': str(RESOURCES_DIR), 'tmp_path': str(tmp_path)}
    cfg = confidence.load_name(
        RESOURCES_DIR / 'end_to_end' / 'dl-data-params-glass-1-rep-train-validate-test',
        RESOURCES_DIR / 'end_to_end' / 'train-benchmark-glass-1-rep-DL-logreg-model-train-validate-test',
    )
    cfg = confidence.Configuration(test_resources_mapping, cfg)

    num_runs = cfg.trian.num_runs if cfg.train.num_runs else 1

    for _ in range(num_runs):
        train_tlm_dl_model(cfg)

    # just to make sure there is only one path where results are saved to
    output_path = tmp_path / '1'
    assert not os.path.isdir(tmp_path / '0')
    assert os.path.isdir(output_path)
    assert not os.path.isdir(tmp_path / '2')

    # assert results are stable
    num_files = len([f for f in output_path.rglob('*') if f.is_file()])
    assert num_files == 8
    filenames = sorted(f.name for f in output_path.iterdir() if f.is_file())
    assert filenames == [
        'best_ece_epoch=1.png',
        'best_ece_epoch=1_zoom.png',
        'best_llr_histogram_validate_epoch=1_validate.png',
        'best_validate_pav_plot_epoch_1.png',
        'loss_plot.png',
        'results.yaml',
        'toplayersigmoidsubtract.pth',
        'train_at_best_validate_pav_plot_epoch_1.png',
    ]


def test_end_to_end_pairwise_oneproj(tmp_path):
    model_name = "tlm one rot, init randn, loss: cllr_train_random_projection"

    def train_tlm_dl_model(cfg: Configuration):
        # get seed from config or from output folder
        seed = cfg.train.seed if cfg.train.seed else get_next_entry(cfg.saving.output_parent)
        # set seed for torch globally
        torch.manual_seed(seed)

        # Get dataloaders
        dataloader_train, dataloader_validate, dataloader_test = get_glass_1_rep_dataloaders_ttv(cfg)

        tlm_size = 18  # n_features of data
        # Define model
        top_layer = TwoLevelNormalNormalOneProj(
            tlm_size,
            cfg.training_data.n_same_source,
            variance_susp="randn",
            variance_cr="randn",
            variance_between="randn",
            n_axes_to_keep=cfg.train.n_axes_to_keep,
        )

        # model = PairToOutput(susp_net, cr_net, top_layer)
        model = top_layer

        # Define loss criterion and optimization parameters
        fr_hard_negs = cfg.training_data.fr_hard_negs
        criterion_train = CLLR(
            fr_hard_negs,
            CllrTrainMode(cfg.cllr_crit_train.mode),
            **cfg.cllr_crit_train.mode_options,
        )
        criterion_validate = CLLREvaluateDynamicallyConcentrated(1, cfg.cllr_crit_eval.threshold_for_focal_evaluate)
        criterion_train_evaluate = CLLREvaluateDynamicallyConcentrated(
            fr_hard_negs, cfg.cllr_crit_eval.threshold_for_focal_evaluate
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
        num_epochs = cfg.train.num_epochs

        input_for_criterion_evaluate = Track('t')

        train_and_validate_llr_head(
            model,
            model_name,
            criterion_train,
            criterion_validate,
            criterion_train_evaluate,
            num_epochs,
            optimizer,
            dataloader_train,
            dataloader_validate,
            cfg,
            tlm_size,
            input_for_criterion_evaluate,
            show_plots=False,
        )

    test_resources_mapping = {'RESOURCES_DIR': str(RESOURCES_DIR), 'tmp_path': str(tmp_path)}
    cfg = confidence.load_name(
        RESOURCES_DIR / 'end_to_end' / 'dl-data-params-glass-1-rep-train-validate-test',
        RESOURCES_DIR / 'end_to_end' / 'train_tlm_one_proj_glass_1_rep_initrandn_naive_train_validate_test',
    )
    cfg = confidence.Configuration(test_resources_mapping, cfg)

    num_runs = cfg.trian.num_runs if cfg.train.num_runs else 1

    for _ in range(num_runs):
        train_tlm_dl_model(cfg)

    # just to make sure there is only one path where results are saved to
    output_path = tmp_path / '1'
    assert not os.path.isdir(tmp_path / '0')
    assert os.path.isdir(output_path)
    assert not os.path.isdir(tmp_path / '2')

    # assert results are stable
    num_files = len([f for f in output_path.rglob('*') if f.is_file()])
    assert num_files == 36
    filenames = sorted(f.name for f in output_path.iterdir() if f.is_file())
    assert filenames == [
        'best_ece_norm_epoch=1.png',
        'best_ece_t_epoch=1.png',
        'best_ece_zoom_norm_epoch=1.png',
        'best_ece_zoom_t_epoch=1.png',
        'best_llr_histogram_norm_all_train_epoch=1.png',
        'best_llr_histogram_norm_all_validate_epoch=1.png',
        'best_llr_histogram_norm_focal_train_epoch=1.png',
        'best_llr_histogram_norm_focal_validate_epoch=1.png',
        'best_llr_histogram_t_focal_validate_epoch=1.png',
        'best_llr_histogram_t_sum_all_validate_epoch=1.png',
        'best_llr_histograms_norm_train_epoch=1.png',
        'best_llr_histograms_norm_validate_epoch=1.png',
        'best_llr_histograms_t_train_epoch=1.png',
        'best_llr_histograms_t_validate_epoch=1.png',
        'best_misleading_evidence_random_train_label0_epoch=1.csv',
        'best_misleading_evidence_random_train_label1_epoch=1.csv',
        'best_misleading_evidence_random_validate_label0_epoch=1.csv',
        'best_misleading_evidence_random_validate_label1_epoch=1.csv',
        'best_misleading_evidence_sorted_train_label0_epoch=1.csv',
        'best_misleading_evidence_sorted_train_label1_epoch=1.csv',
        'best_misleading_evidence_sorted_validate_label0_epoch=1.csv',
        'best_misleading_evidence_sorted_validate_label1_epoch=1.csv',
        'best_qnorm_train_epoch=1z_0.png',
        'best_qnorm_train_epoch=1z_1.png',
        'best_qnorm_validate_epoch=1z_0.png',
        'best_qnorm_validate_epoch=1z_1.png',
        'best_results_epoch=1.json',
        'best_train_norm_pav_plot_epoch_1.png',
        'best_train_t_pav_plot_epoch_1.png',
        'best_validate_norm_pav_plot_epoch_1.png',
        'best_validate_t_pav_plot_epoch_1.png',
        'compare_several_losses.png',
        'configuration.yaml',
        'loss_plot_train.png',
        'loss_plot_validate_t.png',
        'tlm one rot, init randn, loss: cllr_train_random_projection.pth',
    ]


@pytest.mark.parametrize(
    "config_name",
    [
        "train_tlm_one_proj_single_instance_glass_1_rep_initrandn_naive_train_validate_test",
        "train_tlm_one_proj_single_instance_glass_1_rep_initrandn_naive_train_validate_test_no_coherence_filter",
    ],
)
def test_end_to_end_single_instance_oneproj(tmp_path, config_name: str):
    model_name = "tlm single instance one projection, init randn, loss: cllr_train_random_projection"

    def train_tlm_dl_model(cfg: Configuration):
        # get seed from config or from output folder
        seed = cfg.train.seed if cfg.train.seed else get_next_entry(cfg.saving.output_parent)
        # set seed for torch globally
        torch.manual_seed(seed)
        # get data
        features = np.loadtxt(RESOURCES_DIR / "LR_benchmark/glass_duplos/data.csv", delimiter=';')
        labels = np.loadtxt(RESOURCES_DIR / "LR_benchmark/glass_duplos/labels.csv").astype(int)

        # minimize data to keep test fast enough
        features = features[:120, : cfg.train.tlm_size]
        labels = labels[:120]
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

        # get single train data in dataloader
        train_data_single_sources, train_labels_single_sources = reshape_data_labels(
            train_data_single_sources, train_labels_single_sources
        )
        source_dataset_train = SourceDataset(train_data_single_sources, train_labels_single_sources)
        single_source_dataloader_train = CustomDataLoader(
            source_dataset_train, ForensicSingleSampler(batch_size=10, labels=train_labels_single_sources)
        )

        # get single validation data in dataloader
        validate_data_single_sources, validate_labels_single_sources = reshape_data_labels(
            validate_data_single_sources, validate_labels_single_sources
        )
        source_dataset_validate = SourceDataset(validate_data_single_sources, validate_labels_single_sources)
        single_source_dataloader_validate = CustomDataLoader(
            source_dataset_validate, ForensicSingleSampler(batch_size=10, labels=validate_labels_single_sources)
        )

        # get pair data in dataloaders
        pairing_function = SourcePairing()
        data_train_pairs = _get_data_loader(
            pairing_function, train_validate_data, train_validate_labels, meta, train_index, cfg.train.ss_batch_size
        ).dataloader
        data_validate_pairs = _get_data_loader(
            pairing_function,
            train_validate_data,
            train_validate_labels,
            meta,
            validate_index,
            cfg.validate.ss_batch_size,
        ).dataloader
        # data_test_pairs = _get_data_loader(pairing_function, features, labels, meta, test_index,
        # cfg.train.ss_batch_size_test).dataloader

        tlm_size = cfg.train.tlm_size

        # Define model
        top_layer = SingleInstanceTwoLevelNormalNormalOneProj(
            tlm_size,
            train_data_single_sources.shape[0],
            numerator_variance_extra="randn",
            denominator_variance_extra="randn",
            sample_size_sigma2_w=cfg.train.sample_size_sigma2_w,
            n_axes_to_keep=cfg.train.n_axes_to_keep,
            train_coherence_filters=cfg.validate.train_coherence_filters,
            coherence_filter_max_iterations=cfg.validate.coherence_filter_max_iterations,
            coherence_filter_exclude_llrs_smaller_than=cfg.validate.coherence_filter_exclude_llrs_smaller_than,
            coherence_filter_threshold_bias_axis=cfg.validate.coherence_filter_threshold_bias_axis,
            train_clip_denominators=cfg.validate.train_clip_denominators,
            clip_denominator_percentage=cfg.validate.clip_denominator_percentage,
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

        model_dict = train_and_validate_llr_head_single_instance(
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
        )
        return model_dict

    test_resources_mapping = {'RESOURCES_DIR': str(RESOURCES_DIR), 'tmp_path': str(tmp_path)}
    cfg = confidence.load_name(
        RESOURCES_DIR / 'end_to_end' / 'dl-data-params-glass-1-rep-train-validate-test',
        RESOURCES_DIR / 'end_to_end' / config_name,
    )
    cfg = confidence.Configuration(test_resources_mapping, cfg)

    model_dict = train_tlm_dl_model(cfg)

    # just to make sure there is only one path where results are saved to
    output_path = tmp_path / '1'
    assert not os.path.isdir(tmp_path / '0')
    assert os.path.isdir(output_path)
    assert not os.path.isdir(tmp_path / '2')

    # assert results are stable
    post_hoc_folders = [folder for folder in model_dict.keys()]
    for post_hoc_folder in post_hoc_folders:
        output_path_temp = Path(output_path) / post_hoc_folder
        num_files = len([f for f in output_path_temp.rglob('*') if f.is_file()])
        assert num_files == 70
        filenames = sorted(f.name for f in output_path_temp.iterdir() if f.is_file())
        assert filenames == [
            'best_ece_norm_epoch=1.png',
            'best_ece_t_epoch=1.png',
            'best_ece_zoom_norm_epoch=1.png',
            'best_ece_zoom_t_epoch=1.png',
            'best_llr_histogram_norm_all_train_epoch=1.png',
            'best_llr_histogram_norm_all_validate_epoch=1.png',
            'best_llr_histogram_norm_focal_train_epoch=1.png',
            'best_llr_histogram_norm_focal_validate_epoch=1.png',
            'best_llr_histogram_t_focal_validate_epoch=1.png',
            'best_llr_histogram_t_sum_all_validate_epoch=1.png',
            'best_llr_histograms_norm_train_epoch=1.png',
            'best_llr_histograms_norm_validate_epoch=1.png',
            'best_llr_histograms_t_train_epoch=1.png',
            'best_llr_histograms_t_validate_epoch=1.png',
            'best_misleading_evidence_random_train_label0_epoch=1.csv',
            'best_misleading_evidence_random_train_label1_epoch=1.csv',
            'best_misleading_evidence_random_validate_label0_epoch=1.csv',
            'best_misleading_evidence_random_validate_label1_epoch=1.csv',
            'best_misleading_evidence_sorted_train_label0_epoch=1.csv',
            'best_misleading_evidence_sorted_train_label1_epoch=1.csv',
            'best_misleading_evidence_sorted_validate_label0_epoch=1.csv',
            'best_misleading_evidence_sorted_validate_label1_epoch=1.csv',
            'best_qnorm_train_epoch=1z_0.png',
            'best_qnorm_train_epoch=1z_1.png',
            'best_qnorm_validate_epoch=1z_0.png',
            'best_qnorm_validate_epoch=1z_1.png',
            'best_results_epoch=1.json',
            'best_ss_ece_norm_epoch=1.png',
            'best_ss_ece_t_epoch=1.png',
            'best_ss_ece_zoom_norm_epoch=1.png',
            'best_ss_ece_zoom_t_epoch=1.png',
            'best_ss_llr_histogram_norm_all_train_epoch=1.png',
            'best_ss_llr_histogram_norm_all_validate_epoch=1.png',
            'best_ss_llr_histogram_norm_focal_train_epoch=1.png',
            'best_ss_llr_histogram_norm_focal_validate_epoch=1.png',
            'best_ss_llr_histogram_t_focal_validate_epoch=1.png',
            'best_ss_llr_histogram_t_sum_all_validate_epoch=1.png',
            'best_ss_llr_histograms_norm_train_epoch=1.png',
            'best_ss_llr_histograms_norm_validate_epoch=1.png',
            'best_ss_llr_histograms_t_train_epoch=1.png',
            'best_ss_llr_histograms_t_validate_epoch=1.png',
            'best_ss_qnorm_train_epoch=1z_0.png',
            'best_ss_qnorm_train_epoch=1z_1.png',
            'best_ss_qnorm_validate_epoch=1z_0.png',
            'best_ss_qnorm_validate_epoch=1z_1.png',
            'best_ss_train_norm_pav_plot_epoch_1.png',
            'best_ss_train_t_pav_plot_epoch_1.png',
            'best_ss_validate_norm_pav_plot_epoch_1.png',
            'best_ss_validate_t_pav_plot_epoch_1.png',
            'best_train_norm_pav_plot_epoch_1.png',
            'best_train_t_pav_plot_epoch_1.png',
            'best_validate_norm_pav_plot_epoch_1.png',
            'best_validate_t_pav_plot_epoch_1.png',
            'compare_several_losses.png',
            'cs_vs_ss_llr_plot_train_norm.png',
            'cs_vs_ss_llr_plot_train_t.png',
            'cs_vs_ss_llr_plot_validate_norm.png',
            'cs_vs_ss_llr_plot_validate_t.png',
            'integrated_vs_fixed_plot_train_cs.png',
            'integrated_vs_fixed_plot_train_ss.png',
            'integrated_vs_fixed_plot_validate_cs.png',
            'integrated_vs_fixed_plot_validate_ss.png',
            'loss_plot_train.png',
            'loss_plot_validate_norm.png',
            'specific_source_losses.json',
            'tlm single instance one projection, init randn, loss: cllr_train_random_projection.pth',
            'unfairness_plot_train_norm.png',
            'unfairness_plot_train_t.png',
            'unfairness_plot_validate_norm.png',
            'unfairness_plot_validate_t.png',
        ]
