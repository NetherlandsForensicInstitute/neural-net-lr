import sys

sys.path.append("lib/lrbenchmark")
sys.path.append("../lib/lrbenchmark")

from pathlib import Path

import confidence
import numpy as np
import pytest
import torch

from DL_modules.criterion import CLLR
from DL_modules.criterion.axis_llr_correlation import AxisLlrCorrelationLoss
from DL_modules.criterion.cllr import CLLREvaluate, CLLRFromP, CllrTrainMode
from DL_modules.custom_layers import MediumDenseNet, PairToOutput, TopLayerSubtractDenseSigmoid
from DL_modules.dataloader import ForensicPairDataset, ForensicPairSampler, SameSourceDifferentSourceDataLoaderFactory
from DL_modules.utils import ResultTLM

TESTS_DIR = Path(__file__).parent
RESOURCES_DIR = TESTS_DIR / "resources"


@pytest.fixture
def forensic_pair_sampler_bs5_neg60_pos40():
    return ForensicPairSampler(5, [0] * 60 + [1] * 40)


@pytest.fixture
def forensic_pair_sampler_bs2_pos10000_neg20000():
    return ForensicPairSampler(2, [1] * 10000 + [0] * 20000)


@pytest.fixture
def forensic_pair_sampler_bs20_pos702_neg_134567():
    return ForensicPairSampler(20, [1] * 702 + [0] * 134567)


@pytest.fixture
def selected_training_data():
    cfg = confidence.load_name(TESTS_DIR / '../dl-data-params-glass-1-rep-train-test')
    train_data = np.load(
        TESTS_DIR / '../' / cfg.training_data.input_folder / cfg.training_data.file_name, allow_pickle=True
    )
    train_data_cr = train_data['cr']
    train_data_susp = train_data['susp']
    train_labels = train_data['labels']
    fr_hard_negs = cfg.training_data.fr_hard_negs
    return train_data_susp, train_data_cr, train_labels, fr_hard_negs


@pytest.fixture
def test_data():
    cfg = confidence.load_name(TESTS_DIR / '../dl-data-params-glass-1-rep-train-test')
    test_data = np.load(TESTS_DIR / '../' / cfg.test_data.input_folder / cfg.test_data.file_name, allow_pickle=True)
    test_data_cr = test_data['cr']
    test_data_susp = test_data['susp']
    test_labels = test_data['labels']
    return test_data_susp, test_data_cr, test_labels


@pytest.fixture
def forensic_train_dataset(selected_training_data):
    train_data_cr, train_data_susp, train_labels, fr_hard_negs = selected_training_data
    return ForensicPairDataset(train_data_susp, train_data_cr, train_labels)


@pytest.fixture
def data_loader(selected_training_data):
    factory = SameSourceDifferentSourceDataLoaderFactory(
        selected_training_data[0], selected_training_data[1], selected_training_data[2], 20
    )
    dataloader = factory.dataloader
    return dataloader


@pytest.fixture
def pair_to_output_model():
    susp_fsw = MediumDenseNet(18, [10, 10, 5], 5)
    cr_fsw = MediumDenseNet(18, [10, 10, 5], 5)
    top_layer = TopLayerSubtractDenseSigmoid(5)
    pair_to_output_model = PairToOutput(susp_fsw, cr_fsw, top_layer)
    return pair_to_output_model


@pytest.fixture
def cllr_from_p_instance():
    return CLLRFromP(1)


@pytest.fixture
# in this case the usage is where the number of axes in predictions = 4, so only 2 axes are used.
def cllr_instance_train_evenly():
    return CLLR(1, CllrTrainMode.EVENLY)


@pytest.fixture
def cllr_instance_llr_fr_hard_negs_sampled():
    return CLLREvaluate(0.25)


@pytest.fixture
# in this case the usage is where the number of axes in predictions = 2, so only one axes is used.
def cllr_instance_evaluate():
    return CLLREvaluate(1)


@pytest.fixture
def cllr_instance_train_concentrated():
    return CLLR(1, CllrTrainMode.CONCENTRATED)


@pytest.fixture
def cllr_instance_train_focus_spread_sum():
    return CLLR(1, CllrTrainMode.FOCUS_SPREAD_SUM)


@pytest.fixture
def cllr_instance_train_focus_spread_mean_n_axes_2():
    return CLLR(1, CllrTrainMode.FOCUS_SPREAD_MEAN)


@pytest.fixture
def ignorant_ps():
    preds = torch.full((2, 4), 0.5)
    labels = torch.zeros(4)
    labels[:2] = 1
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def quarter_ps():
    preds = torch.tensor([[1 / 4, 1 / 4, 3 / 4, 3 / 4], [1 / 4, 1 / 4, 3 / 4, 3 / 4]])
    labels = torch.zeros(4)
    labels[:2] = 1
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def all_1_or_0_ps():
    preds = torch.tensor([[1, 1, 1, 1], [0, 0, 0, 0]])
    labels = torch.tensor([1, 1, 0, 0])
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def certainty_ps():
    preds = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    labels = torch.tensor([1, 1, 0, 0])
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def ignorant_llrs():
    preds = torch.full((2, 4), 0.0)
    labels = torch.zeros(4)
    labels[:2] = 1
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def some_llrs():
    preds = torch.tensor(
        [
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
        ]
    )
    labels = torch.tensor([1, 0, 0, 1])
    preds = preds.transpose(0, 1)
    return preds, labels


@pytest.fixture
def some_llrs_per_axis():
    llrs_per_axis = torch.tensor(
        [
            [2, 4, 5, 1],
            [7, -3, 3, 1],
            [-2, 1, 1, 0],
        ]
    )
    # 3 axis with each 4 LLRs
    return llrs_per_axis.t()


@pytest.fixture
def correlation_penalty_instance():
    return AxisLlrCorrelationLoss


def to_tlm_result(llrs: torch.Tensor):
    return ResultTLM({"llr_norm": llrs}, {}, {}, {})


@pytest.fixture
def some_llrs_diff_axes():
    preds = torch.tensor(
        [
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(14)],
        ]
    )
    labels = torch.tensor([1, 0, 0, 1])
    preds = preds.transpose(0, 1)
    return to_tlm_result(preds), labels


@pytest.fixture
def some_llrs_3_columns_tlm_result():
    preds = torch.tensor(
        [
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
            [np.log10(3), np.log10(1 / 15), np.log10(1 / 7), np.log10(7)],
        ]
    )
    labels = torch.tensor([1, 0, 0, 1])
    preds = preds.transpose(0, 1)
    return to_tlm_result(preds), labels


@pytest.fixture
def synthetic_data_labels():
    # Synthesized data (300 samples, 10 features)
    rng = np.random.default_rng(20260405)
    X = rng.standard_normal((60, 2))

    # Labels: [1,1,1, 2,2,2, ..., 100,100,100]
    y = np.repeat(np.arange(1, 21), 3)

    return X, y
