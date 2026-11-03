import numpy as np
import pytest
import torch

from DL_modules.dataloader import (
    ForensicPairDataset,
    SameSourceDifferentSourceDataLoaderFactory,
    get_batch_bins,
    get_n_batches,
)
from tests.utils import iterate_forensic_pair_sampler


# tests Customsampler
def test_n_batches_get_batch_bins():
    assert get_n_batches(12, 5) == 2
    assert np.array_equal(get_batch_bins(2, 12), [0, 6, 12])
    assert np.array_equal(get_batch_bins(2, 53), [0, 26, 53])


@pytest.mark.parametrize(
    "forensic_pair_sampler_fixture,expected",
    [
        ("forensic_pair_sampler_bs2_pos10000_neg20000", list(np.arange(30000))),
        ("forensic_pair_sampler_bs20_pos702_neg_134567", list(np.arange(702 + 134567))),
    ],
)
def test_all_labels_unique_and_in_epoch(request, forensic_pair_sampler_fixture, expected):
    forensic_pair_sampler = request.getfixturevalue(forensic_pair_sampler_fixture)
    one_epoch = iterate_forensic_pair_sampler(forensic_pair_sampler)
    one_epoch = sorted(one_epoch)
    assert one_epoch == expected


def test_exactly_batchsize_pos_in_all_batches(forensic_pair_sampler_bs2_pos10000_neg20000):
    batch_size_pos = 2
    n_pos = 10000
    for batch in forensic_pair_sampler_bs2_pos10000_neg20000:
        assert len([idx for idx in batch if idx < n_pos]) == batch_size_pos


def test_exactly_batchsize_pos_in_all_batches_2(forensic_pair_sampler_bs5_neg60_pos40):
    batch_size_pos = 5
    n_neg = 60
    for batch in forensic_pair_sampler_bs5_neg60_pos40:
        assert len([idx for idx in batch if idx >= n_neg]) == batch_size_pos


# tests DataLoader
def test_forensic_data_loader(selected_training_data):
    factory = SameSourceDifferentSourceDataLoaderFactory(
        selected_training_data[0], selected_training_data[1], selected_training_data[2], 20
    )
    dataloader = factory.dataloader
    i = 0
    data = []
    for batch in dataloader:
        i += 1
        data.extend(batch[0].tolist())
    assert i == 702 // 20
    assert len(data) == len(selected_training_data[0])


# tests Dataset
def test_forensic_pair_dataset(selected_training_data):
    forensic_pair_dataset = ForensicPairDataset(
        selected_training_data[0], selected_training_data[1], selected_training_data[2]
    )
    assert len(forensic_pair_dataset) == len(selected_training_data[0])


def test_forensic_pair_dataset_2(test_data):
    forensic_pair_dataset = ForensicPairDataset(test_data[0], test_data[1], test_data[2])
    assert len(forensic_pair_dataset) == len(test_data[0])


def test_batches_are_random(forensic_pair_sampler_bs5_neg60_pos40):
    batch1 = forensic_pair_sampler_bs5_neg60_pos40
    batch2 = forensic_pair_sampler_bs5_neg60_pos40

    for b1, b2 in zip(batch1, batch2):
        # Due to the length of the batches, the probabilty is very small
        # that two are equal.
        assert not torch.equal(b1, b2)


def test_seeded_batches_are_not_random(forensic_pair_sampler_bs5_neg60_pos40):
    # test if batches generated with same seed are equal
    torch.manual_seed(1234)
    batch1 = list(forensic_pair_sampler_bs5_neg60_pos40)
    torch.manual_seed(1234)
    batch2 = list(forensic_pair_sampler_bs5_neg60_pos40)
    for b1, b2 in zip(batch1, batch2):
        assert torch.equal(b1, b2)

    # test if batches generated with different seed are different
    torch.manual_seed(1235)
    batch3 = list(forensic_pair_sampler_bs5_neg60_pos40)
    for b1, b3 in zip(batch1, batch3):
        assert not torch.equal(b1, b3)
