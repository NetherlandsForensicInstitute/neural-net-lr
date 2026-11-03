import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from confidence import Configuration
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

LOG = logging.getLogger(__name__)


def sort_by_ss_first(
    susp_data: list[Any], cr_data: list[Any], labels: list[float]
) -> tuple[list[Any], list[Any], list[float]]:
    """Sort data by labels (0 or 1), 1's first.

    return: sorted data
    """
    # Combine the lists into a list of tuples
    combined = list(zip(susp_data, cr_data, labels))

    # Sort the combined list based on the integer values, putting 1's first
    sorted_combined = sorted(combined, key=lambda x: -x[2])

    # Separate the sorted tuples back into two lists
    susp_data_sorted, cr_data_sorted, labels_sorted = zip(*sorted_combined)

    # Convert back to lists
    susp_data_sorted = list(susp_data_sorted)
    cr_data_sorted = list(cr_data_sorted)
    labels_sorted = list(labels_sorted)

    return susp_data_sorted, cr_data_sorted, labels_sorted


class ForensicPairSampler(Sampler):
    """
    A sampler that deals with a forensic-pair dataset for pairwise LR-calculations,
    with highly unbalanced labels in the train dataset.
    Label = 1 -> SS, and (many) more Labels = 0 -> DS.

    Sampler constructs batches with either ss_batch_size or ss_batch_size + 1 SS labels and DS labels according
    to the label-ratio in the dataset. ss_batch_size does not have to be a multiple of the number of SS labels.
    """

    def __init__(self, ss_batch_size: int, labels: list):
        super().__init__()
        self.labels = labels
        self.ss_batch_size = ss_batch_size
        self.ss_dataset_size = sum(labels)
        self.ds_dataset_size = len(labels) - self.ss_dataset_size
        self.n_batches = get_n_batches(self.ss_dataset_size, self.ss_batch_size)
        self.ss_batch_bins = get_batch_bins(self.n_batches, self.ss_dataset_size)
        self.ds_batch_bins = get_batch_bins(self.n_batches, self.ds_dataset_size)
        """
        :param ss_batch_size: the desired number of same source pairs in a batch.
        :param labels: list filled with 1's and 0's. Either all 1's first or all 0's first.
        """

        # Ensure the batch size is smaller than n_ss, n_ds
        if not self.ss_batch_size <= self.ss_dataset_size:
            raise ValueError(
                f"ForensicPairSampler: Batch size {self.ss_batch_size} larger "
                f"than number of same source samples in dataset {self.ss_dataset_size}"
            )
        if not self.ss_batch_size <= self.ds_dataset_size:
            raise ValueError(
                "ForensicPairSampler: Batch size larger than number of different source samples in dataset"
            )
        if not self.ss_dataset_size <= self.ds_dataset_size:
            raise ValueError("ForensicPairSampler: ss dataset size larger than ds dataset size")

        ss_bin_widths = self.ss_batch_bins[1:] - self.ss_batch_bins[:-1]
        ds_bin_widths = self.ds_batch_bins[1:] - self.ds_batch_bins[:-1]
        LOG.debug(f"ForensicPairSampler: ss bins range from {np.min(ss_bin_widths)} to {np.max(ss_bin_widths)}")
        LOG.debug(f"ForensicPairSampler: ds bins range from {np.min(ds_bin_widths)} to {np.max(ds_bin_widths)}")

    def __iter__(self) -> Tensor:
        """
        Randomize ss idx's and ds idx's seperately.
        Then iterate over batches. For each batch, yield corresponding idx's,
        randomize selected idx's before yield so models van never exploit the fact that the data in the dataloader has
        ss first.
        """
        ss_idx = torch.randperm(self.ss_dataset_size).tolist()
        ds_idx = torch.randperm(self.ds_dataset_size).tolist()
        for batch_number in range(self.n_batches):
            ss_idx_sampled = [
                ss_idx[i] for i in range(self.ss_batch_bins[batch_number], self.ss_batch_bins[batch_number + 1])
            ]
            ds_idx_sampled = [
                ds_idx[i] for i in range(self.ds_batch_bins[batch_number], self.ds_batch_bins[batch_number + 1])
            ]
            if self.labels[0] == 1:
                idx_batch = torch.tensor(ss_idx_sampled + [ds_idx + self.ss_dataset_size for ds_idx in ds_idx_sampled])
            else:
                idx_batch = torch.tensor([ss_idx + self.ds_dataset_size for ss_idx in ss_idx_sampled] + ds_idx_sampled)
            yield idx_batch[torch.randperm(len(idx_batch))]

    def __len__(self) -> int:
        return self.n_batches


class ForensicPairDataset(Dataset):
    """
    Dataset that holds susp_data and cr_labels that should be compared in a pairwise fashion, and their GT labels
    (same_source = 1, different_source = 0).
    """

    def __init__(self, susp_data: list[Any], cr_data: list[Any], labels: list[float]):
        susp_data_sorted, cr_data_sorted, labels_sorted = sort_by_ss_first(susp_data, cr_data, labels)
        self.susp_data = torch.tensor(np.array(susp_data_sorted))
        self.cr_data = torch.tensor(np.array(cr_data_sorted))
        self.labels = torch.tensor(np.array(labels_sorted))

        if not len(self.susp_data) == len(self.cr_data) == len(self.labels):
            raise RuntimeError("ForensicPairDataSet: Lengths of susp_data, cr_data or labels is not equal")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.susp_data[idx], self.cr_data[idx], self.labels[idx]


class CustomDataLoader:
    """
    Dataloader that combines a Dataset and a Sampler.
    """

    def __init__(self, dataset: Dataset, sampler: Sampler):
        self.dataset = dataset
        self.sampler = sampler

    def __iter__(self):
        for batch_ids in self.sampler:
            yield self.dataset[batch_ids]

    def __len__(self):
        return len(self.sampler)


class SameSourceDifferentSourceDataLoaderFactory:
    """
    Factory for torch DataLoader instance. Defines custom sampler and Dataset.
    """

    def __init__(self, susp_data: list[Any], cr_data: list[Any], labels: list[int], ss_batch_size: int, num_workers=0):
        self.susp_data = susp_data
        self.cr_data = cr_data
        self.labels = labels
        self.ss_batch_size = ss_batch_size

    @property
    def dataloader(self) -> DataLoader | CustomDataLoader:
        dataset = ForensicPairDataset(self.susp_data, self.cr_data, self.labels)
        sampler = ForensicPairSampler(self.ss_batch_size, self.labels)
        dataloader = CustomDataLoader(dataset, sampler=sampler)

        return dataloader


def get_glass_1_rep_dataloaders(cfg: Configuration) -> tuple[DataLoader, DataLoader]:
    """return: a Dataloader for train and one for validate data."""
    train_data = np.load(Path(cfg.training_data.input_folder) / cfg.training_data.file_name, allow_pickle=True)
    validate_data = np.load(Path(cfg.validate_data.input_folder) / cfg.validate_data.file_name, allow_pickle=True)

    # unpack data
    train_data_susp = train_data['susp']
    train_data_cr = train_data['cr']
    train_labels = train_data['labels']

    validate_data_susp = validate_data['susp']
    validate_data_cr = validate_data['cr']
    validate_labels = validate_data['labels']

    # define dataloaders
    data_loader_factory_train = SameSourceDifferentSourceDataLoaderFactory(
        train_data_susp, train_data_cr, train_labels, cfg.train.ss_batch_size
    )
    dataloader_train = data_loader_factory_train.dataloader

    data_loader_factory_validate = SameSourceDifferentSourceDataLoaderFactory(
        validate_data_susp, validate_data_cr, validate_labels, cfg.validate.ss_batch_size
    )
    dataloader_validate = data_loader_factory_validate.dataloader

    return dataloader_train, dataloader_validate


def get_glass_1_rep_dataloaders_ttv(cfg: Configuration) -> tuple[CustomDataLoader, CustomDataLoader, CustomDataLoader]:
    """return: A DataLoader for train data, for validate data, and for validation data."""
    train_data = np.load(Path(cfg.training_data.input_folder) / cfg.training_data.file_name, allow_pickle=True)
    validate_data = np.load(Path(cfg.validate_data.input_folder) / cfg.validate_data.file_name, allow_pickle=True)
    test_data = np.load(Path(cfg.test_data.input_folder) / cfg.test_data.file_name, allow_pickle=True)

    # unpack data
    train_data_susp = train_data['susp']
    train_data_cr = train_data['cr']
    train_labels = train_data['labels']

    validate_data_susp = validate_data['susp']
    validate_data_cr = validate_data['cr']
    validate_labels = validate_data['labels']

    test_data_susp = test_data['susp']
    test_data_cr = test_data['cr']
    test_labels = test_data['labels']

    # define dataloaders
    data_loader_factory_train = SameSourceDifferentSourceDataLoaderFactory(
        train_data_susp, train_data_cr, train_labels, cfg.train.ss_batch_size
    )
    dataloader_train = data_loader_factory_train.dataloader

    data_loader_factory_validate = SameSourceDifferentSourceDataLoaderFactory(
        validate_data_susp, validate_data_cr, validate_labels, cfg.validate.ss_batch_size
    )
    dataloader_validate = data_loader_factory_validate.dataloader

    data_loader_factory_test = SameSourceDifferentSourceDataLoaderFactory(
        test_data_susp, test_data_cr, test_labels, cfg.test.ss_batch_size
    )
    dataloader_test = data_loader_factory_test.dataloader

    return dataloader_train, dataloader_validate, dataloader_test


# Define a custom Dataset class
class SourceDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)  # Convert numpy array to torch tensor
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to torch tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


class ForensicSingleSampler(Sampler):
    """
    A sampler that deals with a forensic-pair dataset for single LR-calculations.

    Sampler constructs batches with either ss_batch_size or ss_batch_size + 1 SS labels.
    ss_batch_size does not have to be a multiple of the number of SS labels.
    """

    def __init__(self, batch_size: int, labels: list):
        super().__init__()
        self.labels = labels
        self.batch_size = batch_size
        self.dataset_size = len(labels)
        self.n_batches = get_n_batches(self.dataset_size, self.batch_size)
        self.batch_bins = get_batch_bins(self.n_batches, self.dataset_size)
        """
        :param ss_batch_size: the desired number of same source pairs in a batch.
        :param labels: source labels.
        """

        # Ensure the batch size is smaller than n_ss, n_ds
        if not self.batch_size <= self.dataset_size:
            raise ValueError(
                f"ForensicPairSampler: Batch size {self.batch_size} larger "
                f"than number of same source samples in dataset {self.dataset_size}"
            )

        bin_widths = self.batch_bins[1:] - self.batch_bins[:-1]
        LOG.debug(f"ForensicPairSampler: ss bins range from {np.min(bin_widths)} to {np.max(bin_widths)}")

    def __iter__(self) -> Tensor:
        """
        Randomize source idx's.

        Then iterate over batches. For each batch, yield corresponding idx's,
        randomize selected idx's before yield so models van never exploit the fact that the data in the dataloader has
        """
        idx = torch.randperm(self.dataset_size).tolist()
        for batch_number in range(self.n_batches):
            idx_sampled = [idx[i] for i in range(self.batch_bins[batch_number], self.batch_bins[batch_number + 1])]
            yield torch.tensor(idx_sampled)

    def __len__(self) -> int:
        return self.n_batches


def get_n_batches(dataset_size, batch_size) -> int:
    n_batches = dataset_size // batch_size
    LOG.debug(f"the number of batches is {n_batches}")
    return n_batches


def get_batch_bins(n_batches: int, dataset_size: int) -> np.ndarray:
    bin_width = int(dataset_size // n_batches)
    mod = int(dataset_size % n_batches)
    plusses = np.concatenate((np.zeros(n_batches + 1 - mod), 1 + np.arange(mod)))
    batch_bins = np.arange(n_batches + 1) * bin_width + plusses
    batch_bins = batch_bins.astype(int)

    return batch_bins
