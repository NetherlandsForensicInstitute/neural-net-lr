from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from lrbenchmark.lrsystems.lrsystems import LRSystem
from sklearn.model_selection import GroupShuffleSplit

from DL_modules.dataloader import SameSourceDifferentSourceDataLoaderFactory
from DL_modules.dataloader.data_loader import CustomDataLoader


@dataclass(frozen=True)
class _Data:
    pair_features: np.ndarray
    pair_labels: np.ndarray
    pair_meta: np.ndarray
    ss_batch_size: int

    @property
    def tlm_size(self) -> int:
        return self.pair_features.shape[2]

    @cached_property
    def dataloader(self) -> CustomDataLoader:
        data_susp = self.pair_features[:, 0, :]  # first element is trace/susp
        data_cr = self.pair_features[:, 1, :]  # second element is ref/cr
        data_loader_factory = SameSourceDifferentSourceDataLoaderFactory(
            data_susp, data_cr, self.pair_labels, self.ss_batch_size
        )
        return data_loader_factory.dataloader

    @cached_property
    def n_same_source_pairs(self) -> int:
        return np.sum(self.pair_labels)


class LRSystemNN(LRSystem):
    def __init__(self, name: str):
        super().__init__(name)

    def fit(self, instances: np.ndarray, labels: np.ndarray, meta: np.ndarray):
        pass

    def apply(self, features: Any, labels: np.ndarray | None, meta: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    def _split_in_three(self, features: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> tuple[_Data, _Data, _Data]:
        splitter = GroupShuffleSplit(train_size=1 / 3, n_splits=1, random_state=self.seed)
        ((train_index, validate_test_index),) = splitter.split(features, labels, labels)

        # Step 2: split remaining 2/3 into validate and test (50% each of remaining)
        temp_features = features[validate_test_index]
        temp_labels = labels[validate_test_index]

        splitter2 = GroupShuffleSplit(train_size=0.5, n_splits=1, random_state=self.seed)
        ((validate_index_temp, test_index_temp),) = splitter2.split(temp_features, temp_labels, temp_labels)

        # Map indices back to the original dataset
        validate_index = validate_test_index[validate_index_temp]
        test_index = validate_test_index[test_index_temp]

        return train_index, validate_index, test_index

    def _get_pair_data_loader(
        self, features: np.ndarray, labels: np.ndarray, meta: np.ndarray, indexes: np.ndarray | None, ss_batch_size: int
    ) -> _Data:
        if indexes is not None:
            features, labels, meta = features[indexes], labels[indexes], meta[indexes]

        pair_features, pair_labels, pair_meta = self.pairing_function.pair(features, labels, meta)
        return _Data(pair_features, pair_labels, pair_meta, ss_batch_size)
