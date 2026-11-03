import os
from pathlib import PosixPath

from DL_modules.train_validate_loops.utils import get_next_entry


def iterate_forensic_pair_sampler(forensic_pair_sampler):
    all_idx = []
    for idx in forensic_pair_sampler:
        all_idx.extend(idx.tolist())
    return all_idx


def test_get_next_entry(tmp_path: PosixPath):
    # test if function returns `1` for empty folder
    entry = get_next_entry(str(tmp_path))
    assert entry == 1

    # test if function returns the correct next entry
    os.makedirs(os.path.join(tmp_path, '1/'))
    entry = get_next_entry(str(tmp_path))
    assert entry == 2

    os.makedirs(os.path.join(tmp_path, 'test/'))
    entry = get_next_entry(str(tmp_path))
    assert entry == 2

    os.makedirs(os.path.join(tmp_path, '1234/'))
    entry = get_next_entry(str(tmp_path))
    assert entry == 1235
