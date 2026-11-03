import random

import numpy as np
import torch


def test_torch():
    torch.manual_seed(0)
    assert torch.rand(1).numpy()[0] == 0.4962566
    assert torch.rand(1).numpy()[0] == 0.7682218


def test_numpy():
    rng = np.random.default_rng(seed=0)
    assert rng.random() == 0.6369616873214543
    assert rng.random() == 0.2697867137638703


def test_python():
    random.seed(0)
    assert random.random() == 0.8444218515250481  # noqa: S311
    assert random.random() == 0.7579544029403025  # noqa: S311
