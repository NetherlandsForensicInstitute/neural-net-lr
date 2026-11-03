from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from DL_modules.utils import ResultTLM


def plot_unfairness(ss: np.ndarray, cs: np.ndarray, labels: np.ndarray, output_path: Path, suffix: str):
    delta = cs - ss

    mask_1 = labels == 1
    mask_0 = labels == 0

    # kleinere punten, bv. 5
    point_size = 20

    plt.figure()

    # label == 1 → blue
    plt.scatter(ss[mask_1], delta[mask_1], s=point_size, c='blue', label='H1', alpha=0.6)  # 0.8

    # label == 0 → '#ff7f0e' is orangy-salmon
    plt.scatter(ss[mask_0], delta[mask_0], s=point_size, c='#ff7f0e', label='H0', alpha=0.2)  # 0.4

    plt.xlabel('specific source LLRs')
    plt.ylabel('common source - specific source LLRs')
    plt.legend()
    plt.xlim(max(-5, float(np.min(ss))), float(np.max(ss)))
    plt.ylim(max(-1, float(np.min(delta))), min(float(np.max(delta)), 1))
    plt.grid(True)
    if output_path:
        plt.savefig(output_path / f'unfairness_plot_{suffix}', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_cs_vs_ss_llrs(ss: np.ndarray, cs: np.ndarray, labels: np.ndarray, output_path: Path, suffix: str):
    mask_1 = labels == 1
    mask_0 = labels == 0

    # kleinere punten, bv. 5
    point_size = 20

    plt.figure()

    # add a straight line y = x
    x = np.linspace((-6, 6), (6, 6), 100)
    plt.plot(x, x, '--', c='lightgrey')

    # label == 1 → blue
    plt.scatter(ss[mask_1], cs[mask_1], s=point_size, c='blue', label='H1', alpha=0.6)  # 0.8

    # label == 0 → '#ff7f0e' is orangy-salmon
    plt.scatter(ss[mask_0], cs[mask_0], s=point_size, c='#ff7f0e', label='H0', alpha=0.2)  # 0.4

    plt.xlabel('specific source LLRs')
    plt.ylabel('common source LLRs')
    plt.legend()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.grid(True)
    if output_path:
        plt.savefig(output_path / f'cs_vs_ss_llr_plot_{suffix}', bbox_inches='tight')
    plt.show()
    plt.close()


def make_ss_cs_plots(
    predictions_ss: ResultTLM, predictions_cs: ResultTLM, labels: Tensor, output_dir: Path, suffix: str
):
    plot_unfairness(
        torch.sum(predictions_ss.log_t["llr_t"], dim=1).numpy(),
        torch.sum(predictions_cs.log_t["llr_t"], dim=1).numpy(),
        labels.numpy(),
        output_dir,
        suffix=f"{suffix}_t",
    )
    plot_unfairness(
        torch.sum(predictions_ss.log_norm["llr_norm"], dim=1).numpy(),
        torch.sum(predictions_cs.log_norm["llr_norm"], dim=1).numpy(),
        labels.numpy(),
        output_dir,
        suffix=f"{suffix}_norm",
    )

    plot_cs_vs_ss_llrs(
        torch.sum(predictions_ss.log_t["llr_t"], dim=1).numpy(),
        torch.sum(predictions_cs.log_t["llr_t"], dim=1).numpy(),
        labels.numpy(),
        output_dir,
        suffix=f"{suffix}_t",
    )

    plot_cs_vs_ss_llrs(
        torch.sum(predictions_ss.log_norm["llr_norm"], dim=1).numpy(),
        torch.sum(predictions_cs.log_norm["llr_norm"], dim=1).numpy(),
        labels.numpy(),
        output_dir,
        suffix=f"{suffix}_norm",
    )
