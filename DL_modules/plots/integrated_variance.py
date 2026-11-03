from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from DL_modules.utils import ResultTLM


def plot_integrated_vs_fixed(tlm_result: ResultTLM, labels: np.ndarray, output_path: Path, suffix: str):
    norm = torch.sum(tlm_result.log_norm["llr_norm"], dim=1).numpy()
    t = torch.sum(tlm_result.log_t["llr_t"], dim=1).numpy()
    delta = t - norm

    mask_1 = labels == 1
    mask_0 = labels == 0

    # kleinere punten, bv. 5
    point_size = 20

    plt.figure()

    # label == 1 → blue
    plt.scatter(norm[mask_1], delta[mask_1], s=point_size, c='blue', label='H1', alpha=0.6)  # 0.8

    # label == 0 → '#ff7f0e' is orangy-salmon
    plt.scatter(norm[mask_0], delta[mask_0], s=point_size, c='#ff7f0e', label='H0', alpha=0.2)  # 0.4

    plt.xlabel('LLRs for fixed within_var')
    plt.ylabel('integrated - fixed LLRs')
    plt.legend()
    plt.xlim((-6, 6))
    plt.ylim((-2, 2))
    plt.grid(True)
    if output_path:
        plt.savefig(output_path / f'integrated_vs_fixed_plot_{suffix}', bbox_inches='tight')
    plt.show()
    plt.close()
