import json
from pathlib import Path

import numpy as np
from torch import Tensor

from DL_modules.criterion import CLLREvaluate
from DL_modules.plots.plot_collections import plot_and_save_test_results
from DL_modules.utils import ResultTLM, try_devpav


def save_tlm_nn_test_results(
    result_tlm_test: ResultTLM,
    labels_test: Tensor,
    criterion_test: CLLREvaluate,
    output_dir: Path,
    suffix: str | None = None,
):
    labels_numpy = labels_test.detach().numpy()
    llrs_t = result_tlm_test.log_t['llr_t']
    llrs_t_numpy = llrs_t.numpy()
    llrs_norm = result_tlm_test.log_norm['llr_norm']
    llrs_norm_numpy = llrs_norm.numpy()
    test_loss_t, axes_focus = criterion_test(llrs_t, labels_test)
    test_loss_norm, _ = criterion_test(llrs_norm, labels_test)
    axes_focus_numpy = axes_focus.numpy()
    test_devpav_norm = try_devpav(10 ** np.sum(llrs_norm_numpy, axis=1), labels_numpy)
    test_devpav_t = try_devpav(10 ** np.sum(llrs_t_numpy, axis=1), labels_numpy)

    output_params = {
        # -2 for all axes
        'cllr_norm_test': float(test_loss_norm.squeeze(0).squeeze(0)[-2].numpy()),
        'cllr_norm_test_all_info': test_loss_norm.squeeze(0).numpy().tolist(),
        # -2 for all axes
        'cllr_t_test': float(test_loss_t.squeeze(0).squeeze(0)[-2].numpy()),
        'cllr_t_test_all_info': test_loss_t.squeeze(0).numpy().tolist(),
        'axes_focus': axes_focus_numpy.tolist(),
        'devpav_norm_test': float(test_devpav_norm),
        'devpav_t_test': float(test_devpav_t),
        'variances_h1': result_tlm_test.variances['variances_1'].numpy().tolist(),
        'variances_h0': result_tlm_test.variances['variances_0'].numpy().tolist(),
    }
    filename = 'test_results_at_best_epoch'
    if suffix:
        filename = f"{filename}_{suffix}"
    with open(output_dir / f'{filename}.json', 'w') as file:
        json.dump(output_params, file, indent=4)

    plot_and_save_test_results(
        llrs_t_numpy,
        llrs_norm_numpy,
        result_tlm_test.z,
        labels_test.numpy(),
        test_loss_norm,
        test_loss_t,
        axes_focus_numpy,
        output_dir,
    )


def process_specific_source_results(
    predictions_ss: ResultTLM, labels: Tensor, criterion_validate: CLLREvaluate, output_dir: Path
):
    save_tlm_nn_test_results(predictions_ss, labels, criterion_validate, output_dir, 'specific_source')
