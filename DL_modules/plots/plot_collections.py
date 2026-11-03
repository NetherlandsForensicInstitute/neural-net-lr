from pathlib import Path

import lir
import numpy as np
import torch
from torch import Tensor

from DL_modules.plots import calibration_plot, ece_plot, plot_llr_distributions, plot_z_qnorms
from DL_modules.utils import ResultTLM


def show_and_save_plots(
    epoch: int,
    experiment_path: Path | str,
    current_train_loss_on_validate_crit: Tensor,
    current_validate_loss: Tensor,
    all_predictions_validate: ResultTLM,
    all_predictions_train: ResultTLM,
    axes: list[int],
    all_labels_validate: Tensor,
    all_labels_train: Tensor,
    prefix: str,
    show_plots: bool = True,
):
    train_loss = current_train_loss_on_validate_crit
    validate_loss = current_validate_loss
    labels_validate = all_labels_validate.numpy()
    labels_train = all_labels_train.numpy()
    llrs_norm_validate = all_predictions_validate.log_norm['llr_norm'].numpy()
    llrs_norm_train = all_predictions_train.log_norm['llr_norm'].numpy()
    llrs_t_validate = all_predictions_validate.log_t['llr_t'].numpy()
    llrs_t_train = all_predictions_train.log_t['llr_t'].numpy()
    llrs_t_sum_focal_validate = torch.sum(all_predictions_validate.log_t['llr_t'][:, axes], dim=1).numpy()
    llrs_t_sum_all_validate = torch.sum(all_predictions_validate.log_t['llr_t'], dim=1).numpy()
    llrs_t_sum_all_train = torch.sum(all_predictions_train.log_t['llr_t'], dim=1).numpy()
    llrs_norm_sum_focal_validate = torch.sum(all_predictions_validate.log_norm['llr_norm'][:, axes], dim=1).numpy()
    llrs_norm_sum_all_validate = torch.sum(all_predictions_validate.log_norm['llr_norm'], dim=1).numpy()
    llrs_norm_sum_all_train = torch.sum(all_predictions_train.log_norm['llr_norm'], dim=1).numpy()
    z_train = all_predictions_train.z
    llrs_norm_sum_focal_train = torch.sum(all_predictions_train.log_norm['llr_norm'][:, axes], dim=1).numpy()
    z_validate = all_predictions_validate.z

    ece_plot(
        llrs_t_sum_all_validate,
        labels_validate,
        pathfile_save=experiment_path / f'{prefix}_ece_t_epoch={epoch}',
        show_plot=show_plots,
    )

    ece_plot(
        llrs_t_sum_all_validate,
        labels_validate,
        # -2 for all axes
        ylim=(0, 2.0 * float(torch.mean(validate_loss, dim=0)[0, -2].numpy())),
        pathfile_save=experiment_path / f'{prefix}_ece_zoom_t_epoch={epoch}',
        show_plot=show_plots,
    )

    ece_plot(
        llrs_norm_sum_all_validate,
        labels_validate,
        pathfile_save=experiment_path / f'{prefix}_ece_norm_epoch={epoch}',
        show_plot=show_plots,
    )

    ece_plot(
        llrs_norm_sum_all_validate,
        labels_validate,
        # -2 for all axes
        ylim=(0, 2.0 * float(torch.mean(validate_loss, dim=0)[0, -2].numpy())),
        pathfile_save=experiment_path / f'{prefix}_ece_zoom_norm_epoch={epoch}',
        show_plot=show_plots,
    )

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_t_sum_focal_validate, 10**-15, 10**15), labels_validate)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_t_focal_validate_epoch={epoch}")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_t_sum_all_validate, 10**-15, 10**15), labels_validate)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_t_sum_all_validate_epoch={epoch}")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_norm_sum_focal_validate, 10**-15, 10**15), labels_validate)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_norm_focal_validate_epoch={epoch}")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_norm_sum_all_validate, 10**-15, 10**15), labels_validate)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_norm_all_validate_epoch={epoch}")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_norm_sum_all_train, 10**-15, 10**15), labels_train)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_norm_all_train_epoch={epoch}")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_norm_sum_focal_train, 10**-15, 10**15), labels_train)
        ax.xlim(-6, 6)
        ax.savefig(experiment_path / f"{prefix}_llr_histogram_norm_focal_train_epoch={epoch}")

    calibration_plot(
        llrs_t_sum_all_validate,
        labels_validate,
        title=f'PAV-plot validate t, epoch = {epoch}',
        show_plot=show_plots,
        pathfile_save=experiment_path / f"{prefix}_validate_t_pav_plot_epoch_{epoch}",
        xlim=(-15, 15),
    )

    calibration_plot(
        llrs_norm_sum_all_validate,
        labels_validate,
        title=f'PAV-plot validate norm, epoch = {epoch}',
        show_plot=show_plots,
        pathfile_save=experiment_path / f"{prefix}_validate_norm_pav_plot_epoch_{epoch}",
        xlim=(-15, 15),
    )

    calibration_plot(
        llrs_t_sum_all_train,
        labels_train,
        title=f'PAV-plot train t, epoch = {epoch}',
        show_plot=show_plots,
        pathfile_save=experiment_path / f"{prefix}_train_t_pav_plot_epoch_{epoch}",
        xlim=(-15, 15),
    )

    calibration_plot(
        llrs_norm_sum_all_train,
        labels_train,
        title=f'PAV-plot train norm, epoch = {epoch}',
        show_plot=show_plots,
        pathfile_save=experiment_path / f"{prefix}_train_norm_pav_plot_epoch_{epoch}",
        xlim=(-15, 15),
    )

    plot_z_qnorms(
        z_train,
        labels_train,
        train_loss.squeeze(0).numpy().tolist(),
        f'qnorm train at {prefix} epoch',
        [int(idx) for idx in axes],
        experiment_path / f"{prefix}_qnorm_train_epoch={epoch}",
        show_plot=show_plots,
    )

    plot_z_qnorms(
        z_validate,
        labels_validate,
        validate_loss.squeeze(0).numpy().tolist(),
        f'qnorm validate at {prefix} epoch',
        [int(idx) for idx in axes],
        experiment_path / f"{prefix}_qnorm_validate_epoch={epoch}",
        show_plot=show_plots,
    )

    plot_llr_distributions(
        llrs_norm_train,
        labels_train,
        torch.mean(train_loss, dim=1).squeeze(0).numpy().tolist(),
        f'llr distributions norm train at {prefix} epoch',
        experiment_path / f"{prefix}_llr_histograms_norm_train_epoch={epoch}",
        show_plot=show_plots,
    )

    plot_llr_distributions(
        llrs_norm_validate,
        labels_validate,
        torch.mean(validate_loss, dim=1).squeeze(0).numpy().tolist(),
        f'llr distributions norm validate at {prefix} epoch',
        experiment_path / f"{prefix}_llr_histograms_norm_validate_epoch={epoch}",
        show_plot=show_plots,
    )

    plot_llr_distributions(
        llrs_t_train,
        labels_train,
        torch.mean(train_loss, dim=1).squeeze(0).numpy().tolist(),
        f'llr distributions t train at {prefix} epoch',
        experiment_path / f"{prefix}_llr_histograms_t_train_epoch={epoch}",
        show_plot=show_plots,
    )

    plot_llr_distributions(
        llrs_t_validate,
        labels_validate,
        torch.mean(validate_loss, dim=1).squeeze(0).numpy().tolist(),
        f'llr distributions t validate at {prefix} epoch',
        experiment_path / f"{prefix}_llr_histograms_t_validate_epoch={epoch}",
        show_plot=show_plots,
    )


def plot_and_save_test_results(
    llrs_t: np.ndarray,
    llrs_norm: np.ndarray,
    z: dict[str, Tensor],
    labels: np.ndarray,
    test_loss_norm: Tensor,
    test_loss_t: Tensor,
    axes_focus: np.ndarray,
    output_dir: Path,
):
    llrs_norm_sum = np.sum(llrs_norm, axis=1)
    llrs_t_sum = np.sum(llrs_t, axis=1)

    ece_plot(
        llrs_norm_sum,
        labels,
        pathfile_save=output_dir / 'test_ece_norm_best_epoch',
        show_plot=False,
    )

    ece_plot(
        llrs_norm_sum,
        labels,
        # -2 for all axes
        ylim=(0, 2.0 * float(test_loss_norm[-2])),
        pathfile_save=output_dir / 'test_ece_zoom_norm_best_epoch=',
        show_plot=False,
    )

    ece_plot(
        llrs_t_sum,
        labels,
        pathfile_save=output_dir / 'test_ece_t_best_epoch',
        show_plot=False,
    )

    ece_plot(
        llrs_t_sum,
        labels,
        # -2 for all axes
        ylim=(0, 2.0 * float(test_loss_t[-2])),
        pathfile_save=output_dir / 'test_ece_zoom_t_best_epoch=',
        show_plot=False,
    )

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_norm_sum, 10**-15, 10**15), labels)
        ax.xlim(-6, 6)
        ax.savefig(output_dir / "test_llr_histogram_norm_best_epoch")

    with lir.plotting.show() as ax:
        ax.lr_histogram(np.clip(10**llrs_t_sum, 10**-15, 10**15), labels)
        ax.xlim(-6, 6)
        ax.savefig(output_dir / "test_llr_histogram_t_best_epoch")

    calibration_plot(
        llrs_norm_sum,
        labels,
        title='PAV-plot test norm, best_epoch',
        show_plot=False,
        pathfile_save=output_dir / "test_norm_pav_plot_best_epoch",
        xlim=(-15, 15),
    )

    calibration_plot(
        llrs_t_sum,
        labels,
        title='PAV-plot test integrate_variance, best_epoch',
        show_plot=False,
        pathfile_save=output_dir / "test_t_pav_plot_best_epoch",
        xlim=(-15, 15),
    )

    plot_z_qnorms(
        z,
        labels,
        test_loss_t.unsqueeze(0).numpy().tolist(),
        'qnorm test (with loss_t) at best_epoch',
        [int(idx) for idx in axes_focus],
        output_dir / "test_qnorm_best_epoch",
        show_plot=False,
    )

    plot_llr_distributions(
        llrs_norm,
        labels,
        test_loss_norm.numpy().tolist(),
        'llr distributions test normal_model at best_epoch',
        output_dir / "test_llr_histograms_norm_best_epoch",
        show_plot=False,
    )

    plot_llr_distributions(
        llrs_t,
        labels,
        test_loss_t.numpy().tolist(),
        'llr distributions test integrated_variance at best_epoch',
        output_dir / "test_llr_histograms_t_test_best_epoch",
        show_plot=False,
    )
