import copy
import json
import logging
import os
from collections.abc import Callable
from enum import Enum
from pathlib import Path

import numpy as np
import torch
from confidence import Configuration
from torch.nn import Module
from torch.optim import Optimizer

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.custom_layers import merge_results_tlm
from DL_modules.custom_layers.top_layers import SingleInstanceTwoLevelNormalNormalNProj, TwoLevelNormalNormalNProj
from DL_modules.custom_layers.two_level_normal_normal_sub_modules import ResultTLM
from DL_modules.dataloader.data_loader import CustomDataLoader
from DL_modules.plots import (
    calibration_plot,
    plot_several_validate_losses,
    plot_train_losses_multiple_axes,
    plot_validate_losses_multiple_axes,
)
from DL_modules.plots.plot_collections import show_and_save_plots
from DL_modules.post_hoc import CoherenceFilter
from DL_modules.train_validate_loops.utils import get_output_path
from DL_modules.utils import (
    misleading_evidence_collection_to_csv,
    save_configuration,
    save_results,
)

LOG = logging.getLogger(__name__)


class Track(Enum):
    T = "t"
    Norm = "norm"
    Predictions = "predictions"


def run_validation(
    model: SingleInstanceTwoLevelNormalNormalNProj | TwoLevelNormalNormalNProj, dataloader: CustomDataLoader
) -> tuple[ResultTLM, torch.Tensor]:
    """Apply the given model to the data in the given dataloader. Return a resultTLM object and tensor with all labels.

    :param model: a model that returns a ResultTLM object.
    :param dataloader: data which to run through the model.
    """
    all_tlmresults = ResultTLM({}, {}, {}, {})
    all_labels = torch.empty(0)
    with torch.no_grad():
        for all_data in dataloader:
            susp, cr, labels = all_data
            result_tlm = model(susp, cr, use_z_clip=True)
            all_tlmresults = merge_results_tlm(all_tlmresults, result_tlm)
            all_labels = torch.cat((all_labels, labels))

    return all_tlmresults, all_labels


def train_and_validate_llr_head(
    model: Module,
    model_name: str,
    criterion_train: TlmLossFunction,
    criterion_validate: Callable,
    criterion_train_evaluate: Callable,
    num_epochs: int,
    optimizer: Optimizer,
    dataloader_train: CustomDataLoader,
    dataloader_validate: CustomDataLoader,
    cfg: Configuration,
    tlm_size: int,
    input_for_criterion_validate: Track,
    gradient_clip_value: float | None = None,
    gradient_clip_norm: float | None = None,
    show_plots: bool = True,
    output_dir_lr_benchmark: Path | None = None,
    coherence_filter: bool = False,
    coherence_filter_threshold_bias_axis: float | None = None,
    coherence_filter_max_iterations: int | None = None,
    coherence_filter_exclude_llrs_smaller_than: float | None = None,
):
    if output_dir_lr_benchmark:
        output_path = f"{output_dir_lr_benchmark}/"
    elif cfg.saving.save:
        output_path = get_output_path(cfg.saving.output_parent)
    else:
        output_path = None

    if output_path:
        # save configuation
        os.makedirs(Path(output_path), exist_ok=True)
        save_configuration(cfg, Path(output_path) / "configuration.yaml")

    # Initalize best_axes. It will be overwritten after first epoch
    best_axes_focus = None
    # Initialize best_epoch
    best_epoch = 0

    n_axes_to_keep = cfg.train.n_axes_to_keep if cfg.train.n_axes_to_keep else tlm_size
    # Initialize validate loss tracking
    if input_for_criterion_validate == Track.Predictions:
        # Then prediction objects are tracked and loss is defined per hypotheses. (
        # For e.g. KLDivergence, that works per ax and hypotheses)
        # Therefore, we monitor 2 (hypotheses) *  n_axes
        validate_loss: torch.Tensor = torch.empty(0, 2, n_axes_to_keep + 2)
        train_loss_validate_crit = torch.empty(0, 2, n_axes_to_keep + 2)
    else:
        # when 'norm' and 't' are used
        validate_loss_norm = torch.empty(0, n_axes_to_keep + 2)
        validate_loss_t = torch.empty(0, n_axes_to_keep + 2)
        # train loss always for norm variant
        train_loss_validate_crit = torch.empty(0, 1, n_axes_to_keep + 2)

    axes_to_monitor = cfg.cllr_crit_eval.axes_to_monitor
    if axes_to_monitor not in ["all", "focus"]:
        raise ValueError(f'axes_to_monitor should be "all" or "focus"; found: {axes_to_monitor}')
    match axes_to_monitor:
        case "focus":
            evaluate_column = -1
        case "all":
            evaluate_column = -2

    filter = (
        CoherenceFilter(
            threshold_bias_axis=coherence_filter_threshold_bias_axis,
            max_iterations=coherence_filter_max_iterations,
            exclude_llrs_smaller_than=coherence_filter_exclude_llrs_smaller_than,
        )
        if coherence_filter
        else None
    )

    # Start training
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss_train_crit = 0.0
        num_batches = len(dataloader_train)

        for batch_idx, (inputs) in enumerate(dataloader_train):
            optimizer.zero_grad()
            susp, cr, labels = inputs
            # moving tensors to GPU does not speed things up a lot. Probably since the model is so simple.
            # susp.to('cuda')
            # cr.to('cuda')
            # labels.to('cuda')
            result_tlm = model(susp, cr)
            loss = criterion_train(result_tlm, labels.float())

            loss.backward()

            # Perform gradient clipping by value, by norm, or none
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clip_value)
            elif gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

            optimizer.step()
            total_train_loss_train_crit += loss.item()

            # Show progress
            if (batch_idx + 1) % 10 == 0:
                LOG.debug(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{num_batches}]")

        if (epoch + 1) % 10 == 0:
            calibration_plot(
                torch.sum(result_tlm.log_norm['llr_norm'], dim=1).detach().numpy(),
                labels.numpy(),
                title=f"PAV-plot train, epoch = {epoch}",
                xlim=(-15, 15),
                show_plot=show_plots,
            )

        # Compute validate and train loss per epoch for all axes and sum_llr
        # First validate losses
        model.eval()

        # collect all tlm_results for train and validate
        all_tlmresults_validate, all_labels_validate = run_validation(model, dataloader_validate)
        all_tlmresults_train, all_labels_train = run_validation(model, dataloader_train)

        # calculate train and validate loss for current epoch on all predictions
        if input_for_criterion_validate == Track.Predictions:
            current_train_loss_on_validate_crit, _ = criterion_train_evaluate(
                all_tlmresults_train, all_labels_train.float()
            )
            current_train_loss_on_validate_crit = current_train_loss_on_validate_crit.unsqueeze(0)
            current_validate_loss, _ = criterion_validate(all_tlmresults_validate, all_labels_validate)
        else:  # then we have tracked 't' or 'norm' llrs
            current_train_loss_on_validate_crit, _ = criterion_train_evaluate(
                all_tlmresults_train.log_norm["llr_norm"], all_labels_train.float()
            )
            current_train_loss_on_validate_crit = current_train_loss_on_validate_crit.unsqueeze(0).unsqueeze(0)

            # apply coherence filter if yaml indicates so.
            # Its purpose is to correct for incoherence in LLRs over axes, i.e. that some are positive while others
            #   are negative.
            if filter:
                all_tlmresults_train.log_norm['llr_norm'] = filter.apply(
                    all_tlmresults_train.log_norm['llr_norm'],
                    current_train_loss_on_validate_crit.flatten()[:n_axes_to_keep],
                )

                all_tlmresults_validate.log_t['llr_t'] = filter.apply(
                    all_tlmresults_validate.log_t['llr_t'],
                    current_train_loss_on_validate_crit.flatten()[:n_axes_to_keep],
                )
                all_tlmresults_validate.log_norm['llr_norm'] = filter.apply(
                    all_tlmresults_validate.log_norm['llr_norm'],
                    current_train_loss_on_validate_crit.flatten()[:n_axes_to_keep],
                )

            current_validate_loss_t, _ = criterion_validate(
                all_tlmresults_validate.log_t["llr_t"], all_labels_validate.float()
            )

            current_validate_loss_norm, _ = criterion_validate(
                all_tlmresults_validate.log_norm["llr_norm"],
                all_labels_validate.float(),
            )

            # Calculate current train loss on filtered predictions, since above we need to use unfiltered
            #   current_train_loss_on_validate_criterion
            current_train_loss_on_validate_crit, axes_focus = criterion_train_evaluate(
                all_tlmresults_train.log_norm["llr_norm"], all_labels_train.float()
            )
            current_train_loss_on_validate_crit = current_train_loss_on_validate_crit.unsqueeze(0).unsqueeze(0)
            train_loss_validate_crit = torch.cat((train_loss_validate_crit, current_train_loss_on_validate_crit), dim=0)
            validate_loss_norm = torch.cat((validate_loss_norm, current_validate_loss_norm.unsqueeze(0)), dim=0)
            validate_loss_t = torch.cat((validate_loss_t, current_validate_loss_t.unsqueeze(0)), dim=0)

        if input_for_criterion_validate == Track.Predictions:
            LOG.debug(
                f"Axes monitored: {axes_to_monitor}\n"
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train Loss on validate crit: {torch.mean(train_loss_validate_crit[-1, :, evaluate_column]):.4f},"
                f"Validate Loss: {torch.mean(current_validate_loss[:, evaluate_column]):.4f}"
            )
        else:  # i.e. we tracked 't' or 'norm'.
            LOG.debug(
                f"Axes monitored: {axes_to_monitor}\n"
                f"Epoch [{epoch}/{num_epochs}], "
                f"Train loss on train crit last minibatch: {float(loss.detach()):.4f}, "
                f"Loss from validation criteria on complete Train set: "
                f"{torch.mean(train_loss_validate_crit[-1, :, evaluate_column]):.4f}, "
                f"Loss from validation criteria on complete Validation set: "
                f"{current_validate_loss_t[evaluate_column].item():.4f}"
            )

        # Save calibration plots of train and validate at every tenth of the number of epochs
        if output_path:
            if (epoch + 1) % np.floor(num_epochs / 10) == 0:
                train_sum_llrs = torch.sum(all_tlmresults_train.log_norm["llr_norm"], dim=1).numpy()
                calibration_plot(
                    train_sum_llrs,
                    all_labels_train.numpy(),
                    title=f"PAV-plot train, epoch = {epoch}",
                    show_plot=False,
                    pathfile_save=f"{output_path}pav_plot_train_epoch_{epoch}_batch_{batch_idx}.png",
                    xlim=(-15, 15),
                )
                if input_for_criterion_validate in (Track.Predictions, Track.Norm):
                    validate_sum_llrs = torch.sum(all_tlmresults_validate.log_norm["llr_norm"], dim=1).numpy()
                elif input_for_criterion_validate == Track.T:
                    validate_sum_llrs = torch.sum(all_tlmresults_validate.log_t["llr_t"], dim=1).numpy()

                calibration_plot(
                    validate_sum_llrs,
                    all_labels_validate.numpy(),
                    title=f"PAV-plot validate, epoch = {epoch + 1} llrs from {input_for_criterion_validate.value}",
                    show_plot=False,
                    pathfile_save=f"{output_path}pav_plot_validate_epoch_{epoch + 1}"
                    f"_llrs_from_{input_for_criterion_validate.value}.png",
                    xlim=(-15, 15),
                )

        # Show calibration plot of validate data every tenth epoch
        if (epoch + 1) % 10 == 0:
            if input_for_criterion_validate in (Track.Norm, Track.Predictions):
                validate_sum_llrs = torch.sum(all_tlmresults_validate.log_norm["llr_norm"], dim=1).numpy()
            elif input_for_criterion_validate == Track.T:
                validate_sum_llrs = torch.sum(all_tlmresults_validate.log_t["llr_t"], dim=1).numpy()

            calibration_plot(
                validate_sum_llrs,
                all_labels_validate.numpy(),
                title=f"PAV-plot validate, epoch = {epoch}, llrs from {input_for_criterion_validate.value}",
                xlim=(-15, 15),
                show_plot=show_plots,
            )

        # Keep track of best results
        if input_for_criterion_validate == Track.T:
            tracked_losses = validate_loss_t.unsqueeze(1)
            current_loss = current_validate_loss_t.unsqueeze(0).unsqueeze(0)
        elif input_for_criterion_validate == Track.Norm:
            tracked_losses = validate_loss_norm.unsqueeze(1)
            current_loss = current_validate_loss_norm.unsqueeze(0).unsqueeze(0)
        elif input_for_criterion_validate == Track.Predictions:
            tracked_losses = validate_loss
            current_loss = current_validate_loss.unsqueeze(0)

        if epoch == 0 or float(torch.mean(current_loss, dim=1)[0, evaluate_column]) < np.min(
            torch.mean(tracked_losses, dim=1)[:-1, evaluate_column].numpy()
        ):
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            best_axes_focus = axes_focus

            best_train_loss = current_train_loss_on_validate_crit
            best_validate_loss = current_loss
            best_labels_validate = all_labels_validate
            best_labels_train = all_labels_train

            best_results_tlm_train = all_tlmresults_train
            best_results_tlm_validate = all_tlmresults_validate
            best_variances_1 = all_tlmresults_validate.variances["variances_1"].numpy()
            best_variances_0 = all_tlmresults_validate.variances["variances_0"].numpy()

        # Save results when overfitting at epoch_for_saving_overfit
        epoch_for_saving_overfit = (
            cfg.save_epoch_past_best_epoch + best_epoch if cfg.save_epoch_past_best_epoch else None
        )
        if output_path and epoch == epoch_for_saving_overfit:
            predictions_with_labels_train, predictions_with_labels_validate = save_results(
                model_name,
                current_train_loss_on_validate_crit,
                current_loss,
                all_labels_validate,
                all_labels_train,
                all_tlmresults_validate,
                all_tlmresults_train,
                axes_focus,
                epoch,
                Path(output_path),
                evaluate_column,
                Track(input_for_criterion_validate).value,
                n_axes_to_keep,
                "overfit",
                cfg.threshold_for_mean_cllr,
            )

            show_and_save_plots(
                epoch,
                Path(output_path),
                current_train_loss_on_validate_crit,
                current_loss,
                all_tlmresults_validate,
                all_tlmresults_train,
                axes_focus,
                all_labels_validate,
                all_labels_train,
                "overfit",
                cfg.show_plots,
            )

            misleading_evidence_collection_to_csv(
                predictions_with_labels_train,
                predictions_with_labels_validate,
                all_tlmresults_validate.variances,
                current_train_loss_on_validate_crit[0, 0, :n_axes_to_keep],
                output_path,
                50,
                epoch,
                "overfit",
                n_axes_to_keep,
            )

    # After all epochs, save loss plots and best_results
    if output_path:
        torch.save(best_model, Path(output_path) / f"{model_name}.pth")

        plot_validate_losses_multiple_axes(
            num_epochs,
            train_loss_validate_crit[:, :, evaluate_column].numpy(),
            tracked_losses.numpy(),
            best_epoch=best_epoch,
            title=f"validate_{input_for_criterion_validate.value}",
            pathfile_save=f"{output_path}loss_plot_validate_t.png",
            show_plot=show_plots,
        )
        plot_train_losses_multiple_axes(
            num_epochs,
            train_loss_validate_crit.numpy(),
            best_epoch=best_epoch,
            pathfile_save=f"{output_path}loss_plot_train.png",
            show_plot=show_plots,
        )

        show_and_save_plots(
            best_epoch,
            Path(output_path),
            best_train_loss,
            best_validate_loss,
            best_results_tlm_validate,
            best_results_tlm_train,
            best_axes_focus,
            best_labels_validate,
            best_labels_train,
            "best",
            cfg.show_plots,
        )

        # Save results to a json file
        if input_for_criterion_validate == Track.Predictions:
            output_params = {
                "model": model_name,
                "best_epoch": best_epoch,
                "best_KLdiv_validate": float(torch.mean(best_validate_loss, dim=1).squeeze(0)[-1].numpy()),
                "best_cllr_validate_all_info": best_validate_loss.squeeze(0).numpy().tolist(),
                "accompanying_train_loss_on_validate_criterion_all_info": best_train_loss.squeeze(0).numpy().tolist(),
                "axes focus": best_axes_focus.numpy().tolist(),
                "validate_losses": tracked_losses.numpy().tolist(),
                "train_ss_batch_size": cfg.train.ss_batch_size,
                "validate_ss_batch_size": cfg.validate.ss_batch_size,
                "best_variances_h1": best_variances_1.tolist(),
                "best_variances_h0": best_variances_0.tolist(),
                "tracked_for_loss": input_for_criterion_validate,
            }
            with open(output_path / f"best_results_epoch={best_epoch}.json", "w") as file:
                json.dump(output_params, file, indent=4)

        else:  # Then we have tracked t or norm and not KLdiv. We save a lot more in this case
            validate_losses = {"norm": validate_loss_norm, "t": validate_loss_t}
            plot_several_validate_losses(
                validate_losses,
                num_epochs,
                pathfile_save=f"{output_path}compare_several_losses.png",
                show_plot=show_plots,
            )

            predictions_with_labels_train, predictions_with_labels_validate = save_results(
                model_name,
                best_train_loss,
                best_validate_loss,
                best_labels_validate,
                best_labels_train,
                best_results_tlm_validate,
                best_results_tlm_train,
                best_axes_focus,
                best_epoch,
                Path(output_path),
                evaluate_column,
                Track(input_for_criterion_validate).value,
                n_axes_to_keep,
                "best",
                cfg.threshold_for_mean_cllr,
                tracked_losses=tracked_losses,
            )

            misleading_evidence_collection_to_csv(
                predictions_with_labels_train,
                predictions_with_labels_validate,
                best_results_tlm_validate.variances,
                best_train_loss[0, 0, :n_axes_to_keep],
                output_path,
                50,
                best_epoch,
                "best",
                n_axes_to_keep,
            )

    else:  # If you don't want to save anything and just show plots...
        plot_validate_losses_multiple_axes(
            num_epochs,
            train_loss_validate_crit.numpy(),
            tracked_losses.numpy(),
            best_epoch=best_epoch,
            show_plot=cfg.show_plots,
        )
        plot_train_losses_multiple_axes(
            num_epochs,
            train_loss_validate_crit.numpy(),
            best_epoch=best_epoch,
            show_plot=cfg.show_plots,
        )

    # After training completes, print train_loss_train_crit and validate_loss and best
    LOG.debug(f"Train Losses: {train_loss_validate_crit[:, :, evaluate_column]}")
    LOG.debug(f"Validate Losses: {tracked_losses[:, :, evaluate_column]}")
    LOG.debug(
        f"Axes monitored: {axes_to_monitor}\n"
        f"Best validate loss at {float(torch.mean(best_validate_loss, dim=1)[0, evaluate_column]):.4f}, "
        f"and accompanying train loss at "
        f"{float(torch.mean(best_train_loss, dim=1)[0, evaluate_column]):.4f} at epoch {best_epoch}."
    )

    return best_model
