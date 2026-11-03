import copy
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch import Tensor
from torch.optim import Optimizer
from tqdm import tqdm

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.custom_layers import merge_results_tlm
from DL_modules.custom_layers.top_layers import SingleInstanceTwoLevelNormalNormalNProj
from DL_modules.dataloader.data_loader import LOG, CustomDataLoader
from DL_modules.plots import (
    calibration_plot,
    plot_several_validate_losses,
    plot_train_losses_multiple_axes,
    plot_validate_losses_multiple_axes,
)
from DL_modules.plots.integrated_variance import plot_integrated_vs_fixed
from DL_modules.plots.plot_collections import show_and_save_plots
from DL_modules.plots.specific_source import make_ss_cs_plots
from DL_modules.post_hoc import CoherenceFilter
from DL_modules.train_validate_loops.utils import EmptyResultTLM, get_output_path
from DL_modules.utils import ResultTLM, misleading_evidence_collection_to_csv, save_results, save_results_ss


def train_and_validate_llr_head_single_instance(
    model: SingleInstanceTwoLevelNormalNormalNProj,
    model_name: str,
    criterion_train_single_instance: TlmLossFunction,
    criterion_validate: Callable,
    num_epochs: int,
    optimizer: Optimizer,
    dataloader_train_single_source: CustomDataLoader,
    dataloader_validate_single_source: CustomDataLoader,
    dataloader_train_pairs: CustomDataLoader,
    dataloader_validate_pairs: CustomDataLoader,
    threshold_for_mean_cllr: float,
    save: bool,
    show_plots: bool,
    output_parent: str | None = None,
    save_epoch_past_best_epoch: int | None = None,
    output_dir_lr_benchmark: Path | None = None,
    also_predict_specific_source: bool = False,
):
    if output_dir_lr_benchmark:
        output_path = f"{output_dir_lr_benchmark}/"
    elif save and output_parent:
        output_path = get_output_path(output_parent)
        os.makedirs(Path(output_path), exist_ok=True)
    else:
        output_path = None

    if output_path:
        # save configuation
        os.makedirs(Path(output_path), exist_ok=True)

    # legacy code to be able to use axes_to_monitor == "focus"
    axes_to_monitor = "all"
    if axes_to_monitor not in ["all", "focus"]:
        raise ValueError(f'axes_to_monitor should be "all" or "focus"; found: {axes_to_monitor}')
    match axes_to_monitor:
        case "focus":
            evaluate_column = -1
        case "all":
            evaluate_column = -2

    n_axes_to_keep = model.n_axes_to_keep

    tracked_train_losses_all_axes = {}
    tracked_validate_losses_all_axes = {}
    best_losses = {}
    post_hoc_configurations = {}
    axes_focus_dict = {}

    coherence_clip_configurations = model.build_coherence_clip_config_dict()
    for post_hoc_folder, post_hoc_config in coherence_clip_configurations.items():
        if output_path:
            os.makedirs(Path(output_path) / post_hoc_folder, exist_ok=True)
        tracked_train_losses_all_axes[post_hoc_folder] = torch.empty(0, 1, n_axes_to_keep + 2)
        tracked_validate_losses_all_axes[post_hoc_folder] = torch.empty(0, 1, n_axes_to_keep + 2)
        best_losses[post_hoc_folder] = np.inf
        post_hoc_configurations[post_hoc_folder] = _get_configuration(
            train_coherence_filters=post_hoc_config.train_coherence_filters,
            threshold_bias_axis=post_hoc_config.threshold_bias_axis,
            max_iterations=post_hoc_config.max_iterations,
            exclude_llrs_smaller_than=post_hoc_config.exclude_llrs_smaller_than,
            train_clip_denominators=post_hoc_config.train_clip_denominators,
            clip_denominator_percentage=post_hoc_config.clip_denominator_percentage,
        )

    losses_dict = {}
    best_model_and_results = {}

    # Start training
    if model.empirical_init:
        for data, labels in dataloader_train_single_source:
            model.initialize_white_and_projector(data, labels)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss_train_crit = 0.0
        num_batches = len(dataloader_train_single_source)

        for batch_idx, (inputs) in enumerate(dataloader_train_single_source):
            optimizer.zero_grad()
            data, labels = inputs
            result_tlm, single_llrs_labels = model(data, labels)

            loss = criterion_train_single_instance(result_tlm, torch.tensor(single_llrs_labels))
            loss.backward()
            optimizer.step()

            total_train_loss_train_crit += loss.item()

            # Show progress
            if (batch_idx + 1) % 10 == 0:
                LOG.debug(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{num_batches}]")

        if (epoch + 1) % 10 == 0:
            calibration_plot(
                torch.sum(result_tlm.log_norm["llr_norm"], dim=1).detach().numpy(),
                single_llrs_labels,
                title=f"PAV-plot single_llrs_train, epoch = {epoch}",
                xlim=(-15, 15),
                show_plot=show_plots,
            )

        # Compute validate and train loss per epoch for all axes and sum_llr
        # First validate losses
        model.eval()

        # collect all tlm_results for train and validate
        all_tlmresults_pair_train_dict, all_labels_train = _run_validation_single_post_hocs(
            model, dataloader_train_pairs
        )
        all_tlmresults_pair_validate_dict, all_labels_validate = _run_validation_single_post_hocs(
            model, dataloader_validate_pairs
        )

        for post_hoc_folder in all_tlmresults_pair_train_dict.keys():
            current_train_loss_on_validate_crit, axes_focus = criterion_validate(
                all_tlmresults_pair_train_dict[post_hoc_folder].log_norm["llr_norm"], all_labels_train.float()
            )

            current_tracked_validate_losses_all_axes, _ = criterion_validate(
                all_tlmresults_pair_validate_dict[post_hoc_folder].log_norm["llr_norm"],
                all_labels_validate.float(),
            )

            current_train_loss_on_validate_crit_all_axes = current_train_loss_on_validate_crit.unsqueeze(0).unsqueeze(0)
            current_validate_losses_all_axes = current_tracked_validate_losses_all_axes.unsqueeze(0).unsqueeze(0)
            axes_focus_dict[post_hoc_folder] = axes_focus
            losses_dict[post_hoc_folder] = Losses(
                validate_loss=torch.mean(current_validate_losses_all_axes[-1, :, evaluate_column]).item(),
                train_loss=torch.mean(current_train_loss_on_validate_crit_all_axes[-1, :, evaluate_column]).item(),
                train_loss_all_axes=current_train_loss_on_validate_crit_all_axes,
                validate_loss_all_axes=current_validate_losses_all_axes,
            )
            tracked_train_losses_all_axes[post_hoc_folder] = torch.cat(
                (tracked_train_losses_all_axes[post_hoc_folder], current_train_loss_on_validate_crit_all_axes), dim=0
            )
            tracked_validate_losses_all_axes[post_hoc_folder] = torch.cat(
                (tracked_validate_losses_all_axes[post_hoc_folder], current_validate_losses_all_axes), dim=0
            )
        print(
            f"Axes monitored: {axes_to_monitor}\n"
            f"Epoch [{epoch}/{num_epochs}], "
            f"Train loss on train crit last minibatch: {float(loss.detach()):.4f}, "
        )
        for post_hoc_folder, losses in losses_dict.items():
            print(
                f"Loss from validation criteria on complete Train set for {post_hoc_folder}: "
                f"loss: {losses.train_loss:.4f} "
                f"and validation loss: "
                f"loss: {losses.validate_loss:.4f}"
            )

        # plot one of the dict values for sanity check, just take the first.
        if (epoch + 1) % 10 == 0:
            first_item, first_value = next(iter(all_tlmresults_pair_train_dict.items()))
            train_sum_pair_llrs = torch.sum(first_value.log_norm["llr_norm"], dim=1).numpy()
            calibration_plot(
                train_sum_pair_llrs,
                all_labels_train.numpy(),
                title=f"PAV-plot pair_llrs_train {first_item}, epoch = {epoch}",
                show_plot=show_plots,
                xlim=(-15, 15),
            )
            first_item, first_value = next(iter(all_tlmresults_pair_validate_dict.items()))
            validate_sum_pair_llrs = torch.sum(first_value.log_norm["llr_norm"], dim=1).numpy()
            calibration_plot(
                validate_sum_pair_llrs,
                all_labels_validate.numpy(),
                title=f"PAV-plot pair_llrs_validate {first_item}, epoch = {epoch} llrs from {"norm"}",
                show_plot=show_plots,
                xlim=(-15, 15),
            )

        # Save calibration plots of train and validate at every tenth of the number of epochs
        if output_path:
            if (epoch + 1) % np.floor(num_epochs / 10) == 0:
                for post_hoc_folder, results in all_tlmresults_pair_train_dict.items():
                    train_sum_pair_llrs = torch.sum(results.log_norm["llr_norm"], dim=1).numpy()
                    calibration_plot(
                        train_sum_pair_llrs,
                        all_labels_train.numpy(),
                        title=f"PAV-plot train_sum_pair_llrs, epoch = {epoch}",
                        show_plot=False,
                        pathfile_save=Path(output_path)
                        / post_hoc_folder
                        / f"pav_plot_train_sum_pair_llrs_epoch_{epoch}_batch_{batch_idx}.png",
                        xlim=(-15, 15),
                    )
                for post_hoc_folder, results in all_tlmresults_pair_validate_dict.items():
                    validate_sum_pair_llrs = torch.sum(results.log_norm["llr_norm"], dim=1).numpy()
                    calibration_plot(
                        validate_sum_pair_llrs,
                        all_labels_validate.numpy(),
                        title=f"PAV-plot validate_sum_pair_llrs, epoch = {epoch} llrs from {"norm"}",
                        show_plot=False,
                        pathfile_save=Path(output_path)
                        / post_hoc_folder
                        / f"pav_plot_validate_sum_pair_llrs_epoch_{epoch}_llrs_from_norm.png",
                        xlim=(-15, 15),
                    )

        # update best_models and best_losses
        for post_hoc_folder, losses in losses_dict.items():
            if losses.validate_loss < best_losses[post_hoc_folder]:
                best_losses[post_hoc_folder] = losses.validate_loss
                _update_best_model_and_results(
                    best_model_and_results,
                    post_hoc_folder,
                    post_hoc_configurations[post_hoc_folder],
                    model,
                    epoch,
                    axes_focus_dict[post_hoc_folder],
                    losses,
                )

        # Save results when overfitting at epoch_for_saving_overfit
        for post_hoc_folder, best_model_result in best_model_and_results.items():
            epoch_for_saving_overfit = (
                save_epoch_past_best_epoch + best_model_result.best_epoch if save_epoch_past_best_epoch else None
            )

            if output_path and epoch == epoch_for_saving_overfit:
                model_freezed = _freeze_single_instance_model(model, post_hoc_configurations[post_hoc_folder])
                (
                    all_tlmresults_train,
                    all_labels_train,
                    all_tlmresults_validate,
                    all_labels_validate,
                    all_tlmresults_train_ss,
                    all_tlmresults_validate_ss,
                ) = _also_integrate_out_variance(
                    model_freezed,
                    dataloader_train_single_source,
                    dataloader_train_pairs,
                    dataloader_validate_pairs,
                    also_predict_specific_source=False,
                )

                # process common source results
                validate_loss_integrate_variance, _ = criterion_validate(
                    all_tlmresults_validate.log_t["llr_t"], all_labels_validate.float()
                )

                predictions_with_labels_train, predictions_with_labels_validate = save_results(
                    model_name,
                    losses_dict[post_hoc_folder].train_loss_all_axes,
                    losses_dict[post_hoc_folder].validate_loss_all_axes,
                    all_labels_validate,
                    all_labels_train,
                    all_tlmresults_validate,
                    all_tlmresults_train,
                    axes_focus_dict[post_hoc_folder],
                    epoch,
                    Path(output_path) / post_hoc_folder,
                    evaluate_column,
                    "norm",
                    n_axes_to_keep,
                    "overfit_cs",
                    threshold_for_mean_cllr,
                    validate_loss_integrate_variance,
                )

                show_and_save_plots(
                    epoch,
                    Path(output_path) / post_hoc_folder,
                    losses_dict[post_hoc_folder].train_loss_all_axes,
                    losses_dict[post_hoc_folder].validate_loss_all_axes,
                    all_tlmresults_validate,
                    all_tlmresults_train,
                    axes_focus_dict[post_hoc_folder],
                    all_labels_validate,
                    all_labels_train,
                    "overfit_cs",
                    show_plots,
                )

                misleading_evidence_collection_to_csv(
                    predictions_with_labels_train,
                    predictions_with_labels_validate,
                    all_tlmresults_validate.variances,
                    current_train_loss_on_validate_crit_all_axes[0, 0, :n_axes_to_keep],
                    Path(output_path) / post_hoc_folder,
                    50,
                    epoch,
                    "overfit_cs",
                    n_axes_to_keep,
                )

    # After all epochs, calculate_integrated_variance_llrs, save loss plots and best_results
    for post_hoc_folder, best_model_result in best_model_and_results.items():
        (
            best_results_tlm_train,
            best_labels_train,
            best_results_tlm_validate,
            best_labels_validate,
            best_results_tlm_train_ss,
            best_results_tlm_validate_ss,
        ) = _also_integrate_out_variance(
            best_model_result.model,
            dataloader_train_single_source,
            dataloader_train_pairs,
            dataloader_validate_pairs,
            also_predict_specific_source=also_predict_specific_source,
        )

        if output_path:
            torch.save(best_model_result.model, Path(output_path) / post_hoc_folder / f"{model_name}.pth")

            # save results for common source LLRs
            final_validate_loss_on_validate_crit, _ = criterion_validate(
                best_results_tlm_validate.log_t["llr_t"], best_labels_validate.float()
            )

            plot_validate_losses_multiple_axes(
                num_epochs,
                tracked_train_losses_all_axes[post_hoc_folder][:, :, evaluate_column].numpy(),
                tracked_validate_losses_all_axes[post_hoc_folder].numpy(),
                best_epoch=best_model_result.best_epoch,
                title=f"validate_{"norm"}",
                pathfile_save=Path(output_path) / post_hoc_folder / "loss_plot_validate_norm.png",
                show_plot=show_plots,
            )
            plot_train_losses_multiple_axes(
                num_epochs,
                tracked_train_losses_all_axes[post_hoc_folder].numpy(),
                best_epoch=best_model_result.best_epoch,
                pathfile_save=Path(output_path) / post_hoc_folder / "loss_plot_train.png",
                show_plot=show_plots,
            )

            show_and_save_plots(
                best_model_result.best_epoch,
                Path(output_path) / post_hoc_folder,
                best_model_result.losses.train_loss_all_axes,
                best_model_result.losses.validate_loss_all_axes,
                best_results_tlm_validate,
                best_results_tlm_train,
                best_model_result.axes_focus,
                best_labels_validate,
                best_labels_train,
                "best",
                show_plots,
            )

            validate_losses = {"norm": tracked_validate_losses_all_axes[post_hoc_folder][:, 0, :]}  # squeeze second dim
            plot_several_validate_losses(
                validate_losses,
                num_epochs,
                pathfile_save=Path(output_path) / post_hoc_folder / "compare_several_losses.png",
                show_plot=show_plots,
            )

            predictions_with_labels_train, predictions_with_labels_validate = save_results(
                model_name,
                best_model_result.losses.train_loss_all_axes,
                best_model_result.losses.validate_loss_all_axes,
                best_labels_validate,
                best_labels_train,
                best_results_tlm_validate,
                best_results_tlm_train,
                best_model_result.axes_focus,
                best_model_result.best_epoch,
                Path(output_path) / post_hoc_folder,
                evaluate_column,
                "norm",
                n_axes_to_keep,
                "best",
                threshold_for_mean_cllr,
                final_validate_loss_on_validate_crit,
                tracked_validate_losses_all_axes[post_hoc_folder],
            )

            misleading_evidence_collection_to_csv(
                predictions_with_labels_train,
                predictions_with_labels_validate,
                best_results_tlm_validate.variances,
                best_model_result.losses.train_loss_all_axes[0, 0, :n_axes_to_keep],
                Path(output_path) / post_hoc_folder,
                50,
                best_model_result.best_epoch,
                "best",
                n_axes_to_keep,
            )

            plot_integrated_vs_fixed(
                best_results_tlm_validate,
                best_labels_validate.numpy(),
                Path(output_path) / post_hoc_folder,
                suffix='validate_cs',
            )
            plot_integrated_vs_fixed(
                best_results_tlm_train,
                best_labels_train.numpy(),
                Path(output_path) / post_hoc_folder,
                suffix='train_cs',
            )

            if also_predict_specific_source:
                # save results for specific source LLRs
                final_train_loss_on_validate_crit_ss_norm, _ = criterion_validate(
                    best_results_tlm_train_ss.log_norm["llr_norm"], best_labels_train.float()
                )
                final_validate_loss_on_validate_crit_ss_norm, _ = criterion_validate(
                    best_results_tlm_validate_ss.log_norm["llr_norm"], best_labels_validate.float()
                )
                final_train_loss_on_validate_crit_ss_t, _ = criterion_validate(
                    best_results_tlm_train_ss.log_t["llr_t"], best_labels_train.float()
                )
                final_validate_loss_on_validate_crit_ss_t, _ = criterion_validate(
                    best_results_tlm_validate_ss.log_t["llr_t"], best_labels_validate.float()
                )

                show_and_save_plots(
                    best_model_result.best_epoch,
                    Path(output_path) / post_hoc_folder,
                    final_train_loss_on_validate_crit_ss_t.unsqueeze(0).unsqueeze(0),
                    final_validate_loss_on_validate_crit_ss_t.unsqueeze(0).unsqueeze(0),
                    best_results_tlm_validate_ss,
                    best_results_tlm_train_ss,
                    best_model_result.axes_focus,
                    best_labels_validate,
                    best_labels_train,
                    "best_ss",
                    show_plots,
                )

                save_results_ss(
                    model_name,
                    final_train_loss_on_validate_crit_ss_norm,
                    final_validate_loss_on_validate_crit_ss_norm,
                    final_train_loss_on_validate_crit_ss_t,
                    final_validate_loss_on_validate_crit_ss_t,
                    evaluate_column,
                    experiment_path=Path(output_path) / post_hoc_folder,
                )

                make_ss_cs_plots(
                    best_results_tlm_train_ss,
                    best_results_tlm_train,
                    best_labels_train,
                    Path(output_path) / post_hoc_folder,
                    "train",
                )
                make_ss_cs_plots(
                    best_results_tlm_validate_ss,
                    best_results_tlm_validate,
                    best_labels_validate,
                    Path(output_path) / post_hoc_folder,
                    "validate",
                )

                plot_integrated_vs_fixed(
                    best_results_tlm_validate_ss,
                    best_labels_validate.numpy(),
                    Path(output_path) / post_hoc_folder,
                    suffix='validate_ss',
                )
                plot_integrated_vs_fixed(
                    best_results_tlm_train_ss,
                    best_labels_train.numpy(),
                    Path(output_path) / post_hoc_folder,
                    suffix='train_ss',
                )

        else:  # If you don't want to save anything and just show plots...
            plot_validate_losses_multiple_axes(
                num_epochs,
                tracked_train_losses_all_axes[post_hoc_folder].numpy(),
                tracked_validate_losses_all_axes[post_hoc_folder].numpy(),
                best_epoch=best_model_result.best_epoch,
                show_plot=show_plots,
            )
            plot_train_losses_multiple_axes(
                num_epochs,
                tracked_train_losses_all_axes[post_hoc_folder].numpy(),
                best_epoch=best_model_result.best_epoch,
                show_plot=show_plots,
            )

        # After training completes, print train_loss_train_crit and validate_loss and best
        print(
            f"Results for post_hoc_config {post_hoc_folder}\n"
            f"Axes monitored: {axes_to_monitor}\n"
            f"Best validate loss at {
                float(torch.mean(best_model_result.losses.validate_loss_all_axes, dim=1)[0, evaluate_column]):.4f}, "
            f"and accompanying train loss at "
            f"{float(torch.mean(best_model_result.losses.train_loss_all_axes, dim=1)[0, evaluate_column]):.4f} at "
            f"epoch {best_model_result.best_epoch}, "
            f"and accompanying validate integrate_variance loss at "
            f"{float(final_validate_loss_on_validate_crit[evaluate_column]):.4f}."
        )

    return {post_hoc_folder: result.model for post_hoc_folder, result in best_model_and_results.items()}


@dataclass(frozen=True)
class Losses:
    train_loss: float
    validate_loss: float
    train_loss_all_axes: torch.Tensor
    validate_loss_all_axes: torch.Tensor


@dataclass(frozen=True)
class BestModelResults:
    config: str
    model: SingleInstanceTwoLevelNormalNormalNProj
    best_epoch: int
    axes_focus: Tensor
    losses: Losses


class PostHocConfiguration(BaseModel, arbitrary_types_allowed=True):
    coherence_filter: CoherenceFilter | None
    clip_denominator: Annotated[float | None, Field(ge=0, le=1)]


InitResultTLMDict = {"init": ResultTLM({}, {}, {}, {})}


def _get_configuration(
    train_coherence_filters: bool,
    threshold_bias_axis: float | None,
    max_iterations: int | None,
    exclude_llrs_smaller_than: int | None,
    train_clip_denominators: bool,
    clip_denominator_percentage: float | None,
):
    if clip_denominator_percentage is not None and clip_denominator_percentage > 1:
        raise ValueError(
            f"clip_denominator_percentage {clip_denominator_percentage} is not valid. It should be a float between 0 "
            f"and 1."
        )
    coherence_filter = (
        CoherenceFilter(
            threshold_bias_axis=threshold_bias_axis,
            max_iterations=max_iterations,
            exclude_llrs_smaller_than=exclude_llrs_smaller_than,
        )
        if train_coherence_filters
        else None
    )
    # TODO initiate clip_denominator

    post_hoc_configuration = PostHocConfiguration(coherence_filter=coherence_filter, clip_denominator=None)
    return post_hoc_configuration


def _freeze_single_instance_model(
    model: SingleInstanceTwoLevelNormalNormalNProj, post_hoc_configuration: PostHocConfiguration
) -> SingleInstanceTwoLevelNormalNormalNProj:
    # 1. Recreate architecture (dummy variance init values are fine)
    new_model = type(model)(
        model.n_axes,
        model.n_sources_train_data,
        0.0,  # dummy placeholder for numerator_variance_extra
        0.0,  # dummy placeholder for denominator_variance_extra
        model.sample_size_sigma2_w,
        model.empirical_init,
        n_axes_to_keep=model.n_axes_to_keep,
    )

    # 2. Restore full state (params + buffers + Variance.log_var etc.)
    new_model.load_state_dict(model.state_dict(), strict=True)

    # 3. Copy python objects
    new_model.embeddings = copy.deepcopy(model.embeddings)
    new_model.means = copy.deepcopy(model.means)
    new_model.all_data = copy.deepcopy(model.all_data_to_integrate_out_var)

    # 4. Copy non-registered tensors
    new_model.alpha = model.alpha.detach().clone()
    new_model.beta = model.beta.detach().clone()
    new_model.sigma2_w_prior_sample = model.sigma2_w_prior_sample.detach().clone()

    # we have a trained model. Set it to predict ResultTLM and specific post_hoc_configuration
    new_model.predict_result_tlm = True
    new_model.coherence_filter = post_hoc_configuration.coherence_filter
    new_model.clip_denominator = post_hoc_configuration.clip_denominator

    return new_model


def _run_validation_single_post_hocs(
    model: SingleInstanceTwoLevelNormalNormalNProj,
    dataloader: CustomDataLoader,
) -> tuple[dict[str, ResultTLM], torch.Tensor]:
    """Apply the given model to the data in the given dataloader. Return a resultTLM object and tensor with all labels.

    :param model: a model that returns a ResultTLM object.
    :param dataloader: data which to run through the model.
    """
    all_tlmresults_cs_dict = InitResultTLMDict
    all_labels = torch.empty(0)
    with torch.no_grad():
        for all_data in dataloader:
            susp, cr, labels = all_data
            result_tlm_batch_cs_dict = model.predict_cs_llr(susp, cr, also_integrate_variance=False)
            all_tlmresults_cs_dict = _merge_results_tlm_single_post_hoc(
                all_tlmresults_cs_dict, result_tlm_batch_cs_dict
            )
            all_labels = torch.cat((all_labels, labels))
    return all_tlmresults_cs_dict, all_labels


def _run_validation_single(
    model: SingleInstanceTwoLevelNormalNormalNProj,
    dataloader: CustomDataLoader,
    also_integrate_variance: bool = False,
    also_predict_ss_llrs: bool = False,
) -> tuple[ResultTLM, torch.Tensor, ResultTLM]:
    """Apply the given model to the data in the given dataloader. Return a ResultTLM object and tensor with all labels.

    :param model: a model that returns a ResultTLM object.
    :param dataloader: data which to run through the model.
    :param also_integrate_variance: boolean indicating whether also return prediction based on integrating out variance
        uncertainty
    :param also_predict_ss_llrs: boolean. Also predict specific source llrs when True
    """
    all_tlmresults_cs = EmptyResultTLM
    all_tlmresults_ss = EmptyResultTLM
    all_labels = torch.empty(0)
    iterator = (
        tqdm(dataloader, desc="Evaluating with SS LLRs", total=len(dataloader), position=0)
        if also_predict_ss_llrs
        else dataloader
    )
    with torch.no_grad():
        for all_data in iterator:
            susp, cr, labels = all_data
            result_tlm_batch_cs = model.predict_cs_llr(susp, cr, also_integrate_variance)
            all_tlmresults_cs = merge_results_tlm(all_tlmresults_cs, result_tlm_batch_cs)
            all_labels = torch.cat((all_labels, labels))
            if also_predict_ss_llrs:
                result_tlm_batch_ss = model.predict_ss_llr(susp, cr, also_integrate_variance)
                all_tlmresults_ss = merge_results_tlm(all_tlmresults_ss, result_tlm_batch_ss)

    return all_tlmresults_cs, all_labels, all_tlmresults_ss


def _merge_results_tlm_single_post_hoc(
    first_dict: ResultTLM | dict[str, ResultTLM], second_dict: dict[str, ResultTLM]
) -> dict[str, ResultTLM]:
    # concatenate tlm predictions over batches.
    if first_dict == InitResultTLMDict:
        return second_dict
    else:
        result = {}
        for config_folder, second_value in second_dict.items():
            first_value = first_dict[config_folder]
            for i, dictionary in enumerate(first_value):
                if i != 3:  # variances do not need to be merged
                    for key in dictionary:
                        first_value[i][key] = torch.cat((dictionary[key], second_value[i][key]), dim=0)
            result[config_folder] = first_value
        return result


def _update_best_model_and_results(
    best_model_and_results: dict[str, BestModelResults],
    post_hoc_folder: str,
    post_hoc_configuration: PostHocConfiguration,
    model: SingleInstanceTwoLevelNormalNormalNProj,
    epoch: int,
    axes_focus: Tensor,
    losses: Losses,
):
    # todo: initiate CoherenceFilter in this model
    updated = BestModelResults(
        config=post_hoc_folder,
        model=_freeze_single_instance_model(model, post_hoc_configuration),
        best_epoch=epoch,
        axes_focus=axes_focus,
        losses=losses,
    )
    best_model_and_results[post_hoc_folder] = updated


def _also_integrate_out_variance(
    model: SingleInstanceTwoLevelNormalNormalNProj,
    dataloader_single_source: CustomDataLoader,
    dataloader_train_pairs: CustomDataLoader,
    dataloader_validate_pairs: CustomDataLoader,
    also_predict_specific_source: bool,
) -> tuple[ResultTLM, torch.Tensor, ResultTLM, torch.Tensor, ResultTLM, ResultTLM]:
    """
    Also predict LLRs assuming within_variance is not constant over sources.

    :param model: TLM single_instance model
    :param dataloader_single_source: dataloader for single sources, used to fit distribution of within variances
    :param dataloader_train_pairs: dataloader used to calculate LLRs for train pairs
    :param dataloader_validate_pairs: data loader used to calculate LLRs for validate pairs
    :param also_predict_specific_source: to also calculate specific-source LLRs.

    :returns: TLMResult for train, labels for train, TLMResult for validate, labels for validate, TLMResult for
        specific source LLRs for train, TLMResult for specific source LLRs for validate
    """
    # TODO run_validation_single -> run_validation
    with torch.no_grad():
        for data, _ in dataloader_single_source:
            model.initialize_integrate_out_variance(data, len(dataloader_single_source.dataset))

        # collect all tlm_results for train, including integration_of_variance
        best_results_tlm_train, best_labels_train, best_results_tlm_train_ss = _run_validation_single(
            model,
            dataloader_train_pairs,
            also_integrate_variance=True,
            also_predict_ss_llrs=also_predict_specific_source,
        )

        # collect all tlm_results for validate, including integration_of_variance
        best_results_tlm_validate, best_labels_validate, best_results_tlm_validate_ss = _run_validation_single(
            model,
            dataloader_validate_pairs,
            also_integrate_variance=True,
            also_predict_ss_llrs=also_predict_specific_source,
        )

        return (
            best_results_tlm_train,
            best_labels_train,
            best_results_tlm_validate,
            best_labels_validate,
            best_results_tlm_train_ss,
            best_results_tlm_validate_ss,
        )
