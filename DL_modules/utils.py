import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import yaml
from confidence import Configuration
from lir.metrics import devpav
from pandas import DataFrame
from torch import Tensor


class ResultTLM(NamedTuple):
    # TLM predictionclass.
    # We predict: log_normal likelihoods, log_t_distributed likelihoods, z_scores and variances
    # (although the latter are model :parameters).
    log_norm: dict[str, torch.Tensor]
    log_t: dict[str, torch.Tensor]
    z: dict[str, torch.Tensor]
    variances: dict[str, torch.Tensor]


def odds_to_logodds(odds: np.ndarray) -> np.ndarray:
    return np.log10(odds)


def probability_to_logodds(p: np.ndarray) -> np.ndarray:
    """
    Convert probability values to their log odds with base 10.
    """
    with np.errstate(divide="ignore"):
        complement = 1 - p
        return np.log10(p) - np.log10(complement)


def result_to_csv(result_tlm: ResultTLM, labels: np.ndarray, filepath: Path) -> pd.DataFrame:
    data = labels.reshape((-1, 1))
    n_axis = list(range(result_tlm.log_norm['llh_norm_0'].shape[1]))
    headers = ['labels']
    for name, data_dict in result_tlm._asdict().items():
        if name != 'variances':
            for key, value in data_dict.items():
                temp = [f"{key}_ax_{axis}" for axis in n_axis]
                headers.extend(temp)
                data = np.concatenate((data, value), axis=1)

    df = DataFrame(data, columns=headers)
    # TODO make saving this a separate cfg parameter?
    # at the moment we do not want to save this, since it takes too much diskspace
    # df.to_csv(filepath, index=False)
    return df


def save_configuration(cfg: Configuration, filepath: Path):
    cfg_dict = cfg.__dict__['_source']
    with open(filepath, 'w') as file:
        yaml.safe_dump(cfg_dict, file)


def misleading_evidence_to_csv(
    predictions_with_labels: pd.DataFrame,
    variances: dict,
    train_losses_per_axis: Tensor,
    n_comparisons_to_save: int,
    csv_output: Path,
    label_value: int,
    sort: bool,
    n_axes_to_keep: int,
):
    variances_h1 = variances['variances_0'].numpy()
    variances_h0 = variances['variances_1'].numpy()
    train_losses_per_axis = train_losses_per_axis.numpy()

    cllrs_variances_ratios = []
    for i, (value1, value0) in enumerate(zip(variances_h1, variances_h0)):
        ratio = value0 / value1
        cllrs_variances_ratios.append((train_losses_per_axis[i], value1, value0, ratio))

    cllrs_variances_ratios = pd.DataFrame(
        cllrs_variances_ratios, columns=['cllr_train', 'variances_h1', 'variances_h0', 'ratio']
    )
    cllrs_variances_ratios = cllrs_variances_ratios.transpose()
    cllrs_variances_ratios.columns = [f'ax_{i}' for i in range(n_axes_to_keep)]
    cllrs_variances_ratios.index = ['cllr_train', 'variances_h1', 'variances_h0', 'ratio']

    misleading_evidence_results = []

    # get llr_norm_ax, z_0_ax and z_1_ax for a number of llr sums (row indices) over axes for H1 or H0
    filtered_df = predictions_with_labels.loc[predictions_with_labels['labels'] == label_value].copy()

    column_names = [f'llr_norm_ax_{i}' for i in range(n_axes_to_keep)]
    sum_series = filtered_df[column_names].sum(axis=1)
    filtered_df.loc[:, 'llr_norm_sum'] = sum_series
    sort_ascending = label_value == 1.0
    if sort:
        sorted_df = (
            filtered_df.nsmallest(n_comparisons_to_save, 'llr_norm_sum')
            if sort_ascending
            else filtered_df.nlargest(n_comparisons_to_save, 'llr_norm_sum')
        )
    else:
        sorted_df = filtered_df.head(n_comparisons_to_save)

    for row_index in range(n_comparisons_to_save):
        selected_row = sorted_df.iloc[row_index : row_index + 1]
        llr_sum_value = sorted_df.loc[selected_row.index, 'llr_norm_sum'].iloc[0]

        result_per_ax_dfs = []
        for i in range(0, n_axes_to_keep):
            llr_column = f'llr_norm_ax_{i}'
            z_0_column = f'z_0_ax_{i}'
            z_1_column = f'z_1_ax_{i}'

            subset_df = selected_row[[llr_column, z_0_column, z_1_column]]
            transposed_df = subset_df.transpose()
            transposed_df.columns = [f'ax_{i}']
            transposed_df.index = ['llr_norm_ax', 'z_0_ax', 'z_1_ax']
            result_per_ax_dfs.append(transposed_df)

        all_axes_df = pd.concat(result_per_ax_dfs, axis=1)
        llr_sum_row = pd.DataFrame(
            [[llr_sum_value] * len(all_axes_df.columns)], columns=all_axes_df.columns, index=['llr_sum']
        )
        all_axes_df = pd.concat([llr_sum_row, all_axes_df], axis=0)
        empty_row_for_index = pd.DataFrame(
            [[np.nan] * len(all_axes_df.columns)],
            columns=all_axes_df.columns,
            index=[f'Row index: {row_index}'],
        )
        row_index_df = pd.concat([all_axes_df, empty_row_for_index])

        misleading_evidence_results.append(row_index_df)

    csv_df = pd.concat(misleading_evidence_results, axis=0)

    combined_dfs = pd.concat([cllrs_variances_ratios, csv_df], axis=0)
    combined_dfs.to_csv(csv_output, index=True)


def misleading_evidence_collection_to_csv(
    predictions_with_labels_train: pd.DataFrame,
    predictions_with_labels_validate: pd.DataFrame,
    variances: dict,
    train_losses_per_axis: Tensor,
    experiment_path: Path | str,
    n_comparisons_to_save: int,
    epoch: int,
    prefix: str,
    n_axes_to_keep: int,
):
    """
    Write misleading evidence results of train and validate versus labels 0 and 1 to file.

    :param predictions_with_labels_train: DataFrame that holds for train data predictions (rows) per axis (columns) and
        a column with labels
    :param predictions_with_labels_validate: DataFrame that holds for validation data predictions (rows) per axis
        (columns) and a column with labels
    variances: the fitted variances by the model, per axis and hypotheses
    param train_losses_per_axis: Tensor that holds for train data losses per axis, only axes_to_keep
    param n_comparisons_to_save: number of comparisons to save in csv per label and filestem
    param epoch: epoch number
    param prefix: prefix for csv files
    :param n_axes_to_keep: number of axes to keep
    """
    n_comparisons_to_save = min(n_comparisons_to_save, np.sum(predictions_with_labels_train.labels == 1.0))
    misleading_evidence_to_csv(
        predictions_with_labels_train,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_sorted_train_label0_epoch={epoch}.csv',
        0,
        True,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_train,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_sorted_train_label1_epoch={epoch}.csv',
        1,
        True,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_validate,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_sorted_validate_label0_epoch={epoch}.csv',
        0,
        True,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_validate,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_sorted_validate_label1_epoch={epoch}.csv',
        1,
        True,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_train,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_random_train_label0_epoch={epoch}.csv',
        0,
        False,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_train,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_random_train_label1_epoch={epoch}.csv',
        1,
        False,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_validate,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_random_validate_label0_epoch={epoch}.csv',
        0,
        False,
        n_axes_to_keep,
    )

    misleading_evidence_to_csv(
        predictions_with_labels_validate,
        variances,
        train_losses_per_axis,
        n_comparisons_to_save,
        Path(experiment_path) / f'{prefix}_misleading_evidence_random_validate_label1_epoch={epoch}.csv',
        1,
        False,
        n_axes_to_keep,
    )


def save_results(
    model_name: str,
    train_loss_on_validate_crit: Tensor,
    loss_validate: Tensor,
    all_labels_validate: Tensor,
    all_labels_train: Tensor,
    all_predictions_validate: ResultTLM,
    all_predictions_train: ResultTLM,
    axes_focus: Tensor,
    epoch: int,
    experiment_path: Path,
    evaluate_column: int,
    norm_or_t_tracked: str,
    n_axes_to_keep: int,
    prefix: str,
    cllr_threshold: float,
    loss_integrate_variance: Tensor | None = None,
    tracked_losses: Tensor | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Save data for specific epoch to dict.

    Param model_name:  model name
    Param train_loss_on_validate_crit: epoch train loss
    Param loss_validate: validate loss
    Param all_labels_validate: validation set labels
    Param all_labels_train: train set labels
    Param all_predictions_validate: ResultTLM predictions validation set
    Param all_predictions_train: ResultTLM predictions train set
    Param axes_focus: chosen axes for which cllr_train < cllr_threshold
    Param epoch: epoch
    Param experiment_path: experiment path
    param evaluate_column: column in losses that is used for validate_loss_tracking: either column for
        'all' (one_but_last) or 'focal' (last)
    param norm_or_t_tracked: either llrs based on 'norm' or 't' distributions were tracked for loss_tracking
    param n_axes_to_keep: number of axes to keep in TwoLevelModel
    param prefix: prefix for file_names (e.g. 'overfit' or 'best')
    param cllr_threshold: # when calculating the mean of cllr_train over axes, skip axes with cllr > this value
    param loss_integrate_variance: loss when lls are calculated for which the model integates over within variance
        uncertainty
    param tracked_losses: all losses for all axes and epochs (only used when training is completed)
    """
    llrs_norm_sum_validate = torch.sum(all_predictions_validate.log_norm['llr_norm'], dim=1).numpy()
    devpav_validate = try_devpav(10**llrs_norm_sum_validate, all_labels_validate.numpy())
    train_losses = train_loss_on_validate_crit.squeeze(0).squeeze(0).numpy()
    output_params = {
        'model': model_name,
        'epoch': epoch,
        'cllr_validate': float(loss_validate.squeeze(0).squeeze(0)[evaluate_column].numpy()),
        'cllr_validate_all_info': loss_validate.squeeze(0).numpy().tolist(),
        'accompanying_train_loss_on_validate_criterion': train_losses.tolist(),
        'axes_focus': axes_focus.numpy().tolist(),
        'devpav_validate': float(devpav_validate),
        'variances_h1': all_predictions_validate.variances['variances_1'].numpy().tolist(),
        'variances_h0': all_predictions_validate.variances['variances_0'].numpy().tolist(),
        'norm_or_t_tracked_for_loss': norm_or_t_tracked,
        'n_axes_to_keep': n_axes_to_keep,
        'sd_cllr_train_focal': float(np.std(train_losses[train_losses < cllr_threshold])),
    }

    if tracked_losses is not None:
        output_params = output_params | {
            'validate_losses_per_axis_and_epoch': tracked_losses.squeeze(1).numpy().tolist(),
        }
    if loss_integrate_variance is not None:
        output_params = output_params | {
            'loss_integrate_variance': float(loss_integrate_variance[evaluate_column]),
            "loss_integrate_variance_all_info": loss_integrate_variance.numpy().tolist(),
        }

    with open(experiment_path / f'{prefix}_results_epoch={epoch}.json', 'w') as file:
        json.dump(output_params, file, indent=4)

    predictions_with_labels_train = result_to_csv(
        all_predictions_train, all_labels_train, experiment_path / f'{prefix}_predictions_train_{epoch}.csv'
    )
    predictions_with_labels_validate = result_to_csv(
        all_predictions_validate, all_labels_validate, experiment_path / f'{prefix}_predictions_validate_{epoch}.csv'
    )

    return predictions_with_labels_train, predictions_with_labels_validate


def save_results_ss(
    model_name: str,
    train_loss_on_validate_crit_ss_norm: Tensor,
    validate_loss_on_validate_crit_ss_norm: Tensor,
    train_loss_on_validate_crit_ss_t: Tensor,
    validate_loss_on_validate_crit_ss_t: Tensor,
    evaluate_column: int,
    experiment_path: Path,
):
    """
    Save losses for specific source LLRs to file.

    :Param model_name:  model name
    :Param train_loss_on_validate_crit_ss_norm: train split loss for specific source LLRs without variance integration
    :Param validate_loss_on_validate_crit_ss_norm: validate split loss for specific source LLRs without variance
        integration
    :Param train_loss_on_validate_crit_ss_t: train split loss for specific source LLRs with variance integration
    :Param validate_loss_on_validate_crit_ss_t: validate split loss for specific source LLRs with variance integration
    :Param evaluate_column: column in losses that is used for validate_loss_tracking: either column for
        'all' (one_but_last) or 'focal' (last)
    :Param experiment_path: experiment path
    """
    output_params = {
        'model': model_name,
        'cllr_validate_ss_norm': float(
            validate_loss_on_validate_crit_ss_norm.squeeze(0).squeeze(0)[evaluate_column].numpy()
        ),
        'cllr_validate_ss_norm_all_info': validate_loss_on_validate_crit_ss_norm.squeeze(0).numpy().tolist(),
        'cllr_train_ss_norm': float(train_loss_on_validate_crit_ss_norm.squeeze(0).squeeze(0)[evaluate_column].numpy()),
        'cllr_train_ss_norm_all_info': train_loss_on_validate_crit_ss_norm.squeeze(0).numpy().tolist(),
        'cllr_validate_ss_t': float(validate_loss_on_validate_crit_ss_t.squeeze(0).squeeze(0)[evaluate_column].numpy()),
        'cllr_validate_ss_t_all_info': validate_loss_on_validate_crit_ss_norm.squeeze(0).numpy().tolist(),
        'cllr_train_ss_t': float(train_loss_on_validate_crit_ss_t.squeeze(0).squeeze(0)[evaluate_column].numpy()),
        'cllr_train_ss_t_all_info': train_loss_on_validate_crit_ss_t.squeeze(0).numpy().tolist(),
    }

    with open(experiment_path / 'specific_source_losses.json', 'w') as file:
        json.dump(output_params, file, indent=4)


def reshape_data_labels(data, labels):
    # Sort indices based on labels
    sorted_indices = np.argsort(labels)

    # Sort data and labels using sorted indices
    sorted_data = data[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # calculate the number of repetitions
    n_rep = int(len(sorted_labels) / np.unique(sorted_labels).size)
    n_features = data.shape[1]

    # Reshape sorted data to (n/r, r, f)
    reshaped_data = sorted_data.reshape(-1, n_rep, n_features)

    # Reshape sorted labels to get unique labels per group
    # Select the first label of each group assuming they are now sequentially grouped
    reshaped_labels = sorted_labels[::n_rep]

    return reshaped_data, reshaped_labels


def try_devpav(LRs: np.ndarray, labels: np.ndarray):
    """
    Try devpav and catch error.

    : param LRs: numpy array of LRs, shape(n,)
    : param labels: numpy array of labels, shape(n,)
    """
    try:
        result = devpav(LRs, labels)
    except (ValueError, AttributeError):
        # rarely, PAVtransform is non-monotonic due to rounding errors. In that case it errors.
        result = np.nan
    return result
