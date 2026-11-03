from collections.abc import Mapping

import lir.plotting
import matplotlib.colors as mcolors
import numpy as np
import statsmodels.api as sm
import torch
from matplotlib import pyplot as plt
from torch import Tensor

# Choose a colormap
colormap = plt.get_cmap('viridis')


def plot_losses(
    num_epochs: int, train_loss: list[float], validate_loss: list[float], pathfile_save=None, show_plot: bool = False
):
    """Plot training and validate loss as a function of epoch.

    :param num_epochs: number of epochs used for training
    :param train_loss: training losses
    :param validate_loss: validate losses
    :param pathfile_save: path to save the plot to
    :param show_plot: show plot or not
    """
    # let's plot results
    epochs = range(0, num_epochs)
    plt.figure()
    plt.suptitle("base_network")
    plt.plot(epochs, train_loss, "bo", label="Training loss")
    plt.plot(epochs, validate_loss, "b", label="Validate loss")
    plt.xlabel("epoch")
    plt.yscale('log')
    plt.yticks(rotation=90)
    plt.ylabel("cllr")
    plt.legend()
    plt.axhline(y=1, color='r', linestyle='--', linewidth=2)
    if pathfile_save:
        plt.savefig(pathfile_save, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()


def plot_validate_losses_multiple_axes(
    num_epochs: int,
    train_loss: np.ndarray,
    validate_loss: np.ndarray,
    best_epoch=None,
    title=None,
    pathfile_save=None,
    show_plot: bool = True,
):
    """Plot training and validate loss as a function of epoch. Plots validate losses also per axis.

    :param num_epochs: number of epochs used for training
    :param train_loss: training losses shape: (n_epochs, )
    :param validate_loss: validate losses
        shape: (n_epochs, 1 or 2 (depending on loss per h or for both hs), n_axes + 2).
        The + 2 is one loss for all axes combined, and one for focus_axes combined
    :param best_epoch: the epoch where the validate loss was minimal.
    :param pathfile_save: path to save the plot to
    :param show_plot: show plot or not
    """
    # check whether a loss was calculated for instances from both hs, or that a loss was calculated for instances per h
    one_loss_per_h_or_one_for_both = train_loss.shape[1]
    if one_loss_per_h_or_one_for_both == 0 or one_loss_per_h_or_one_for_both > 2:
        raise ValueError(
            f"The second dimension of train_loss should be of length 1 or 2; found: {one_loss_per_h_or_one_for_both}"
        )

    for i in np.arange(one_loss_per_h_or_one_for_both):
        # let's plot results
        n_axes = validate_loss.shape[2] - 2  # last one is foxus axes and last_but_one is all_axes
        # Normalize the integer
        norm = mcolors.Normalize(vmin=0, vmax=n_axes)
        epochs = range(0, num_epochs)
        plt.figure()
        if one_loss_per_h_or_one_for_both == 1:
            plt.suptitle("tlm")
        if one_loss_per_h_or_one_for_both == 2:
            plt.suptitle(f"tlm h_{i}")
        plt.plot(epochs, train_loss[:, i], "bo", label="Train")
        plt.plot(epochs, validate_loss[:, i, -1], "b", label="Validate_focus_axes")
        plt.plot(epochs, validate_loss[:, i, -2], "g", label="Validate_all_axes")
        for axis in range(n_axes):
            color = colormap(norm(axis))
            plt.plot(epochs, validate_loss[:, i, axis], label=f"Validate {axis}", color=color)
        if best_epoch is not None:
            plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1)
        plt.xlabel("epoch")
        plt.yscale('log')
        plt.yticks(rotation=90)
        plt.ylabel("cllr")
        plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='small', title="loss")
        plt.axhline(y=1, color='r', linestyle='--', linewidth=2)
        if title:
            plt.title(title)
        if pathfile_save and one_loss_per_h_or_one_for_both == 1:
            plt.savefig(pathfile_save, bbox_inches='tight')
        if pathfile_save and one_loss_per_h_or_one_for_both == 2:
            name, extension = pathfile_save.split('.')
            plt.savefig(f'{name}_h_{i}.{extension}', bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()


def plot_train_losses_multiple_axes(
    num_epochs: int, train_loss: np.ndarray, best_epoch=None, pathfile_save=None, show_plot: bool = True
):
    """
    Plot training losses per axis and for all axes combined as a function of epoch.

    :param num_epochs: number of epochs used for training
    :param train_loss: train loss, per axis and with a last dimension the loss for all axes combined.
        shape: (n_epochs, 1 or 2 (depending on loss per h or for both hs), n_axes + 2).
        The + 2 is one loss for all axes combined, and one for focus_axes combined
    :param best_epoch: the epoch where the validate loss was minimal.
    :param pathfile_save: path to save the plot to
    :param show_plot: show plot or not
    """
    # check whether a loss was calculated for instances from both hs, or that a loss was calculated for instances per h
    one_loss_per_h_or_one_for_both = train_loss.shape[1]
    if one_loss_per_h_or_one_for_both == 0 or one_loss_per_h_or_one_for_both > 2:
        raise ValueError("The second dimension of train_loss should be of length 1 or 2")

    for i in np.arange(one_loss_per_h_or_one_for_both):
        # let's plot results
        n_axes = train_loss.shape[2] - 2  # last one is foxus axes and last_but_one is all_axes
        # Normalize the integer
        norm = mcolors.Normalize(vmin=0, vmax=n_axes)
        epochs = range(0, num_epochs)
        plt.figure()
        if one_loss_per_h_or_one_for_both == 1:
            plt.suptitle("tlm")
        if one_loss_per_h_or_one_for_both == 2:
            plt.suptitle(f"tlm h_{i}")
        plt.plot(epochs, train_loss[:, i, -1], "bo", label="Train_focus_axes")
        plt.plot(epochs, train_loss[:, i, -2], "go", label="Train_all")

        for axis in range(n_axes):
            color = colormap(norm(axis))
            plt.plot(epochs, train_loss[:, i, axis], label=f"Axis {axis}", color=color)
        if best_epoch is not None:
            plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1)
        plt.xlabel("epoch")
        plt.yscale('log')
        plt.yticks(rotation=90)
        plt.ylabel("cllr")
        plt.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='small', title="loss")
        plt.axhline(y=1, color='r', linestyle='--', linewidth=2)
        if pathfile_save and one_loss_per_h_or_one_for_both == 1:
            plt.savefig(pathfile_save, bbox_inches='tight')
        if pathfile_save and one_loss_per_h_or_one_for_both == 2:
            name, extension = pathfile_save.split('.')
            plt.savefig(f'{name}_h_{i}.{extension}', bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()


def plot_llr_distributions(
    llrs: np.array,
    labels: np.array,
    loss_per_axis: list[float],
    suptitle: str,
    pathfile_save: str | None = None,
    show_plot: bool = True,
):
    """
    Plot SS and DS distributions of LLR, per axis. Include the cllr of the axis in the plot title.
    :param llrs: array of LLRs. Size (n, m). Columns denote axes.
    :param labels: array of labels. Size (n,).
    :param loss_per_axis: array of loss_per_axis. Size (m,).
    :param suptitle: title of the plots.
    :param pathfile_save: path (including file extension) to save the plots to.
    :param show_plot: show plot or not.
    """
    n_axes = llrs.shape[1]
    fig, axes = plt.subplots(nrows=4, ncols=int(np.ceil(n_axes / 4)), figsize=(15, 10), sharex='all')
    fig.suptitle(suptitle)
    for i, ax in enumerate(axes.ravel()):
        if i < n_axes:
            lir.plotting.lr_histogram(np.clip(10 ** llrs[:, i], 10**-15, 10**15), labels, ax=ax)
            ax.set_xlim(-6, 6)
            ax.grid()
            ax.set_title(f'axes {i + 1}, with loss {loss_per_axis[i]:.3f}')
        else:
            ax.remove()

    plt.subplots_adjust(hspace=0.4)
    if pathfile_save is not None:
        plt.savefig(pathfile_save)
    if show_plot:
        plt.show()
    plt.close()


def plot_z_qnorms(
    z: Mapping[str, torch.tensor],
    labels: np.ndarray,
    loss_per_axis: list[float],
    suptitle: str,
    axes_focus: list[int] | None = None,
    pathfile_save: str | None = None,
    show_plot: bool = True,
):
    """
    Q-norm plots for (hopefully) standard normally distributed data, per axis and for the summed output of axes_focus
    (divided by the square root of the number of axes_focus). Loss_per_axis is used in the titles of the sub-plots.
    :param z: TLM output of z, under h0 and h1.
    :param labels: array of labels. Size (n,).
    :param loss_per_axis: array of loss_per_axis.
        If same loss for H0 and H1 axis (such as cllr), the list has one element with m floats.
        If different loss for H0 and H1 axis (such as KL-divergence), the list has two elements with m floats each.
    :param suptitle: title of the plots.
    :param axes_focus: list of indices of the axes to consider.
    :param pathfile_save: path (including file extension) to save the plots to.
    :param show_plot: show plot or not.
    """
    if len(loss_per_axis) == 1:
        loss_per_axis = loss_per_axis * 2
    for loss_h, (key, z_h) in zip(loss_per_axis, z.items()):
        n_axes = z_h.shape[1]
        n_plots = n_axes + 2

        z_ones = z_h.numpy()[labels == 1]
        xlim_abs = np.ceil(np.max(np.abs(z_ones)))
        fig, axes = plt.subplots(nrows=4, ncols=int(np.ceil(n_plots / 4)), figsize=(15, 10), sharex='all')
        fig.suptitle(f'{key} - {suptitle}')
        for i, ax in enumerate(axes.ravel()):
            if i < n_axes:
                sm.qqplot(z_ones[:, i], line='45', loc=0, scale=1, ax=ax)
                ax.set_xlim(-xlim_abs, xlim_abs)
                ax.grid()
                ax.set_title(f'axes {i + 1}, with loss {loss_h[i]:.3f}')
            elif i == n_axes:
                # this is for all axes
                scaled_sum = np.sum(z_ones, axis=1) / np.sqrt(len(axes_focus))
                sm.qqplot(scaled_sum, line='45', loc=0, scale=1, ax=ax)
                ax.set_xlim(-xlim_abs, xlim_abs)
                ax.grid()
                title = "sum_z for all-axes"
                ax.set_title(title)
            elif i == n_axes + 1 and axes_focus is not None:
                # this is for axes_focus
                z_focus = z_ones[:, axes_focus]
                if not np.array_equal(z_focus, z_ones):  # else this plot is redundant since z_focus covers all axes
                    scaled_sum = np.sum(z_focus, axis=1) / np.sqrt(len(axes_focus))
                    sm.qqplot(scaled_sum, line='45', loc=0, scale=1, ax=ax)
                    ax.set_xlim(-xlim_abs, xlim_abs)
                    ax.grid()
                    title = f'sum_z for axes {[ax + 1 for ax in axes_focus]}'
                    ax.set_title(title)
                else:
                    ax.remove()
            else:
                ax.remove()
        plt.subplots_adjust(hspace=0.4)
        if pathfile_save is not None:
            plt.savefig(str(pathfile_save) + f'{key}.png')
        if show_plot:
            plt.show()
        plt.close()


def plot_several_validate_losses(
    losses: dict[str, Tensor], num_epochs: int, pathfile_save=None, show_plot: bool = True
):
    """
    Compare multiple validate losses, for example a loss for TLM normal LLRs and a loss for TLM t-distributed LLRs.
    :param losses: Dictionary of loss-type to losses. Losses have dimension (n_epochs, n_axis)
    :param num_epochs: number of epochs.
    :param pathfile_save: path (including extension) to save the file to.
    :param show_plot: show plot or not.
    """
    plt.figure()
    epochs = range(0, num_epochs)
    norm = mcolors.Normalize(vmin=0, vmax=len(losses) * 2)
    for i, (kind, loss) in enumerate(losses.items()):
        color_1 = colormap(norm(2 * i))
        color_2 = colormap(norm(2 * i + 1))
        argmin_1 = int(np.argmin(loss[:, -1]))
        argmin_2 = int(np.argmin(loss[:, -2]))
        plt.plot(epochs, loss[:, -1], label=f"{kind}_focus_axes, b_epoch={argmin_1}", color=color_1)
        plt.plot(epochs, loss[:, -2], label=f"{kind}_all_axes, b_epoch={argmin_2}", color=color_2)
        plt.scatter(epochs[argmin_1], loss[argmin_1, -1], color=color_1)
        plt.scatter(epochs[argmin_2], loss[argmin_2, -2], color=color_2)
        plt.yscale('log')
        plt.yticks(rotation=90)
        plt.ylabel("cllr")
        plt.legend()
    if pathfile_save:
        plt.savefig(pathfile_save, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
