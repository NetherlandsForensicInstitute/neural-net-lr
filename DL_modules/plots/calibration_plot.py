import warnings
from pathlib import Path

import lir
import numpy as np
from lrbenchmark.calibration import IsotonicCalibrator
from matplotlib import pyplot as plt


def calibration_plot(
    llrs: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float | int, float | int] = (-6, 6),
    ylim: tuple[float | int, float | int] = (-6, 6),
    spline_means: str | None = None,
    show_plot: bool = True,
    title: str | None = None,
    pathfile_save: Path | str | None = None,
):
    """Plot isotonic regression result. Include identity line.

    :param input_data nd_array of p's, lr's or llr's. Dimension (n)
    :param y nd_array of labels (0, 1). Dimension (n)
    :param from_p_llr_or_lr: either 'from_p', 'from_llr' or 'from_lr'.
    :param x_lim : lower and upper of x_axis
    :param y_lim: lower and upper of y_axis
    :param spline_means: whether to include spline means in the plot
    :param show_plot: whether to show the plot or not
    :param title: optional title of the plot
    :param pathfile_save: path_file including extension to save the plot to

    """
    # sort input_data in ascending order
    sorted_index = llrs.argsort()
    y = y[sorted_index]
    llrs = llrs[sorted_index]

    # Define the filter to supress warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # fit isotonic regression
    cal = IsotonicCalibrator()
    LLRPAV = cal.fit_transform(llrs, y)

    # start with 1-dimensional scatter plots of data
    X0, X1 = lir.util.Xy_to_Xn(llrs, y)
    delta = ylim[1] - ylim[0]
    legendH1 = "H1, N = " + str(len(X1))
    legendH2 = "H2, N = " + str(len(X0))
    label_scatter_H1 = plt.scatter(
        X1, np.repeat(ylim[0] + 0.06666 * delta, len(X1)), c='#1f77b4', marker="+", label=legendH1
    )
    label_scatter_H2 = plt.scatter(
        X0, np.repeat(ylim[0] + 0.03333 * delta, len(X0)), c='#1f77b4', marker="|", label=legendH2
    )

    # find y-values that are smaller than ylim[0].
    indices = np.where(LLRPAV < ylim[0])[0]
    # If present, find the largest index and replace LLRPAV[largest_index] by ylim[0] and remove smaller indices.
    # Adjust llrs[largest_index] accordingly
    if indices.size > 0:
        largest_index = np.max(indices)
        fraction_shift = (ylim[0] - LLRPAV[largest_index + 1]) / (LLRPAV[largest_index] - LLRPAV[largest_index + 1])
        x_new = llrs[largest_index + 1] + fraction_shift * (llrs[largest_index] - llrs[largest_index + 1])
        LLRPAV[largest_index] = ylim[0]
        LLRPAV = LLRPAV[largest_index:]
        llrs = llrs[largest_index:]
        llrs[0] = x_new

    # do the same but then for larger than ylim[1]
    indices = np.where(LLRPAV > ylim[1])[0]
    # Find the smallest index and replace LLRPAV[smallest_index] by ylim[1] and remove larger indices.
    # Adjust llrs[smallest_index] accordingly
    if indices.size > 0:
        smallest_index = np.min(indices)
        fraction_shift = (ylim[1] - LLRPAV[smallest_index - 1]) / (LLRPAV[smallest_index] - LLRPAV[smallest_index - 1])
        x_new = llrs[smallest_index - 1] + fraction_shift * (llrs[smallest_index] - llrs[smallest_index - 1])

        LLRPAV[smallest_index] = ylim[1]
        LLRPAV = LLRPAV[: smallest_index + 1]
        llrs = llrs[: smallest_index + 1]
        llrs[-1] = x_new

    # start with straight line y = x
    x = np.linspace(np.min((xlim, ylim)), np.max((xlim, ylim)), 100)
    plt.plot(x, x, c='#FFA500')

    # plot PAV transform
    (label_PAV_transform,) = plt.plot(llrs, LLRPAV, c='#1f77b4', label="PAV-transform")
    # plot limits
    label_limits_plot = plt.scatter(
        (np.min(X1), np.max(X1), np.min(X0), np.max(X0)),
        (ylim[0] + 0.1 * delta, ylim[0] + 0.1 * delta, ylim[0] + 0.9 * delta, ylim[0] + 0.9 * delta),
        c='#FFA500',
        marker="x",
        label="data limits",
    )

    # plot interpolation splines (optional)
    if spline_means is not None:
        # plt.plot(edge_coordinates[:,0], edge_coordinates[:,1], c='#FFA500')
        label_interpolated: str = "spline interpolated"
        (label_interpolation_plot,) = plt.plot(
            spline_means[:, 0], spline_means[:, 1], c='red', label=label_interpolated
        )
        plt.legend(
            handles=[
                label_PAV_transform,
                label_interpolation_plot,
                label_scatter_H1,
                label_scatter_H2,
                label_limits_plot,
            ],
            loc=2,
        )
    else:
        plt.legend(handles=[label_PAV_transform, label_scatter_H1, label_scatter_H2, label_limits_plot], loc=2)

    # finish plot
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid()
    plt.xlabel("LLR/input_data")
    plt.ylabel("LLR PAV")
    plt.title(title)

    if pathfile_save:
        plt.savefig(pathfile_save, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
