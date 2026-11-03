import matplotlib.pyplot as plt
import numpy as np
import torch
from lir import calibration
from torch import logsumexp


def ece_plot(
    llrs: np.ndarray,
    y: np.ndarray,
    log_prior_odds_range: list[float] | None = None,
    ylim: tuple[float, float] | None = (0, None),
    pathfile_save: str | None = None,
    show_plot: bool = True,
):
    """
    Generate an ECE plot for a set of LRs and corresponding ground-truth
    labels.

    The x-axis indicates the log prior odds of a sample being drawn from class
    1; the y-axis shows the entropy for (1) a non-informative system (dotted
    line), (2) the set of LR values (line), and (3) the set of LR values after
    PAV-transformation (Pool Adjacent Violators, dashed line).

    :param llrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param log_prior_odds_range: the range of prior odds (tuple of two values,
        indicating both ends of the range on the x-axis)
    :param ylim: ylim of the plot
    :param pathfile_save: the path of the file (including extension) to save the plot to
    """
    if log_prior_odds_range is None:
        log_prior_odds_range = (-3, 3)

    log_prior_odds = np.arange(*log_prior_odds_range, 0.01)

    # plot reference
    plt.plot(
        log_prior_odds, calculate_ece(np.zeros(2), np.array([1, 0]), log_prior_odds), linestyle=':', label='reference'
    )

    # plot LRs
    plt.plot(log_prior_odds, calculate_ece(llrs, y, log_prior_odds), linestyle='-', label='LRs')

    # plot PAV LRs
    pav_lrs = calibration.IsotonicCalibrator().fit_transform(llrs, y)
    with np.errstate(divide='ignore'):
        pav_llrs = np.log10(pav_lrs)
    plt.plot(log_prior_odds, calculate_ece(pav_llrs, y, log_prior_odds), linestyle='--', label='PAV LRs')

    plt.xlabel("prior log$_{10}$(odds)")
    plt.ylabel("empirical cross-entropy")
    plt.ylim(ylim)
    plt.xlim(log_prior_odds_range)
    plt.legend()
    plt.grid(True, linestyle=':')
    if pathfile_save is not None:
        plt.savefig(pathfile_save)
    if show_plot:
        plt.show()
    plt.close()


def calculate_ece(llrs: np.ndarray, y: np.ndarray, log_prior_odds: np.ndarray):
    """
    Calculate the empirical cross-entropy (ECE) of a set of LRs and
    corresponding ground-truth labels.

    An entropy is calculated for each element of `priors`.

    :param llrs: an array of Log10LRs, shape (n,)
    :param y: an array of ground-truth labels of the LRs (values 0 for Hd or 1
        for Hp); shape (n,)
    :param log_prior_odds: an array of log prior odds of the samples being drawn
        from class 1, shape (m,)
    :returns: an array of entropy values of the same length as `priors`, shape (m, )
    """
    if not np.all(np.unique(y) == np.array([0, 1])):
        raise ValueError("label set must be [0, 1]")
    ln_lrs = llrs * np.log(10.0)
    ln_prior_odds = np.repeat(log_prior_odds * np.log(10.0), len(ln_lrs)).reshape((len(log_prior_odds), len(ln_lrs)))
    prior_odds = np.power(10, log_prior_odds)
    priors = prior_odds / (1 + prior_odds)
    ln_posterior_odds = ln_prior_odds + ln_lrs
    zeros_np_array = np.zeros(ln_posterior_odds.shape)
    ln_posterior_odds_with_zeros = np.stack((ln_posterior_odds, zeros_np_array), axis=-1)
    ln_to_base_2 = 1 / np.log(2.0)
    logsumexp_result = logsumexp(torch.tensor(ln_posterior_odds_with_zeros), dim=2).numpy()
    ece0 = -(1 - priors.reshape((len(priors), 1))) * -logsumexp_result[:, y == 0]
    with np.errstate(invalid='ignore'):
        ece1 = -priors.reshape((len(priors), 1)) * (ln_posterior_odds[:, y == 1] - logsumexp_result[:, y == 1])

    # filter out inf - inf. Should be 0. not nan.
    ece1[np.isinf(ln_posterior_odds[:, y == 1])] = 0

    avg0 = np.average(ece0, axis=1)
    avg1 = np.average(ece1, axis=1)

    return ln_to_base_2 * (avg0 + avg1)
