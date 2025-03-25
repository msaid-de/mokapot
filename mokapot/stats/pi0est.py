from __future__ import annotations

import logging
from collections import namedtuple
from typing import Literal

import numpy as np
import scipy as sp
from typeguard import typechecked

from mokapot.stats.histdata import TDHistData
from mokapot.stats.pvalues import empirical_pvalues

LOGGER = logging.getLogger(__name__)

Pi0EstStorey = namedtuple(
    "Pi0Est", ["pi0", "pi0s_smooth", "pi0s_raw", "lambdas", "mse"]
)


@typechecked
def pi0_from_scores_storey(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    *,
    method: Literal["smoother", "bootstrap", "fixed"] = "smoother",
    lambdas: np.ndarray[float] = np.arange(0.2, 0.8, 0.01),
    eval_lambda: float = 0.5,
) -> float:
    """
    Estimate the proportion of true null hypotheses (pi0) from the input scores and
    targets using Storey's method. The method computes empirical p-values for the
    given scores and derives the proportion of true null hypotheses based on the
    provided parameters. It supports smoothing, bootstrap, and fixed methods for
    calculation.

    Parameters
    ----------
    scores:
        Array of score values, where each score corresponds to a hypothesis test.
    targets:
        Boolean array indicating which scores correspond to the hypothesized
        signal values (true) and which correspond to the background or
        null hypothesis (false).
    method:
        Method used to estimate pi0. Options include "smoother" for smoothed
        estimation, "bootstrap" for bootstrap-based estimation, and "fixed" for a
        pre-defined estimate. Default is "smoother".
    lambdas:
        Array of lambda values used for tuning and smoothing purposes.
        Default values range from 0.2 to 0.8 with a step size of 0.01.
    eval_lambda:
        The lambda value at which to evaluate pi0. Default is 0.5.

    Returns
    -------
    float
        The estimated pi0 value, representing the proportion of true null
        hypotheses.

    """
    pvalues = empirical_pvalues(scores[targets], scores[~targets])
    pi0est = pi0_from_pvalues_storey(
        pvalues, method=method, lambdas=lambdas, eval_lambda=eval_lambda
    )
    return pi0est.pi0


@typechecked
def pi0_from_pvalues_storey(
    pvals: np.ndarray[float],
    *,
    method: Literal["smoother", "bootstrap", "fixed"] = "smoother",
    lambdas: np.ndarray[float] = np.arange(0.2, 0.8, 0.01),
    eval_lambda: float = 0.5,
) -> Pi0EstStorey:
    """
    Estimate pi0 from p-value using Storey's method.

    Parameters
    ----------
    pvals:
        Array of p-values for which the proportion of null hypotheses (pi0) is
        estimated.
    method:
        The method used for smoothing ('smoother' or 'bootstrap'). Default is
        'smoother'.
    lambdas:
        An array of lambda values used to estimate pi0. Default is an array
        from 0.05 to 0.95 with step 0.05. For the meaning of lambda and eval_lambda
        please the paper [1].
    eval_lambda:
        A value of lambda used to evaluate pi0. Default is 0.5.

    Returns
    -------
    A namedtuple with fields
    - pi0 : float
        The estimated pi0 value.
    - pi0s_smoothed : np.ndarray[float]
        Array of smoothed pi0 values.
    - pi0s_lambda : np.ndarray[float]
        Array of raw pi0 estimates.
    - lambdas : np.ndarray[float]
        Array of lambdas used to estimate pi0.

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """
    N = len(pvals)
    lambdas = np.sort(lambdas)

    assert min(pvals) >= 0 and max(pvals) <= 1
    assert min(lambdas) >= 0 and max(lambdas) <= 1
    assert len(lambdas) >= 4

    if max(pvals) < max(lambdas):
        LOGGER.warning(
            f"The maximum p-value ({max(pvals)}) should be larger than the "
            f"maximum lambda ({max(lambdas)})"
        )

    pi0s_smooth = pi0s_raw = mse = None

    if method == "fixed":
        pvals_exceeding_lambda = sum(pvals >= eval_lambda)
        pi0 = pvals_exceeding_lambda / (N * (1.0 - eval_lambda))
    else:
        # Estimate raw pi0 values ("contaminated" for small lambdas by the true
        # target distribution
        W = count_larger(pvals, lambdas)
        pi0s_raw = W / (N * (1.0 - lambdas))

        if method == "smoother":
            # Now smooth it with a smoothing spline and evaluate
            pi0_spline_est = sp.interpolate.UnivariateSpline(
                lambdas, pi0s_raw, k=3, ext=0
            )
            pi0s_smooth = pi0_spline_est(lambdas)
            pi0 = pi0_spline_est(eval_lambda)
        elif method == "bootstrap":
            pi0_min = np.quantile(pi0s_raw, 0.1)
            mse = (
                W / (N**2 * (1 - lambdas) ** 2) * (1 - W / N)
                + (pi0s_raw - pi0_min) ** 2
            )
            pi0 = pi0s_raw[np.argmin(mse)]
        else:
            raise ValueError(f"Unknown pi0-estimation method {method}")

    pi0 = np.clip(pi0, 0, 1)

    return Pi0EstStorey(pi0, pi0s_smooth, pi0s_raw, lambdas, mse)


@typechecked
def pi0_from_hist_storey(
    td_hist_data: TDHistData,
    *,
    method: Literal["smoother", "bootstrap", "fixed"] = "smoother",
    lambdas: np.ndarray[float] = np.arange(0.2, 0.8, 0.01),
    eval_lambda: float = 0.5,
) -> float:
    """
    Estimate pi0 from p-value using Storey's method adapted to histograms.

    Parameters
    ----------
    td_hist_data:
        Target decoy histogram data.
    method:
        The method used for smoothing ('smoother' or 'bootstrap'). Default is
        'smoother'.
    lambdas:
        An array of lambda values used to estimate pi0. Default is an array
        from 0.05 to 0.95 with step 0.05. For the meaning of lambda and eval_lambda
        please the paper [1].
    eval_lambda:
        A value of lambda used to evaluate pi0. Default is 0.5.

    Returns
    -------
    float:
        The estimated pi0 value.

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """
    # todo: it would be nice, though not trivial, to integrate this function
    #  with the normal Storey pi0 estimation function.

    _, target_counts, decoy_counts = td_hist_data.as_counts()

    targets_sum = np.flip(target_counts).cumsum()
    decoys_sum = np.flip(decoy_counts).cumsum()
    N = targets_sum[-1]

    # We need to append the last value once for the last bin_edge
    targets_sum = np.append(targets_sum, targets_sum[-1])
    decoys_sum = np.append(decoys_sum, decoys_sum[-1])

    lambdas = np.sort(lambdas)
    pvals = decoys_sum / decoys_sum[-1]

    def num_pvals_by_lambda(lmbda):
        return np.interp(lmbda, pvals, targets_sum.max() - targets_sum)

    assert min(pvals) >= 0 and max(pvals) <= 1
    assert min(lambdas) >= 0 and max(lambdas) <= 1
    assert len(lambdas) >= 4

    if max(pvals) < max(lambdas):
        LOGGER.warning(
            f"The maximum p-value ({max(pvals)}) should be larger than the "
            f"maximum lambda ({max(lambdas)})"
        )

    if method == "fixed":
        # pvals_exceeding_lambda = sum(pvals >= eval_lambda)
        pvals_exceeding_lambda = num_pvals_by_lambda(eval_lambda)

        pi0 = pvals_exceeding_lambda / (N * (1.0 - eval_lambda))
    else:
        # Estimate raw pi0 values ("contaminated" for small lambdas by the true
        # target distribution
        W = num_pvals_by_lambda(lambdas)

        pi0s_raw = W / (N * (1.0 - lambdas))

        if method == "smoother":
            # Now smooth it with a smoothing spline and evaluate
            pi0_spline_est = sp.interpolate.UnivariateSpline(
                lambdas, pi0s_raw, k=3, ext=0
            )
            pi0 = pi0_spline_est(eval_lambda)
        elif method == "bootstrap":
            pi0_min = np.quantile(pi0s_raw, 0.1)
            mse = (
                W / (N**2 * (1 - lambdas) ** 2) * (1 - W / N)
                + (pi0s_raw - pi0_min) ** 2
            )
            pi0 = pi0s_raw[np.argmin(mse)]
        else:
            raise ValueError(f"Unknown pi0-estimation method {method}")

    pi0 = np.clip(pi0, 0, 1)

    return pi0


def pi0_from_pdfs_by_slope(
    target_pdf: np.ndarray[float],
    decoy_pdf: np.ndarray[float],
    threshold: float = 0.9,
):
    r"""Estimate pi0 using the slope of decoy vs target PDFs.

    The idea is that :math:`f_T(s) = \pi_0 f_D(s) + (1-\pi_0) f_{TT}(s)` and
    that for small scores `s` and a scoring function that sufficiently
    separates targets and decoys (or false targets) it holds that
    :math:`f_T(s) \simeq \pi_0 f_D(s)`.
    The algorithm works by determining the maximum of the decoy distribution
    and then estimating the slope of the target vs decoy density for all scores
    left of 90% of the maximum of the decoy distribution.

    Parameters
    ----------
    target_pdf:
        An estimate of the target PDF.
    decoy_pdf:
        An estimate of the decoy PDF.
    threshold:
        The threshold for selecting decoy PDF values (default is 0.9).

    Returns
    -------
    float:
        The estimated value of pi0.
    """
    max_decoy = np.max(decoy_pdf)
    last_index = np.argmax(decoy_pdf >= threshold * max_decoy)
    try:
        pi0_est, _ = np.polyfit(decoy_pdf[:last_index], target_pdf[:last_index], 1)
    except RuntimeError:
        return 1.0
    return np.clip(pi0_est, 1e-10, 1.0)


@typechecked
def pi0est_from_scores_by_slope(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    bins: np.ndarray[float] | int | str | None,
    slope_threshold: float = 0.9,
):
    """
    Estimates the proportion of false positives (pi0) in a given set of scores and
    targets by analyzing the slope of the density functions derived from the score
    distribution.

    This function computes histograms for the provided scores with respect to their
    target and decoy classifications, estimates the density functions, and calculates
    the proportion of null hypotheses (pi0) based on the slope of these densities.

    Parameters
    ----------
    scores:
        A collection of numeric scores representing the statistical significance or
        other measures associated with some experiment or process.

    targets:
        A binary array or collection where each value indicates whether a
        corresponding score is from the target or decoy class.

    bins:
        The number of histogram bins to use or the specific bin edges to divide
        the scores into.

    slope_threshold:
        A threshold value for the slope used to determine the proportion of null
        hypotheses (pi0) from the density estimates of the target and decoy
        distributions.

    Returns
    -------
    float
        The estimated proportion of null hypotheses (pi0) based on the provided
        scores, targets, and density slope threshold.

    """
    hist_data = TDHistData.from_scores(bins, scores, targets)
    hist_data.as_densities()
    _, target_density, decoy_density = hist_data.as_densities()
    return pi0_from_pdfs_by_slope(
        target_density, decoy_density, threshold=slope_threshold
    )


def pi0est_from_counts(scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
    """
    Estimates the proportion of null (decoy) elements relative to the non-null
    (target) elements based on their counts. The function calculates the ratio
    of decoys to targets in the given dataset. If the count of decoys is zero,
    a warning is logged, and the ratio cannot be meaningfully estimated.

    Parameters
    ----------
    scores : np.ndarray[float]
        An array representing the scores of the elements. This input is not
        directly utilized in the current implementation but assumed to align
        element-wise with the `targets` array.

    targets : np.ndarray[bool]
        A boolean array indicating whether each element in the dataset is a
        target (True) or a decoy (False). The size of `targets` must match
        the size of `scores`.

    Returns
    -------
    float
        The calculated estimate of the decoy-to-target ratio. If no decoys
        are present, the ratio may not provide meaningful information.
    """
    targets_count = targets.sum()
    decoys_count = (~targets).sum()
    if decoys_count == 0:
        LOGGER.warning(
            f"Can't estimate pi0 with zero decoys (targets={targets_count}, "
            f"decoys={decoys_count}, total={len(targets)})"
        )
    decoy_target_ratio = decoys_count / targets_count
    return decoy_target_ratio


@typechecked
def count_larger(pvals, lambdas):
    """
    Counts the number of elements in `pvals` that are greater than or equal to
    each value in `lambdas`.

    This function calculates the cumulative count of `pvals` values that are
    larger than or equal to each quantile in the `lambdas` array. The `lambdas`
    array is expected to be in strictly ascending order.

    Parameters
    ----------
    pvals : array_like
        Array of values for which cumulative counts are calculated against the
        `lambdas` thresholds.

    lambdas : array_like
        Array of threshold values in strictly ascending order. The counts of
        `pvals` larger than or equal to these values will be computed.

    Returns
    -------
    cumulative_counts : ndarray
        Array of cumulative counts, where each element corresponds to the count
        of `pvals` that are greater than or equal to the corresponding value in
        `lambdas`.
    """
    assert np.all(lambdas[1:] > lambdas[:-1])
    bin_edges = np.append(lambdas, np.inf)
    hist_counts, _ = np.histogram(pvals, bins=bin_edges)
    cumulative_counts = np.cumsum(hist_counts[::-1])[::-1]
    return cumulative_counts


@typechecked
def pi0_from_scores_bootstrap(
    scores: np.ndarray[float], targets: np.ndarray[bool], N: int = 100000
) -> float:
    """
    Estimates the proportion of true null hypotheses (pi0) using bootstrapping.

    This function calculates an estimate of pi0, the proportion of true null
    hypotheses, based on bootstrap sampling. It takes as input an array of
    scores, a boolean array indicating target values, and an optional number
    of samples for the bootstrapping process. By performing resampling and
    comparing scores between target true and target false groups, the function
    computes a value proportional to the likelihood of scores from true null
    hypothesis cases.

    Parameters
    ----------
    scores : np.ndarray[float]
        Array of scores corresponding to the data points.
    targets : np.ndarray[bool]
        Boolean array where True represents the target class, and False
        represents the non-target class.
    N : int, optional
        The number of samples to be drawn for bootstrapping. The default is
        100000.

    Returns
    -------
    float
        An estimate of the proportion of true null hypotheses (pi0).
    """
    A = np.random.choice(scores[targets], size=N, replace=True)
    B = np.random.choice(scores[~targets], size=N, replace=True)
    return 2 * (sum(B > A) + 0.5 * sum(B == A)) / N


@typechecked
def pi0_from_hist_bootstrap(td_hist_data: TDHistData, N: int = 100000) -> float:
    """
    Estimates the proportion of true null hypotheses (pi0) using bootstrapping.

    This function calculates an estimate of pi0, the proportion of true null
    hypotheses, based on bootstrap sampling. It takes as input an array of
    scores, a boolean array indicating target values, and an optional number
    of samples for the bootstrapping process. By performing resampling and
    comparing scores between target true and target false groups, the function
    computes a value proportional to the likelihood of scores from true null
    hypothesis cases.

    Parameters
    ----------
    scores : np.ndarray[float]
        Array of scores corresponding to the data points.
    targets : np.ndarray[bool]
        Boolean array where True represents the target class, and False
        represents the non-target class.
    N : int, optional
        The number of samples to be drawn for bootstrapping. The default is
        100000.

    Returns
    -------
    float
        An estimate of the proportion of true null hypotheses (pi0).
    """
    A = td_hist_data.targets.sample(N)
    B = td_hist_data.decoys.sample(N)
    return 2 * (sum(B > A) + 0.5 * sum(B == A)) / N
