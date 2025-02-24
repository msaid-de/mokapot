from __future__ import annotations

import logging
from collections import namedtuple
from typing import Literal

import numpy as np
import scipy as sp
from typeguard import typechecked

from mokapot.stats.histdata import hist_data_from_scores

LOGGER = logging.getLogger(__name__)

Pi0EstStorey = namedtuple(
    "Pi0Est", ["pi0", "pi0s_smooth", "pi0s_raw", "lambdas", "mse"]
)


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
    pvals : np.ndarray[float]
        Array of p-values for which the proportion of null hypotheses (pi0) is
        estimated.
    method : str, optional
        The method used for smoothing ('smoother' or 'bootstrap'). Default is
        'smoother'.
    lambdas : np.ndarray, optional
        An array of lambda values used to estimate pi0. Default is an array
        from 0.05 to 0.95 with step 0.05.

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


def pi0est_from_scores_by_slope(scores, targets, bins, slope_threshold):
    hist_data = hist_data_from_scores(scores, targets, bins=bins)
    hist_data.as_densities()
    _, target_density, decoy_density = hist_data.as_densities()
    return pi0_from_pdfs_by_slope(
        target_density, decoy_density, threshold=slope_threshold
    )


def pi0est_from_counts(scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
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
