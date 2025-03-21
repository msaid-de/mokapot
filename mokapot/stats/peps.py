from __future__ import annotations

import logging
import os
import shutil
from typing import Callable

import numpy as np
from triqler import qvality
from typeguard import typechecked

from mokapot.stats.histdata import TDHistData
from mokapot.stats.monotonize import fit_nnls, monotonize_nnls
from mokapot.stats.pi0est import pi0_from_pdfs_by_slope
from mokapot.stats.utils import pdfs_from_scores

LOGGER = logging.getLogger(__name__)


class PepsConvergenceError(Exception):
    """Raised when nnls iterations do not converge."""

    pass


@typechecked
def peps_from_scores_qvality(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    is_tdc: bool,
    use_binary: bool = False,
    pi0: float | None = None,
) -> np.ndarray[float]:
    """Compute PEPs from scores using the triqler pep algorithm.

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    use_binary:
        Whether to the binary (Percolator) version of qvality (True), or the
        Python (triqler) version (False). Defaults to False. If True, the
        compiled `qvality` binary must be on the shell search path.

    Returns
    -------
    array:
        A numpy array containing the posterior error probabilities (PEPs)
        calculated using the qvality method. The PEPs are calculated based on
        the provided scores and targets, and are returned in the same order as
        the targets array.
    """

    # Triqler returns the peps for reverse sorted scores, so we sort the scores
    # ourselves and later sort them back
    index = np.argsort(scores)[::-1]
    scores_sorted, targets_sorted = scores[index], targets[index]

    try:
        old_verbosity, qvality.VERB = qvality.VERB, 0
        target_scores = scores_sorted[targets_sorted]
        decoy_scores = scores_sorted[~targets_sorted]
        args = [target_scores, decoy_scores]
        kwargs = dict(includeDecoys=True, includePEPs=True, tdcInput=is_tdc)
        if use_binary:
            if pi0 is not None:
                raise ValueError("Cannot use `use_binary` with `pi0`")
            _, peps_sorted = qvality.getQvaluesFromScoresQvality(*args, **kwargs)
            peps_sorted = np.array(peps_sorted, dtype=float)
        else:
            # includePEPs is ignored in triqler, tdcInput also
            pi0 = 1.0 if pi0 is None else pi0
            _, peps_sorted = qvality.getQvaluesFromScores(*args, **kwargs, pi0=pi0)

        inverse_idx = np.argsort(index)
        peps = peps_sorted[inverse_idx]
    except SystemExit as msg:
        if "no decoy hits available for PEP calculation" in str(msg):
            peps = np.zeros_like(scores)
        else:
            raise
    finally:
        qvality.VERB = old_verbosity

    return peps


def is_qvality_on_path():
    """
    Check whether qvality is executable and on the system path.

    Returns:
        bool: True if the qvality is callable, False otherwise.
    """
    # Check if the executable exists on the path
    path = shutil.which("qvality")
    if not path:
        return False

    # Check if the file is executable
    return os.access(path, os.X_OK)


@typechecked
def peps_from_scores_kde_nnls(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    pi0: float | None = None,
    *,
    num_eval_scores: int = 500,
) -> np.ndarray[float]:
    """Estimate peps from scores using density estimates and monotonicity.

    This method computes the estimated probabilities of target being
    incorrect (peps) based on the given scores and targets. It uses the
    following steps:

        1. Compute evaluation scores, target probability density function
           (PDF), and decoy probability density function evaluated at the
           given scores.
        2. Estimate pi0 and the number of correct targets using the target
           PDF, decoy PDF, and pi0EstThresh.
        3. Calculate the number of correct targets by subtracting the decoy
           PDF multiplied by pi0Est from the target PDF, and clip it to
           ensure non-negative values.
        4. Estimate peps by dividing the number of correct targets by the
           target PDF, and clip the result between 0 and 1.
        5. Monotonize the pep estimates.
        6. Linearly interpolate the pep estimates from the evaluation
           scores to the given scores of interest.
        7. Return the estimated probabilities of target being incorrect
           (peps).

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or
        a decoy (False).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    num_eval_scores:
        The number of evaluation scores to be computed. Default is 500.
    pi0_estimation_threshold:
        The threshold for pi0 estimation. Default is 0.9.

    Returns
    -------
    array:
        The estimated probabilities of target being incorrect (peps) for the
        given scores.
    """

    # Compute evaluation scores, and target and decoy pdfs
    # (evaluated at given scores)
    eval_scores, target_pdf, decoy_pdf = pdfs_from_scores(
        scores, targets, num_eval_scores
    )

    if pi0 is None:
        # Estimate pi0 and estimate number of correct targets
        pi0 = pi0_from_pdfs_by_slope(target_pdf, decoy_pdf)

    correct = target_pdf - decoy_pdf * pi0
    correct = np.clip(correct, 0, None)

    # Estimate peps from #correct targets, clip it
    pepEst = np.clip(1.0 - correct / target_pdf, 0.0, 1.0)

    # Now monotonize using the NNLS algo putting more weight on areas with high
    # target density
    pepEst = monotonize_nnls(pepEst, w=target_pdf, ascending=False)

    # Linearly interpolate the pep estimates from the eval points to the scores
    # of interest.
    peps = np.interp(scores, eval_scores, pepEst)
    peps = np.clip(peps, 0.0, 1.0)
    return peps


@typechecked
def peps_from_scores_hist_nnls(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    pi_factor: float = 1.0,
    scale_to_one: bool = False,
    weight_exponent: float = -1.0,
    bin_edges: str | int | np.ndarray[float] | None = None,
):
    """Calculate the PEP (Posterior Error Probability) estimates from scores
    and targets using the NNLS (Non-negative Least Squares) method.

    The algorithm follows the steps outlined below:
        1. Define joint bins for targets and decoys.
        2. Estimate the number of trials and successes inside each bin.
        3. Perform a monotone fit by minimizing the objective function
           || n - diag(p) * k || with weights n over monotone descending p.
        4. If scaleToOne is True and the first element of the pepEst array is
           less than 1, scale the pepEst array by dividing it by the first
           element.
        5. Linearly interpolate the pep estimates from the evaluation points to
           the scores of interest.
        6. Clip the interpolated pep estimates between 0 and 1 in case they
           went slightly out of bounds.
        7. Return the interpolated and clipped PEP estimates.

    Parameters
    ----------
    scores:
        numpy array containing the scores of interest.
    targets:
        numpy array containing the target values corresponding to each score.
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    scale_to_one:
        Boolean value indicating whether to scale the PEP estimates to 1 for
        small score values. Default is False.

    Returns
    -------
    array:
        Array of PEP estimates at the scores of interest.
    """

    hist_data = TDHistData.from_scores_targets(scores, targets, bin_edges)
    peps_func = peps_func_from_hist_nnls(
        hist_data, pi_factor, scale_to_one, weight_exponent
    )
    return peps_func(scores)


@typechecked
def peps_func_from_hist_nnls(
    hist_data: TDHistData,
    pi_factor: float,
    scale_to_one: bool = False,
    weight_exponent: float = -1.0,
) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    """Compute a function that calculates the PEP (Posterior Error Probability)
    estimates from scores and targets using the NNLS (Non-negative Least
    Squares) method.

    For a description see `peps_from_scores_hist_nnls`.

    Parameters
    ----------
    hist_data:
        Histogram data as `TDHistData` object.
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    scale_to_one: Scale the result if the maximum PEP is smaller than 1.
         (Default value = False)
    weight_exponent:
         The weight exponent for the `fit_nnls` fit (see there, default 1).

    Returns
    -------
    function:
        A function that computes PEPs, given scores as input. Input must be an
        numpy array.
    """
    # Define joint bins for targets and decoys
    eval_scores, target_counts, decoy_counts = hist_data.as_counts()

    n, k = estimate_trials_and_successes(
        decoy_counts, target_counts, pi_factor, restrict=False
    )

    # Do monotone fit, minimizing || n - diag(p) * k || with weights n over
    # monotone descending p
    try:
        pep_est = fit_nnls(n, k, ascending=False, weight_exponent=weight_exponent)
    except RuntimeError as e:
        raise PepsConvergenceError from e

    if scale_to_one and pep_est[0] < 1:
        pep_est = pep_est / pep_est[0]

    # Linearly interpolate the pep estimates from the eval points to the scores
    # of interest (keeping monotonicity) clip in case we went slightly out of
    # bounds
    return lambda scores: np.clip(np.interp(scores, eval_scores, pep_est), 0, 1)


@typechecked
def estimate_trials_and_successes(
    decoy_counts: np.ndarray[int],
    target_counts: np.ndarray[int],
    pi_factor: float,
    restrict: bool = True,
):
    """Estimate trials/successes (assuming a binomial model) from decoy and
    target counts.

    Parameters
    ----------
    decoy_counts:
        A numpy array containing the counts of decoy occurrences (histogram).
    target_counts:
        A numpy array containing the counts of target occurrences (histogram).
    is_tdc:
        Scores and targets come from competition, rather than separate search.
    restrict:
        A boolean indicating whether to restrict the estimated trials and
        successes per bin. If True, the estimated values will be bounded by a
        minimum of 0 and a maximum of the corresponding target count. If False,
        the estimated values will be unrestricted.

    Returns
    -------
    tuple:
        A tuple (n, k) where n is a numpy array representing the estimated
        trials per bin, and k is a numpy array representing the estimated
        successes per bin.
    """

    # Estimate trials and successes per bin
    if restrict:
        n = np.maximum(target_counts, 1)
        k = np.ceil(pi_factor * decoy_counts).astype(int)
        k = np.clip(k, 0, n)
    else:
        n = target_counts
        k = pi_factor * decoy_counts
    return n, k
