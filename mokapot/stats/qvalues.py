"""
This module estimates q-values.
"""

from typing import Callable

import numpy as np
from typeguard import typechecked

import mokapot.stats.pi0est
import mokapot.stats.pvalues as pvalues
from mokapot.stats.histdata import TDHistData
from mokapot.stats.monotonize import monotonize_simple
from mokapot.stats.pi0est import pi0_from_pvalues_storey


@typechecked
def qvalues_from_counts_tdc(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    desc: bool = True,
):
    """Estimate q-values using target decoy competition.

    Estimates q-values using the simple target decoy competition method.
    For set of target and decoy PSMs meeting a specified score threshold,
    the false discovery rate (FDR) is estimated as:

    ...math:
        FDR = \frac{Decoys + 1}{Targets}

    More formally, let the scores of target and decoy PSMs be indicated as
    :math:`f_1, f_2, ..., f_{m_f}` and :math:`d_1, d_2, ..., d_{m_d}`,
    respectively. For a score threshold :math:`t`, the false discovery
    rate is estimated as:

    ...math:
        E\\{FDR(t)\\} = \frac{|\\{d_i > t; i=1, ..., m_d\\}| + 1}
        {\\{|f_i > t; i=1, ..., m_f|\\}}

    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.

    Parameters
    ----------
    scores : numpy.ndarray of float
        A 1D array containing the score to rank by
    targets : numpy.ndarray of bool
        A 1D array indicating if the entry is from a target or decoy
        hit. This should be boolean, where `True` indicates a target
        and `False` indicates a decoy. `target[i]` is the label for
        `metric[i]`; thus `target` and `metric` should be of
        equal length.
    desc : bool
        Are higher scores better? `True` indicates that they are,
        `False` indicates that they are not.

    Returns
    -------
    numpy.ndarray
        A 1D array with the estimated q-value for each entry. The
        array is the same length as the `scores` and `target` arrays.
    """
    # todo: I think the allowed data types are way to general and lenient. scores
    # should me maximally integer|floating (but better just float) and targets
    # should only be bool, nothing else. The rest is the job of the calling code.

    # todo: should be removed and only qvalues_from_counts be used
    return qvalues_from_counts(scores, targets, pi_factor=1.0, desc=desc)


@typechecked
def qvalues_from_peps(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    peps: np.ndarray[float],
):
    r"""Compute q-values from peps.

    Computation is done according to Käll et al. (Section 3.2, first formula)
    Non-parametric estimation of posterior error probabilities associated with
    peptides identified by tandem mass spectrometry Bioinformatics, Volume 24,
    Issue 16, August 2008, Pages i42-i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{PEP}(x^t) = \min_{x'\ge {x^t}}
        \frac{\sum_{x\in\{y|y\ge x',y\in T\}}P(H_0|X=x)}
        {|\{y|y\ge x',y\in T\}|}

    Note: the formula in the paper has an :math:`x^t` in the denominator, which
    does not make a whole lot of sense. Shall probably be :math:`x'` instead.

    Parameters
    ----------
    scores:
        Array-like object representing the scores.
    targets:
        Boolean array-like object indicating the targets.
    peps:
        Array-like object representing the posterior error probabilities
        associated with the peptides. Default is None (then it's computed via
        the HistNNLS algorithm).

    Returns
    -------
    array:
        Array of q-values computed from peps.
    """

    # We need to sort scores in descending order for the formula to work
    # (it involves to cumsum's from the maximum scores downwards)
    ind = np.argsort(-scores)
    scores_sorted = scores[ind]
    targets_sorted = targets[ind]
    peps_sorted = peps[ind]

    target_scores = scores_sorted[targets_sorted]
    target_peps = peps_sorted[targets_sorted]
    target_fdr = target_peps.cumsum() / np.arange(1, len(target_peps) + 1, dtype=float)
    target_qvalues = monotonize_simple(target_fdr, ascending=True, reverse=True)

    # Note: we need to flip the arrays again, since interp needs scores in
    #   ascending order
    qvalues = np.interp(scores, np.flip(target_scores), np.flip(target_qvalues))
    return qvalues


@typechecked
def qvalues_from_counts(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    *,
    desc: bool = True,
    pi_factor: float = 1,
):
    r"""
    Compute qvalues from target/decoy counts.

    Computed according to Käll et al. (Section 3.2, second formula)
    Non-parametric estimation of posterior error probabilities associated with
    peptides identified by tandem mass spectrometry Bioinformatics, Volume 24,
    Issue 16, August 2008, Pages i42–i48
    https://doi.org/10.1093/bioinformatics/btn294

    The formula used is:

    .. math:: q_{TD}(x^t) = \pi_0 \frac{\#T}{\#D} \min_{x\ge {x^t}}
        {|\{y|y\ge x, y\in D\}|}/
        {|\{y|y\ge x, y\in T\}|}

    Note: the factor :math:`\frac{\#T}{\#D}` is not in the original equation,
    but should be there to account of lists of targets and decoys of different
    lengths.

    Note 2: for tdc the estimator #D/#T is used for $\pi_0$, effectively
    cancelling out the factor.

    Parameters
    ----------
    scores :
        Array-like object representing the scores.
    targets :
        Boolean array-like object indicating the targets.
    is_tdc:
        Scores and targets come from competition, rather than separate search.

    Returns
    -------
    array:
        Array of q-values computed from peps.
    """
    if not desc:
        scores = -scores

    # Sort by score, but take also care of multiple targets/decoys per score
    ind = np.lexsort((targets, -scores))
    targets_sorted = targets[ind]
    scores_sorted = scores[ind]
    fdr_sorted = (
        pi_factor
        * ((~targets_sorted).cumsum() + 1)
        / np.maximum(targets_sorted.cumsum(), 1)
    )
    qvalues_sorted = monotonize_simple(fdr_sorted, ascending=True, reverse=True)

    # Extract unique scores and take qvalue from the last of them
    # (for each unique score we need the unique qvalue, the extra return values
    # are needed to get the correct one and to map them back later to the full
    # array. See np.unique and do the math ;)
    # Note: if all scores are uniq this step could be skipped and maybe some
    # cpu cycles saved.
    scores_uniq, idx_uniq, rev_uniq, cnt_uniq = np.unique(
        scores_sorted,
        return_index=True,
        return_counts=True,
        return_inverse=True,
    )
    qvalues_uniq = qvalues_sorted[idx_uniq + cnt_uniq - 1]

    # Map unique values back, but in original order
    qvalues = np.zeros_like(qvalues_sorted)
    qvalues[ind] = qvalues_uniq[rev_uniq]

    return np.clip(qvalues, 0.0, 1.0)


@typechecked
def qvalues_func_from_hist(
    hist_data: TDHistData, pi_factor: float
) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    r"""Compute q-values from histogram counts.

    Compute qvalues from target/decoy counts according to Käll et al. (Section
    3.2, second formula), but from the histogram counts.

    Note that the formula is exact for the left edges of each histogram bin.
    For the interiors of the bins the q-values are linearly interpolated.

    Parameters
    ----------
    scores :
        Array-like object representing the scores.
    targets :
        Boolean array-like object indicating the targets.
    hist_data:
        Histogram data in form of a `TDHistData` object.
    is_tdc:
        Scores and targets come from competition, rather than separate search.

    Returns
    -------
    function:
        Function the computes an array of q-values from an array of scores.
    """

    _, target_counts, decoy_counts = hist_data.as_counts()

    targets_sum = np.flip(target_counts).cumsum()
    decoys_sum = np.flip(decoy_counts).cumsum()

    # We need to append the last value once for the last bin_edge
    targets_sum = np.append(targets_sum, targets_sum[-1])
    decoys_sum = np.append(decoys_sum, decoys_sum[-1])

    fdr_flipped = pi_factor * (decoys_sum + 1) / np.maximum(targets_sum, 1)
    fdr_flipped = np.clip(fdr_flipped, 0.0, 1.0)
    qvalues_flipped = monotonize_simple(fdr_flipped, ascending=True, reverse=True)
    qvalues = np.flip(qvalues_flipped)

    eval_scores = hist_data.targets.bin_edges
    return lambda scores: np.interp(scores, eval_scores, qvalues)


@typechecked
def qvalues_from_storeys_algo(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    pi0: float | None = None,
    decoy_qvals_by_interp: bool = True,
    pvalue_method: str = "conservative",
):
    """
    Calculates q-values for a set of scores using Storey's algorithm.

    The function computes empirical p-values, estimates the proportion of
    null hypotheses (`pi0`), and subsequently calculates the q-values,
    which are adjusted p-values used in multiple hypothesis testing.

    Parameters
    ----------
    scores : np.ndarray[float]
        Array of scores for which q-values are computed. Must be of
        type float.
    targets : np.ndarray[bool]
        Binary array indicating whether a score is a target (`True`) or
        a decoy (`False`). Must have the same shape as `scores`.
    pi0 : float, optional
        Proportion of null hypotheses in the dataset. If not provided,
        it is estimated from the p-values using the "smoother" method
        with evaluation at lambda = 0.5.
    decoy_qvals_by_interp : bool, optional
        Whether to calculate q-values for decoys by interpolating from
        target q-values (`True`) or calculate decoy q-values independently
        as 1:1 p-values (`False`). Defaults to `True`.
    pvalue_method : str, optional
        Method to calculate empirical p-values. Acceptable values are
        mode-based methods such as "conservative". Defaults to
        "conservative".

    Returns
    -------
    np.ndarray[float]
        An array of q-values corresponding to the input `scores`, where
        lower scores generally indicate a higher likelihood of being a
        target. The q-values are adjusted for multiple hypothesis testing.
    """

    stat1 = scores[targets]
    stat0 = scores[~targets]

    pvals1 = pvalues.empirical_pvalues(stat1, stat0, mode=pvalue_method)

    if pi0 is None:
        pi0est = mokapot.stats.pi0est.pi0_from_pvalues_storey(
            pvals1, method="smoother", eval_lambda=0.5
        )
        pi0 = pi0est.pi0

    qvals1 = qvalues_from_pvalues(pvals1, pi0=pi0, small_p_correction=False)

    if decoy_qvals_by_interp:
        ind = np.argsort(stat1)
        qvals0 = np.interp(stat0, stat1[ind], qvals1[ind])
    else:
        pvals0 = pvalues.empirical_pvalues(stat0, stat0, mode=pvalue_method)
        qvals0 = qvalues_from_pvalues(pvals0, pi0=1)

    qvals = np.zeros_like(scores, dtype=float)
    qvals[targets] = qvals1
    qvals[~targets] = qvals0
    return qvals


@typechecked
def qvalues_from_pvalues(
    pvals: np.ndarray[float],
    *,
    pi0: float | None = None,
    small_p_correction: bool = False,
) -> np.ndarray[float]:
    """
    Calculates q-values, which are an estimate of false discovery rates (FDR),
    for an array of p-values. This function is based on Storey's implementation
    in the R package "qvalue".

    Parameters
    ----------
    pvals : np.ndarray[float]
        An array of p-values to compute q-values for. The array should contain
        values between 0 and 1.
    pi0 : float, optional
        Proportion of true null hypotheses. If not provided, it is estimated
        using the bootstrap method (see estimate_pi0).
    small_p_correction : bool, optional
        Whether to apply a small p-value correction (Storey's pfdr parameter),
        which adjusts for very small p-values in the dataset.

    Returns
    -------
    np.ndarray[float]
        An array of q-values corresponding to the input p-values. The q-values
        are within the range [0, 1].

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] John D. Storey, A direct approach to false discovery rates,
        J. R. Statist. Soc. B (2002), 64, Part 3, pp. 479–498
    .. [3] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """
    N = len(pvals)
    order = np.argsort(pvals)
    pvals_sorted = pvals[order]
    if pi0 is None:
        pi0 = pi0_from_pvalues_storey(pvals, method="bootstrap")

    fdr_sorted = pi0 * pvals_sorted * N / np.linspace(1, N, N)
    if small_p_correction:
        fdr_sorted /= 1 - (1 - pvals) ** N

    # Note that the monotonization takes also correctly care of repeated pvalues
    # so that they always get the same qvalue
    qvalues_sorted = monotonize_simple(fdr_sorted, ascending=True, reverse=True)

    qvalues = np.zeros_like(qvalues_sorted)
    qvalues[order] = qvalues_sorted

    qvalues = np.clip(qvalues, 0.0, 1.0)
    return qvalues


@typechecked
def qvalues_func_from_hist_storey(
    td_hist_data: TDHistData,
    pi0: float,
    *,
    small_p_correction: bool = False,
) -> Callable[[np.ndarray[float]], np.ndarray[float]]:
    """
    Calculates q-values, which are an estimate of false discovery rates (FDR),
    for an array of p-values. This function is based on Storey's implementation
    in the R package "qvalue" and modified to work on (approximately) on
    histograms.

    Parameters
    ----------
    td_hist_data : TDHistData
        A TDHistData object.
    pi0 : float
        Proportion of true null hypotheses.
    small_p_correction : bool, optional
        Whether to apply a small p-value correction (Storey's pfdr parameter),
        which adjusts for very small p-values in the dataset.

    Returns
    -------
    function
        A function that returns q-values given scores as input.

    References
    ----------
    .. [1] John D. Storey, Robert Tibshirani, Statistical significance for
        genomewide studies,pp. 9440 –9445 PNAS August 5, 2003, vol. 100, no. 16
        www.pnas.org/cgi/doi/10.1073/pnas.1530509100
    .. [2] John D. Storey, A direct approach to false discovery rates,
        J. R. Statist. Soc. B (2002), 64, Part 3, pp. 479–498
    .. [3] Storey JD, Bass AJ, Dabney A, Robinson D (2024). qvalue: Q-value
        estimation for false discovery rate control. R package version 2.38.0,
        http://github.com/jdstorey/qvalue.
    """

    _, target_counts, decoy_counts = td_hist_data.as_counts()

    targets_sum = np.flip(target_counts).cumsum()
    decoys_sum = np.flip(decoy_counts).cumsum()
    N = targets_sum[-1]

    # We need to append the last value once for the last bin_edge
    targets_sum = np.append(targets_sum, targets_sum[-1])
    decoys_sum = np.append(decoys_sum, decoys_sum[-1])

    pvals = decoys_sum / decoys_sum[-1]
    fdr = pi0 * pvals * N / np.maximum(targets_sum, 1)
    if small_p_correction:
        fdr /= 1 - (1 - pvals) ** N

    # Note that the monotonization takes also correctly care of repeated pvalues
    # so that they always get the same qvalue
    qvalues = monotonize_simple(fdr, ascending=True, reverse=True)
    qvalues = np.clip(qvalues, 0.0, 1.0)
    qvalues = np.flip(qvalues)

    eval_scores = td_hist_data.targets.bin_edges
    return lambda scores: np.interp(scores, eval_scores, qvalues)
