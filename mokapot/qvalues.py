"""
This module estimates q-values.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import numba as nb


def tdc(metric: np.ndarray, target: np.ndarray, desc: bool = True) \
        -> np.ndarray:
    """
    Estimate q-values using target decoy competition.

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
        E\{FDR(t)\} = \frac{|\{d_i > t; i=1, ..., m_d\}| + 1}
        {\{|f_i > t; i=1, ..., m_f|\}}

    The reported q-value for each PSM is the minimum FDR at which that
    PSM would be accepted.

    Parameters
    ----------
    metric : numpy.ndarray
        A 1D array containing the score to rank by (`float`)

    target : numpy.ndarray
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
        array is the same length as the `metric` and `target` arrays.
    """
    # Check arguments
    if isinstance(metric, pd.Series):
        metric = metric.values

    if isinstance(target, pd.Series):
        target = target.values

    msg = "'{}' must be a 1D numpy.ndarray or pandas.Series"
    if not isinstance(metric, np.ndarray) or len(metric.shape) != 1:
        raise ValueError(msg.format("metric"))

    if not isinstance(target, np.ndarray) or len(target.shape) != 1:
        raise ValueError(msg.format("target"))

    if not isinstance(desc, bool):
        raise ValueError("'desc' must be boolean (True or False)")

    if metric.shape[0] != target.shape[0]:
        raise ValueError("'metric' and 'target' must be the same length")

    try:
        target = np.array(target, dtype=bool)
    except ValueError:
        raise ValueError("'target' should be boolean.")

    # Sort and estimate FDR
    if desc:
        srt_idx = np.argsort(-metric)
    else:
        srt_idx = np.argsort(metric)

    metric = metric[srt_idx]
    target = target[srt_idx]
    cum_targets = target.cumsum()
    cum_decoys = ((target-1)**2).cumsum()
    num_total = cum_targets + cum_decoys

    # Handles zeros in denominator
    fdr = np.divide((cum_decoys + 1), cum_targets,
                    out=np.ones_like(cum_targets, dtype=float),
                    where=(cum_targets != 0))

    # Calculate q-values
    unique_metric, indices = np.unique(metric, return_counts=True)

    # Some arrays need to be flipped so that we can loop through from
    # worse to best score.
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(fdr, num_total, unique_metric, indices)
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


def crosslink_tdc(metric: np.ndarray, num_targets: np.ndarray,
                  desc: bool = True) -> np.ndarray:
    """
    Estimate q-values using the Walzthoeni et al method.

    This q-value method is specifically for cross-linked peptides.

    Parameters
    ----------
    num_targets : numpy.ndarray
        The number of target sequences in the cross-linked pair.

    metric : numpy.ndarray
        The metric to used to rank elements.

    desc : bool
        Is a higher metric better?

    Returns
    -------
    numpy.ndarray
        A 1D array of q-values in the same order as `num_targets` and
        `metric`.
    """
    if isinstance(metric, pd.Series):
        metric = metric.values

    if isinstance(num_targets, pd.Series):
        num_targets = num_targets.values

    msg = "'{}' must be a 1D numpy.ndarray or pandas.Series"
    if not isinstance(metric, np.ndarray) or len(metric.shape) != 1:
        raise ValueError(msg.format("metric"))

    if not isinstance(num_targets, np.ndarray) or len(num_targets.shape) != 1:
        raise ValueError(msg.format("num_targets"))

    if not isinstance(desc, bool):
        raise ValueError("'desc' must be boolean (True or False)")

    if metric.shape[0] != num_targets.shape[0]:
        raise ValueError("'metric' and 'num_targets' must be the same length.")

    if desc:
        srt_idx = np.argsort(-metric)
    else:
        srt_idx = np.argsort(metric)

    metric = metric[srt_idx]
    num_targets = num_targets[srt_idx]
    num_total = np.ones(len(num_targets)).cumsum()
    cum_targets = (num_targets == 2).astype(int).cumsum()
    one_decoy = (num_targets == 1).astype(int).cumsum()
    two_decoy = (num_targets == 0).astype(int).cumsum()

    fdr = np.divide((one_decoy - two_decoy), cum_targets,
                    out=np.ones_like(cum_targets, dtype=float),
                    where=(cum_targets != 0))

    fdr[fdr < 0] = 0

    # Calculate q-values
    unique_metric, indices = np.unique(metric, return_counts=True)

    # Flip arrays to loop from worst to best score
    fdr = np.flip(fdr)
    num_total = np.flip(num_total)
    if not desc:
        unique_metric = np.flip(unique_metric)
        indices = np.flip(indices)

    qvals = _fdr2qvalue(fdr, num_total, unique_metric, indices)
    qvals = np.flip(qvals)
    qvals = qvals[np.argsort(srt_idx)]

    return qvals


@nb.njit
def _fdr2qvalue(fdr: np.ndarray, num_total: np.ndarray,
                met: np.ndarray, indices: Tuple[np.ndarray]) \
        -> np.ndarray:
    """
    Quickly turn a list of FDRs to q-values.

    All of the inputs are assumed to be sorted.

    Parameters
    ----------
    fdr : numpy.ndarray
        A vector of all unique FDR values.

    num_total : numpy.ndarray
        A vector of the cumulative number of PSMs at each score.

    met : numpy.ndarray
        A vector of the scores for each PSM.

    indices : tuple of numpy.ndarray
        Tuple where the vector at index i indicates the PSMs that
        shared the unique FDR value in `fdr`.

    Returns
    -------
    numpy.ndarray
        A vector of q-values.
    """
    min_q = 1
    qvals = np.ones(len(fdr))
    group_fdr = np.ones(len(fdr))
    prev_idx = 0
    for idx in range(met.shape[0]):
        next_idx = prev_idx + indices[idx]
        group = slice(prev_idx, next_idx)
        prev_idx = next_idx

        fdr_group = fdr[group]
        n_group = num_total[group]
        curr_fdr = fdr_group[np.argmax(n_group)]
        if curr_fdr < min_q:
            min_q = curr_fdr

        group_fdr[group] = curr_fdr
        qvals[group] = min_q

    return qvals