from __future__ import annotations

import logging
from datetime import time

import numpy as np
from scipy import stats as stats
from typeguard import typechecked

LOGGER = logging.getLogger(__name__)


@typechecked
def pdfs_from_scores(
    scores: np.ndarray[float],
    targets: np.ndarray[bool],
    num_eval_scores: int = 500,
) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
    """Compute target and decoy probability density functions (PDFs) from
    scores using kernel density estimation (KDE).

    Parameters
    ----------
    scores:
        A numpy array containing the scores for each target and decoy peptide.
    targets:
        A boolean array indicating whether each peptide is a target (True) or a
        decoy (False).
    num_eval_scores:
        Number of evaluation scores to compute in the PDFs. Defaults to 500.

    Returns
    -------
    tuple:
        A tuple containing the evaluation scores, the target PDF, and the decoy
        PDF at those scores.
    """
    # Compute score range and evaluation points
    min_score = min(scores)
    max_score = max(scores)
    eval_scores = np.linspace(min_score, max_score, num=num_eval_scores)

    # Compute target and decoy pdfs
    target_scores = scores[targets]
    decoy_scores = scores[~targets]
    target_pdf_estimator = stats.gaussian_kde(target_scores)
    decoy_pdf_estimator = stats.gaussian_kde(decoy_scores)
    target_pdf = target_pdf_estimator.pdf(eval_scores)
    decoy_pdf = decoy_pdf_estimator.pdf(eval_scores)
    return eval_scores, target_pdf, decoy_pdf


__tictoc_t0: float


def tic():
    global __tictoc_t0
    __tictoc_t0 = time.time()


def toc():
    global __tictoc_t0
    elapsed = time.time() - __tictoc_t0
    logging.info(f"Elapsed time: {elapsed}")


def bernoulli_zscore1(n, k, p):
    p1 = k / n
    z = (p1 - p) / np.sqrt(p * (1 - p) / n)
    return z


def bernoulli_zscore2(n1, k1, n2, k2):
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z
