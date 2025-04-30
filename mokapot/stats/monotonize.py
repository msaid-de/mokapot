from __future__ import annotations

import logging
from typing import List, TypeVar

import numpy as np
from scipy.optimize._nnls import _nnls
from typeguard import typechecked

NumericArray = TypeVar(
    "NumericArray", np.ndarray[np.floating], np.ndarray[np.integer], List[float | int]
)

LOGGER = logging.getLogger(__name__)


def nnls(A, b, max_iter=None):
    try:
        x, rnorm, mode = _nnls(A, b, max_iter, atol=1e-5)
    except TypeError:
        x, rnorm, mode = _nnls(A, b, max_iter)
    if mode == 1:
        LOGGER.debug("\t - Warning: nnls went into loop. Taking last solution.")
    return x, rnorm


@typechecked
def monotonize_simple(
    x: NumericArray, ascending: bool, reverse: bool = False
) -> NumericArray:
    """Monotonizes the input array `x` in either ascending or descending order
    beginning from the start of the array.

    Parameters
    ----------
    x:
        Input array to be monotonized.
    ascending:
        Specifies whether to monotonize in ascending order (`True`) or
        descending order (`False`). Direction is always w.r.t. to the start of
        the array, independently of `reverse`.
    reverse:
         Specify whether the process should run from the start (`False`) or the
         end (`True`) of the array. Defaults to `False`.

    Returns
    -------
    array:
        Monotonized array `x`
    """
    if reverse:
        return monotonize_simple(x[::-1], not ascending, False)[::-1]

    if ascending:
        return np.maximum.accumulate(x)
    else:
        return np.minimum.accumulate(x)


@typechecked
def monotonize(
    x: np.ndarray[float], ascending: bool, simple_averaging: bool = False
) -> np.ndarray[float]:
    """Monotonizes the input array `x` in either ascending or descending order
    averaging over both directions.

    Note: it makes a difference whether you start with monotonization from the
    start or the end of the array.

    Parameters
    ----------
    x:
        Input array to be monotonized.
    ascending:
        Specifies whether to monotonize in ascending order (`True`) or
        descending order (`False`).
    simple_averaging:
        Specifies whether to use a simple average (`True`) or weighted average
        (`False`) when computing the average of the monotonized arrays. Only
        used if `average` is `True`. Default is `False`. Note: the weighted
        average tries to minimize the L2 difference in the change between the
        original and the returned arrays.

    Returns
    -------
    array:
        Monotonized array `x` based on the specified parameters.
    """
    x1 = monotonize_simple(x, ascending)
    if np.all(x1 == x):
        return x  # nothing to do here
    x2 = monotonize_simple(x[::-1], not ascending)[::-1]
    alpha = (
        0.5
        if simple_averaging
        else np.sum((x - x2) * (x1 - x2)) / np.sum((x1 - x2) * (x1 - x2))
    )
    return alpha * x1 + (1 - alpha) * x2


@typechecked
def monotonize_nnls(
    x: np.ndarray[np.floating],
    w: np.ndarray[np.floating | np.integer] | None = None,
    ascending: bool = True,
) -> np.ndarray[float]:
    """Monotonizes a given array `x` using non-negative least squares (NNLS)
    optimization.

    The returned array is the monotone array `y` that minimized `x-y` in the
    L2-norm. The of all monotone arrays `y` is such that `x-y` has minimum
    mean squared error (MSE).

    Parameters
    ----------
    x:
        numpy array to be monotonized.
    w:
        numpy array containing weights. If None, equal weights are assumed.
    ascending:
        Boolean indicating whether the monotonized array should be in ascending
        order.

    Returns
    -------
    array:
        The monotonized array.
    """
    if not ascending:
        # If descending, just return the reversed output of the algo with
        # reversed inputs.
        return monotonize_nnls(x[::-1], None if w is None else w[::-1])[::-1]

    # Basically the algorithm works by solving
    #   x1 = d1
    #   x2 = d1 + d2
    #   x3 = d1 + d2 + d3
    # and so on for all non-negative di - or rather minimizing the sum of the
    # squared differences - and afterwards # taking xm1 = d1, xm2 = d1 + d2,
    # and so on as the monotonized values.
    # The first is the same as minimizing
    # ||x - A d|| for the matrix A = [1 0 0 0...; 1 1 0 0...; 1 1 1 0...; ...].
    # The second is just computing the cumsum of the d vector.
    N = len(x)
    A = np.tril(np.ones((N, N)))
    if w is not None:
        # We do the weighting by multiplying both sides (i.e. A and x) by
        # a diagonal matrix consisting of the square roots of the weights
        D = np.diag(np.sqrt(w))
        A = D.dot(A)
        x = D.dot(x)
    d, _ = nnls(A, x)

    xm = np.cumsum(d)
    return xm


def fit_nnls(n, k, ascending=True, *, weight_exponent=-1.0, erase_zeros=False):
    """Do monotone nnls fit on binomial model.

    This method performs a non-negative least squares (NNLS) fit on given
    input parameters 'n' and 'k', such that `k[i]` is close to `p[i] * n[i]` in
    a weighted least squared sense (weight is determined by
    `n[i] ** weightExponent`) and the `p[i]` are monotone.

    Note: neither `n` nor `k` need to integer valued or positive nor does `k`
    need to be between `0` and `n`.

    Parameters
    ----------
    n:
        The input array of length N
    k:
        The input array of length N
    ascending:
        Optional bool (Default value = True). Whether the result should be
        monotone ascending or descending.
    weight_exponent:
        Optional (Default value = -1.0). The weight exponent to use.
    erase_zeros:
        Optional (Default value = False). Whether 0s in `n` should be erased
        prior to fitting or not.

    Returns
    -------
    array:
        The monotonically increasing or decreasing array `p` of length N.

    """
    # For the basic idea of this algorithm (i.e. monotonize under constraints),
    # see the `monotonize_nnls` algorithm.  This is more or less the same, just
    # that the functional to be minimized is different here.
    if not ascending:
        n = n[::-1]
        k = k[::-1]

    N = len(n)
    D = np.diag(n)
    A = D @ np.tril(np.ones((N, N)))
    w = np.zeros_like(n, dtype=float)
    w[n != 0] = n[n != 0] ** (0.5 * weight_exponent)
    W = np.diag(w)
    R = np.eye(N)

    zeros = (n == 0).nonzero()[0]
    if len(zeros) > 0:
        A[zeros, zeros] = 1
        A[zeros, np.minimum(zeros + 1, N - 1)] = -1
        w[zeros] = 1
        k[zeros] = 0
        W = np.diag(w)

        if erase_zeros:
            # The following lines remove variables that will end up being the
            # same (matrix R) as well as equations that are essentially zero on
            # both sides U). However, since this is a bit tricky, and difficult
            # to maintain and does not seem to lower the condition of the
            # matrix substantially, it is only activated on demand and left
            # here more for further reference, in case it will be needed in the
            # future.
            nnz = n != 0
            nnz[-1] = True
            nnz2 = np.insert(nnz, 0, True)[:-1]

            def make_perm_mat(rows, cols):
                M = np.zeros((np.max(rows) + 1, np.max(cols) + 1))
                M[rows, cols] = 1
                return M

            R = make_perm_mat(np.arange(N), nnz2.cumsum() - 1)
            U = make_perm_mat(np.arange(sum(nnz)), nnz.nonzero()[0])
            W = U @ W

    d, _ = nnls(W @ A @ R, W @ k)
    p = np.cumsum(R @ d)

    if not ascending:
        return p[::-1]
    else:
        return p
