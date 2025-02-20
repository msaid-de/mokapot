import numpy as np
import pytest
import scipy as sp
from numpy.ma.core import allclose

from mokapot.stats.tdmodel import STDSModel, TDCModel


def bernoulli_zscore(n1, k1, n2, k2):
    p1 = k1 / n1
    p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return z


@pytest.mark.parametrize("rho0", [0.01, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("discrete", [False, True])
@pytest.mark.parametrize("is_tdc", [False, True])
def test_tdmodel(discrete, is_tdc, rho0):
    N = 1000000
    np.random.seed(123)
    if discrete:
        R1 = sp.stats.binom(10, 0.7)
        R0 = sp.stats.binom(8, 0.3)
    else:
        R1 = sp.stats.norm(0, 1)
        R0 = sp.stats.norm(-1, 1)

    if is_tdc:
        model = TDCModel(R0, R1, rho0)
    else:
        model = STDSModel(R0, R1, rho0)

    scores, targets, is_fd = model.sample_scores(N)

    num_targets = targets.sum()
    num_decoys = (~targets).sum()
    num_false_targets = (targets & is_fd).sum()
    num_true_targets = (targets & ~is_fd).sum()

    # Test sampling
    # Some basic stuff here
    assert num_false_targets + num_true_targets == num_targets
    if is_tdc:
        assert num_targets + num_decoys == N
        assert num_false_targets >= N * rho0 / 2
    else:
        assert num_targets == N
        assert num_decoys == N
        assert num_false_targets >= N * rho0

    if discrete:
        assert (scores == np.floor(scores)).all()

    # We test whether the "decoy distribution==false target distribution" with
    # respect to the total number of decoys and false targets.
    # Assumption holds (which of course only holds sharply for TDC and
    # approximately for STDS with small rho0 and good separation)

    if is_tdc:
        z = bernoulli_zscore(N, num_decoys, N, num_false_targets)
        # Use 1.96 for confidence level 0.05, and 2.58 for alpha=0.01
        assert abs(z) < 1.96

    # Test that pi0 is correct (sampled value and theoretical prediction match)
    num_false_targets_expect = np.round(num_targets * model.approx_pi0())
    assert allclose(num_false_targets, num_false_targets_expect, rtol=0.01)

    # Compare the distributions of the decoy and false target distribution
    # (but only, if the target decoy assumption holds, which is not always the
    # case for stds)
    if is_tdc:
        td_assumption = True
    else:
        # for separate search we make a crude checked (outlined in my paper)
        # whether the target decoy assumption holds sufficiently well
        pi0 = model.approx_pi0()
        F1 = model.decoy_cdf(scores)
        s1 = scores[F1 > 0.99].min()  # more or less max of the decoy distribution
        thresh = (1 - pi0) / pi0 * model.true_target_cdf(s1)
        td_assumption = thresh < 0.1

    if td_assumption:
        # Get a crude approximation of the distribution depending on whether
        # it's discrete or continuous
        if discrete:
            x1, y1 = np.unique(scores[~targets], return_counts=True)
            x2, y2 = np.unique(scores[targets & is_fd], return_counts=True)
        else:
            bin_edges = np.histogram_bin_edges(scores, bins=100)
            x1 = x2 = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            y1 = np.histogram(scores[~targets], bins=bin_edges)[0]
            y2 = np.histogram(scores[targets & is_fd], bins=bin_edges)[0]

        if not is_tdc:
            # For stds we just compare the shape by multiplying the decoy to
            # false target ratio
            y2 = y2 * num_decoys / num_false_targets

        # Now compare for the point masses or the bins for which there
        # sufficiently many decoys or false targets (the 100 and the maximum
        # relative deviation 0.2 is just some heuristic that seems to work)
        assert (x1 == x2).all()
        d = abs(y1 - y2) / (y1 + y2)
        d = d[(y1 + y2) > 100]
        assert (d < 0.2).all()
