import numpy as np
import pytest
import scipy as sp

from mokapot.stats.pvalues import empirical_pvalues
from unit_tests.stats.helpers import create_tdmodel


def test_empirical_pvalues():
    # Basic test for computing empirical pvalues, compare results to
    # results from Storey's R implementation (qvalue::empPvals)
    # NOTE: Storey's implementation is wrong IMHO when a target and a decoy
    # happen to have the same score, because he's using Pr(S>s|H0) instead of
    # Pr(S>=s|H0) for the p-values (which can make spuriously low p-values in
    # the case of discrete distributions)
    s = np.arange(0, 13, dtype=float)
    s0 = np.arange(-99, 101, dtype=float)
    p = 0.505 - s / len(s0)

    def empPvals(s, s0):
        return empirical_pvalues(s, s0, mode="storey")

    np.testing.assert_almost_equal(empPvals(s, s0), p)

    # Now shift and compare
    delta = 1e-12
    np.testing.assert_almost_equal(empPvals(s + delta, s0), p - 0.005)
    np.testing.assert_almost_equal(empPvals(s + 1 - delta, s0), p - 0.005)
    np.testing.assert_almost_equal(empPvals(s + 1, s0), p - 0.005)
    np.testing.assert_almost_equal(empPvals(s + 1 + delta, s0), p - 0.01)

    np.testing.assert_almost_equal(empPvals(s - delta, s0), p)
    np.testing.assert_almost_equal(empPvals(s - 1, s0), p + 0.005)

    # Test the different p-value computation modes
    s = np.arange(-200, 200, 23)
    s0 = np.arange(-99, 101)
    N = len(s0)
    p = np.clip(0.505 - s / N, 0, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="unbiased"), p)
    p = np.clip(0.505 - s / N, 1.0 / N, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="storey"), p)
    p = np.clip((102 - s) / (1 + N), 1.0 / (N + 1), 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="conservative"), p)


def test_empirical_pvalues_discrete():
    s = np.arange(0, 5)
    s0 = np.repeat(s, 100)
    np.random.shuffle(s0)
    N = len(s0)

    p = empirical_pvalues(s, s0, mode="unbiased")
    p_unb = 1.0 - (s / 5.0)
    np.testing.assert_almost_equal(p, p_unb)

    p = empirical_pvalues(s, s0, mode="storey")
    np.testing.assert_almost_equal(p, p_unb)

    p = empirical_pvalues(s, s0, mode="conservative")
    p_cons = p_unb + (1 - p_unb) / (N + 1)  # add bias term
    np.testing.assert_almost_equal(p, p_cons)


def test_empirical_pvalues_repetitions():
    # Test that pvalues are correct for repeated values (i.e. repeated stats
    # get the same pvalue)
    np.random.seed(42)
    N = 1000
    s = np.round(np.random.rand(N) * 60)
    s0 = np.round(np.random.rand(N) * 20 + 20)
    assert np.sum(s == 25) > 1 and np.sum(s0 == 25) > 1  # make sure test is meaningfull

    p = empirical_pvalues(s, s0, mode="conservative")
    assert np.all(np.diff(p[s == 25]) == 0)
    p = empirical_pvalues(s, s0, mode="unbiased")
    assert np.all(np.diff(p[s == 25]) == 0)
    p = empirical_pvalues(s, s0, mode="storey")
    assert np.all(np.diff(p[s == 25]) == 0)


@pytest.mark.parametrize("rho0", [0.01, 0.3, 0.8, 0.95])
@pytest.mark.parametrize(
    ("discrete", "is_tdc"), [(False, False), (True, False), (False, True)]
)
def test_empirical_pvalues_on_tdmodel(discrete, is_tdc, rho0):
    N = 1000000
    model = create_tdmodel(is_tdc, rho0, discrete)
    scores, targets, is_fd = model.sample_scores(N)
    stat = scores[targets]
    stat0 = scores[~targets]
    pvalues = empirical_pvalues(stat, stat0)

    if is_tdc:
        # Compute decoy cdf in output via numerical integration of the
        # output decoy pdf. Todo: move part of this to tdmodel test and tdmodel
        # Note: no formula for p-values of discrete tdc output decoy distribution
        # yet, but this is enough for testing the p-value estimation.
        xi = np.linspace(min(scores) - 1, max(scores) + 1, 100000)
        T_pdf, TT_pdf, FT_pdf, D_pdf, FDR = model.get_sampling_pdfs(xi)
        FT_cdf = sp.integrate.cumulative_trapezoid(FT_pdf, xi, initial=0.0)

        def decoy_cdf(x):
            return np.interp(x, xi, FT_cdf)

        # assert allclose(FT_cdf[0], 0)
        # assert allclose(FT_cdf[-1], 1, 1e-3)
    else:
        decoy_cdf = model.decoy_cdf

    # negative offset needed as a hack for discrete distributions
    pvalues_expect = 1 - decoy_cdf(stat - 1e-10)
    np.testing.assert_allclose(pvalues, pvalues_expect, atol=1e-2)
