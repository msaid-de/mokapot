import numpy as np

from mokapot.stats.pvalues import empirical_pvalues


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
