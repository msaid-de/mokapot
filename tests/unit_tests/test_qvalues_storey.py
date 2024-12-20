import json

import numpy as np
from pytest import approx

from mokapot.qvalues_storey import empirical_pvalues, estimate_pi0, qvalues


def test_empirical_pvalues():
    # Basic test for computing empirical pvalues, compare results to
    # results from Storey's R implementation (qvalue::empPvals)
    s = np.arange(0, 13)
    s0 = np.arange(-99, 101)
    p = 0.5 - s / len(s0)

    def empPvals(s, s0):
        return empirical_pvalues(s, s0, mode="storey")

    np.testing.assert_almost_equal(empPvals(s, s0), p)

    # Now shift and compare
    delta = 1e-12
    np.testing.assert_almost_equal(empPvals(s + 1, s0), p - 0.005)
    np.testing.assert_almost_equal(empPvals(s + 1 - delta, s0), p)
    np.testing.assert_almost_equal(empPvals(s + delta, s0), p)
    np.testing.assert_almost_equal(empPvals(s - delta, s0), p + 0.005)
    np.testing.assert_almost_equal(empPvals(s - 1, s0), p + 0.005)

    # Test the different p-value computation modes
    s = np.arange(-200, 200, 23)
    s0 = np.arange(-99, 101)
    N = len(s0)
    p = np.clip(0.5 - s / N, 0, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="unbiased"), p)
    p = np.clip(0.5 - s / N, 1.0 / N, 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="storey"), p)
    p = np.clip((101 - s) / (1 + N), 1.0 / (N + 1), 1)
    np.testing.assert_almost_equal(empirical_pvalues(s, s0, mode="conservative"), p)


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


def test_estimate_pi0_from_R():
    # with open("data/samples.json", "r") as file:
    with open("data/hedenfalk.json", "r") as file:
        data = json.load(file)
    targets = np.array(data["stat"])
    decoys = np.array(data["stat0"])

    # Preliminary check that pvalues match
    pvalues = empirical_pvalues(targets, decoys, mode="storey")
    pvals_exp = np.array(data["pvalues"])
    np.testing.assert_almost_equal(pvalues, pvals_exp)

    # Compute pi0 with bootstrap method
    lambdas = np.arange(0.05, 1, 0.05)
    pi0est = estimate_pi0(pvalues, method="bootstrap", lambdas=lambdas)
    assert pi0est.pi0 == approx(0.6763407)
    assert pi0est.mse is not None
    assert pi0est.pi0s_raw[np.argmin(pi0est.mse)] == pi0est.pi0

    # Compute pi0 by smoothing (the comparison value is from the qvalue package
    # in R, since the smoothing method is different the difference is relatively
    # large)
    lambdas = np.arange(0.2, 0.95, 0.01)
    pi0est = estimate_pi0(pvalues, lambdas=lambdas, eval_lambda=0.8)
    assert pi0est.pi0 == approx(0.6931328, abs=0.01)

    # Compute pi0 with fixed lambda
    pi0est = estimate_pi0(pvalues, method="fixed", eval_lambda=0.7)
    assert pi0est.pi0 == approx(0.701367)
    pi0est = estimate_pi0(pvalues, method="fixed", eval_lambda=0.3)
    assert pi0est.pi0 == approx(0.7138351)


def test_qvalues_storey():
    with open("data/hedenfalk.json", "r") as file:
        data = json.load(file)
    pvals = np.array(data["pvalues"])
    qvals_expect = np.array(data["qvalues"])

    pi0est = estimate_pi0(pvals, method="bootstrap", lambdas=np.arange(0.05, 1, 0.05))
    qvals = qvalues(pvals, pi0=pi0est.pi0)
    np.testing.assert_almost_equal(qvals, qvals_expect)