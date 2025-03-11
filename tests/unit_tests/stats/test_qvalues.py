"""
These tests verify that our q-value calculations are correct.
"""

import json

import numpy as np
import pytest
from scipy import stats

from mokapot.stats.histdata import TDHistData
from mokapot.stats.peps import peps_from_scores_hist_nnls
from mokapot.stats.pi0est import pi0_from_pvalues_storey
from mokapot.stats.qvalues import (
    qvalues_from_counts,
    qvalues_from_counts_tdc,
    qvalues_from_peps,
    qvalues_from_pvalues,
    qvalues_from_storeys_algo,
    qvalues_func_from_hist,
)
from mokapot.stats.tdmodel import STDSModel
from tests.helpers.utils import TestOutcome


@pytest.fixture
def desc_scores():
    """Create a series of descending scores and their q-values"""
    scores = np.array([10, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1], dtype=float)
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    qvals = np.array([
        1 / 4,
        1 / 4,
        1 / 4,
        1 / 4,
        2 / 6,
        2 / 6,
        2 / 6,
        3 / 7,
        3 / 7,
        4 / 7,
        5 / 8,
        5 / 8,
        1,
        1,
        1,
        1,
    ])
    return scores, target, qvals


@pytest.fixture
def asc_scores():
    """Create a series of ascending scores and their q-values"""
    scores = np.array([1, 1, 2, 3, 4, 4, 5, 6, 7, 8, 9, 9, 10, 10, 10, 10], dtype=float)
    target = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    qvals = np.array([
        1 / 4,
        1 / 4,
        1 / 4,
        1 / 4,
        2 / 6,
        2 / 6,
        2 / 6,
        3 / 7,
        3 / 7,
        4 / 7,
        5 / 8,
        5 / 8,
        1,
        1,
        1,
        1,
    ])
    return scores, target, qvals


def qvalues_are_valid(qvalues, scores=None) -> TestOutcome:
    """Helper function for tests on qvalues"""
    if not np.all(qvalues >= 0):
        return TestOutcome.fail("'qvalues must be >= 0'")
    if not np.all(qvalues <= 1):
        return TestOutcome.fail("'qvalues must be <= 1'")

    if scores is not None:
        ind = np.argsort(scores)
        diff_qvals = np.diff(qvalues[ind])
        diff_scores = np.diff(scores[ind])
        if not np.all(diff_qvals * diff_scores <= 0):
            return TestOutcome.fail(
                "'qvalues are monotonically decreasing with higher scores'"
            )

        if not np.all((diff_scores != 0) | (diff_qvals == 0)):
            # Note that "!A | B" is the same as the implication "A => B"
            # When two scores are equal they must have the same qvalue, but if
            # they are different the may have the same qvalue
            return TestOutcome.fail("'equal scores must have equal qvalues'")

    return TestOutcome.success()


def test_tdc_descending(desc_scores):
    """Test that q-values are correct for descending scores"""
    scores, target, true_qvals = desc_scores
    qvals = qvalues_from_counts_tdc(scores, target, desc=True)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_tdc_ascending(asc_scores):
    """Test that q-values are correct for ascending scores"""
    scores, target, true_qvals = asc_scores

    qvals = qvalues_from_counts_tdc(scores, target, desc=False)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_tdc_diff_len():
    """If the arrays are different lengths, should get a ValueError"""
    scores = np.array([1, 2, 3, 4, 5], dtype=float)
    targets = np.array([True] * 3 + [False] * 3)
    with pytest.raises(ValueError):
        qvalues_from_counts_tdc(scores, targets)


@pytest.fixture
def rand_scores_stds():
    np.random.seed(1240)  # this produced an error with failing iterations
    N = 5000
    rho0 = 0.7
    R0 = stats.norm(loc=-4, scale=2)
    R1 = stats.norm(loc=3, scale=2)
    model = STDSModel(R0, R1, rho0)
    scores, targets, is_fd = model.sample_scores(N)
    return scores, targets, is_fd


def all_sorted(arrays, desc=True):
    sortIdx = np.argsort(-arrays[0] if desc else arrays[0])
    return (arr[sortIdx] for arr in arrays)


def rounded(arrays, decimals=1):
    scores, *args = arrays
    scores = np.round(scores, decimals=decimals)
    return scores, *args


@pytest.fixture
def rand_scores_rounded(rand_scores_stds):
    all_scores, is_target, is_fd = rand_scores_stds
    sortIdx = np.argsort(-all_scores)
    return all_scores[sortIdx], is_target[sortIdx], is_fd[sortIdx]


@pytest.mark.parametrize("is_tdc", [True, False])
def test_qvalues_from_peps(rand_scores_stds, is_tdc):
    # Note: we should also test against some known truth
    #   (of course, up to some error margin and fixing the random seed),
    #   and also against shuffeling of the target/decoy sequences.
    scores, targets, _ = rand_scores_stds
    peps = peps_from_scores_hist_nnls(scores, targets, is_tdc)
    qvalues = qvalues_from_peps(scores, targets, peps)
    assert qvalues_are_valid(qvalues, scores)


@pytest.mark.parametrize("pi_factor", [1.0, 0.9])
def test_qvalues_from_counts(rand_scores_stds, pi_factor):
    scores, targets, _ = rand_scores_stds
    qvalues = qvalues_from_counts(scores, targets, pi_factor=pi_factor)
    assert qvalues_are_valid(qvalues, scores)


def test_qvalues_from_storey(rand_scores_stds):
    scores, targets, _ = rand_scores_stds
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=True)
    assert qvalues_are_valid(qvalues, scores)

    # Test with rounded scores, so that there are scores
    scores, targets, _ = rounded(rand_scores_stds, decimals=0)
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=True)
    assert qvalues_are_valid(qvalues, scores)

    # Test with separate target/decoy qvalue evaluation (qvalues not globally sorted)
    qvalues = qvalues_from_storeys_algo(scores, targets, decoy_qvals_by_interp=False)
    assert qvalues_are_valid(qvalues[targets], scores[targets])
    assert qvalues_are_valid(qvalues[~targets], scores[~targets])


def test_qvalues_from_counts_descending(desc_scores):
    """Test that q-values are correct for descending scores"""
    scores, target, true_qvals = desc_scores
    targets = target == 1
    qvals = qvalues_from_counts(scores, targets)
    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


def test_qvalues_from_hist_desc(desc_scores):
    scores, target, true_qvals = desc_scores
    targets = target == 1
    # Use small bins covering every interval + the scores as bin edges
    bin_edges = np.linspace(0, 11, num=371)
    bin_edges = np.unique(np.sort(np.concatenate((bin_edges, scores))))

    hist_data = TDHistData.from_scores_targets(scores, targets, bin_edges)
    qvalue_func = qvalues_func_from_hist(hist_data, pi_factor=1)
    qvals = qvalue_func(scores)

    np.testing.assert_allclose(qvals, true_qvals, atol=1e-7)


@pytest.mark.parametrize("pi_factor", [1.0, 0.7])
def test_compare_rand_qvalues_from_hist_vs_count(rand_scores_stds, pi_factor):
    # Compare the q-values computed via counts with those computed via
    # histogram on a dataset of a few thousand random scores
    scores, targets, _ = rand_scores_stds
    hist_data = TDHistData.from_scores_targets(scores, targets)
    qvalue_func = qvalues_func_from_hist(hist_data, pi_factor=pi_factor)
    qvals_hist = qvalue_func(scores)
    qvals_counts = qvalues_from_counts(scores, targets, pi_factor=pi_factor)

    np.testing.assert_allclose(qvals_hist, qvals_counts, atol=0.02)


def test_qvalues_discrete(rand_scores_stds):
    scores, targets, _ = rand_scores_stds
    scores = np.asarray(scores > scores.mean(), dtype=float)

    qvals_counts = qvalues_from_counts(scores, targets, pi_factor=1)

    # A tolerance of 0.1 is okay in the following tests, since the methods are
    # widely different.
    qvals_st1 = qvalues_from_storeys_algo(scores, targets, pvalue_method="conservative")
    np.testing.assert_allclose(qvals_st1, qvals_counts, atol=0.1)

    qvals_st2 = qvalues_from_storeys_algo(scores, targets, pvalue_method="storey")
    np.testing.assert_allclose(qvals_st2, qvals_counts, atol=0.1)


def test_qvalues_storey():
    with open("data/hedenfalk.json", "r") as file:
        data = json.load(file)
    pvals = np.array(data["pvalues"])
    qvals_expect = np.array(data["qvalues"])

    pi0est = pi0_from_pvalues_storey(
        pvals, method="bootstrap", lambdas=np.arange(0.05, 1, 0.05)
    )
    qvals = qvalues_from_pvalues(pvals, pi0=pi0est.pi0)
    np.testing.assert_almost_equal(qvals, qvals_expect)
