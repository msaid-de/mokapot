import numpy as np
import pytest
from scipy import stats

import mokapot.stats.peps as peps
from mokapot.stats.tdmodel import STDSModel
from tests.helpers.utils import TestOutcome
from .helpers import create_tdmodel


def rand_scores_stds():
    N = 5000
    rho0 = 0.7
    R0 = stats.norm(loc=-4, scale=2)
    R1 = stats.norm(loc=3, scale=2)
    model = STDSModel(R0, R1, rho0)
    scores, targets, is_fd = model.sample_scores(N)
    return scores, targets, is_fd


def peps_are_valid(peps, scores=None) -> TestOutcome:
    """Helper function for tests on qvalues"""
    if not np.all(peps >= 0):
        return TestOutcome.fail("'peps must be >= 0'")
    if not np.all(peps <= 1):
        return TestOutcome.fail("'peps must be <= 1'")

    if scores is not None:
        ind = np.argsort(scores)
        diff_peps = np.diff(peps[ind])
        diff_scores = np.diff(scores[ind])
        if not np.all(diff_peps * diff_scores <= 0):
            return TestOutcome.fail(
                "'peps are monotonically decreasing with higher scores'"
            )

        if not np.all((diff_scores != 0) | (diff_peps == 0)):
            # Note that "!A | B" is the same as the implication "A => B"
            # When two scores are equal they must have the same peps, but if
            # they are different the may have the same pep
            return TestOutcome.fail("'equal scores must have equal peps'")

    return TestOutcome.success()


@pytest.mark.parametrize("rho0", [0.01, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("is_tdc", [False])
def test_peps_qvality(is_tdc, rho0):
    N = 1000000
    model = create_tdmodel(is_tdc, rho0, False, delta=2)
    scores, targets, is_fd = model.sample_scores(N)

    peps_values = peps.peps_from_scores_qvality(
        scores, targets, is_tdc=False, use_binary=True
    )
    assert peps_are_valid(peps_values, scores=scores)


@pytest.mark.parametrize("rho0", [0.1, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("is_tdc", [True, False])
def test_peps_kde_nnls(is_tdc, rho0):
    model = create_tdmodel(is_tdc, rho0, False, delta=2, max_dev_z=10)
    N = 100000
    scores, targets, is_fd = model.sample_scores(N)
    peps_exact = model.true_pep(scores)

    pi0 = model.pi0_from_data(targets, is_fd)

    peps_values = peps.peps_from_scores_kde_nnls(scores, targets, pi0)
    assert peps_are_valid(peps_values, scores=scores)
    np.testing.assert_allclose(peps_values, peps_exact, atol=0.1)

    peps_values2 = peps.peps_from_scores_qvality(
        scores, targets, is_tdc=False, pi0=pi0, use_binary=False
    )
    assert peps_are_valid(peps_values2, scores=scores)
    np.testing.assert_allclose(peps_values, peps_exact, atol=0.1)

    peps_values3 = peps.peps_from_scores_hist_nnls(
        scores, targets, pi_factor=pi0 * sum(targets) / sum(~targets)
    )
    assert peps_are_valid(peps_values3, scores=scores)
    np.testing.assert_allclose(peps_values, peps_exact, atol=0.1)


@pytest.mark.parametrize(
    "seed",
    # Those were collected seeds from random experiments where nnls failed
    [1253, 41908, 39831, 21706, 38306, 23020, 46079, 96127, 23472, 21706]
    + [38306, 23020, 46079, 96127, 23472, 21706, 38306, 23020, 46079, 96127]
    + [23472, 21706, 38306, 23020, 46079, 96127, 23472, 21706, 38306, 23020]
    + [46079, 96127, 23472, 21706, 38306, 23020, 46079, 96127, 23472, 21706],
)
def test_peps_hist_nnls(seed):
    np.random.seed(seed)
    scores, targets, _ = rand_scores_stds()

    peps_values = peps.peps_from_scores_hist_nnls(scores, targets, False)
    assert peps_are_valid(peps_values, scores=scores)


def test_peps_from_qvality_sorting():
    # Check that qvality works with differently sorted scores
    rng = np.random.Generator(np.random.PCG64(42))

    N = 1126
    targets = (rng.uniform(0, 1, N)) > 0.7
    scores = targets * rng.normal(1, 1, N) + (1 - targets) * rng.normal(-1, 1, N)

    peps0 = peps.peps_from_scores_qvality(scores, targets, True, use_binary=False)
    assert peps_are_valid(peps0, scores=scores)

    index = np.argsort(scores)[::-1]
    scores_sorted, targets_sorted = scores[index], targets[index]

    peps1 = peps.peps_from_scores_qvality(
        scores_sorted, targets_sorted, True, use_binary=False
    )

    assert np.all(peps0[index] == peps1)
    assert peps_are_valid(peps1, scores=scores_sorted)
