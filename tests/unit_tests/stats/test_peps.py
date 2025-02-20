import numpy as np
import pytest
from scipy import stats

import mokapot.stats.peps as peps


def get_target_decoy_data():
    N = 5000
    pi0 = 0.7
    R0 = stats.norm(loc=-4, scale=2)
    R1 = stats.norm(loc=3, scale=2)
    NT0 = int(np.round(pi0 * N))
    NT1 = N - NT0
    target_scores = np.concatenate((
        np.maximum(R1.rvs(NT1), R0.rvs(NT1)),
        R0.rvs(NT0),
    ))
    decoy_scores = R0.rvs(N)
    all_scores = np.concatenate((target_scores, decoy_scores))
    is_target = np.concatenate((
        np.full(len(target_scores), True),
        np.full(len(decoy_scores), False),
    ))

    sortIdx = np.argsort(-all_scores)
    return [all_scores[sortIdx], is_target[sortIdx]]


@pytest.mark.parametrize("is_tdc", [True, False])
def test_peps_qvality(is_tdc):
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_qvality(scores, targets, is_tdc)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)


@pytest.mark.parametrize("is_tdc", [True, False])
def test_peps_kde_nnls(is_tdc):
    np.random.seed(1253)  # this produced an error with failing iterations in nnls
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_kde_nnls(scores, targets, is_tdc)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)

    np.random.seed(1245)  # this produced an assertion error due to peps over 1.0
    scores, targets = get_target_decoy_data()
    peps_values = peps.peps_from_scores_kde_nnls(scores, targets, is_tdc)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)


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
    scores, targets = get_target_decoy_data()

    peps_values = peps.peps_from_scores_hist_nnls(scores, targets, False)
    assert np.all(peps_values >= 0)
    assert np.all(peps_values <= 1)
    assert np.all(np.diff(peps_values) * np.diff(scores) <= 0)


def test_peps_from_qvality_sorting():
    # Check that qvality works with differently sorted scores
    rng = np.random.Generator(np.random.PCG64(42))

    N = 1126
    targets = (rng.uniform(0, 1, N)) > 0.7
    scores = targets * rng.normal(1, 1, N) + (1 - targets) * rng.normal(-1, 1, N)

    peps0 = peps.peps_from_scores_qvality(scores, targets, True, use_binary=False)

    index = np.argsort(scores)[::-1]
    scores_sorted, targets_sorted = scores[index], targets[index]

    peps1 = peps.peps_from_scores_qvality(
        scores_sorted, targets_sorted, True, use_binary=False
    )

    assert np.all(peps0[index] == peps1)
