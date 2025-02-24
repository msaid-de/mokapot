import json

import numpy as np
import pytest
from pytest import approx

from mokapot.stats.pi0est import pi0_from_pvalues_storey, pi0_from_scores_storey
from mokapot.stats.pvalues import empirical_pvalues
from .helpers import create_tdmodel


def test_estimate_pi0_from_R():
    # Compare against Storey's method with the Hedenfalk data.

    # with open("data/samples.json", "r") as file:
    with open("data/hedenfalk.json", "r") as file:
        data = json.load(file)
    targets = np.array(data["stat"])
    decoys = np.array(data["stat0"])

    # Preliminary check that pvalues match
    # Note: A not so slight problem is, that there is an issue with the empPVals
    # function in the qvalue package when some target and decoy scores are
    # identical, which can become a major problem for discrete features. This
    # needed to be fixed which is why some of the results here differ a bit more.
    pvalues = empirical_pvalues(targets, decoys, mode="storey")
    pvals_exp = np.array(data["pvalues"])
    np.testing.assert_allclose(pvalues, pvals_exp, atol=5e-6)

    # Compute pi0 with bootstrap method
    lambdas = np.arange(0.05, 1, 0.05)
    pi0est = pi0_from_pvalues_storey(pvalues, method="bootstrap", lambdas=lambdas)
    assert pi0est.pi0 == approx(0.6763407)
    assert pi0est.mse is not None
    assert pi0est.pi0s_raw[np.argmin(pi0est.mse)] == pi0est.pi0

    # Compute pi0 by smoothing (the comparison value is from the qvalue package
    # in R, since the smoothing method is different the difference is relatively
    # large)
    lambdas = np.arange(0.2, 0.95, 0.01)
    pi0est = pi0_from_pvalues_storey(pvalues, lambdas=lambdas, eval_lambda=0.8)
    assert pi0est.pi0 == approx(0.6931328, abs=0.01)

    # Compute pi0 with fixed lambda
    pi0est = pi0_from_pvalues_storey(pvalues, method="fixed", eval_lambda=0.7)
    assert pi0est.pi0 == approx(0.701367)
    pi0est = pi0_from_pvalues_storey(pvalues, method="fixed", eval_lambda=0.3)
    assert pi0est.pi0 == approx(0.7138351)


@pytest.mark.parametrize("rho0", [0.01, 0.3, 0.8, 0.95])
@pytest.mark.parametrize("discrete", [True, False])
@pytest.mark.parametrize("is_tdc", [True, False])
def test_pi0est_storey(discrete, is_tdc, rho0):
    if is_tdc and not discrete and rho0 < 0.1:
        pytest.skip("Does not work for small rho0/pi0 in this case")

    N = 1000000
    model = create_tdmodel(is_tdc, rho0, discrete, delta=2)
    scores, targets, is_fd = model.sample_scores(N)

    pi0 = pi0_from_scores_storey(scores, targets, method="bootstrap")

    pi0_actual = model.pi0_from_data(targets, is_fd)
    assert pi0 == approx(pi0_actual, abs=0.1)

    # pi0_expect = model.approx_pi0()
    # assert pi0 == approx(pi0_expect, abs=2 * 0.05)
