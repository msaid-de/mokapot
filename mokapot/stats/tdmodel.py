import typing
from abc import ABC, abstractmethod

import numpy as np
import scipy as sp
from typeguard import typechecked


# This file (class) is only for checking validity of TD modelling assumptions.


class RandomVar(typing.Protocol):
    """Interface definition for random variables."""

    def rvs(self, N: int) -> int: ...
    def cdf(self, x: float | np.ndarray[float]) -> float: ...
    def sf(self, x: float | np.ndarray[float]) -> float: ...


@typechecked
def set_mu_std(dist: sp.stats.rv_continuous, mu: float, std: float):
    """Modifies distribution parameters to have specified mean and standard
    deviation.

    Note: the input distribution needs to have finite mean and standard
    deviation for this method to work.

    Parameters
    ----------
    dist : sp.stats.rv_continuous
        The continuous random variable distribution object.
    mu : float
        The desired mean value for the distribution.
    std : float
        The desired standard deviation value for the distribution.

    Returns
    -------
    dist
        The distribution object with updated mean and standard deviation.
    """
    kwds = dist.kwds
    kwds["loc"] = 0
    kwds["scale"] = 1
    rv0 = dist.dist(**kwds)
    kwds["scale"] = std / rv0.std()
    rv1 = dist.dist(**kwds)
    kwds["loc"] = mu - rv1.mean()
    return dist.dist(**kwds)


@typechecked
def set_support(dist, lower: float, upper: float):
    """Modifies distribution object to have fixed support.

    Note: the input distribution must have finite support already.

    Parameters
    ----------
    dist : sp.stats.rv_continuous
        The continuous random variable distribution object.
    lower : float
        The new lower limit of the support.
    upper : float
        The new upper limit of the support.

    Returns
    -------
    dist
        The distribution object with updated support.
    """
    kwds = dist.kwds
    kwds["loc"] = 0
    kwds["scale"] = 1
    rv0 = dist.dist(**kwds)
    kwds["scale"] = (upper - lower) / (rv0.support()[1] - rv0.support()[0])
    rv1 = dist.dist(**kwds)
    kwds["loc"] = lower - rv1.support()[0]
    return dist.dist(**kwds)


def is_discrete(RV: RandomVar) -> bool:
    return hasattr(RV, "pmf")


@typechecked
class TDModel(ABC):
    """Abstract base class for target-decoy models.

    Attributes:
        R0 (object): The distribution model for decoy scores.
        R1 (object): The distribution model for true target scores.
        rho0 (float): The fraction of foreign spectra (i.e. spectra for which
          the generating spectrum is not in the database, in [Keich 2015] this
          is pi0, but that does not coincide with Kaell or Storey).

    Methods:
        sample_decoys(N):
            Generates N decoy scores from the (initial) decoy score
            distribution.

        sample_true_targets(N):
            Generates N true target scores from the (initial) target score
            distribution.

        sample_targets(N, include_is_fd=True, shuffle_result=True):
            Generates N target scores by sampling from both the target
            and decoy score distributions.

        sample_scores(N):
            Abstract method for generating N target and decoy scores.

        decoy_pdf(x):
            Computes the probability density function of the decoy score
            distribution at x.

        true_target_pdf(x):
            Computes the probability density function of the target score
            distribution at x.

        decoy_cdf(x):
            Computes the cumulative distribution function of the decoy score
            distribution at x.

        true_target_cdf(x):
            Computes the cumulative distribution function of the target score
            distribution at x.

        true_pep(x):
            Computes the posterior error probability for a given score x.

        true_fdr(x):
            Computes the false discovery rate for a given score x.

        get_sampling_pdfs(x):
            Abstract method for getting the sampling PDFs for a given score x.
    """

    def __init__(self, R0: RandomVar, R1: RandomVar, rho0: float):
        self.R0 = R0
        self.R1 = R1
        self.U = sp.stats.uniform()

        if is_discrete(R0) != is_discrete(R1):
            raise ValueError(
                "Random variables must be of same type "
                "(both discrete or both continuous)"
            )
        self.is_discrete = is_discrete(R0)
        self.rho0 = rho0

    def _perturb(self, samples, delta=1e-10):
        # Add some perturbation in case of discrete distributions
        if self.is_discrete:
            return samples + delta * self.U.rvs(len(samples))
        else:
            return samples

    def _unperturb(self, samples):
        if self.is_discrete:
            return np.floor(samples)
        else:
            return samples

    def _sample_from(self, R: RandomVar, N: int, perturb: bool):
        samples = R.rvs(N).astype(float)
        return self._perturb(samples) if perturb else samples

    def sample_true_targets(self, N: int, perturb: bool = False) -> np.ndarray[float]:
        return self._sample_from(self.R1, N, perturb)

    def sample_false_targets(self, N: int, perturb: bool = False) -> np.ndarray[float]:
        return self._sample_from(self.R0, N, perturb)

    def sample_decoys(self, N: int, perturb: bool = False) -> np.ndarray[float]:
        # This is by the central target-decoy method assumption
        return self.sample_false_targets(N, perturb)

    def _sample_targets(
        self, N: int, include_is_fd: bool = True
    ) -> tuple[np.ndarray[float], np.ndarray[bool]] | np.ndarray[float]:
        NT = N

        X = self.sample_true_targets(NT, perturb=True)
        is_foreign = self.U.rvs(NT) < self.rho0
        X[is_foreign] = -np.inf

        Y = self.sample_false_targets(NT, perturb=True)
        Z = np.maximum(X, Y)

        target_scores = Z
        is_fd = Z == Y

        return target_scores, is_fd

    @abstractmethod
    def sample_scores(self, N: int):
        # Depends on whether target decoy competition or separate search is employed
        pass

    def _sample_both(self, NT: int, ND: int):
        target_scores, is_fd = self._sample_targets(NT)
        decoy_scores = self.sample_decoys(ND, perturb=True)
        return target_scores, decoy_scores, is_fd

    @staticmethod
    def _sort_and_return(
        scores: np.ndarray[float], is_target: np.ndarray[bool], is_fd: np.ndarray[bool]
    ) -> tuple[np.ndarray[float], np.ndarray[bool], np.ndarray[bool]]:
        sort_idx = np.argsort(-scores)
        sorted_scores = scores[sort_idx]
        is_target = is_target[sort_idx]
        is_fd = is_fd[sort_idx]
        return sorted_scores, is_target, is_fd

    def decoy_pdf(self, x: float | np.ndarray[float]) -> float | np.ndarray[float]:
        if self.is_discrete:
            return self.R0.pmf(x)
        else:
            return self.R0.pdf(x)

    def true_target_pdf(
        self, x: float | np.ndarray[float]
    ) -> float | np.ndarray[float]:
        if self.is_discrete:
            return self.R1.pmf(x)
        else:
            return self.R1.pdf(x)

    def decoy_cdf(self, x: float | np.ndarray[float]) -> float | np.ndarray[float]:
        return 1.0 - self.R0.sf(x)
        return self.R0.cdf(x)

    def true_target_cdf(
        self, x: float | np.ndarray[float]
    ) -> float | np.ndarray[float]:
        return 1.0 - self.R1.sf(x)
        return self.R1.cdf(x)

    def true_pep(self, x: float | np.ndarray[float]) -> float | np.ndarray[float]:
        T_pdf, TT_pdf, FT_pdf, D_pdf, rho0 = self.get_sampling_pdfs(x)
        return rho0 * FT_pdf / T_pdf

    def true_fdr(self, x: float | np.ndarray[float]) -> float | np.ndarray[float]:
        if any(np.diff(x) < 0):
            raise ValueError("x must be non-decreasing, but wasn't'")

        # This pi0 is in both cases the Storey pi0, not the Keich pi0
        T_pdf, TT_pdf, FT_pdf, D_pdf, FDR = self.get_sampling_pdfs(x)

        fdr = FDR * np.flip(np.cumsum(np.flip(FT_pdf)) / np.cumsum(np.flip(T_pdf)))

        return fdr

    def _get_integration_points(self, R: RandomVar | None = None, N=10000):
        R = R or self.R0
        y = np.linspace(0.0, 1.0, num=N)
        x0 = R.ppf(y)
        x = np.unique(x0[np.isfinite(x0)])
        return x

    def _integrate0(self, f, x):
        if self.is_discrete:
            assert np.unique(x).size == x.size
            return sum(f)
        else:
            return sp.integrate.trapz(f, x)

    def _integrate1(self, F1, f1, f, x):
        if self.is_discrete:
            assert np.unique(x).size == x.size
            return sum((F1 - f1 / 2) * f)
        else:
            return sp.integrate.trapz(F1 * f, x)

    def _integrate2(self, F1, f1, F2, f2, f, x):
        if self.is_discrete:
            # Note: the discrete stuff comes from stretching the discrete
            # distribution a bit, integrating then, and letting the stretch then
            # go to zero (well, you find that the integral is independent of the
            # stretch anyway, but it's way to length and complicated to put the
            # derivation into a comment here)
            assert np.unique(x).size == x.size
            return sum((F1 * F2 - (f1 * F2 + F1 * f2) / 2 + f1 * f2 / 3) * f)
        else:
            return sp.integrate.trapz(F1 * F2 * f, x)

    def _get_input_pdfs_and_cdfs(self, x):
        rho0 = self.rho0
        X0_pdf = self.true_target_pdf(x)
        X0_cdf = self.true_target_cdf(x)
        X_pdf = (1 - rho0) * X0_pdf
        X_cdf = rho0 + (1 - rho0) * X0_cdf
        Y_pdf = self.decoy_pdf(x)
        Y_cdf = self.decoy_cdf(x)
        return X_pdf, X_cdf, Y_pdf, Y_cdf

    def test_pdf(self, pdf, x):
        val = self._integrate0(pdf, x)
        if not (0.99 <= val <= 1.01):
            raise ValueError(
                f"Problem with pdf: integral gives {val} (not in [0.9, 1.1])"
            )

    @abstractmethod
    def get_sampling_pdfs(self, x):
        pass

    @abstractmethod
    def approx_pi0(self):
        pass


@typechecked
class TDCModel(TDModel):
    """A TDModel class for target decoy competition or concatenated search"""

    def sample_scores(
        self, N: int
    ) -> tuple[np.ndarray[float], np.ndarray[bool], np.ndarray[bool]]:
        target_scores, decoy_scores, is_fd = self._sample_both(N, N)

        is_target = target_scores >= decoy_scores

        all_scores = np.where(is_target, target_scores, decoy_scores)
        all_scores = self._unperturb(all_scores)
        return self._sort_and_return(all_scores, is_target, is_fd)

    def approx_fdr_and_dp(self):
        x = self._get_integration_points()
        X_pdf, X_cdf, Y_pdf, Y_cdf = self._get_input_pdfs_and_cdfs(x)
        self.test_pdf(Y_pdf, x)
        DP = self._integrate2(X_cdf, X_pdf, Y_cdf, Y_pdf, Y_pdf, x)
        FDR = DP / (1 - DP)
        return FDR, DP

    def approx_pi0(self):
        return self.approx_fdr_and_dp()[0]

    def get_sampling_pdfs(self, x):
        DP, FDR = self.approx_fdr_and_dp()
        X_pdf, X_cdf, Y_pdf, Y_cdf = self._get_input_pdfs_and_cdfs(x)

        T_pdf = (X_pdf * Y_cdf**2 + X_cdf * Y_cdf * Y_pdf) / (1 - DP)
        TT_pdf = X_pdf * Y_cdf**2 / (1 - 2 * DP)
        FT_pdf = (X_cdf * Y_cdf * Y_pdf) / DP
        D_pdf = FT_pdf
        return T_pdf, TT_pdf, FT_pdf, D_pdf, FDR


@typechecked
class STDSModel(TDModel):
    """A TDModel class for separate search"""

    def sample_scores(
        self, NT: int, ND: int | None = None
    ) -> tuple[
        np.ndarray[float], np.ndarray[bool], np.ndarray[bool] | np.ndarray[bool]
    ]:
        ND = NT if ND is None else ND
        target_scores, decoy_scores, is_fd = self._sample_both(NT, ND)
        all_scores = np.concatenate((target_scores, decoy_scores))
        is_target = np.concatenate((np.full(NT, True), np.full(ND, False)))
        is_fd = np.concatenate((
            is_fd,
            np.full(ND, False),
        ))  # is_fd value for decoys is irrelevant

        all_scores = self._unperturb(all_scores)
        return self._sort_and_return(all_scores, is_target, is_fd)

    def approx_pi0(self):
        x = self._get_integration_points()
        X_pdf, X_cdf, Y_pdf, _ = self._get_input_pdfs_and_cdfs(x)
        self.test_pdf(Y_pdf, x)
        FDR = self._integrate1(X_cdf, X_pdf, Y_pdf, x)
        return FDR

    def get_sampling_pdfs(self, x):
        FDR = self.approx_pi0()

        X_pdf, X_cdf, Y_pdf, Y_cdf = self._get_input_pdfs_and_cdfs(x)
        T_pdf = X_pdf * Y_cdf + X_cdf * Y_pdf
        TT_pdf = X_pdf * Y_cdf / (1 - FDR)
        FT_pdf = X_cdf * Y_pdf / FDR
        D_pdf = Y_pdf
        return T_pdf, TT_pdf, FT_pdf, D_pdf, FDR
