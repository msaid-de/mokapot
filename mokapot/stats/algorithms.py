"""
The idea of this module is to encapsulate all algorithms including their
"standard" parameters into callable objects.

Currently, there is only the QvalueAlgorithm for training here. But we could
also put more algo's here (e.g. for the peps, differentiate between qvalues for
training and for confidence estimation, etc). Either by specific classes or
maybe by some algorithm registry.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

import numpy as np
from typeguard import typechecked

import mokapot.stats.peps as pepsmod
import mokapot.stats.pi0est as pi0est
import mokapot.stats.pvalues as pvalues
import mokapot.stats.qvalues as qvalues

LOGGER = logging.getLogger(__name__)

# todo: change ABC to Protocol, and separate singleton from implementations


## Algorithms for pi0 estimation
@typechecked
class Pi0EstAlgorithm(ABC):
    # Derived classes: StoreyPi0Algorithm, TDSlopePi0Algorithm
    pi0_algo = None

    @abstractmethod
    def estimate(self, scores, targets):
        raise NotImplementedError

    def estimate_pi_factor(
        self, scores: np.ndarray[float], targets: np.ndarray[bool]
    ) -> float:
        pi0 = self.estimate(scores, targets)
        targets_count = targets.sum()
        decoys_count = (~targets).sum()
        return pi0 * targets_count / decoys_count

    @classmethod
    def set_algorithm(cls, pi0_algo: Pi0EstAlgorithm):
        cls.pi0_algo = pi0_algo

    @classmethod
    def long_desc(cls):
        return cls.pi0_algo.long_desc()


@typechecked
class TDCPi0Algorithm(Pi0EstAlgorithm):
    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        targets_count = targets.sum()
        decoys_count = (~targets).sum()
        if decoys_count == 0:
            LOGGER.warning(
                f"Can't estimate pi0 with zero decoys (targets={targets_count}, "
                f"decoys={decoys_count}, total={len(targets)})"
            )
        decoy_target_ratio = decoys_count / targets_count
        return decoy_target_ratio

    def estimate_pi_factor(
        self, scores: np.ndarray[float], targets: np.ndarray[bool]
    ) -> float:
        return 1.0

    def long_desc(self) -> str:
        return "decoy_target_ratio"


@typechecked
class StoreyPi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, method: str, eval_lambda: float):
        self.method = method
        self.eval_lambda = eval_lambda

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        pvals = pvalues.empirical_pvalues(
            scores[targets], scores[~targets], mode="conservative"
        )
        pi0 = pi0est.pi0_from_pvalues_storey(
            pvals,
            method=self.method,
            lambdas=np.arange(0.2, 0.8, 0.01),
            eval_lambda=self.eval_lambda,
        ).pi0
        return pi0

    def long_desc(self) -> str:
        return f"storey_pi0(method={self.method}, lambda={self.eval_lambda})"


@typechecked
class SlopePi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, hist_bins="scott", slope_threshold: float = 0.9):
        self.bins = hist_bins
        self.slope_threshold = slope_threshold

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        return pi0est.pi0est_from_scores_by_slope(
            scores, targets, bins=self.bins, slope_threshold=self.slope_threshold
        )

    def long_desc(self) -> str:
        return (
            f"pi0_by_slope(hist_bins={self.bins}, "
            f"slope_threshold={self.slope_threshold})"
        )


@typechecked
class BootstrapPi0Algorithm(Pi0EstAlgorithm):
    def __init__(self, N: int = 100000):
        self.N = N

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        return pi0est.pi0_from_bootstrap(scores, targets, self.N)

    def long_desc(self) -> str:
        return f"pi0_by_bootstrap(N={self.N})"


@typechecked
class FixedPi0(Pi0EstAlgorithm):
    """Not really an estimation algorithm, but just returning a fixed pi0."""

    def __init__(self, pi0: float):
        self.pi0 = pi0

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> float:
        return self.pi0

    def long_desc(self) -> str:
        return f"fixed_pi0({self.pi0})"


@typechecked
class Pi0EstimationMixin:
    pi0_algo: Pi0EstAlgorithm

    def __init__(self, pi0_algo: Pi0EstAlgorithm, **kwargs):
        super().__init__(**kwargs)
        self.pi0_algo = pi0_algo

    def estimate_pi0(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        # todo: move into mixin class
        pi0_algo = self.pi0_algo or Pi0EstAlgorithm.pi0_algo
        pi0 = pi0_algo.estimate(scores, targets)
        LOGGER.debug(f"pi0-estimate: pi0={pi0}, algo={pi0_algo.long_desc()}")
        return pi0

    def estimate_pi_factor(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        # todo: move into mixin class
        pi0_algo = self.pi0_algo or Pi0EstAlgorithm.pi0_algo
        pi_factor = pi0_algo.estimate_pi_factor(scores, targets)
        LOGGER.debug(
            f"pi-factor estimate: pi_factor={pi_factor}, algo={pi0_algo.long_desc()}"
        )
        return pi_factor


## Algorithms for qvalue computation
@typechecked
class QvalueAlgorithm(ABC, Pi0EstimationMixin):
    qvalue_algo = None

    def __init__(self, pi0_algo: Pi0EstAlgorithm):
        super().__init__(pi0_algo=pi0_algo)

    @abstractmethod
    def estimate(self, scores, targets, desc):
        raise NotImplementedError

    @classmethod
    def set_algorithm(cls, qvalue_algo: QvalueAlgorithm):
        cls.qvalue_algo = qvalue_algo

    @classmethod
    def eval(cls, scores, targets, desc=True):
        if cls.qvalue_algo is None:
            raise ValueError("qvalue_algorithm is not set")
        return cls.qvalue_algo.estimate(scores, targets, desc)

    @classmethod
    def long_desc(cls):
        if cls.qvalue_algo is None:
            raise ValueError("qvalue_algorithm is not set")
        return cls.qvalue_algo.long_desc()


@typechecked
class CountsQvalueAlgorithm(QvalueAlgorithm):
    def __init__(self, *, pi0_algo: Pi0EstAlgorithm):
        super().__init__(pi0_algo)

    def estimate(
        self, scores: np.ndarray[float], targets: np.ndarray[bool], desc: bool
    ):
        if not desc:
            scores = -scores
        pi_factor = super().estimate_pi_factor(scores, targets)
        qvals = qvalues.qvalues_from_counts(scores, targets, pi_factor=pi_factor)
        return qvals

    def long_desc(self):
        return "qvalue_by_counts"


@typechecked
class StoreyQvalueAlgorithm(QvalueAlgorithm):
    def __init__(self, *, pvalue_method="best", pi0_algo=None):
        super().__init__(pi0_algo)
        self.pvalue_method = pvalue_method

    def estimate(
        self, scores: np.ndarray[float], targets: np.ndarray[bool], desc: bool
    ):
        if not desc:
            scores = -scores
        pi0 = super().estimate_pi0(scores, targets)
        qvals = qvalues.qvalues_from_storeys_algo(scores, targets, pi0)
        return qvals

    def long_desc(self):
        return "qvalues_by_storeys_algo"


# Algoritms for pep computation
@typechecked
class PepsAlgorithm(ABC, Pi0EstimationMixin):
    # Derived classes: TriqlerPEPAlgorithm, HistNNLSAlgorithm, KDENNLSAlgorithm
    # Not yet: StoreyLFDRAlgorithm (probit, logit)
    peps_algo: PepsAlgorithm | None = None
    peps_error: bool = True

    def __init__(self, pi0_algo: Pi0EstAlgorithm):
        super().__init__(pi0_algo=pi0_algo)

    @abstractmethod
    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        raise NotImplementedError

    @classmethod
    def set_algorithm(cls, peps_algo: PepsAlgorithm):
        cls.peps_algo = peps_algo

    @classmethod
    def eval(cls, scores, targets):
        if cls.peps_algo is None:
            raise ValueError("peps_algorithm is not set")
        try:
            peps = cls.peps_algo.estimate(scores, targets)
        except pepsmod.PepsConvergenceError:
            LOGGER.warning(
                f"\t- Encountered convergence problems in `{cls.peps_algo}`. "
                "Falling back to triqler ...",
            )
            peps_algo = TriqlerPepsAlgorithm(pi0_algo=Pi0EstAlgorithm.pi0_algo)
            peps = peps_algo.estimate(scores, targets)

        if all(peps == 1):
            if cls.peps_error:
                raise ValueError("PEP values are all equal to 1.")
            else:
                LOGGER.warning("PEP values are all equal to 1.")

        return peps

    @classmethod
    def long_desc(cls):
        if cls.peps_algo is None:
            raise ValueError("peps_algorithm is not set")
        return cls.peps_algo.long_desc()


@typechecked
class TriqlerPepsAlgorithm(PepsAlgorithm):
    def __init__(self, *, pi0_algo=None):
        super().__init__(pi0_algo)

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        pi0 = super().estimate_pi0(scores, targets)
        peps = pepsmod.peps_from_scores_qvality(
            scores, targets, use_binary=False, pi0=pi0, is_tdc=True
        )
        return peps

    def long_desc(self):
        return "peps_by_triqler"


@typechecked
class QvalityPepsAlgorithm(PepsAlgorithm):
    def __init__(self, *, is_tdc=True):
        self.is_tdc = is_tdc
        if not pepsmod.is_qvality_on_path():
            raise ValueError(
                "The `qvality` binary is not on the path or not executable"
            )

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        peps = pepsmod.peps_from_scores_qvality(
            scores, targets, use_binary=True, is_tdc=self.is_tdc
        )
        return peps

    def long_desc(self):
        return "peps_by_qvality"


@typechecked
class KdeNNLSPepsAlgorithm(PepsAlgorithm):
    def __init__(self, *, pi0_algo=None):
        super().__init__(pi0_algo)

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        pi_factor = super().estimate_pi_factor(scores, targets)
        peps = pepsmod.peps_from_scores_kde_nnls(scores, targets, pi_factor=pi_factor)
        return peps

    def long_desc(self):
        return "peps_by_kde_nnls"


@typechecked
class HistNNLSPepsAlgorithm(PepsAlgorithm):
    def __init__(self, *, pi0_algo=None):
        super().__init__(pi0_algo)

    def estimate(self, scores: np.ndarray[float], targets: np.ndarray[bool]):
        pi_factor = super().estimate_pi_factor(scores, targets)
        peps = pepsmod.peps_from_scores_hist_nnls(scores, targets, pi_factor=pi_factor)
        return peps

    def long_desc(self):
        return "peps_by_hist_nnls"


# Configuration of algorithms via command line arguments
def configure_algorithms(config):
    is_tdc = config.tdc

    pi0_algorithm = config.pi0_algorithm
    if pi0_algorithm is None or pi0_algorithm == "default":
        pi0_algorithm = "ratio" if is_tdc else "bootstrap"

    match pi0_algorithm:
        case "ratio":
            if not is_tdc:
                msg = "Can't use 'ratio' for pi0 estimation, when 'tdc' is false"
                raise ValueError(msg)
            pi0_algorithm = TDCPi0Algorithm()
        case "slope":
            pi0_algorithm = SlopePi0Algorithm()
        case "fixed":
            pi0_algorithm = FixedPi0(config.pi0_value)
        case "bootstrap":
            pi0_algorithm = BootstrapPi0Algorithm()
        case "storey_smoother":
            pi0_algorithm = StoreyPi0Algorithm("smoother", config.pi0_eval_lambda)
        case "storey_fixed":
            pi0_algorithm = StoreyPi0Algorithm("fixed", config.pi0_eval_lambda)
        case "storey_bootstrap":
            pi0_algorithm = StoreyPi0Algorithm("bootstrap", config.pi0_eval_lambda)
        case _:
            raise ValueError(f"Unknown pi0 algorithm '{pi0_algorithm}'")
    Pi0EstAlgorithm.set_algorithm(pi0_algorithm)

    qvalue_algorithm = config.qvalue_algorithm
    if qvalue_algorithm is None or qvalue_algorithm == "default":
        qvalue_algorithm = "from_counts" if is_tdc else "storey"

    match qvalue_algorithm:
        case "from_counts":
            qvalue_algorithm = CountsQvalueAlgorithm(pi0_algo=pi0_algorithm)
        case "storey":
            qvalue_algorithm = StoreyQvalueAlgorithm(pi0_algo=pi0_algorithm)
        case _:
            raise ValueError(f"Unknown qvalue algorithm '{qvalue_algorithm}'")
    QvalueAlgorithm.set_algorithm(qvalue_algorithm)

    peps_algorithm = config.peps_algorithm
    match peps_algorithm:
        case "triqler":
            peps_algorithm = TriqlerPepsAlgorithm(pi0_algo=pi0_algorithm)
        case "qvality":
            if config.pi0_algorithm != "default":
                warnings.warn(
                    "PEPs-algorithm `qvality` cannot use pi0-algorithm "
                    f"'{config.pi0_algorithm}', but will only use its internal one."
                )
            peps_algorithm = QvalityPepsAlgorithm(is_tdc=is_tdc)
        case "kde_nnls":
            peps_algorithm = KdeNNLSPepsAlgorithm(pi0_algo=pi0_algorithm)
        case "hist_nnls":
            peps_algorithm = HistNNLSPepsAlgorithm(pi0_algo=pi0_algorithm)
        case _:
            raise ValueError(f"Unknown peps algorithm '{peps_algorithm}'")
    PepsAlgorithm.set_algorithm(peps_algorithm)
    PepsAlgorithm.peps_error = config.peps_error

    LOGGER.debug(f"pi0 algorithm: {pi0_algorithm.long_desc()}")
    LOGGER.debug(f"q-value algorithm: {qvalue_algorithm.long_desc()}")
    LOGGER.debug(f"peps algorithm: {peps_algorithm.long_desc()}")


Pi0EstAlgorithm.set_algorithm(TDCPi0Algorithm())
QvalueAlgorithm.set_algorithm(CountsQvalueAlgorithm(pi0_algo=Pi0EstAlgorithm.pi0_algo))
PepsAlgorithm.set_algorithm(TriqlerPepsAlgorithm(pi0_algo=Pi0EstAlgorithm.pi0_algo))
