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
from abc import ABC, abstractmethod

import numpy as np
from typeguard import typechecked

import mokapot.stats.pi0est as pi0est
import mokapot.stats.pvalues as pvalues
import mokapot.stats.qvalues as qvalues

LOGGER = logging.getLogger(__name__)

# todo: change ABC to Protocol, and separate singleton from implementations


## Algorithms for pi0 estimation
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


## Algorithms for qvalue computation
@typechecked
class QvalueAlgorithm(ABC):
    qvalue_algo = None

    def __init__(self, pi0_algo):
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
    def __init__(self, tdc: bool, pi0_algo: Pi0EstAlgorithm):
        super().__init__(pi0_algo)
        self.tdc = tdc

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
class PEPAlgorithm(ABC):
    # Derived classes: TriqlerPEPAlgorithm, HistNNLSAlgorithm, KDENNLSAlgorithm
    # Not yet: StoreyLFDRAlgorithm (probit, logit)
    pass


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
            raise NotImplementedError
    Pi0EstAlgorithm.set_algorithm(pi0_algorithm)

    qvalue_algorithm = config.qvalue_algorithm
    if qvalue_algorithm is None or qvalue_algorithm == "default":
        qvalue_algorithm = "from_counts" if is_tdc else "storey"

    match qvalue_algorithm:
        case "from_counts":
            qvalue_algorithm = CountsQvalueAlgorithm(is_tdc, pi0_algo=pi0_algorithm)
        case "storey":
            qvalue_algorithm = StoreyQvalueAlgorithm(pi0_algo=pi0_algorithm)
        case _:
            raise NotImplementedError
    QvalueAlgorithm.set_algorithm(qvalue_algorithm)

    LOGGER.debug(f"pi0 algorithm: {pi0_algorithm.long_desc()}")
    LOGGER.debug(f"q-value algorithm: {qvalue_algorithm.long_desc()}")


QvalueAlgorithm.set_algorithm(CountsQvalueAlgorithm(True, TDCPi0Algorithm()))
