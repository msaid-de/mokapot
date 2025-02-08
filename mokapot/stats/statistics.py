import math
from collections import namedtuple

import numpy as np
from typeguard import typechecked

SummaryStatistics = namedtuple(
    "SummaryStatistics", ("n", "min", "max", "sum", "mean", "var", "sd")
)


@typechecked
class OnlineStatistics:
    """
    @class Statistics:
        A class for performing basic statistical calculations.

    @attribute min:
        The minimum value encountered so far. Initialized to positive infinity.

    @attribute max:
        The maximum value encountered so far. Initialized to negative infinity.

    @attribute n:
        The number of values encountered so far. Initialized to 0.

    @attribute sum:
        The sum of all values encountered so far. Initialized to 0.0.

    @attribute mean:
        The mean value calculated based on the encountered values. Initialized
        to 0.0.

    @attribute var:
        The variance value calculated based on the encountered values.
        Initialized to 0.0.

    @attribute sd:
        The standard deviation value calculated based on the encountered
        values. Initialized to 0.0.

    @attribute M2n:
        The intermediate value used in calculating variance. Initialized to
        0.0.

    @method update(vals: np.ndarray):
        Updates the statistics with an array of values.

    Args:
        vals (np.ndarray): An array of values to update the statistics.

    Returns:
        None.
    """

    min: float = math.inf
    max: float = -math.inf
    n: int = 0
    sum: float = 0.0
    mean: float = 0.0

    M2n: float = 0.0
    ddof: float = 1.0

    @property
    def var(self) -> float:
        return self.M2n / (self.n - self.ddof)

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    def __init__(self, unbiased: bool = True):
        if unbiased:
            self.ddof = 1  # Use unbiased variance estimator
        else:
            self.ddof = 0  # Use maximum likelihood (best L2) variance estimator

    def update(self, vals: np.ndarray) -> None:
        """
        Update the statistics with an array of values.

        For updating the variance a variant of Welford's algo is used (see e.g.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm).

        Parameters
        ----------
        vals : np.ndarray
            The array of values to update the statistics with.
        """  # noqa: E501

        self.min = min(self.min, vals.min())
        self.max = max(self.max, vals.max())
        self.n += len(vals)
        self.sum += vals.sum()
        old_mean = self.mean
        self.mean = self.sum / self.n
        self.M2n += ((vals - old_mean) * (vals - self.mean)).sum()

    def update_single(self, val):
        # Note: type checking is too tricky due to all the different numeric
        # data type in vanilla python and in numpy
        self.min = min(self.min, val)
        self.max = max(self.max, val)
        self.n += 1
        self.sum += val
        old_mean = self.mean
        self.mean = self.sum / self.n
        self.M2n += (val - old_mean) * (val - self.mean)

    def describe(self):
        return SummaryStatistics(
            self.n, self.min, self.max, self.sum, self.mean, self.var, self.sd
        )
