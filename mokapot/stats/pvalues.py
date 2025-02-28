import numpy as np
import scipy as sp
from typeguard import typechecked


@typechecked
def empirical_pvalues(
    s: np.ndarray[float | int],
    s0: np.ndarray[float | int],
    *,
    mode: str = "conservative",
) -> np.ndarray[float]:
    """
    Computes the empirical p-values for a set of values.

    Parameters
    ----------
    s : np.ndarray[float | int]
        Array of data values/test statistics (typically scores) for which
        p-values are to be computed.

    s0 : np.ndarray[float | int]
        Array of data values (scores) simulated under the null hypothesis
        against which the data values `s` are to be compared.

    mode : str
        Can be one of "unbiased", "conservative", or "storey". Default is
        "conservative".
        "unbiased" means to use $r/n$ as an estimator, where $r$ is the rank of
        the data value in the array of simulated samples. This is an unbiased
        and good when the data value is a concrete value, and not sampled.
        $conservative$ means $(r+1)/(n+1)$ which is a slightly biased, but
        conservative when the data values are sampled from the null
        distribution, in which case it is to be preferred.
        $storey$ is similar to "unbiased" but includes the "hack" for $r=0$
        found in the implementation of the `qvalue` package. It is neither
        completely unbiased nor convervative.

    Returns
    -------
    np.ndarray[float]
        Array of empirical p-values corresponding to the input data array `s`.

    References
    ----------

    .. [1] B V North, D Curtis, P C Sham, A Note on the Calculation of
       Empirical P Values from Monte Carlo Procedures,  Am J Hum Genet, 2003
       Feb;72(2):498–499. doi: 10.1086/346173
    .. [2] https://en.wikipedia.org/wiki/P-value#Definition
    """
    N = len(s0)

    # The p-value of some test statistic is the probability this or a higher
    # value would be attained under the null hypothesis, i.e. p=Pr(S>=s|H0) (see
    # [2], or p=Pr(S0>=s) if we denote by S0 the sample distribution under H0.
    # The cumulative distribution function (CDF) of S0 is given by
    # F_{S0}(s) = Pr(S0<=s).
    # Since the event S0<=s happens exactly when -S0>=-s, we see that
    # p=Pr(S0>=s)=Pr(-S0<=-s)=F_{-S0}(-s).
    # Note: some derivations out in the wild are not correct, as they don't
    # consider discrete or mixed distributions and compute the p-value via the
    # survival function SF_{S0}(s)=Pr(S0>s), which is okay for continuous
    # distributions, but problematic otherwise, if the distribution has
    # non-zero probability mass at s.
    emp_null = sp.stats.ecdf(-s0)
    p = emp_null.cdf.evaluate(-s)

    mode = mode.lower()
    if mode == "unbiased":
        return p
    elif mode == "storey":
        # Apply Storey's correction for 0 p-values
        return np.maximum(p, 1.0 / N)
    elif mode == "conservative":
        # Make p-values slightly biased, but conservative (see [1])
        return (p * N + 1) / (N + 1)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Must be either 'conservative', 'unbiased' or"
            " 'storey'."
        )
