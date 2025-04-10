# Algorithms for confidence assignment

## TDC vs non-TDC data

In mokapot and brew_rollup you can specify whether you data comes from target decoy competition (TDC) by using the
`--tdc` switch (which is the default for backwards compatibility with old mokapot versions) or from other decoy
strategies like e.g. separate search (STDS) which you can specify by using the `--no-tdc` switch.
The only difference by now is how defaults are set for algorithm selection (i.e. for pi0-estimation, q-value
estimation and peps estimation) and whether certain algorithms are available at all (currently this affects only `ratio`
for pi0-estimation which is obviously not available for non-TDC data, since the decoy-target ratio here is completely
arbitrary).

Currently, the defaults are:

- TDC (chosen for backwards compatibility with mokapot)
    - pi0-estimation: `ratio` just the number of decoys divided by the number of targets
    - qvalue-estimation: `from_counts` basically equivalent to the old mokapot `tdc` function. However, can also work
      with histograms and other pi0's.
    - peps-estimation: `triqler` the qvality port to Python, except if streaming is enabled, then it is `hist_nnls`
- Non-TDC (chosen for (perceived) robustness and stability)
    - pi0-estimation: `bootstrap` a (non-deterministic) bootstrapping
    - qvalue-estimatiion: `storey` (NB: could just as well be `from_counts`, should check whether this is sensible)
    - peps-estimation: 'hist_nnls' histogram and non-negative least squares based algorithm

## Description of the algorithms

In the description of the algorithms $D$ usually stands for number of decoys, e.g. $D_i$ would be number of decoys in
bin $i$ in a histogram based algorithm, $p_D(s)$ for the probability density function of the decoys at score $s$,
or $\#(D>s)$ for the number of decoys larger than $s$. Same for the number of targets $T$.

### pi0 algorithms

- `ratio`: Estimate $\pi_0$ by $\#D$ over $\#T$ as in TDC the decoys can be assumed to be equally distributed as the
  false targets. Works easily also with histograms, however, not with non-TDC methods, because than $\#D$ and $\#T$ can
  be totally arbitrary.
- `slope`: for small scores $s$, the probability density of the targets can be assumed to be completely dominated by
  false targets, as true targets usually achieve higher scores. That means that $p_D(s) \propto p_T(s)$ with
  proportinality constant $\pi_0$ for small enough $s$, so that $\pi_0$ can be determined by fitting a straight line (
  i.e. linear regression) to a graph of $p_T$ over $p_D$.
- `fixed`: no estimate at all, but expressly specifying a value for $\pi_0$ everywhere.
- `bootstrap`: when sampling from the decoy density and the target density (i.e. bootstrapping) and letting both compete
  with each other, the relative number of times the "decoys win" times two is a conservative estimate of $\pi_0$,
  becoming more and more accurate the better the distributions of targets and decoys are separated.
- `storey_*`: All "Storey-based" pi0 estimates first employ the computation of p-values for all targets, then
  compute the number of p-values that are larger than some $\lambda$, say $p(\lambda)=\#(p\geq \lambda)$. Since the
  p-values of false targets follow a uniform distribution this number should be approximately $\pi_0 \#T (1-\lambda)$,
  so that we solve $\pi_0(\lambda) = p(\lambda) / (N(1-\lambda))$. The different "Storey-based" methods, differ in how
  they get an approximate $\pi_0$ from this $\pi_0(\lambda)$. Note, that a) in theory $\pi_0(\lambda)$ should become
  flat for "larger" $\lambda$ values, which in practice it often doesn't unfortunately, and b)
  that $\pi_0(\lambda)$ should be getting close to the real $\pi_0$ as $\lambda$ approaches one (bias get smaller), but
  variability gets higher. I.e. we have a bias-variance trade off that does not favor $\lambda$ very close to
  one.
- `storey_smoother`: Use the method sketched above, fit a smoothed third-order spline through $\pi_0(\lambda)$ and
  try to estimate it at some $\lambda$ value close to one. Storey uses 0.8 or 0.95 (depending on code or publication),
  while the default here is 0.5, being more conservative but also being somewhat more stable.
- `storey_fixed`: Use the method sketched above, but evaluate at a fixed $\lambda$. Currently, the
  chosen $\lambda$ is 0.5.
- `storey_bootstrap`: Evaluate at a $\lambda$ where the MSE is smallest, i.e. the bias-variance tradeoff gives the least
  mean-squared error. This was initially evaluated by bootstrapping, but got replaced by an explicit formula for the
  location of $\lambda_{MSE}$.

### qvalue algorithms

- `from_counts`: Estimate q-values from the "standard" TDC formula $\#(D\geq s)/\#(T\geq s)$ and monotonizing the
  result. This formula works out of the box for TDC. For non-TDC or TDC with different pi0 estimation (i.e. not "ratio")
  this is scaled by $\pi_0 \#T/\#D$ (which, of course, equals 1 for "ratio").
- `storey`: Estimate q-values for targets by first getting their p-values by approximating the cumulative density (CDF)
  of the decoys (as a stand-in for the null-hypothesis) and evaluating the scores of the targets with this estimated
  density.

### peps algorithms

- `qvality`: Works by storing score data in a file and running the qvality binary (compiled from C++ sources) over it.
  Obviously, can work on streamed data/histograms.
- `triqler`: Uses the Python qvality implementation triqler in order to compute peps. Does not work on streamed
  data/histograms, too. However, since internally a histogram is used for the fit, it could in principle be made
  streamable. However, this histogram is very homegrown and it's questinable whether this would be worth the effort.
  Note: The idea behind qvality/triqler is to make a smoothed third-order spline fit to $\pi_0 D_i / T_i$, where $D_i$
  and $T_i$ are the decoy and target counts in bin $i$, and determine a smoothing parameter that balances smoothness of
  the approximation with closeness to the initial estimates in a GLM fashion. Since, monotonicity is only an
  after-thought in this algorithm, and peps can go up again for large scores, it may happen that the final "fit" equals
  constant one.
- `hist_nnls`: Works on histograms. Tries to fit the pep at each bin with an estimate $\pi_0 D_i / T_i$, and doing a
  best-fitting monotonic linear interpolating on it. Since the peps are always positive and monotonically increasing
  when starting from the highest scores, this can be solved by a non-negative least-squares solver (NNLS) for the
  minimum pep and pep differences.
- `kde_nnls`: Approximate the false target density and the target density by kernel density estimation using Gaussian
  kernels. Evaluate the resulting peps $\pi_0 p_D(s_i) / p_T(s_i)$ at certain equally spaced estimation points $s_i$ and
  monotonize the result by NNLS, putting higher weights on areas with large densities, thus avoiding problems from areas
  with poor statistics for very high or very low scores, which can lead to problems e.g. in algorithms like qvality.
