from __future__ import annotations

from typing import Iterator, Literal

import numpy as np
from typeguard import typechecked

from mokapot.stats.statistics import OnlineStatistics


@typechecked
class HistData:
    """
    Representation of histogram data with utility methods for analysis.

    This class provides a structured way to store histogram data,
    including bin edges and counts, and includes methods for calculating
    bin centers, density values, and appropriate binning strategies based
    on different statistical rules.

    Attributes
    ----------
    bin_edges : np.ndarray[float]
        Array containing the edges of the histogram bins. It must have one
        more element than the `counts` attribute.
    counts : np.ndarray[int]
        Array containing the count of data points within each histogram bin.
    """

    bin_edges: np.ndarray[float]
    counts: np.ndarray[int]

    def __init__(
        self, bin_edges: np.ndarray[float], counts: np.ndarray[int] | None = None
    ) -> None:
        if counts is None:
            counts = np.zeros(len(bin_edges) - 1, dtype=int)
        elif len(bin_edges) != len(counts) + 1:
            raise ValueError(
                "`bin_edges` must have one more element than `counts` "
                f"({len(bin_edges)=}, {len(counts)=})"
            )

        self.bin_edges = bin_edges
        self.counts = counts

    def update(self, data: np.ndarray[float]) -> None:
        counts, _ = np.histogram(data, bins=self.bin_edges)
        self.counts += counts

    @property
    def bin_centers(self) -> np.ndarray[float]:
        """
        Calculates the centers of histogram bins based on bin edges.

        Given an array of bin edges, this property calculates the center points
        for each bin in the histogram by averaging the corresponding start and
        end edges of each bin. The output is a NumPy array containing the bin
        centers.

        Returns
        -------
        np.ndarray of float
            A NumPy array containing the calculated centers of the histogram bins.
        """
        return 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    @property
    def density(self) -> np.ndarray[float]:
        """
        Computes the density of the histogram.

        This property calculates the density of the histogram based on its bin
        counts and bin edges. The density is determined by normalizing the
        counts in each bin by the width of the bin and the total counts,
        providing a probability density function.

        Returns
        -------
        np.ndarray[float]
            Array of density values for each bin in the histogram.
        """
        dx = np.diff(self.bin_edges)
        counts = self.counts.astype(float)
        return counts / (dx * counts.sum())

    @staticmethod
    def bin_size_sturges(stats: OnlineStatistics) -> float:
        """
        Computes the bin size for a histogram using Sturges' rule.

        This function calculates an appropriate bin size for histogram
        construction based on Sturges' rule, which determines the number of bins
        using the logarithm of the number of data points and scales the bin size
        based on the data range.

        Not recommended: see e.g.
          https://en.wikipedia.org/wiki/Sturges%27s_rule#Criticisms

        Parameters
        ----------
        stats : OnlineStatistics
            Instance of the OnlineStatistics class that provides statistical
            information including the minimum value, maximum value, and the
            total number of data points.

        Returns
        -------
        float
            The bin size computed using Sturges' formula.
        """
        return (stats.max - stats.min) / (np.log2(stats.n) + 1.0)

    @staticmethod
    def bin_size_scott(stats: OnlineStatistics) -> float:
        """
        Calculate the bin size for a histogram using Scott's rule.

        Scott's rule is a method for selecting the bin size for a histogram. It aims
        to minimize the mean integrated squared error (MISE) by considering the
        standard deviation of the data and the number of observations. This
        method is simple to estimate and commonly used in statistical data analysis.

        Parameters
        ----------
        stats : OnlineStatistics
            An object containing statistical information about the data, including
            the standard deviation (`sd`) and the total number of observations (`n`).

        Returns
        -------
        float
            The bin size calculated using Scott's rule.
        """
        factor = (24 * 24 * np.pi) ** (1.0 / 6.0)
        return factor * stats.sd * stats.n ** (-1.0 / 3.0)

    @staticmethod
    def bin_size_terrell_scott(stats: OnlineStatistics) -> float:
        """
        Calculate the bin size using Terrell-Scott formula.

        This method computes the bin size for a histogram based on the statistical
        data provided. The Terrell-Scott rule is used, which is derived to determine
        reasonable bin sizes for accurately capturing the distribution of the data.

        Parameters
        ----------
        stats : OnlineStatistics
            An object that holds the statistical properties of the data, including
            minimum value, maximum value, and the number of samples.

        Returns
        -------
        float
            The computed bin size for the histogram, calculated using the
            Terrell-Scott formula.
        """
        num_bins = np.ceil((2.0 * stats.n) ** (1.0 / 3.0))
        return (stats.max - stats.min) / num_bins

    @staticmethod
    def get_bin_edges(
        stats: OnlineStatistics,
        name: Literal["scott", "terrell_scott", "sturges", "auto"] = "scott",
        clip: tuple[int, int] | None = None,
        extend: bool = False,
    ):
        if name == "scott":
            bin_size = HistData.bin_size_scott(stats)
        elif name == "terrell_scott":
            bin_size = HistData.bin_size_terrell_scott(stats)
        elif name == "sturges":
            bin_size = HistData.bin_size_sturges(stats)
        elif name == "auto":
            bin_size = HistData.bin_size_scott(stats)
        else:
            # Just in case that type-checking is disabled
            raise ValueError(f"Unrecognized binning algorithm name: {name}")

        range = (stats.min, stats.max)
        num_bins = int(np.ceil((range[1] - range[0]) / bin_size))
        if clip is not None:
            num_bins = np.clip(num_bins, *clip)

        if extend:
            bin_size = (range[1] - range[0]) / num_bins
            num_bins += 1
            if clip is not None:
                num_bins = np.clip(num_bins, *clip)
            range = (stats.min - 0.5 * bin_size, stats.max + 0.5 * bin_size)

        bin_edges = np.histogram_bin_edges([], bins=num_bins, range=range)
        return bin_edges


@typechecked
class TDHistData:
    """
    Represents target and decoy histogram data for statistical analysis.

    This class is used to store and manipulate histogram data specific to
    targets and decoys. It provides functionality to initialize this data
    either directly via bin edges and counts or indirectly through scores
    and targets. The data is internally represented using the HistData
    class for both target and decoy entities. It also enables access to
    the histogram as counts or densities, along with corresponding bin
    centers.

    It also makes sure, that histogram data for targets and decoys uses the
    same bin edges.

    Attributes
    ----------
    targets : HistData
        Histogram data related to target entities. It includes bin edges,
        counts, and density information for targets.
    decoys : HistData
        Histogram data related to decoy entities. It includes bin edges,
        counts, and density information for decoys.
    """

    targets: HistData
    decoys: HistData

    def __init__(
        self,
        bin_edges: np.ndarray[float],
        target_counts: np.ndarray[int] | None = None,
        decoy_counts: np.ndarray[int] | None = None,
    ):
        self.targets = HistData(bin_edges, target_counts)
        self.decoys = HistData(bin_edges, decoy_counts)

    def update(self, scores: np.ndarray[float], targets: np.ndarray[bool]) -> None:
        self.targets.update(scores[targets])
        self.decoys.update(scores[~targets])

    @staticmethod
    def from_scores_targets(
        scores: np.ndarray[float],
        targets: np.ndarray[bool],
        bin_edges: np.ndarray[float] | int | str | None = None,
    ) -> TDHistData:
        """Generate histogram data from scores and target/decoy information.

        Parameters
        ----------
        scores:
            A numpy array containing the scores for each target and decoy peptide.
        targets:
            A boolean array indicating whether each peptide is a target (True) or a
            decoy (False).
        bin_edges:
            Either: The number of bins to use for the histogram. Or: the edges of
            the bins to take. Or: None, which lets numpy determines the bins from
            all scores (which is the default).

        Returns
        -------
        TDHistData:
            A `TDHistData` object, encapsulating the histogram data.
        """
        if isinstance(bin_edges, np.ndarray):
            bin_edges = bin_edges
        else:
            bin_edges = np.histogram_bin_edges(scores, bins=bin_edges or "scott")

        td_hist_data = TDHistData(bin_edges)
        td_hist_data.update(scores, targets)
        return td_hist_data

    @staticmethod
    def from_score_target_iterator(
        score_target_iterator: Iterator, bin_edges: np.ndarray[float]
    ) -> TDHistData:
        """Generate histogram data from scores and target/decoy information
        provided by an iterator.

        This is for streaming algorithms.

        Parameters
        ----------
        score_target_iterator:
            An iterator that yields scores and target/decoy information. For each
            iteration a tuple consisting of a score array and a corresponding
            target must be yielded.
        bin_edges:
            The bins to use for the histogram. Must be provided (since they cannot
            be determined at the start of the algorithm).

        Returns
        -------
        TDHistData:
            A `TDHistData` object, encapsulating the histogram data.
        """
        td_hist_data = TDHistData(bin_edges)

        for scores, targets in score_target_iterator:
            td_hist_data.update(scores, targets)
        return td_hist_data

    def as_counts(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[int], np.ndarray[int]]:
        """Return bin centers and target and decoy counts."""
        return (
            self.targets.bin_centers,
            self.targets.counts,
            self.decoys.counts,
        )

    def as_densities(
        self,
    ) -> tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]]:
        """Return bin centers and target and decoy densities."""
        return (
            self.targets.bin_centers,
            self.targets.density,
            self.decoys.density,
        )
