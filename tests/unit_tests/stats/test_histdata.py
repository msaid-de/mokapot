import numpy as np
import pytest
from pytest import approx
from typeguard import TypeCheckError

from mokapot.stats.histdata import (
    HistData,
    TDHistData,
)
from mokapot.stats.statistics import OnlineStatistics


def test_hist_data_from_iterator():
    scores = np.random.uniform(-3, 10, 1127)
    targets = (np.random.uniform(0, 1, len(scores))) > 0.7

    def score_iterator(scores, targets, chunksize=5):
        for i in range(0, len(scores), chunksize):
            yield scores[i : i + chunksize], targets[i : i + chunksize]

    bin_edges = np.histogram_bin_edges(scores, bins=31)
    e0, t0, d0 = TDHistData.from_scores_targets(scores, targets, bin_edges).as_counts()
    e1, t1, d1 = TDHistData.from_score_target_iterator(
        score_iterator(scores, targets), bin_edges
    ).as_counts()
    assert e0 == approx(e1)
    assert t0 == approx(t1)
    assert d0 == approx(d1)

    bin_edges = np.histogram_bin_edges(scores, bins=17)
    e0, t0, d0 = TDHistData.from_scores_targets(
        scores, targets, bin_edges
    ).as_densities()
    e1, t1, d1 = TDHistData.from_score_target_iterator(
        score_iterator(scores, targets, chunksize=7), bin_edges
    ).as_densities()
    assert e0 == approx(e1)
    assert t0 == approx(t1)
    assert d0 == approx(d1)


def test_hist_data():
    N = 1000
    x = np.concatenate([np.random.normal(size=N), np.random.normal(2, size=N)])
    counts, bin_edges = np.histogram(x, bins="scott")
    hist = HistData(bin_edges, counts)

    density, _ = np.histogram(x, bins=bin_edges, density=True)
    assert density == approx(hist.density)

    bin_centers = hist.bin_centers
    assert len(bin_centers) == len(counts)
    assert bin_centers[0] == approx((bin_edges[0] + bin_edges[1]) / 2)
    assert bin_centers[-1] == approx((bin_edges[-2] + bin_edges[-1]) / 2)


def test_binning():
    N = 10000
    x = np.random.normal(2, 3, size=N)

    stats = OnlineStatistics()
    stats.update(x)

    assert HistData.get_bin_edges(stats, "scott") == approx(
        np.histogram_bin_edges(x, bins="scott")
    )
    assert HistData.get_bin_edges(stats, "sturges") == approx(
        np.histogram_bin_edges(x, bins="sturges")
    )
    assert HistData.get_bin_edges(stats, "auto") == approx(
        np.histogram_bin_edges(x, bins="scott")
    )

    # If we extend the bins, the new bins should have the same center, it
    # should be one bin more, and they should all have the same size
    edges0 = HistData.get_bin_edges(stats, "scott")
    edges1 = HistData.get_bin_edges(stats, "scott", extend=True)
    assert len(edges1) == len(edges0) + 1
    assert np.diff(edges1).mean() == approx(np.diff(edges0).mean())
    assert edges1.mean() == approx(edges0.mean())

    assert HistData.get_bin_edges(stats, clip=(2, 3)) == approx(
        np.histogram_bin_edges(x, bins=3)
    )
    assert HistData.get_bin_edges(stats, clip=(200, 202)) == approx(
        np.histogram_bin_edges(x, bins=200)
    )

    with pytest.raises((ValueError, TypeCheckError, TypeError)):
        HistData.get_bin_edges(stats, name="xyz")
