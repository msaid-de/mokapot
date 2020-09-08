"""
One of the primary purposes of mokapot is to assign confidence estimates to PSMs.
This task is accomplished by ranking PSMs according to a score or metric and
using an appropriate confidence estimation procedure for the type of data
(currently, linear and cross-linked PSMs are supported). In either case,
mokapot can provide confidence estimates based any score, regardless of
whether it was the result of a learned :py:func:`mokapot.model.Model`
instance or provided independently.

The following classes store the confidence estimates for a dataset based on the
provided score. In either case, they provide utilities to access, save, and
plot these estimates for the various relevant levels (i.e. PSMs, peptides, and
proteins). The :py:func:`LinearConfidence` class is appropriate for most
proteomics datasets, whereas the :py:func:`CrossLinkedConfidence` is
specifically designed for crosslinked peptides.
"""
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality

from . import qvalues

# Import dask if available:
try:
    import dask.array as da
except ImportError:
    da = None


LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class Confidence():
    """
    Estimate and store the statistical confidence for a collection of
    PSMs.

    :meta private:
    """
    _level_labs = {"psms": "PSMs",
                   "peptides": "Peptides",
                   "proteins": "Proteins",
                   "csms": "Cross-Linked PSMs",
                   "peptide_pairs": "Peptide Pairs"}

    def __init__(self, psms, scores, desc):
        """
        Initialize a PsmConfidence object.
        """
        self._data = psms.metadata
        self._score_column = _new_column("score", self._data)

        # Flip sign of scores if not descending
        self._data = _assign(self._data, self._score_column,
                             scores * (desc*2 - 1))

        # This attribute holds the results as DataFrames:
        self._confidence_estimates = {}

    def __getattr__(self, attr):
        try:
            return self._confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    @property
    def levels(self):
        """
        The available levels for confidence estimates.
        """
        return list(self._confidence_estimates.keys())

    def to_txt(self, dest_dir=None, file_root=None, sep="\t"):
        """
        Save confidence estimates to delimited text files.

        Parameters
        ----------
        dest_dir : str or None, optional
            The directory in which to save the files. The default is the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files.
        sep : str
            The delimiter to use.

        Returns
        -------
        list of str
            The paths to the saved files.
        """
        file_base = "mokapot"
        if file_root is not None:
            file_base = file_root + "." + file_base
        if dest_dir is not None:
            file_base = os.path.join(dest_dir, file_base)

        out_files = []
        for level, qvals in self._confidence_estimates.items():
            out_file = file_base + f".{level}.txt"
            try:
                # Dask
                qvals.to_csv(out_file, single_file=True, sep=sep, index=False)
            except TypeError:
                # Pandas
                qvals.to_csv(out_file, sep=sep, index=False)

            out_files.append(out_file)

        return out_files

    def _perform_tdc(self, psm_columns):
        """
        Perform target-decoy competition.

        Parameters
        ----------
        psm_columns : str or list of str
            The columns that define a PSM.
        """
        psm_idx = _groupby_max(self._data, psm_columns, self._score_column)
        self._data = self._data.loc[psm_idx, :]


    def plot_qvalues(self, level, threshold=0.1, ax=None, **kwargs):
        """
        Plot the cumulative number of discoveries over range of q-values.

        The available levels can be found using
        :py:meth:`~mokapot.confidence.Confidence.levels` attribute.

        Parameters
        ----------
        level : str, optional
            The level of q-values to report.
        threshold : float, optional
            Indicates the maximum q-value to plot.
        ax : matplotlib.pyplot.Axes, optional
            The matplotlib Axes on which to plot. If `None` the current
            Axes instance is used.
        **kwargs : dict, optional
            Arguments passed to :py:func:`matplotlib.pyplot.plot`.

        Returns
        -------
        matplotlib.pyplot.Axes
            An :py:class:`matplotlib.axes.Axes` with the cumulative
            number of accepted target PSMs or peptides.
        """
        ax = plot_qvalues(self._confidence_estimates[level]["mokapot q-value"],
                          threshold=threshold, ax=ax, **kwargs)
        ax.set_xlabel("q-value")
        ax.set_ylabel(f"Accepted {self._level_labs[level]}")

        return ax


class LinearConfidence(Confidence):
    """
    Assign confidence estimates to a set of PSMs

    Estimate q-values and posterior error probabilities (PEPs) for PSMs
    and peptides when ranked by the provided scores.

    Parameters
    ----------
    psms : LinearPsmDataset object
        A collection of PSMs.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.

    Attributes
    ----------
    levels : list of str
    psms : pandas.DataFrame
        Confidence estimates for PSMs in the dataset.
    peptides : pandas.DataFrame
        Confidence estimates for peptides in the dataset.
    """
    def __init__(self, psms, scores, desc=True, eval_fdr=0.01):
        """Initialize a a LinearPsmConfidence object"""
        LOGGER.info("=== Assigning Confidence ===")
        super().__init__(psms, scores, desc)
        self._target_column = _new_column("target", self._data)
        self._data[self._target_column] = psms.targets
        self._psm_columns = psms._spectrum_columns
        self._peptide_column = psms._peptide_column
        self._eval_fdr = eval_fdr

        # Set an index:
        LOGGER.info("Performing target-decoy competition...")
        LOGGER.info("Keeping the best match per %s columns...",
                    "+".join(self._psm_columns))

        self._perform_tdc(self._psm_columns)
        LOGGER.info("  - Found %i PSMs from unique spectra.",
                    len(self._data))

        self._assign_confidence(desc=desc)

    def __repr__(self):
        """How to print the class"""
        pass_psms = np.array(self.psms["mokapot q-value"]
                             <= self._eval_fdr).sum()
        pass_peps = np.array(self.peptides["mokapot q-value"]
                             <= self._eval_fdr).sum()

        return ("A mokapot.confidence.LinearConfidence object:\n"
                f"\t- PSMs at q<={self._eval_fdr:g}: {pass_psms}\n"
                f"\t- Peptides at q<={self._eval_fdr:g}: {pass_peps}")

    def _assign_confidence(self, desc=True):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        desc : bool
            Are higher scores better?
        """
        peptide_idx = _groupby_max(self._data, self._peptide_column,
                                   self._score_column)

        peptides = self._data.loc[peptide_idx, :]
        LOGGER.info("\t- Found %i unique peptides.", len(peptides))

        for level, data in zip(("PSMs", "peptides"), (self._data, peptides)):
            scores = data.loc[:, self._score_column].values
            targets = data.loc[:, self._target_column].astype(bool).values

            # Estimate q-values and assign to dataframe
            LOGGER.info("Assiging q-values to %s.", level)
            qvals = qvalues.tdc(scores, targets, desc=True)
            data = _assign(data, "mokapot q-value", qvals)

            # Make the table prettier
            not_target_cols = [c for c in data.columns
                               if c != self._target_column]
            data = (data.loc[targets, :]
                        .reset_index(drop=True)
                        .loc[:, not_target_cols]
                        .rename(columns={self._score_column: "mokapot score"}))

            # Set scores to be the correct sign again:
            data["mokapot score"] = data["mokapot score"] * (desc*2 - 1)

            # A nice logging update.
            pass_targets = (qvals[targets] <= self._eval_fdr).sum()
            LOGGER.info("  - Found %i %s with q<=%g", pass_targets,
                        level, self._eval_fdr)

            # Calculate PEPs
            LOGGER.info("Assiging PEPs to %s.", level)
            targ_scores = np.array(scores[targets])
            targ_idx = np.argsort(-targ_scores)

            dec_scores = np.array(scores[~targets])
            dec_idx = np.argsort(-dec_scores)

            _, pep = qvality.getQvaluesFromScores(targ_scores[targ_idx],
                                                  dec_scores[dec_idx])

            # Assign PEPs to dataframe
            data = _assign(data, "mokapot PEP", pep[np.argsort(targ_idx)])

            # Sort values
            try:
                data = data.sort_values("mokapot score", ascending=(not desc))
            except AttributeError:
                logging.info("Results are not sorted when using the "
                             "dask backend.")

            self._confidence_estimates[level.lower()] = data


class CrossLinkedConfidence(Confidence):
    """
    Assign confidence estimates to a set of cross-linked PSMs

    Estimate q-values and posterior error probabilities (PEPs) for
    cross-linked PSMs (CSMs) and the peptide pairs when ranked by the
    provided scores.

    Parameters
    ----------
    psms : CrossLinkedPsmDataset object
        A collection of cross-linked PSMs.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?

    Attributes
    ----------
    csms : pandas.DataFrame
        Confidence estimates for cross-linked PSMs in the dataset.
    peptide_pairs : pandas.DataFrame
        Confidence estimates for peptide pairs in the dataset.
    """
    def __init__(self, psms, scores, desc=True):
        """Initialize a CrossLinkedConfidence object"""
        super().__init__(psms, scores)
        self._data[len(self._data.columns)] = psms.targets
        self._target_column = self._data.columns[-1]
        self._psm_columns = psms._spectrum_columns
        self._peptide_column = psms._peptide_column

        self._perform_tdc(self._psm_columns)
        self._assign_confidence(desc=desc)

    def _assign_confidence(self, desc=True):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        desc : bool
            Are higher scores better?
        """
        peptide_idx = _groupby_max(self._data, self._peptide_columns,
                                   self._score_column)

        peptides = self._data.loc[peptide_idx]
        levels = ("csms", "peptide_pairs")

        for level, data in zip(levels, (self._data, peptides)):
            scores = data.loc[:, self._score_column].values
            targets = data.loc[:, self._target_column].astype(bool).values
            data["mokapot q-value"] = qvalues.crosslink_tdc(scores, targets,
                                                            desc)

            data = data.loc[targets, :] \
                       .sort_values(self._score_column, ascending=(not desc)) \
                       .reset_index(drop=True) \
                       .drop(self._target_column, axis=1) \
                       .rename(columns={self._score_column: "mokapot score"})

            _, pep = qvality.getQvaluesFromScores(scores[targets == 2],
                                                  scores[~targets])
            data["mokapot PEP"] = pep
            self._confidence_estimates[level] = data


# Functions -------------------------------------------------------------------
def plot_qvalues(qvalues, threshold=0.1, ax=None, **kwargs):
    """
    Plot the cumulative number of discoveries over range of q-values.

    Parameters
    ----------
    qvalues : numpy.ndarray
        The q-values to plot.
    threshold : float, optional
        Indicates the maximum q-value to plot.
    ax : matplotlib.pyplot.Axes, optional
        The matplotlib Axes on which to plot. If `None` the current
        Axes instance is used.
    **kwargs : dict, optional
        Arguments passed to :py:func:`matplotlib.axes.Axes.plot`.

    Returns
    -------
    matplotlib.pyplot.Axes
        An :py:class:`matplotlib.axes.Axes` with the cumulative
        number of accepted target PSMs or peptides.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate cumulative targets at each q-value
    qvals = pd.Series(qvalues, name="qvalue")
    qvals = qvals.sort_values(ascending=True).to_frame()
    qvals["target"] = 1
    qvals["num"] = qvals["target"].cumsum()
    qvals = qvals.groupby(["qvalue"]).max().reset_index()
    qvals = qvals[["qvalue", "num"]]

    zero = pd.DataFrame({"qvalue": qvals["qvalue"][0],
                        "num": 0}, index=[-1])
    qvals = pd.concat([zero, qvals], sort=True).reset_index(drop=True)

    xmargin = threshold * 0.05
    ymax = qvals.num[qvals["qvalue"] <= (threshold + xmargin)].max()
    ymargin = ymax * 0.05

    # Set margins
    curr_ylims = ax.get_ylim()
    if curr_ylims[1] < ymax + ymargin:
        ax.set_ylim(0 - ymargin, ymax + ymargin)

    ax.set_xlim(0 - xmargin, threshold + xmargin)
    ax.set_xlabel("q-value")
    ax.set_ylabel(f"Discoveries")

    ax.step(qvals["qvalue"].values,
            qvals.num.values, where="post", **kwargs)

    return ax


# Utility Functions -----------------------------------------------------------
def _groupby_max(df, by_cols, max_col):
    """
    Quickly get the indices for the maximum value of col.

    Here the sampling is needed to ensure ties are broken
    randomly. Unfortunatly, this makes the dask backend
    very slow.
    """
    # Sometimes we can skip this (such as when looking at PSMs searched)
    # using a concatenated database:
    if len(df) == len(df.drop_duplicates(list(by_cols))):
        return df.index

    try:
        #raise AttributeError
        # This is much faster for smallish pandas dataframes:
        idx = (df.sort_values(list(by_cols)+[max_col], axis=0)
                 .drop_duplicates(list(by_cols), keep="last")
                 .index)
    except AttributeError:
        # Dask does not have 'sort_values':
        idx = (df.loc[:, list(by_cols)+[max_col]]
                 .groupby(list(by_cols))
                 .idxmax()
                 .values
                 .compute()
                 .flatten())

    return idx


def _assign(df, name, vals):
    """Assign a new column"""
    try:
        df[name] = vals
    except TypeError:
        # Assumes partitions are sized evenly.
        chunks = len(vals) // df.npartitions
        vals = da.from_array(vals, chunks=chunks)
        df[name] = vals

    return df


def _new_column(name, df):
    """Add a new column, ensuring a unique name"""
    new_name = name
    cols = set(df.columns)
    i = 0
    while new_name in cols:
        new_name = name + "_" + str(i)
        i += 1

    return new_name
