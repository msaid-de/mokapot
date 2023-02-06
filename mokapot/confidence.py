"""One of the primary purposes of mokapot is to assign confidence estimates to
PSMs. This task is accomplished by ranking PSMs according to a score and using
an appropriate confidence estimation procedure for the type of data. mokapot
can provide confidence estimates based any score, regardless of whether it was
the result of a learned :py:func:`~mokapot.model.Model` instance or provided
independently.

The following classes store the confidence estimates for a dataset based on the
provided score. They provide utilities to access, save, and plot these
estimates for the various relevant levels (i.e. PSMs, peptides, and proteins).
The :py:func:`LinearConfidence` class is appropriate for most data-dependent
acquisition proteomics datasets.

We recommend using the :py:func:`~mokapot.brew()` function or the
:py:meth:`~mokapot.LinearPsmDataset.assign_confidence()` method to obtain these
confidence estimates, rather than initializing the classes below directly.
"""
import os
import glob
from pathlib import Path
import logging
import pandas as pd
import matplotlib.pyplot as plt
from triqler import qvality
from joblib import Parallel, delayed

from . import qvalues
from . import utils
from .dataset import find_best_feature
from .picked_protein import picked_protein
from .writers import to_flashlfq, to_txt
from .parsers.pin import read_file, convert_targets_column, read_file_in_chunks
from .constants import CONFIDENCE_CHUNK_SIZE

LOGGER = logging.getLogger(__name__)

# Classes ---------------------------------------------------------------------
class GroupedConfidence:
    """Perform grouped confidence estimation for a collection of PSMs.

    Groups are defined by the :py:class:`~mokapot.dataset.LinearPsmDataset`.
    Confidence estimates for each group can be retrieved by using the group
    name as an attribute, or from the
    :py:meth:`~GroupedConfidence.group_confidence_estimates` property.

    Parameters
    ----------
    psms_info : Dict
        Dict contain information about percolator input.
    scores : np.ndarray
        A vector containing the score of each PSM.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    file_root : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "{level}.txt" where "{level}" indicates the level at
        which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    combine : bool, optional
            Should groups be combined into a single file?
    Attributes
    ----------
    groups: List
    group_confidence_estimates: Dict
    """

    def __init__(
        self,
        psms_info,
        scores,
        desc=True,
        eval_fdr=0.01,
        decoys=False,
        dest_dir=None,
        file_root=None,
        sep="\t",
        proteins=None,
        combine=False,
    ):
        """Initialize a GroupedConfidence object"""
        psms = read_file(
            psms_info["file"],
            use_cols=list(psms_info["feature_columns"])
            + list(psms_info["metadata_columns"]),
        )
        self.group_column = psms_info["group_column"]
        psms_info["group_column"] = None
        scores = scores * (desc * 2 - 1)

        # Do TDC
        scores = (
            pd.Series(scores, index=psms.index).sample(frac=1).sort_values()
        )

        idx = (
            psms.loc[scores.index, :]
            .drop_duplicates(psms_info["spectrum_columns"], keep="last")
            .index
        )

        self._group_confidence_estimates = {}
        for group, group_df in psms.groupby(self.group_column):
            LOGGER.info("Group: %s == %s", self.group_column, group)
            tdc_winners = group_df.index.intersection(idx)
            group_psms = group_df.loc[tdc_winners, :]
            group_scores = scores.loc[group_psms.index].values + 1
            psms_info["file"] = "group_psms.csv"
            group_psms.to_csv(psms_info["file"], sep="\t", index=False)
            assign_confidence(
                psms_info,
                group_scores * (2 * desc - 1),
                desc=desc,
                eval_fdr=eval_fdr,
                dest_dir=dest_dir,
                file_root=file_root,
                sep=sep,
                decoys=decoys,
                proteins=proteins,
                group_column=group,
                combine=combine,
            )

    @property
    def group_confidence_estimates(self):
        """A dictionary of the confidence estimates for each group."""
        return self._group_confidence_estimates

    @property
    def groups(self):
        """The groups for confidence estimation"""
        return list(self._group_confidence_estimates.keys())

    def to_txt(
        self,
        dest_dir=None,
        file_root=None,
        sep="\t",
        decoys=False,
        combine=False,
    ):
        """Save confidence estimates to delimited text files.

        Parameters
        ----------
        dest_dir : str or None, optional
            The directory in which to save the files. `None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will be "mokapot.{level}.txt", where "{level}" indicates the level
            at which confidence estimation was performed (i.e. PSMs, peptides,
            proteins) if :code:`combine=True`. If :code:`combine=False` (the
            default), additionally the group value is prepended, yeilding a
            suffix "{group}.mokapot.{level}.txt".
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        combine : bool, optional
            Should groups be combined into a single file?

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        if combine:
            res = self.group_confidence_estimates.values()
            ret_files = to_txt(
                res,
                dest_dir=dest_dir,
                file_root=file_root,
                sep=sep,
                decoys=decoys,
            )
            return ret_files

        ret_files = []
        for group, res in self.group_confidence_estimates.items():
            prefix = file_root + f".{group}"
            new_files = res.to_txt(
                dest_dir=dest_dir, file_root=prefix, sep=sep, decoys=decoys
            )
            ret_files.append(new_files)

        return ret_files

    def __repr__(self):
        """Nice printing."""
        ngroups = len(self.group_confidence_estimates)
        lines = [
            "A mokapot.confidence.GroupedConfidence object with "
            f"{ngroups} groups:\n"
        ]

        for group, conf in self.group_confidence_estimates.items():
            lines += [f"Group: {self.group_column} == {group}"]
            lines += ["-" * len(lines[-1])]
            lines += [str(conf)]

        return "\n".join(lines)

    def __getattr__(self, attr):
        """Make groups accessible easily"""
        try:
            return self.group_confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    def __len__(self):
        """Report the number of groups"""
        return len(self.group_confidence_estimates)


class Confidence:
    """Estimate and store the statistical confidence for a collection of PSMs.

    :meta private:
    """

    _level_labs = {
        "psms": "PSMs",
        "peptides": "Peptides",
        "proteins": "Proteins",
        "csms": "Cross-Linked PSMs",
        "peptide_pairs": "Peptide Pairs",
    }

    def __init__(self, psms_info, proteins=None):
        """Initialize a PsmConfidence object."""
        self._score_column = "score"
        self._target_column = psms_info["target_column"]
        self._protein_column = "proteinIds"
        self._group_column = psms_info["group_column"]
        self._metadata_column = psms_info["metadata_columns"]

        self.scores = None
        self.targets = None
        self.qvals = None
        self.peps = None

        self._proteins = proteins

        # This attribute holds the results as DataFrames:
        self.confidence_estimates = {}
        self.decoy_confidence_estimates = {}

    def __getattr__(self, attr):
        try:
            return self.confidence_estimates[attr]
        except KeyError:
            raise AttributeError

    @property
    def levels(self):
        """
        The available levels for confidence estimates.
        """
        return list(self.confidence_estimates.keys())

    def to_txt(
        self, data_path, columns, level, decoys, file_root, dest_dir, sep
    ):
        """Save confidence estimates to delimited text files.
        Parameters
        ----------
        data_path : Path
            File of unique psms or peptides.
        level : str
            the level at which confidence estimation was performed
        dest_dir : str or None, optional
            The directory in which to save the files. `None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix
            will always be "mokapot.{level}.txt", where "{level}" indicates the
            level at which confidence estimation was performed (i.e. PSMs,
            peptides, proteins).
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?

        Returns
        -------
        list of str
            The paths to the saved files.

        """
        reader = read_file_in_chunks(
            file=data_path,
            chunk_size=CONFIDENCE_CHUNK_SIZE,
            use_cols=[i for i in columns if i != self._target_column],
        )

        self.scores = utils.create_chunks(
            self.scores, chunk_size=CONFIDENCE_CHUNK_SIZE
        )
        self.qvals = utils.create_chunks(
            self.qvals, chunk_size=CONFIDENCE_CHUNK_SIZE
        )
        self.peps = utils.create_chunks(
            self.peps, chunk_size=CONFIDENCE_CHUNK_SIZE
        )
        self.targets = utils.create_chunks(
            self.targets, chunk_size=CONFIDENCE_CHUNK_SIZE
        )

        if file_root is not None:
            dest_dir = Path(dest_dir, file_root)
        outfile_t = str(dest_dir) + f"targets.{level}"
        outfile_d = str(dest_dir) + f"decoys.{level}"

        columns.remove(self._target_column)
        output_columns = columns + ["q-value", "posterior_error_prob"]
        if level != "proteins" and self._protein_column is not None:
            output_columns.remove(self._protein_column)
            output_columns.append(self._protein_column)
        if not os.path.exists(outfile_t):
            with open(outfile_t, "w") as fp:
                fp.write(f"{sep.join(output_columns)}\n")
        if decoys and not os.path.exists(outfile_d):
            with open(outfile_d, "w") as fp:
                fp.write(f"{sep.join(output_columns)}\n")

        for data_in, score_in, qvals_in, pep_in, target_in in zip(
            reader, self.scores, self.qvals, self.peps, self.targets
        ):
            data_in = data_in.apply(pd.to_numeric, errors="ignore")
            data_in["score"] = score_in
            data_in["qvals"] = qvals_in
            data_in["PEP"] = pep_in
            if level != "proteins" and self._protein_column is not None:
                data_in[self._protein_column] = data_in.pop(
                    self._protein_column
                )
            data_in.loc[target_in, :].to_csv(
                outfile_t, sep=sep, index=False, mode="a", header=None
            )
            if decoys:
                data_in.loc[~target_in, :].to_csv(
                    outfile_d, sep=sep, index=False, mode="a", header=None
                )
        os.remove(data_path)

    def _perform_tdc(self, psms, psm_columns):
        """Perform target-decoy competition.

        Parameters
        ----------
        psms : Dataframe
            Dataframe of percolator with metadata columns [SpecId, Label, ScanNr, ExpMass, Peptide, score, Proteins].
        psm_columns : str or list of str
            The columns that define a PSM.
        """
        psm_idx = utils.groupby_max(psms, psm_columns, self._score_column)
        return psms.loc[psm_idx, :]

    def plot_qvalues(self, level="psms", threshold=0.1, ax=None, **kwargs):
        """Plot the cumulative number of discoveries over range of q-values.

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
        qvals = self.confidence_estimates[level]["mokapot q-value"]
        if qvals is None:
            raise ValueError(f"{level}-level estimates are unavailable.")

        ax = plot_qvalues(qvals, threshold=threshold, ax=ax, **kwargs)
        ax.set_xlabel("q-value")
        ax.set_ylabel(f"Accepted {self._level_labs[level]}")

        return ax


class LinearConfidence(Confidence):
    """Assign confidence estimates to a set of PSMs

    Estimate q-values and posterior error probabilities (PEPs) for PSMs and
    peptides when ranked by the provided scores.

    Parameters
    ----------
    psms_info : Dict
        Dict contain information about percolator input.
    psms_path : Path
            File with unique psms.
    peptides_path : Path
            File with unique peptides.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report performance. This parameter
        has no affect on the analysis itself, only logging messages.
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    file_root : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "{level}.txt" where "{level}" indicates the level at
        which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    combine : bool, optional
            Should groups be combined into a single file?
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.

    """

    def __init__(
        self,
        psms_info,
        psms_path,
        peptides_path,
        desc=True,
        eval_fdr=0.01,
        dest_dir=None,
        file_root=None,
        decoys=None,
        deduplication=True,
        proteins=None,
        sep="\t",
        group_column=None,
        combine=False,
    ):
        """Initialize a a LinearPsmConfidence object"""
        super().__init__(psms_info, proteins)
        self._target_column = psms_info["target_column"]
        self._psm_columns = psms_info["spectrum_columns"]
        self._peptide_column = psms_info["peptide_column"]
        self._protein_column = "proteinIds"
        self._eval_fdr = eval_fdr
        self.deduplication = deduplication

        self._assign_confidence(
            psms_path,
            peptides_path,
            desc=desc,
            dest_dir=dest_dir,
            file_root=file_root,
            decoys=decoys,
            sep=sep,
            group_column=group_column,
            combine=combine,
        )

        self.accepted = {}
        for level in self.levels:
            self.accepted[level] = self._num_accepted(level)

    def __repr__(self):
        """How to print the class"""
        base = (
            "A mokapot.confidence.LinearConfidence object:\n"
            f"\t- PSMs at q<={self._eval_fdr:g}: {self.accepted['psms']}\n"
            f"\t- Peptides at q<={self._eval_fdr:g}: "
            f"{self.accepted['peptides']}\n"
        )

        if self._proteins:
            base += (
                f"\t- Protein groups at q<={self._eval_fdr:g}: "
                f"{self.accepted['proteins']}\n"
            )

        return base

    def _num_accepted(self, level):
        """Calculate the number of accepted discoveries"""
        disc = self.confidence_estimates[level]
        if disc is not None:
            return (disc["q-value"] <= self._eval_fdr).sum()
        else:
            return None

    def _assign_confidence(
        self,
        psms_path,
        peptides_path,
        desc=True,
        decoys=False,
        file_root=None,
        dest_dir=None,
        sep="\t",
        group_column=None,
        combine=False,
    ):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        psms_path : Path
            File with unique psms.
        peptides_path : Path
            File with unique peptides.
        desc : bool
            Are higher scores better?
        dest_dir : str or None, optional
            The directory in which to save the files. :code:`None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix will
            always be "{level}.txt" where "{level}" indicates the level at
            which confidence estimation was performed (i.e. PSMs, peptides,
            proteins).
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        combine : bool, optional
                Should groups be combined into a single file?
        group_column : str, optional
            A factor to by which to group PSMs for grouped confidence
            estimation.
        """
        levels = ["PSMs"]
        level_data_path = [psms_path]

        if self.deduplication:
            levels.append("peptides")
            level_data_path.append(peptides_path)

        if self._proteins:
            data = read_file(peptides_path)
            data = data.apply(pd.to_numeric, errors="ignore")
            convert_targets_column(
                data=data, target_column=self._target_column
            )
            proteins = picked_protein(
                data,
                self._target_column,
                self._peptide_column,
                self._score_column,
                self._proteins,
            )
            proteins_path = "proteins.csv"
            proteins.to_csv(proteins_path, index=False, sep=sep)
            levels += ["proteins"]
            level_data_path += [proteins_path]
            LOGGER.info("\t- Found %i unique protein groups.", len(proteins))

        for level, data_path in zip(levels, level_data_path):
            data = read_file(data_path)
            data = data.apply(pd.to_numeric, errors="ignore")
            data_columns = list(data.columns)
            convert_targets_column(
                data=data, target_column=self._target_column
            )
            self.scores = data.loc[:, self._score_column].values
            self.targets = data.loc[:, self._target_column].astype(bool).values
            del data
            if all(self.targets):
                LOGGER.warning(
                    "No decoy PSMs remain for confidence estimation. "
                    "Confidence estimates may be unreliable."
                )

            # Estimate q-values and assign to dataframe
            LOGGER.info("Assiging q-values to %s...", level)
            self.qvals = qvalues.tdc(self.scores, self.targets, desc=True)

            # Set scores to be the correct sign again:
            self.scores = self.scores * (desc * 2 - 1)
            # Logging update on q-values
            LOGGER.info(
                "\t- Found %i %s with q<=%g",
                (self.qvals[self.targets] <= self._eval_fdr).sum(),
                level,
                self._eval_fdr,
            )

            # Calculate PEPs
            LOGGER.info("Assiging PEPs to %s...", level)
            try:
                _, self.peps = qvality.getQvaluesFromScores(
                    self.scores[self.targets],
                    self.scores[~self.targets],
                    includeDecoys=True,
                )
            except SystemExit as msg:
                if "no decoy hits available for PEP calculation" in str(msg):
                    self.peps = 0
                else:
                    raise

            logging.info(f"Writing {level} results...")
            if group_column and not combine:
                file_root = f"{group_column}."
                self.to_txt(
                    data_path,
                    data_columns,
                    level.lower(),
                    decoys,
                    file_root,
                    dest_dir,
                    sep,
                )
            else:
                self.to_txt(
                    data_path,
                    data_columns,
                    level.lower(),
                    decoys,
                    file_root,
                    dest_dir,
                    sep,
                )

    def to_flashlfq(self, out_file="mokapot.flashlfq.txt"):
        """Save confidenct peptides for quantification with FlashLFQ.

        `FlashLFQ <https://github.com/smith-chem-wisc/FlashLFQ>`_ is an
        open-source tool for label-free quantification. For mokapot to save
        results in a compatible format, a few extra columns are required to
        be present, which specify the MS data file name, the theoretical
        peptide monoisotopic mass, the retention time, and the charge for each
        PSM. If these are not present, saving to the FlashLFQ format is
        disabled.

        Note that protein grouping in the FlashLFQ results will be more
        accurate if proteins were added for analysis with mokapot.

        Parameters
        ----------
        out_file : str, optional
            The output file to write.

        Returns
        -------
        str
            The path to the saved file.

        """
        return to_flashlfq(self, out_file)


class CrossLinkedConfidence(Confidence):
    """
    Assign confidence estimates to a set of cross-linked PSMs

    Estimate q-values and posterior error probabilities (PEPs) for
    cross-linked PSMs (CSMs) and the peptide pairs when ranked by the
    provided scores.

    Parameters
    ----------
    psms_path : Path
            File with unique psms.
    peptides_path : Path
            File with unique peptides.
    psms_info : Dict
        Dict contain information about percolator input.
    desc : bool
        Are higher scores better?

    Attributes
    ----------
    csms : pandas.DataFrame
        Confidence estimates for cross-linked PSMs in the dataset.
    peptide_pairs : pandas.DataFrame
        Confidence estimates for peptide pairs in the dataset.

    :meta private:
    """

    def __init__(
        self,
        psms_info,
        psms_path,
        peptides_path,
        desc=True,
        dest_dir=None,
        file_root=None,
        decoys=None,
        sep="\t",
        group_column=None,
        combine=False,
    ):
        """Initialize a CrossLinkedConfidence object"""
        super().__init__(psms_info)
        self._target_column = psms_info["target_column"]
        self._psm_columns = psms_info["spectrum_columns"]
        self._peptide_column = psms_info["peptide_column"]

        self._assign_confidence(
            psms_path,
            peptides_path,
            desc=desc,
            dest_dir=dest_dir,
            file_root=file_root,
            decoys=decoys,
            sep=sep,
            group_column=group_column,
            combine=combine,
        )

    def _assign_confidence(
        self,
        psms_path,
        peptides_path,
        desc=True,
        decoys=False,
        file_root=None,
        dest_dir=None,
        sep="\t",
        group_column=None,
        combine=False,
    ):
        """
        Assign confidence to PSMs and peptides.

        Parameters
        ----------
        psms_path : Path
            File with unique psms.
        peptides_path : Path
            File with unique peptides.
        desc : bool
            Are higher scores better?
        dest_dir : str or None, optional
            The directory in which to save the files. :code:`None` will use the
            current working directory.
        file_root : str or None, optional
            An optional prefix for the confidence estimate files. The suffix will
            always be "{level}.txt" where "{level}" indicates the level at
            which confidence estimation was performed (i.e. PSMs, peptides,
            proteins).
        sep : str, optional
            The delimiter to use.
        decoys : bool, optional
            Save decoys confidence estimates as well?
        combine : bool, optional
                Should groups be combined into a single file?
        group_column : str, optional
            A factor to by which to group PSMs for grouped confidence
            estimation.
        """

        levels = ("csms", "peptide_pairs")

        for level, data_path in zip(levels, [psms_path, peptides_path]):
            data = read_file(
                data_path, use_cols=self._metadata_column + ["score"]
            )
            data = data.apply(pd.to_numeric, errors="ignore")
            convert_targets_column(
                data=data, target_column=self._target_column
            )
            self.scores = data.loc[:, self._score_column].values
            self.targets = data.loc[:, self._target_column].astype(bool).values
            self.qvals = qvalues.crosslink_tdc(self.scores, self.targets, desc)

            _, self.peps = qvality.getQvaluesFromScores(
                self.scores[self.targets == 2], self.scores[~self.targets]
            )
            logging.info(f"Writing {level} results...")
            if group_column and not combine:
                file_root = f"{group_column}."
                self.to_txt(
                    data_path, level.lower(), decoys, file_root, dest_dir, sep
                )
            else:
                self.to_txt(
                    data_path, level.lower(), decoys, file_root, dest_dir, sep
                )


# Functions -------------------------------------------------------------------
def assign_confidence(
    psms_info,
    scores=None,
    desc=True,
    eval_fdr=0.01,
    dest_dir=None,
    file_root=None,
    sep="\t",
    decoys=False,
    deduplication=True,
    proteins=None,
    group_column=None,
    combine=False,
):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Parameters
    ----------
    psms_info : dict
        All info about the input data
    scores : numpy.ndarray
        The scores by which to rank the PSMs. The default, :code:`None`,
        uses the feature that accepts the most PSMs at an FDR threshold of
        `eval_fdr`.
    desc : bool
        Are higher scores better?
    eval_fdr : float
        The FDR threshold at which to report and evaluate performance. If
        `scores` is not :code:`None`, this parameter has no affect on the
        analysis itself, but does affect logging messages and the FDR
        threshold applied for some output formats, such as FlashLFQ.
    dest_dir : str or None, optional
        The directory in which to save the files. :code:`None` will use the
        current working directory.
    file_root : str or None, optional
        An optional prefix for the confidence estimate files. The suffix will
        always be "{level}.txt" where "{level}" indicates the level at
        which confidence estimation was performed (i.e. PSMs, peptides,
        proteins).
    sep : str, optional
        The delimiter to use.
    decoys : bool, optional
        Save decoys confidence estimates as well?
    combine : bool, optional
            Should groups be combined into a single file?
    group_column : str, optional
        A factor to by which to group PSMs for grouped confidence
        estimation.

    Returns
    -------
    LinearConfidence
        A :py:class:`~mokapot.confidence.LinearConfidence` object storing
        the confidence estimates for the collection of PSMs.
    """
    if scores is None:
        feat, _, _, desc = find_best_feature(psms_info, eval_fdr)
        LOGGER.info("Selected %s as the best feature.", feat)
        scores = read_file(file_name=psms_info["file"], use_cols=[feat]).values

    if psms_info["group_column"] is None:
        reader = read_file_in_chunks(
            file=psms_info["file"],
            chunk_size=CONFIDENCE_CHUNK_SIZE,
            use_cols=psms_info["metadata_columns"],
        )
        scores_slices = utils.create_chunks(
            scores, chunk_size=CONFIDENCE_CHUNK_SIZE
        )

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(save_sorted_metadata_chunks)(
                chunk_metadata,
                score_chunk,
                psms_info,
                deduplication,
                i,
                sep,
            )
            for chunk_metadata, score_chunk, i in zip(
                reader, scores_slices, range(len(scores_slices))
            )
        )

        psms_path = "psms.csv"
        peptides_path = "peptides.csv"
        scores_metadata_paths = glob.glob("scores_metadata_*")
        iterable_sorted = utils.merge_sort(
            scores_metadata_paths, col_score="score", sep=sep
        )
        LOGGER.info("Assigning confidence...")
        LOGGER.info("Performing target-decoy competition...")
        LOGGER.info(
            "Keeping the best match per %s columns...",
            "+".join(psms_info["spectrum_columns"]),
        )
        metadata_columns = ["PSMId", "Label", "Peptide", "proteinIds", "score"]
        with open(psms_path, "w") as f_psm:
            f_psm.write(f"{sep.join(metadata_columns)}\n")

        if deduplication:
            with open(peptides_path, "w") as f_peptide:
                f_peptide.write(f"{sep.join(metadata_columns)}\n")

            unique_psms, unique_peptides = utils.get_unique_psms_and_peptides(
                iterable=iterable_sorted,
                out_psms="psms.csv",
                out_peptides="peptides.csv",
                sep=sep,
            )
            LOGGER.info("\t- Found %i PSMs from unique spectra.", unique_psms)
            LOGGER.info("\t- Found %i unique peptides.", unique_peptides)
        else:
            n_psms = 0
            for row in iterable_sorted:
                n_psms += 1
                with open(psms_path, "a") as f_psm:
                    f_psm.write(
                        sep.join([row[0], row[1], row[-3], row[-2], row[-1]])
                    )
            LOGGER.info("\t- Found %i PSMs.", n_psms)

        [os.remove(sc_path) for sc_path in scores_metadata_paths]

        return LinearConfidence(
            psms_info=psms_info,
            psms_path=psms_path,
            peptides_path=peptides_path,
            eval_fdr=eval_fdr,
            desc=desc,
            dest_dir=dest_dir,
            file_root=file_root,
            sep=sep,
            decoys=decoys,
            deduplication=deduplication,
            proteins=proteins,
            group_column=group_column,
            combine=combine,
        )
    else:
        LOGGER.info("Assigning confidence within groups...")
        return GroupedConfidence(
            psms_info,
            scores,
            eval_fdr=eval_fdr,
            desc=desc,
            dest_dir=dest_dir,
            file_root=file_root,
            sep=sep,
            decoys=decoys,
            proteins=proteins,
            combine=combine,
        )


def save_sorted_metadata_chunks(
    chunk_metadata, score_chunk, psms_info, deduplication, i, sep
):
    chunk_metadata["score"] = score_chunk
    chunk_metadata.sort_values(by="score", ascending=False, inplace=True)
    if deduplication:
        chunk_metadata = chunk_metadata.drop_duplicates(
            psms_info["spectrum_columns"]
        )
    chunk_metadata.to_csv(
        f"scores_metadata_{i}.csv",
        sep=sep,
        index=False,
        mode="w",
    )


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

    zero = pd.DataFrame({"qvalue": qvals["qvalue"][0], "num": 0}, index=[-1])
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

    ax.step(qvals["qvalue"].values, qvals.num.values, where="post", **kwargs)

    return ax


def _new_column(name, df):
    """Add a new column, ensuring a unique name"""
    new_name = name
    cols = set(df.columns)
    i = 0
    while new_name in cols:
        new_name = name + "_" + str(i)
        i += 1

    return new_name
