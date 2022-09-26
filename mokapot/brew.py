"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from .model import PercolatorModel
from . import utils
from .dataset import LinearPsmDataset
from .confidence import (
    LinearConfidence,
    GroupedConfidence,
)

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def brew(
    psms,
    model=None,
    test_fdr=0.01,
    folds=3,
    max_workers=1,
    subset_max_train=None,
):
    """
    Re-score one or more collection of PSMs.

    The provided PSMs analyzed using the semi-supervised learning
    algorithm that was introduced by
    `Percolator <http://percolator.ms>`_. Cross-validation is used to
    ensure that the learned models to not overfit to the PSMs used for
    model training. If a multiple collections of PSMs are provided, they
    are aggregated for model training, but the confidence estimates are
    calculated separately for each collection.

    Parameters
    ----------
    psms : PsmDataset object or list of PsmDataset objects
        One or more :doc:`collections of PSMs <dataset>` objects.
        PSMs are aggregated across all of the collections for model
        training, but the confidence estimates are calculated and
        returned separately.
    model: Model object, optional
        The :py:class:`mokapot.Model` object to be fit. The default is
        :code:`None`, which attempts to mimic the same support vector
        machine models used by Percolator.
    test_fdr : float, optional
        The false-discovery rate threshold at which to evaluate
        the learned models.
    folds : int, optional
        The number of cross-validation folds to use. PSMs originating
        from the same mass spectrum are always in the same fold.
    max_workers : int, optional
        The number of processes to use for model training. More workers
        will require more memory, but will typically decrease the total
        run time. An integer exceeding the number of folds will have
        no additional effect. Note that logging messages will be garbled
        if more than one worker is enabled.
    subset_max_train : int or None, optional
        Use only a random subset of the PSMs for training. This is useful
        for very large datasets or models that scale poorly with the
        number of PSMs. The default, :code:`None` will use all of the
        PSMs.

    Returns
    -------
    Confidence object or list of Confidence objects
        An object or a list of objects containing the
        :doc:`confidence estimates <confidence>` at various levels
        (i.e. PSMs, peptides) when assessed using the learned score.
        If a list, they will be in the same order as provided in the
        `psms` parameter.
    list of Model objects
        The learned :py:class:`~mokapot.model.Model` objects, one
        for each fold.
    """
    if model is None:
        model = PercolatorModel()

    try:
        iter(psms)
    except TypeError:
        psms = [psms]

    # Check that all of the datasets have the same features:
    # feat_set = set(psms[0].features.columns)
    # if not all([set(p.features.columns) == feat_set for p in psms]):
    # raise ValueError("All collections of PSMs must use the same features.")

    cols = psms["spectrum_columns"]
    df_spectra = pd.concat(
        [
            pd.read_csv(pin_file, sep="\t", usecols=cols).reset_index(
                drop=True
            )
            for pin_file in psms["files"]
        ]
    )

    if len(df_spectra) > 1:
        LOGGER.info("")
        LOGGER.info("Found %i total PSMs.", len(df_spectra))

    LOGGER.info("Splitting PSMs into %i folds...", folds)
    test_idx = [_split(df_spectra, folds)]
    train_sets = make_train_sets(
        psms=psms,
        test_idx=test_idx,
        subset_max_train=subset_max_train,
        data_size=len(df_spectra),
    )

    if max_workers != 1:
        # train_sets can't be a generator for joblib :(
        train_sets = list(train_sets)

    train_psms = _parse_in_chunks(psms=psms, train_idx=train_sets)

    if type(model) is list:
        models = [[m, False] for m in model]
    else:
        models = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(_fit_model)(d, psms, copy.deepcopy(model), f)
            for f, d in enumerate(train_psms)
        )
    del train_psms

    # sort models to have deterministic results with multithreading.
    # Only way I found to sort is using intercept values
    models.sort(key=lambda x: x[0].estimator.intercept_)
    # Determine if the models need to be reset:
    reset = any([m[1] for m in models])
    if reset:
        # If we reset, just use the original model on all the folds:
        scores = [
            p._calibrate_scores(model.predict(p), test_fdr) for p in psms
        ]
    elif all([m[0].is_trained for m in models]):
        # If we don't reset, assign scores to each fold:
        models = [m for m, _ in models]
        scores = [_predict(p, psms, models, test_fdr) for p in test_idx]
    else:
        # If model training has failed
        scores = [np.zeros(len(p.data)) for p in psms]
    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if type(model) is not list and not model.override:
        best_feats = [find_best_feature(p, test_fdr) for p in [psms]]
        feat_total = sum([best_feat[1] for best_feat in best_feats])
    else:
        feat_total = 0

    preds = [_update_labels(p, s, test_fdr) for p, s in zip([psms], scores)]
    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        scores = []
        descs = []
        for dat, (feat, _, _, desc) in zip(psms["files"], best_feats):
            df = pd.read_csv(dat, sep="\t", usecols=[feat])
            scores.append(df.values)
            descs.append(desc)

    else:
        using_best_feat = False
        descs = [True] * len(psms)

    if using_best_feat:
        logging.warning(
            "Learned model did not improve over the best feature. "
            "Now scoring by the best feature for each collection "
            "of PSMs."
        )
    elif reset:
        logging.warning(
            "Learned model did not improve upon the pretrained "
            "input model. Now re-scoring each collection of PSMs "
            "using the original model."
        )

    LOGGER.info("")
    res = [
        assign_confidence(p, s, eval_fdr=test_fdr, desc=d)
        for p, s, d in zip([psms], scores, descs)
    ]

    if len(res) == 1:
        return res[0], models

    return res, models


# Utility Functions -----------------------------------------------------------
def _split(data, folds):
    """
    Get the indices for random, even splits of the dataset.

    Each tuple of integers contains the indices for a random subset of
    PSMs. PSMs are grouped by spectrum, such that all PSMs from the same
    spectrum only appear in one split. The typical use for this method
    is to split the PSMs into cross-validation folds.

    Parameters
    ----------
    folds: int
        The number of splits to generate.

    Returns
    -------
    A tuple of tuples of ints
        Each of the returned tuples contains the indices  of PSMs in a
        split.
    """
    scans = list(data.groupby(list(data.columns), sort=False).indices.values())
    # or indices in scans:
    # np.random.shuffle(indices)
    np.random.shuffle(scans)
    scans = list(scans)

    # Split the data evenly
    num = len(scans) // folds
    splits = [scans[i : i + num] for i in range(0, len(scans), num)]

    if len(splits[-1]) < num:
        splits[-2] += splits[-1]
        splits = splits[:-1]

    return tuple(utils.flatten(s) for s in splits)


def make_train_sets(psms, test_idx, subset_max_train, data_size):
    """
    Parameters
    ----------
    psms : list of PsmDataset
        The PsmDataset to get a subset of.
    test_idx : list of list of numpy.ndarray
        The indicies of the test sets
    subset_max_train : int or None
        The number of PSMs for training.

    Yields
    ------
    PsmDataset
        The training set.
    """
    all_idx = [set(range(data_size))]
    for idx in zip(*test_idx):
        train_idx = []
        for i, j, dset in zip(idx, all_idx, psms["files"]):
            train_idx = list(j - set(i))
            if subset_max_train is not None:
                if subset_max_train > len(train_idx):
                    LOGGER.warning(
                        "The provided subset value (%i) is larger than the number "
                        "of psms in the training split (%i), so it will be "
                        "ignored.",
                        subset_max_train,
                        len(train_idx),
                    )
                else:
                    LOGGER.info(
                        "Subsetting PSMs (%i) to (%i).",
                        len(train_idx),
                        subset_max_train,
                    )
                    np.random.seed(1)
                    train_idx = np.random.choice(
                        train_idx, subset_max_train, replace=False
                    )

        yield train_idx


def _create_psms(psms, data):
    data[psms["target_column"]] = data[psms["target_column"]].astype(int)
    if any(data[psms["target_column"]] == -1):
        data[psms["target_column"]] = (
            (data[psms["target_column"]] + 1) / 2
        ).astype(bool)
    return LinearPsmDataset(
        psms=data,
        file=psms["files"][0],
        target_column=psms["target_column"],
        spectrum_columns=psms["spectrum_columns"],
        peptide_column=psms["peptide_column"],
        protein_column=psms["protein_column"],
        group_column=psms["group_column"],
        feature_columns=psms["feature_columns"],
        filename_column=psms["filename_column"],
        scan_column=psms["scan_column"],
        calcmass_column=psms["calcmass_column"],
        expmass_column=psms["expmass_column"],
        rt_column=psms["rt_column"],
        charge_column=psms["charge_column"],
        copy_data=False,
    )


def func(psms, eval_fdr, columns, desc):
    with open(psms["files"][0]) as f:
        df = pd.read_csv(f, sep="\t", usecols=[psms["target_column"], columns])
    _data = _create_psms(
        psms=psms, data=df.apply(pd.to_numeric, errors="ignore")
    )
    return (
        _data._update_labels(
            scores=_data.data.loc[:, columns], eval_fdr=eval_fdr, desc=desc
        )
        == 1
    ).sum()


def find_best_feature(psms, eval_fdr):
    best_feat = None
    best_positives = 0
    new_labels = None
    for desc in (True, False):
        labs = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(func)(psms=psms, eval_fdr=eval_fdr, columns=c, desc=desc)
            for c in psms["feature_columns"]
        )

        # print(labs)
        num_passing = pd.Series(labs, index=psms["feature_columns"])
        # print(num_passing)
        feat_idx = num_passing.idxmax()
        num_passing = num_passing[feat_idx]

        if num_passing > best_positives:
            best_positives = num_passing
            best_feat = feat_idx
            with open(psms["files"][0]) as f:
                df = pd.read_csv(
                    f, sep="\t", usecols=[best_feat, psms["target_column"]]
                )
            _data = _create_psms(
                psms=psms, data=df.apply(pd.to_numeric, errors="ignore")
            )
            # print(_data.data.loc[:, best_feat])
            # print(desc)
            new_labels = _data._update_labels(
                scores=_data.data.loc[:, best_feat],
                eval_fdr=eval_fdr,
                desc=desc,
            )
            best_desc = desc

    if best_feat is None:
        raise RuntimeError("No PSMs found below the 'eval_fdr'.")

    return best_feat, best_positives, new_labels, best_desc


def _update_labels(psms, scores, eval_fdr=0.01, desc=True):
    with open(psms["files"][0]) as f:
        df = pd.read_csv(f, sep="\t", usecols=[psms["target_column"]])
    _data = _create_psms(
        psms=psms, data=df.apply(pd.to_numeric, errors="ignore")
    )
    return _data._update_labels(scores, eval_fdr, desc)


def assign_confidence(psms, scores=None, desc=True, eval_fdr=0.01):
    """Assign confidence to PSMs peptides, and optionally, proteins.

    Two forms of confidence estimates are calculated: q-values---the
    minimum false discovery rate (FDR) at which a given PSM would be
    accepted---and posterior error probabilities (PEPs)---the probability
    that a given PSM is incorrect. For more information see the
    :doc:`Confidence Estimation <confidence>` page.

    Parameters
    ----------
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

    Returns
    -------
    LinearConfidence
        A :py:class:`~mokapot.confidence.LinearConfidence` object storing
        the confidence estimates for the collection of PSMs.
    """
    if scores is None:
        feat, _, _, desc = find_best_feature(psms, eval_fdr)
        LOGGER.info("Selected %s as the best feature.", feat)
        scores = pd.read_csv(psms["files"][0], sep="\t", usecols=feat).values

    data = pd.read_csv(
        psms["files"][0],
        sep="\t",
        usecols=[
            psms["target_column"],
            psms["peptide_column"],
            psms["protein_column"],
            psms["spectrum_columns"][0],
            psms["spectrum_columns"][1],
        ],
    )
    data = _create_psms(
        psms=psms, data=data.apply(pd.to_numeric, errors="ignore")
    )
    if psms["group_column"] is None:
        LOGGER.info("Assigning confidence...")
        return LinearConfidence(data, scores, eval_fdr=eval_fdr, desc=desc)
    else:
        LOGGER.info("Assigning confidence within groups...")
        return GroupedConfidence(data, scores, eval_fdr=eval_fdr, desc=desc)


def _parse_in_chunks(psms, train_idx, chunk_size=1000000):
    """
    Parse a file in chunks

    Parameters
    ----------
    file_obj : file object
        The file to read lines from.
    columns : list of str
        The columns for each DataFrame.
    chunk_size : int
        The chunk size in bytes.

    Returns
    -------
    pandas.DataFrame
        The chunk of PSMs
    """
    train_psms = [[] for _ in range(len(train_idx))]
    with pd.read_csv(
        psms["files"][0],
        sep="\t",
        chunksize=chunk_size,
        usecols=psms["columns"],
    ) as reader:
        for i, chunk in enumerate(reader):
            # logging.info("chunk %i", i)
            for k, train in enumerate(train_idx):
                idx = list(set(train) & set(chunk.index))
                train_psms[k].append(
                    chunk.loc[idx].apply(pd.to_numeric, errors="ignore")
                )

    return [
        pd.concat(df).reindex(orig_idx)
        for df, orig_idx in zip(train_psms, train_idx)
    ]


def _predict(test_idx, psms, models, test_fdr):
    """
    Return the new scores for the dataset

    Parameters
    ----------
    dset : PsmDataset
        The dataset to rescore
    test_idx : list of numpy.ndarray
        The indicies of the test sets
    models : list of Model
        The models for each dataset and whether it
        was reset or not.
    test_fdr : the fdr to calibrate at.
    """
    scores = []
    for fold_idx, mod in zip(test_idx, models):
        CHUNK_SIZE = 2700000
        index_slices = [
            fold_idx[i : i + CHUNK_SIZE]
            for i in range(0, len(fold_idx), CHUNK_SIZE)
        ]
        for index_slice in index_slices:

            fold_idx = _parse_in_chunks(psms=psms, train_idx=[index_slice])
            fold_idx = _create_psms(psms, fold_idx[0])
            # Don't calibrate if using predict_proba.
            try:
                mod.estimator.decision_function
                scores.append(
                    fold_idx._calibrate_scores(mod.predict(fold_idx), test_fdr)
                )
            except AttributeError:
                scores.append(mod.predict(fold_idx))
            except RuntimeError:
                raise RuntimeError(
                    "Failed to calibrate scores between cross-validation folds, "
                    "because no target PSMs could be found below 'test_fdr'. Try "
                    "raising 'test_fdr'."
                )

    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    return np.concatenate(scores)[rev_idx]


def _fit_model(train_set, psms, model, fold):
    """
    Fit the estimator using the training data.

    Parameters
    ----------
    train_set : PsmDataset
        A PsmDataset that specifies the training data
    model : tuple of Model
        A Classifier to train.

    Returns
    -------
    model : mokapot.model.Model
        The trained model.
    reset : bool
        Whether the models should be reset to their original parameters.
    """
    LOGGER.info("")
    LOGGER.info("=== Analyzing Fold %i ===", fold + 1)
    reset = False
    train_set = _create_psms(psms, train_set)
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
