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
    # test_idx = [p._split(folds) for p in psms]
    # train_sets = _make_train_sets(psms, test_idx, subset_max_train)
    if max_workers != 1:
        # train_sets can't be a generator for joblib :(
        train_sets = list(train_sets)

    print(train_sets[0])

    if type(model) is list:
        models = [[m, False] for m in model]
    else:
        models = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(_fit_model)(d, copy.deepcopy(model), f)
            for f, d in enumerate(train_sets)
        )

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
        print(psms)
        scores = [
            _predict(p, i, models, test_fdr) for p, i in zip([psms], test_idx)
        ]
    else:
        # If model training has failed
        scores = [np.zeros(len(p.data)) for p in psms]

    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if type(model) is not list and not model.override:
        best_feats = [p._find_best_feature(test_fdr) for p in psms]
        feat_total = sum([best_feat[1] for best_feat in best_feats])
    else:
        feat_total = 0

    preds = [p._update_labels(s, test_fdr) for p, s in zip(psms, scores)]
    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        scores = []
        descs = []
        for dat, (feat, _, _, desc) in zip(psms, best_feats):
            scores.append(dat.data[feat].values)
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
        p.assign_confidence(s, eval_fdr=test_fdr, desc=d)
        for p, s, d in zip(psms, scores, descs)
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
    for indices in scans:
        np.random.shuffle(indices)

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
        data = []
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
                    train_idx = np.random.choice(
                        train_idx, subset_max_train, replace=False
                    )
            data.append(
                pd.concat(
                    (
                        c
                        for c in _parse_in_chunks(
                            dset, psms["columns"], train_idx
                        )
                    ),
                    copy=False,
                )
            )
        _data = pd.concat(data, ignore_index=True)
        _data[psms["target_column"]] = _data[psms["target_column"]].astype(int)
        if any(_data[psms["target_column"]] == -1):
            _data[psms["target_column"]] = (
                (_data[psms["target_column"]] + 1) / 2
            ).astype(bool)

        yield LinearPsmDataset(
            psms=_data,
            file=psms["files"],
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


def _parse_in_chunks(filename, columns, train_idx=None, chunk_size=int(1e7)):
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
    with pd.read_csv(
        filename, sep="\t", chunksize=chunk_size, usecols=columns
    ) as reader:
        for i, psms in enumerate(reader):
            logging.info("chunk %i", i)
            idx = list(set(train_idx) & set(psms.index))
            psms = psms.loc[idx]
    yield psms.apply(pd.to_numeric, errors="ignore")


def _predict(psms, test_idx, models, test_fdr):
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
        _data = pd.concat(
            (
                c
                for c in _parse_in_chunks(
                    psms["files"][0], psms["columns"], fold_idx
                )
            ),
            copy=False,
        )
        _data[psms["target_column"]] = _data[psms["target_column"]].astype(int)
        if any(_data[psms["target_column"]] == -1):
            _data[psms["target_column"]] = (
                (_data[psms["target_column"]] + 1) / 2
            ).astype(bool)
        test_set = LinearPsmDataset(
            psms=_data,
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

        # Don't calibrate if using predict_proba.
        try:
            mod.estimator.decision_function
            scores.append(
                test_set._calibrate_scores(mod.predict(test_set), test_fdr)
            )
        except AttributeError:
            scores.append(mod.predict(test_set))
        except RuntimeError:
            raise RuntimeError(
                "Failed to calibrate scores between cross-validation folds, "
                "because no target PSMs could be found below 'test_fdr'. Try "
                "raising 'test_fdr'."
            )

    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    return np.concatenate(scores)[rev_idx]


def _fit_model(train_set, model, fold):
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
    try:
        model.fit(train_set)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
