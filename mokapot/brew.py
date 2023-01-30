"""
Defines a function to run the Percolator algorithm.
"""
import logging
import copy
from operator import itemgetter

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from .model import PercolatorModel
from . import utils
from .dataset import LinearPsmDataset, calibrate_scores, update_labels
from .parsers.pin import (
    read_file,
    parse_in_chunks,
    convert_targets_column,
    read_file_in_chunks,
)
from .constants import (
    CHUNK_SIZE_ROWS_PREDICTION,
    CHUNK_SIZE_READ_ALL_DATA,
)

LOGGER = logging.getLogger(__name__)


# Functions -------------------------------------------------------------------
def brew(
    psms_info,
    model=None,
    test_fdr=0.01,
    folds=3,
    max_workers=1,
    seed=1,
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

    target_column = psms_info["target_column"]
    spectrum_columns = psms_info["spectrum_columns"]
    df_spectra = read_file(
        file_name=psms_info["file"],
        use_cols=spectrum_columns + [target_column],
    ).apply(pd.to_numeric, errors="ignore")
    df_spectra = convert_targets_column(df_spectra, target_column)
    data_size = len(df_spectra)
    if data_size > 1:
        LOGGER.info("")
        LOGGER.info("Found %i total PSMs.", data_size)
        num_targets = (df_spectra[target_column]).sum()
        num_decoys = (~df_spectra[target_column]).sum()
        LOGGER.info(
            "  - %i target PSMs and %i decoy PSMs detected.",
            num_targets,
            num_decoys,
        )
    df_spectra = df_spectra[spectrum_columns]
    LOGGER.info("Splitting PSMs into %i folds...", folds)
    test_idx = _split(df_spectra, folds)
    del df_spectra
    train_sets = list(
        make_train_sets(
            test_idx=test_idx,
            subset_max_train=subset_max_train,
            data_size=data_size,
        )
    )

    train_psms = parse_in_chunks(
        psms_info=psms_info,
        idx=train_sets,
        chunk_size=CHUNK_SIZE_READ_ALL_DATA,
    )
    del train_sets
    if type(model) is list:
        models = [[m, False] for m in model]
    else:
        models = Parallel(n_jobs=max_workers, require="sharedmem")(
            delayed(_fit_model)(d, psms_info, copy.deepcopy(model), f, seed)
            for f, d in enumerate(train_psms)
        )
    del train_psms

    # sort models to have deterministic results with multithreading.
    # Only way I found to sort is using intercept values
    models.sort(key=lambda x: x[0].estimator.intercept_)
    # Determine if the models need to be reset:
    reset = any([m[1] for m in models])
    model_test_idx = [[i] * len(idx) for i, idx in enumerate(test_idx)]
    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    del test_idx
    model_idx = np.concatenate(model_test_idx)[rev_idx]
    del rev_idx
    if reset:
        # If we reset, just use the original model on all the folds:
        scores = [
            p._calibrate_scores(model.predict(p), test_fdr) for p in psms_info
        ]
    elif all([m[0].is_trained for m in models]):
        # If we don't reset, assign scores to each fold:
        models = [m for m, _ in models]
        scores = _predict(model_idx, psms_info, models, test_fdr)
    else:
        # If model training has failed
        scores = np.zeros(data_size)
    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if type(model) is not list and not model.override:
        best_feats = [[m.best_feat, m.feat_pass, m.desc] for m in models]
        best_feat_idx, feat_total = max(
            enumerate(map(itemgetter(1), best_feats)), key=itemgetter(1)
        )
    else:
        feat_total = 0

    preds = update_labels(psms_info, scores, test_fdr)

    pred_total = sum([(preds == 1).sum()])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        feat, _, desc = best_feats[best_feat_idx]
        scores = pd.read_csv(
            psms_info["file"],
            sep="\t",
            usecols=[feat],
            index_col=False,
            on_bad_lines="skip",
        ).values

    else:
        using_best_feat = False
        desc = True

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

    return psms_info, models, scores, desc


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


def make_train_sets(test_idx, subset_max_train, data_size):
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
    chunk_range = 5000000
    for idx in test_idx:
        k = 0
        train_idx = []
        while k + chunk_range < data_size:
            train_idx += list(set(range(k, k + chunk_range)) - set(idx))
            k += chunk_range
        train_idx += list(set(range(k, data_size)) - set(idx))
        train_idx_size = len(train_idx)
        if subset_max_train is not None:
            if subset_max_train > train_idx_size:
                LOGGER.warning(
                    "The provided subset value (%i) is larger than the number "
                    "of psms in the training split (%i), so it will be "
                    "ignored.",
                    subset_max_train,
                    train_idx_size,
                )
            else:
                LOGGER.info(
                    "Subsetting PSMs (%i) to (%i).",
                    train_idx_size,
                    subset_max_train,
                )
                train_idx = np.random.choice(
                    train_idx, subset_max_train, replace=False
                )
        yield train_idx


def _create_psms(psms_info, data, enforce_checks=True):
    convert_targets_column(data=data, target_column=psms_info["target_column"])
    return LinearPsmDataset(
        psms=data,
        target_column=psms_info["target_column"],
        spectrum_columns=psms_info["spectrum_columns"],
        peptide_column=psms_info["peptide_column"],
        protein_column=psms_info["protein_column"],
        group_column=psms_info["group_column"],
        feature_columns=psms_info["feature_columns"],
        filename_column=psms_info["filename_column"],
        scan_column=psms_info["scan_column"],
        calcmass_column=psms_info["calcmass_column"],
        expmass_column=psms_info["expmass_column"],
        rt_column=psms_info["rt_column"],
        charge_column=psms_info["charge_column"],
        copy_data=False,
        enforce_checks=enforce_checks,
    )


def get_index_values(df, col_name, val, orig_idx):
    df = df[df[col_name] == val].drop(col_name, axis=1)
    orig_idx[val] += list(df.index)
    return df


def predict_fold(model, fold, psms, scores):
    scores[fold].append(model.predict(psms))


def _predict(models_idx, psms_info, models, test_fdr):
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

    model_test_idx = utils.create_chunks(
        data=models_idx, chunk_size=CHUNK_SIZE_ROWS_PREDICTION
    )
    n_folds = len(models)
    fold_scores = [[] for _ in range(n_folds)]
    targets = [[] for _ in range(n_folds)]
    orig_idx = [[] for _ in range(n_folds)]
    reader = read_file_in_chunks(
        file=psms_info["file"],
        chunk_size=CHUNK_SIZE_ROWS_PREDICTION,
        use_cols=psms_info["columns"],
    )
    for i, psms_slice in enumerate(reader):
        psms_slice["fold"] = model_test_idx.pop(0)
        psms_slice = [
            get_index_values(psms_slice, "fold", i, orig_idx)
            for i in range(n_folds)
        ]
        psms_slice = [
            _create_psms(psms_info, psm_slice, enforce_checks=False)
            for psm_slice in psms_slice
        ]
        [
            targets[i].append(psm_slice.targets)
            for i, psm_slice in enumerate(psms_slice)
        ]

        Parallel(n_jobs=-1, require="sharedmem")(
            delayed(predict_fold)(
                model=models[mod_idx], fold=mod_idx, psms=p, scores=fold_scores
            )
            for mod_idx, p in enumerate(psms_slice)
        )
        del psms_slice
    del reader
    del model_test_idx
    for mod in models:
        try:
            mod.estimator.decision_function
            scores.append(
                calibrate_scores(
                    np.hstack(fold_scores.pop(0)),
                    np.hstack(targets.pop(0)),
                    test_fdr,
                )
            )
        except AttributeError:
            scores.append(np.hstack(fold_scores.pop(0)))
        except RuntimeError:
            raise RuntimeError(
                "Failed to calibrate scores between cross-validation folds, "
                "because no target PSMs could be found below 'test_fdr'. Try "
                "raising 'test_fdr'."
            )
    del targets
    del fold_scores
    orig_idx = np.argsort(sum(orig_idx, [])).tolist()
    return np.concatenate(scores)[orig_idx]


def _fit_model(train_set, psms_info, model, fold, seed):
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
    train_set = _create_psms(psms_info, train_set)
    try:
        model.fit(train_set, seed)
    except RuntimeError as msg:
        if str(msg) != "Model performs worse after training.":
            raise

        if model.is_trained:
            reset = True

    return model, reset
