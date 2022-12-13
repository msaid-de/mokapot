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
from .dataset import (
    LinearPsmDataset,
    calibrate_scores,
    assign_confidence,
    update_labels,
)
from .parsers.pin import read_file, parse_in_chunks, convert_targets_column

LOGGER = logging.getLogger(__name__)
CHUNK_SIZE_READ_ALL_DATA = 1000000
CHUNK_SIZE_UPDATE_LABELS_COLUMNS = 2
CHUNK_SIZE_ROWS_PREDICTION = 1500000
CHUNK_SIZE_THREAD_PREDICTION = 212500


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

    try:
        iter(psms_info)
    except TypeError:
        psms_info = [psms_info]

    # Check that all of the datasets have the same features:
    feat_set = set(psms_info[0]["feature_columns"])
    if not all([set(p["feature_columns"]) == feat_set for p in psms_info]):
        raise ValueError("All collections of PSMs must use the same features.")

    target_column = psms_info[0]["target_column"]
    spectrum_columns = psms_info[0]["spectrum_columns"]
    df_spectra = pd.concat(
        [
            read_file(
                file=p["file"], use_cols=spectrum_columns + [target_column]
            )
            for p in psms_info
        ],
        ignore_index=True,
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
    # once we check all the datasets have the same features we can use only one metedata information
    psms_info[0]["file"] = [psm["file"] for psm in psms_info]
    psms_info = psms_info[0]
    df_spectra = df_spectra[spectrum_columns]
    LOGGER.info("Splitting PSMs into %i folds...", folds)
    test_idx = _split(df_spectra, folds)
    del df_spectra
    train_sets = make_train_sets(
        test_idx=test_idx,
        subset_max_train=subset_max_train,
        data_size=data_size,
    )
    if max_workers != 1:
        # train_sets can't be a generator for joblib :(
        train_sets = list(train_sets)
    train_psms = parse_in_chunks(
        psms_info=psms_info,
        idx=train_sets,
        chunk_size=CHUNK_SIZE_READ_ALL_DATA,
    )

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
    if reset:
        # If we reset, just use the original model on all the folds:
        scores = [
            p._calibrate_scores(model.predict(p), test_fdr) for p in psms_info
        ]
    elif all([m[0].is_trained for m in models]):
        # If we don't reset, assign scores to each fold:
        models = [m for m, _ in models]
        scores = [_predict(test_idx, psms_info, models, test_fdr)]
    else:
        # If model training has failed
        scores = [np.zeros(len(p.data)) for p in psms_info]
    # Find which is best: the learned model, the best feature, or
    # a pretrained model.
    if type(model) is not list and not model.override:
        best_feats = [[m.best_feat, m.feat_pass, m.desc] for m in models]
        best_feat_idx, feat_total = max(
            enumerate(map(itemgetter(1), best_feats)), key=itemgetter(1)
        )
    else:
        feat_total = 0

    preds = [
        _update_labels(p, s, test_fdr) for p, s in zip([psms_info], scores)
    ]
    pred_total = sum([(pred == 1).sum() for pred in preds])

    # Here, f[0] is the name of the best feature, and f[3] is a boolean
    if feat_total > pred_total:
        using_best_feat = True
        scores = []
        descs = []
        feat, _, desc = best_feats[best_feat_idx]
        for dat in psms_info["file"]:
            df = pd.read_csv(
                dat,
                sep="\t",
                usecols=[feat],
                index_col=False,
                on_bad_lines="skip",
            )
            scores.append(df.values)
            descs.append(desc)

    else:
        using_best_feat = False
        descs = [True] * len(psms_info)

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
        _assign_confidence(p, s, eval_fdr=test_fdr, desc=d)
        for p, s, d in zip([psms_info], scores, descs)
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


def targets_count_by_feature(psms_info, eval_fdr, columns, desc):
    df = pd.concat(
        [
            read_file(
                file=file, use_cols=columns + [psms_info["target_column"]]
            )
            for file in psms_info["file"]
        ],
        ignore_index=True,
    )

    return (
        df.loc[:, columns].apply(
            update_labels,
            targets=df.loc[:, psms_info["target_column"]],
            eval_fdr=eval_fdr,
            desc=desc,
        )
        == 1
    ).sum()


def find_best_feature(psms_info, eval_fdr):
    best_feat = None
    best_positives = 0
    new_labels = None

    col_slices = utils.create_chunks(
        data=psms_info["feature_columns"],
        chunk_size=CHUNK_SIZE_UPDATE_LABELS_COLUMNS,
    )
    for desc in (True, False):
        labs = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(targets_count_by_feature)(
                psms_info=psms_info,
                eval_fdr=eval_fdr,
                columns=list(c),
                desc=desc,
            )
            for c in col_slices
        )
        num_passing = pd.concat(labs)
        feat_idx = num_passing.idxmax()
        num_passing = num_passing[feat_idx]

        if num_passing > best_positives:
            best_positives = num_passing
            best_feat = feat_idx
            df = pd.concat(
                [
                    read_file(
                        file=file,
                        use_cols=[best_feat, psms_info["target_column"]],
                    )
                    for file in psms_info["file"]
                ],
                ignore_index=True,
            )
            new_labels = update_labels(
                scores=df.loc[:, best_feat],
                targets=df[psms_info["target_column"]],
                eval_fdr=eval_fdr,
                desc=desc,
            )
            best_desc = desc

    if best_feat is None:
        raise RuntimeError("No PSMs found below the 'eval_fdr'.")

    return best_feat, best_positives, new_labels, best_desc


def _update_labels(psms_info, scores, eval_fdr=0.01, desc=True):
    df = pd.concat(
        [
            read_file(file=file, use_cols=[psms_info["target_column"]])
            for file in psms_info["file"]
        ],
        ignore_index=True,
    )
    return update_labels(
        scores=scores,
        targets=df[psms_info["target_column"]],
        eval_fdr=eval_fdr,
        desc=desc,
    )


def _predict(test_idx, psms_info, models, test_fdr):
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
        index_slices = utils.create_chunks(
            data=fold_idx, chunk_size=CHUNK_SIZE_ROWS_PREDICTION
        )
        fold_scores = []
        targets = []
        del fold_idx
        for index_slice in index_slices:
            pred_slices = utils.create_chunks(
                data=index_slice, chunk_size=CHUNK_SIZE_THREAD_PREDICTION
            )
            psms_slice = parse_in_chunks(
                psms_info=psms_info,
                idx=pred_slices,
                chunk_size=CHUNK_SIZE_READ_ALL_DATA,
            )
            del pred_slices
            psms_slice = [
                _create_psms(psms_info, psm_slice, enforce_checks=False)
                for psm_slice in psms_slice
            ]
            targets = targets + [psm_slice.targets for psm_slice in psms_slice]
            fold_scores += Parallel(n_jobs=-1, require="sharedmem")(
                delayed(mod.predict)(psms=p) for p in psms_slice
            )
            del psms_slice
        try:
            mod.estimator.decision_function
            scores.append(
                calibrate_scores(
                    np.hstack(fold_scores), np.hstack(targets), test_fdr
                )
            )
        except AttributeError:
            scores.append(np.hstack(fold_scores))
        except RuntimeError:
            raise RuntimeError(
                "Failed to calibrate scores between cross-validation folds, "
                "because no target PSMs could be found below 'test_fdr'. Try "
                "raising 'test_fdr'."
            )
        del index_slices
        del targets
        del fold_scores
    rev_idx = np.argsort(sum(test_idx, [])).tolist()
    del test_idx
    return np.concatenate(scores)[rev_idx]


def _assign_confidence(psms_info, scores=None, desc=True, eval_fdr=0.01):
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

    Returns
    -------
    LinearConfidence
        A :py:class:`~mokapot.confidence.LinearConfidence` object storing
        the confidence estimates for the collection of PSMs.
    """
    if scores is None:
        feat, _, _, desc = find_best_feature(psms_info, eval_fdr)
        LOGGER.info("Selected %s as the best feature.", feat)
        scores = pd.concat(
            [
                read_file(file=file, use_cols=feat)
                for file in psms_info["file"]
            ],
            ignore_index=True,
        ).values

    data = pd.concat(
        [
            read_file(
                file=file,
                use_cols=[
                    psms_info["target_column"],
                    psms_info["peptide_column"],
                    psms_info["protein_column"],
                    psms_info["specId_column"],
                    psms_info["spectrum_columns"][0],
                    psms_info["spectrum_columns"][1],
                ],
            )
            for file in psms_info["file"]
        ],
        ignore_index=True,
    )

    data = data.apply(pd.to_numeric, errors="ignore")
    convert_targets_column(data=data, target_column=psms_info["target_column"])
    return assign_confidence(
        psms=data,
        psms_info=psms_info,
        scores=scores,
        eval_fdr=eval_fdr,
        desc=desc,
    )


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
