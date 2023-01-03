"""
Utility functions
"""
import itertools

import numpy as np
import pandas as pd

from sorted_in_disk import sorted_in_disk
from sorted_in_disk.utils import read_iter_from_file


def groupby_max(df, by_cols, max_col):
    """Quickly get the indices for the maximum value of col"""
    by_cols = tuplize(by_cols)
    idx = (
        df.sample(frac=1)
        .sort_values(list(by_cols) + [max_col], axis=0)
        .drop_duplicates(list(by_cols), keep="last")
        .index
    )

    return idx


def flatten(split):
    """Get the indices from split"""
    return list(itertools.chain.from_iterable(split))


def safe_divide(numerator, denominator, ones=False):
    """Divide ignoring div by zero warnings"""
    if isinstance(numerator, pd.Series):
        numerator = numerator.values

    if isinstance(denominator, pd.Series):
        denominator = denominator.values

    numerator = numerator.astype(float)
    denominator = denominator.astype(float)
    if ones:
        out = np.ones_like(numerator)
    else:
        out = np.zeros_like(numerator)

    return np.divide(numerator, denominator, out=out, where=(denominator != 0))


def tuplize(obj):
    """Convert obj to a tuple, without splitting strings"""
    try:
        _ = iter(obj)
    except TypeError:
        obj = (obj,)
    else:
        if isinstance(obj, str):
            obj = (obj,)

    return tuple(obj)


def create_chunks(data, chunk_size):
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def sort_file_on_disk(file_path, sort_key, sep=",", reverse=False):
    return sorted_in_disk(
        read_iter_from_file(file_path),
        key=lambda row: float(row.split(sep)[sort_key]),
        write_processes=8,
        reverse=reverse,
    )


def get_unique_psms_and_peptides(iterable, out_psms, out_peptides, sep):
    seen_psm = set()
    seen_peptide = set()
    f_psm = open(out_psms, "a")
    f_peptide = open(out_peptides, "a")
    for line in iterable:
        line_list = line.split(sep)
        line_hash_psm = tuple([int(line_list[2]), float(line_list[3])])
        line_hash_peptide = line_list[4]
        if line_hash_psm not in seen_psm:
            seen_psm.add(line_hash_psm)
            f_psm.write(f"{line}\n")
            if line_hash_peptide not in seen_peptide:
                seen_peptide.add(line_hash_peptide)
                f_peptide.write(f"{line}\n")
    f_psm.close()
    f_peptide.close()
    return [len(seen_psm), len(seen_peptide)]
