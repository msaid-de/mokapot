"""
Utility functions
"""
import io
import itertools
import logging

import tqdm
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# Classes ---------------------------------------------------------------------
class TqdmToLogger(io.StringIO):
    """Send tqdm progress bars through the logging module."""
    buf = ""

    def __init__(self, level=logging.INFO):
        super(TqdmToLogger, self).__init__()
        self.logger = LOGGER
        self.level = level

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


# Functions -------------------------------------------------------------------
def pbar(*args, **kwargs):
    """Create a progress bar"""
    return tqdm.tqdm(*args, **kwargs)

def flatten(split):
    """Get the indices from split"""
    return list(itertools.chain.from_iterable(split))


def safe_divide(numerator, denominator, ones=False):
    """Divide ignoring div by zero warnings"""
    if isinstance(numerator, pd.Series):
        numerator = numerator.values

    if isinstance(denominator, pd.Series):
        denominator = denominator.values

    if ones:
        out = np.ones_like(numerator)
    else:
        out = np.zeros_like(numerator)

    return np.divide(numerator, denominator, out=out,
                     where=(denominator != 0))


def tuplize(obj):
    """Convert obj to a tuple, without splitting strings"""
    try:
        _ = iter(obj)
    except TypeError:
        obj = (obj,)
    else:
        if isinstance(obj, str):
            obj = (obj,)
        else:
            tuple(obj)

    return obj
