import numpy as np
import scipy as sp

from mokapot.stats.tdmodel import STDSModel, TDCModel


def create_tdmodel(is_tdc, rho0, discrete, delta=1):
    np.random.seed(123)
    if discrete:
        R1 = sp.stats.binom(10, 0.6 + 0.1 * delta)
        R0 = sp.stats.binom(8, 0.4 - 0.1 * delta)
    else:
        R1 = sp.stats.norm(0, 1)
        R0 = sp.stats.norm(-delta, 1)
    if is_tdc:
        model = TDCModel(R0, R1, rho0)
    else:
        model = STDSModel(R0, R1, rho0)
    return model
