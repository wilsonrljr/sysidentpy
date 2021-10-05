"""Base classes for NARMAX estimator."""

# Authors:
#           Wilson Rocha Lacerda Junior <wilsonrljr@outlook.com>
# License: BSD 3 clause


import numpy as np
from itertools import combinations_with_replacement
from itertools import chain
from collections import Counter



def _get_max_lag(ylag=1, xlag=1):
    """Get the max lag defined by the user.

    Parameters
    ----------
    ylag : int
        The maximum lag of output regressors.
    xlag : int
        The maximum lag of input regressors.

    Returns
    -------
    max_lag = int
        The max lag value defined by the user.
    """
    ny = np.max(list(chain.from_iterable([[ylag]])))
    nx = np.max(list(chain.from_iterable([[xlag]])))
    return np.max([ny, np.max(nx)])


