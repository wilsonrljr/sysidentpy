from itertools import chain, combinations_with_replacement
from typing import List, Tuple

import numpy as np

from .check_arrays import num_features
from .simulation import list_input_regressor_code, list_output_regressor_code


def get_max_ylag(ylag: int = 1):
    """Get maximum ylag.

    Parameters
    ----------
    ylag : ndarray of int
        The range of lags according to user definition.

    Returns
    -------
    ny : list
        Maximum value of ylag.

    """
    ny = np.max(list(chain.from_iterable([[ylag]])))
    return ny


def get_max_xlag(xlag: int = 1):
    """Get maximum value from various xlag structures.

    Parameters
    ----------
    xlag : int, list of int, or nested list of int
        Input that can be a single integer, a list, or a nested list.

    Returns
    -------
    int
        Maximum value found.
    """
    if isinstance(xlag, int):  # Case 1: Single integer
        return xlag

    if isinstance(xlag, list):
        # Case 2: Flat list of integers
        if all(isinstance(i, int) for i in xlag):
            return max(xlag)
        # Case 3: Nested list
        return max(chain.from_iterable(xlag))

    raise ValueError("Unsupported data type for xlag")


def get_lag_from_regressor_code(regressors):
    """Get the maximum lag from array of regressors.

    Parameters
    ----------
    regressors : ndarray of int
        Flattened list of input or output regressors.

    Returns
    -------
    max_lag : int
        Maximum lag of list of regressors.

    """
    lag_list = [int(i) for i in regressors.astype("str") for i in [np.sum(int(i[2:]))]]
    if len(lag_list) != 0:
        return max(lag_list)

    return 1


def get_max_lag_from_model_code(model_code: List[int]) -> int:
    """Create a flattened array of input regressors.

    Parameters
    ----------
    model_code : ndarray of int
        Model defined by the user to simulate.

    Returns
    -------
    max_lag : int
        Maximum lag of list of regressors.

    """
    xlag_code = list_input_regressor_code(model_code)
    ylag_code = list_output_regressor_code(model_code)
    xlag = get_lag_from_regressor_code(xlag_code)
    ylag = get_lag_from_regressor_code(ylag_code)
    return max(xlag, ylag)


def _process_xlag(X: np.ndarray, xlag) -> Tuple[int, List[int]]:
    """Process and validate input lags, ensuring correct formatting.

    Parameters
    ----------
    X : array-like
        Input data used during the training phase.

    Returns
    -------
    n_inputs : int
        The number of input variables.
    x_lag : list of int
        The processed list of lags for input variables.

    Raises
    ------
    ValueError
        If multiple inputs exist but `xlag` is provided as a single integer instead
        of a list.

    """
    n_inputs = num_features(X)
    if isinstance(xlag, int) and n_inputs > 1:
        raise ValueError(f"If n_inputs > 1, xlag must be a nested list. Got {xlag}")

    if isinstance(xlag, int):
        xlag = list(range(1, xlag + 1))

    return n_inputs, xlag


def _process_ylag(ylag) -> List[int]:
    """Create the list of lags to be used for the outputs.

    Returns
    -------
    y_lag : ndarray of int
        The processed list of lags for the output variable.

    """
    if isinstance(ylag, int):
        ylag = list(range(1, ylag + 1))

    return ylag
