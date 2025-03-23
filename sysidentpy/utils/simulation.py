from itertools import chain, combinations_with_replacement
from typing import List

import numpy as np


def get_index_from_regressor_code(regressor_code: np.ndarray, model_code: List[int]):
    """Get the index of user regressor in regressor space.

    Took from: https://stackoverflow.com/questions/38674027/find-the-row-indexes-of-several-values-in-a-numpy-array/38674038#38674038

    Parameters
    ----------
    regressor_code : ndarray of int
        Matrix codification of all possible regressors.
    model_code : ndarray of int
        Model defined by the user to simulate.

    Returns
    -------
    model_index : ndarray of int
        Index of model code in the regressor space.

    """
    dims = regressor_code.max(0) + 1
    model_index = np.where(
        np.in1d(
            np.ravel_multi_index(regressor_code.T, dims),
            np.ravel_multi_index(model_code.T, dims),
        )
    )[0]
    return model_index


def list_output_regressor_code(model_code: List[int]) -> np.ndarray:
    """Create a flattened array of output regressors.

    Parameters
    ----------
    model_code : ndarray of int
        Model defined by the user to simulate.

    Returns
    -------
    regressor_code : ndarray of int
        Flattened list of output regressors.

    """
    regressor_code = [
        code for code in model_code.ravel() if (code != 0) and (str(code)[0] == "1")
    ]

    return np.asarray(regressor_code)


def list_input_regressor_code(model_code: List[int]) -> np.ndarray:
    """Create a flattened array of input regressors.

    Parameters
    ----------
    model_code : ndarray of int
        Model defined by the user to simulate.

    Returns
    -------
    regressor_code : ndarray of int
        Flattened list of output regressors.

    """
    regressor_code = [
        code for code in model_code.ravel() if (code != 0) and (str(code)[0] != "1")
    ]
    return np.asarray(regressor_code)
