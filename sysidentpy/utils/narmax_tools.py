"""Utils methods for NARMAX modeling."""

from typing import Tuple, Optional, Any
import numpy as np

from ..narmax_base import RegressorDictionary
from ..basis_function import Polynomial
from ._check_arrays import _num_features


def regressor_code(
    *,
    X: Optional[np.ndarray] = None,
    xlag: int = 2,
    ylag: int = 2,
    model_type: str = "NARMAX",
    model_representation: Optional[str] = None,
    basis_function: Optional[Any] = None,
) -> np.ndarray:
    """Generate a regressor code based on the provided parameters.

    Parameters
    ----------
    X : np.ndarray, optional
        The input feature matrix.
    xlag : int, optional
        The number of lags for the input features.
    ylag : int, optional
        The number of lags for the target variable.
    model_type : str, optional
        The type of model to be used. Default is "NARMAX".
    model_representation : str, optional
        The model representation to be used.
    basis_function : object, optional
        The basis function object used to transform the regressor space.

    Returns
    -------
    encoding : np.ndarray
        The generated regressor encoding.
    """
    if X is not None:
        n_inputs = _num_features(X)
    else:
        n_inputs = 1  # only used to create the regressor space base

    encoding = RegressorDictionary(
        xlag=xlag, ylag=ylag, model_type=model_type, basis_function=basis_function
    ).regressor_space(n_inputs)

    if not isinstance(basis_function, Polynomial) and basis_function.ensemble:
        repetition = basis_function.n * 2
        basis_code = np.sort(
            np.tile(encoding[1:, :], (repetition, 1)),
            axis=0,
        )
        encoding = np.concatenate([encoding[1:], basis_code])
    elif (
        not isinstance(basis_function, Polynomial) and basis_function.ensemble is False
    ):
        repetition = basis_function.n * 2
        encoding = np.sort(
            np.tile(encoding[1:, :], (repetition, 1)),
            axis=0,
        )

    if (
        isinstance(basis_function, Polynomial)
        and model_representation == "neural_network"
    ):
        return encoding[1:]
    if isinstance(basis_function, Polynomial) and model_representation is None:
        return encoding

    return encoding


def set_weights(
    *,
    static_function: bool = True,
    static_gain: bool = True,
    start: float = -0.01,
    stop: float = -5,
    num: int = 50,
    base: float = 2.71,
) -> np.ndarray:
    """Set log-spaced weights assigned to each objective in the MO optimization.

    Parameters
    ----------
    static_function : bool, optional
        Indicator for the presence of static function data. Default is True.
    static_gain : bool, optional
        Indicator for the presence of static gain data. Default is True.
    start : float, optional
        The starting exponent for the log-spaced weights. Default is -0.01.
    stop : float, optional
        The stopping exponent for the log-spaced weights. Default is -5.
    num : int, optional
        The number of weights to generate. Default is 50.
    base : float, optional
        The base of the logarithm used to generate weights. Default is 2.71.

    Returns
    -------
    weights : ndarray of floats
        An array containing the weights for each objective.

    Notes
    -----
    This method calculates the weights to be assigned to different objectives in
    multi-objective optimization. The choice of weights depends on the presence
    of static function and static gain data. If both are present, a set of weights
    for dynamic, gain, and static objectives is computed. If either static function
    or static gain is absent, a simplified set of weights is generated.

    """
    w1 = np.logspace(start=start, stop=stop, num=num, base=base)
    if static_function is False or static_gain is False:
        w2 = 1 - w1
        return np.vstack([w1, w2])

    w2 = w1[::-1]
    w1_grid, w2_grid = np.meshgrid(w1, w2)
    w3_grid = 1 - (w1_grid + w2_grid)
    mask = w1_grid + w2_grid <= 1
    dynamic_weight = np.flip(w1_grid[mask])
    gain_weight = np.flip(w2_grid[mask])
    static_weight = np.flip(w3_grid[mask])
    return np.vstack([dynamic_weight, gain_weight, static_weight])


def train_test_split(
    X: Optional[np.ndarray], y: np.ndarray, test_size: float = 0.25
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Split the time series dataset into training and testing sets.

    Parameters
    ----------
    X : np.ndarray, optional
        The feature matrix. Can be None if there are no features.
    y : np.ndarray
        The target vector.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.25.

    Returns
    -------
    X_train : np.ndarray or None
        The training set feature matrix, or None if X is None.
    X_test : np.ndarray or None
        The testing set feature matrix, or None if X is None.
    y_train : np.ndarray
        The training set target vector.
    y_test : np.ndarray
        The testing set target vector.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size should be between 0 and 1")

    # Determine the split index
    split_index = int(len(y) * (1 - test_size))

    y_train, y_test = y[:split_index], y[split_index:]

    if X is None:
        return None, None, y_train, y_test

    X_train, X_test = X[:split_index], X[split_index:]

    return X_train, X_test, y_train, y_test
