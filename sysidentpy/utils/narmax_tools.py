""" Utils methods for NARMAX modeling"""

import numpy as np

from ..narmax_base import RegressorDictionary
from ._check_arrays import _num_features


def regressor_code(
    *,
    X=None,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation=None,
    basis_function=None,
):
    if X is not None:
        n_inputs = _num_features(X)
    else:
        n_inputs = 1  # only used to create the regressor space base

    encoding = RegressorDictionary(
        xlag=xlag, ylag=ylag, model_type=model_type, basis_function=basis_function
    ).regressor_space(n_inputs)

    basis_name = basis_function.__class__.__name__
    if basis_name != "Polynomial" and basis_function.ensemble:
        repetition = basis_function.n * 2
        basis_code = np.sort(
            np.tile(encoding[1:, :], (repetition, 1)),
            axis=0,
        )
        encoding = np.concatenate([encoding[1:], basis_code])
    elif basis_name != "Polynomial" and basis_function.ensemble is False:
        repetition = basis_function.n * 2
        encoding = np.sort(
            np.tile(encoding[1:, :], (repetition, 1)),
            axis=0,
        )

    if basis_name == "Polynomial" and model_representation == "neural_network":
        return encoding[1:]
    if basis_name == "Polynomial" and model_representation is None:
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
    """
    Set log-spaced weights assigned to each objective in the multi-objective
    optimization.

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
