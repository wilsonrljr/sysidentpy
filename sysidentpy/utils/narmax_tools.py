from ..narmax_base import RegressorDictionary
from ._check_arrays import _num_features

import numpy as np


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

    regressor_code = RegressorDictionary(
        xlag=xlag, ylag=ylag, model_type=model_type, basis_function=basis_function
    ).regressor_space(n_inputs)

    basis_name = basis_function.__class__.__name__
    if basis_name != "Polynomial" and basis_function.ensemble:
        repetition = basis_function.n * 2
        basis_code = np.sort(
            np.tile(regressor_code[1:, :], (repetition, 1)),
            axis=0,
        )
        regressor_code = np.concatenate([regressor_code[1:], basis_code])
    elif basis_name != "Polynomial" and basis_function.ensemble is False:
        repetition = basis_function.n * 2
        regressor_code = np.sort(
            np.tile(regressor_code[1:, :], (repetition, 1)),
            axis=0,
        )

    if basis_name == "Polynomial" and model_representation == "neural_network":
        return regressor_code[1:]
    elif basis_name == "Polynomial" and model_representation is None:
        return regressor_code
    else:
        return regressor_code
