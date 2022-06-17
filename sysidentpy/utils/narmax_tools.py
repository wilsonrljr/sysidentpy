from ..narmax_base import GenerateRegressors
from ._check_arrays import _num_features

import numpy as np


def regressor_code(
    *,
    X=None,
    degree=1,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation="polynomial",
    basis_function=None,
):
    if X is not None:
        n_inputs = _num_features(X)
    else:
        n_inputs = 1  # only used to create the regressor space base

    regressor_code = GenerateRegressors().regressor_space(
        degree, xlag, ylag, n_inputs, model_type
    )

    if basis_function.__class__.__name__ != "Polynomial" and basis_function.ensemble:
        repetition = basis_function.n * 2
        basis_code = np.sort(
            np.tile(regressor_code[1:, :], (repetition, 1)),
            axis=0,
        )
        regressor_code = np.concatenate([regressor_code[1:], basis_code])
    elif (
        basis_function.__class__.__name__ != "Polynomial"
        and basis_function.ensemble is False
    ):
        repetition = basis_function.n * 2
        regressor_code = np.sort(
            np.tile(regressor_code[1:, :], (repetition, 1)),
            axis=0,
        )

    if model_representation in ("polynomial", "general_regressors"):
        return regressor_code
    elif model_representation == "neural_network":  # exclude the constant term
        return regressor_code[1:]
    else:
        raise ("The model representation is not implemented")
