from ..narmax_base import GenerateRegressors
from ._check_arrays import _num_features


def regressor_code(
    *,
    X=None,
    degree=1,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation="polynomial"
):
    if X is not None:
        n_inputs = _num_features(X)
    else:
        n_inputs = 1  # only used to create the regressor space base

    regressor_code = GenerateRegressors().regressor_space(
        degree, xlag, ylag, n_inputs, model_type
    )
    if model_representation in ("polynomial", "general_regressors"):
        return regressor_code
    elif model_representation == "neural_network":  # exclude the constant term
        return regressor_code[1:]
    else:
        raise ("The model representation is not implemented")
