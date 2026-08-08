# pylint: disable=protected-access,unused-argument,redefined-outer-name,useless-super-delegation,arguments-renamed,abstract-class-instantiated
from typing import Optional

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy import config_context
from sysidentpy.basis_function import (
    Bernstein,
    Bilinear,
    Fourier,
    Hermite,
    HermiteNormalized,
    Laguerre,
    Legendre,
    Polynomial,
)
from sysidentpy.tests._array_api_asserts import (
    assert_allclose as xp_assert_allclose,
    assert_array_equal as xp_assert_array_equal,
)
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.narmax_base import RegressorDictionary, BaseMSS
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.narmax_base import (
    house,
    rowhouse,
)

GR = RegressorDictionary()
bf_polynomial = Polynomial(degree=2)
bf_fourier = Fourier(degree=2, n=1)


def create_test_data():
    r"""Load test data from an external source.

    The dataset is based on a nonlinear autoregressive model
     with exogenous inputs (NARX) given by:

    $$
    y[k] = \theta_4 y[k-1]^2 + \theta_2 y[k-1] x[k-1] + \theta_0 x[k-2]
          + \theta_3 y[k-2] x[k-2] + \theta_1 y[k-2]
    $$

    where:
    - $ x[k] $ is the input at time step $ k $
    - $ y[k] $ is the output at time step $ k $
    - $ \theta = [\theta_0, \theta_1, \theta_2, \theta_3, \theta_4] $
     are model parameters

    Returns
    -------
        x (numpy.ndarray): Input data of shape $ (n, 1) $.
        y (numpy.ndarray): Output data of shape $ (n, 1) $.
        $\theta$ (numpy.ndarray): Model parameters.

    """
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])

    url = "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/testing/data_for_testing.txt"
    data = np.loadtxt(url)

    xt = data[:, 0].reshape(-1, 1)
    yt = data[:, 1].reshape(-1, 1)

    return xt, yt, theta


x, y, _ = create_test_data()

train_percentage = 90

split_data = int(len(x) * (train_percentage / 100))

X_train = x[0:split_data, 0]
X_test = x[split_data::, 0]

y1 = y[0:split_data, 0]
y_test = y[split_data::, 0]
y_train = y1.copy()

y_train = np.reshape(y_train, (len(y_train), 1))
X_train = np.reshape(X_train, (len(X_train), 1))

y_test = np.reshape(y_test, (len(y_test), 1))
X_test = np.reshape(X_test, (len(X_test), 1))


def test_create_narmax_code():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_regressor_space():
    output1 = np.array([[0], [1001], [1002], [2001], [2002]])
    r1 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=1)
    ).regressor_space(n_inputs=1)
    assert_array_equal(output1, r1)
    output2 = np.array(
        [
            [0, 0],
            [1001, 0],
            [1002, 0],
            [2001, 0],
            [2002, 0],
            [1001, 1001],
            [1002, 1001],
            [2001, 1001],
            [2002, 1001],
            [1002, 1002],
            [2001, 1002],
            [2002, 1002],
            [2001, 2001],
            [2002, 2001],
            [2002, 2002],
        ]
    )
    r2 = RegressorDictionary(
        xlag=2, ylag=2, basis_function=Polynomial(degree=2)
    ).regressor_space(n_inputs=1)
    assert_array_equal(output2, r2)
    output3 = np.array(
        [
            [0, 0],
            [1001, 0],
            [1002, 0],
            [2001, 0],
            [2002, 0],
            [3001, 0],
            [3002, 0],
            [1001, 1001],
            [1002, 1001],
            [2001, 1001],
            [2002, 1001],
            [3001, 1001],
            [3002, 1001],
            [1002, 1002],
            [2001, 1002],
            [2002, 1002],
            [3001, 1002],
            [3002, 1002],
            [2001, 2001],
            [2002, 2001],
            [3001, 2001],
            [3002, 2001],
            [2002, 2002],
            [3001, 2002],
            [3002, 2002],
            [3001, 3001],
            [3002, 3001],
            [3002, 3002],
        ]
    )
    r3 = RegressorDictionary(
        xlag=[[1, 2], [1, 2]], ylag=2, basis_function=Polynomial(degree=2)
    ).regressor_space(n_inputs=2)
    assert_array_equal(output3, r3)


def test_regressor_space_polynomial_include_bias_preserves_code_order():
    default = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=Polynomial(degree=2),
    ).regressor_space(n_inputs=1)
    explicit = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=Polynomial(degree=2, include_bias=True),
    ).regressor_space(n_inputs=1)
    without_bias = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=Polynomial(degree=2, include_bias=False),
    ).regressor_space(n_inputs=1)

    expected_without_bias = np.array(
        [
            [1001, 0],
            [2001, 0],
            [1001, 1001],
            [2001, 1001],
            [2001, 2001],
        ]
    )
    assert_array_equal(default, explicit)
    assert_array_equal(without_bias, expected_without_bias)
    assert not np.any(np.all(without_bias == 0, axis=1))


def test_bilinear_regressor_codes_follow_matrix_order_for_noncontiguous_lags():
    basis_function = Bilinear(degree=2, include_bias=False)
    dictionary = RegressorDictionary(
        xlag=[[1, 3], [2]],
        ylag=[1, 4],
        basis_function=basis_function,
    )
    data = np.array(
        [
            [1.0, 2.0, 3.0, 5.0, 7.0, 11.0],
            [1.0, 13.0, 17.0, 19.0, 23.0, 29.0],
            [1.0, 31.0, 37.0, 41.0, 43.0, 47.0],
        ]
    )
    feature_matrix = basis_function.fit(
        data,
        max_lag=1,
        ylag=dictionary.ylag,
        xlag=dictionary.xlag,
    )
    feature_codes = dictionary.regressor_space(
        n_inputs=2,
        n_features=feature_matrix.shape[1],
    )
    x_codes, y_codes = dictionary.create_narmax_code(n_inputs=2)
    base_codes = np.concatenate(([0], y_codes, x_codes))
    code_to_column = {int(code): column for column, code in enumerate(base_codes)}
    column_indices = np.array(
        [
            [code_to_column[int(code)] for code in feature_code]
            for feature_code in feature_codes
        ]
    )
    matrix_from_codes = np.prod(data[1:, column_indices], axis=2)

    assert_array_equal(feature_matrix, matrix_from_codes)
    assert not np.any(np.all(feature_codes == 0, axis=1))


@pytest.mark.parametrize(
    "basis_cls",
    [Bernstein, Hermite, HermiteNormalized, Laguerre, Legendre],
)
def test_univariate_basis_regressor_codes_match_feature_layout(basis_cls):
    basis_function = basis_cls(
        degree=2,
        include_bias=True,
        ensemble=True,
    )
    data = np.array(
        [
            [1.0, 0.1, 0.2],
            [1.0, 0.3, 0.4],
            [1.0, 0.5, 0.6],
        ]
    )
    feature_matrix = basis_function.fit(data, max_lag=1)
    feature_codes = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=basis_function,
    ).regressor_space(n_inputs=1, n_features=feature_matrix.shape[1])

    expected = np.array(
        [
            [1001, 0],
            [2001, 0],
            [0, 0],
            [1001, 0],
            [1001, 1001],
            [2001, 0],
            [2001, 2001],
        ]
    )
    assert_array_equal(feature_codes, expected)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_fourier_regressor_codes_match_internal_polynomial_layout(degree):
    basis_function = Fourier(degree=degree, n=2, ensemble=True)
    data = np.array(
        [
            [1.0, 0.1, 0.2],
            [1.0, 0.3, 0.4],
            [1.0, 0.5, 0.6],
        ]
    )
    feature_matrix = basis_function.fit(data, max_lag=1)
    feature_codes = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=basis_function,
    ).regressor_space(n_inputs=1, n_features=feature_matrix.shape[1])

    if degree == 1:
        data_codes = np.array([[1001], [2001]])
    else:
        data_codes = np.array(
            [
                [1001, 0],
                [2001, 0],
                [1001, 1001],
                [2001, 1001],
                [2001, 2001],
            ]
        )
        if degree == 3:
            data_codes = np.column_stack([data_codes, np.zeros(5, dtype=int)])

    expected = np.vstack([data_codes, np.repeat(data_codes, 4, axis=0)])
    assert_array_equal(feature_codes, expected)


def test_regressor_space_validates_canonical_feature_width():
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=Polynomial(degree=2, include_bias=False),
    )

    with pytest.raises(ValueError, match="feature matrix with 6 columns"):
        dictionary.regressor_space(n_inputs=1, n_features=6)


class _DuckTypedBasisWithoutFeatureCodes:
    degree = 2


class _InvalidFeatureCodeWidthPolynomial(Polynomial):
    def _get_feature_codes(
        self,
        base_codes,
        *,
        xlag=1,
        ylag=1,
        model_type="NARMAX",
    ):
        return np.zeros((5, 1), dtype=int)


@pytest.mark.parametrize("invalid_n_features", [True, 1.5, "2"])
def test_regressor_space_rejects_non_integer_feature_width(invalid_n_features):
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=_DuckTypedBasisWithoutFeatureCodes(),
    )

    with pytest.raises(TypeError, match="n_features must be a non-negative integer"):
        dictionary.regressor_space(n_inputs=1, n_features=invalid_n_features)


def test_regressor_space_rejects_negative_feature_width_for_custom_basis():
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=_DuckTypedBasisWithoutFeatureCodes(),
    )

    with pytest.raises(ValueError, match="n_features must be a non-negative integer"):
        dictionary.regressor_space(n_inputs=1, n_features=-1)


def test_regressor_space_accepts_zero_and_numpy_integer_feature_width():
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=_DuckTypedBasisWithoutFeatureCodes(),
    )

    regressor_code = dictionary.regressor_space(
        n_inputs=1,
        n_features=np.int64(0),
    )

    assert regressor_code.shape == (0, 2)


def test_regressor_space_rejects_invalid_explicit_feature_code_width():
    dictionary = RegressorDictionary(
        xlag=1,
        ylag=1,
        basis_function=_InvalidFeatureCodeWidthPolynomial(
            degree=2,
            include_bias=False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Expected 2 columns, but got 1",
    ):
        dictionary.regressor_space(n_inputs=1, n_features=5)


def test_house():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    )

    output = np.array(
        [
            1,
            0.18970318,
            0.10702653,
            0.33617182,
            0.42495315,
            0.11959832,
            0.3782042,
            0.12995458,
            0.26016588,
            0.08369197,
        ]
    )
    assert_almost_equal(house(a), output)


def test_row_house():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    ).reshape(-1, 1)

    b = np.array(
        [
            0.90009285,
            0.21392929,
            0.58429212,
            0.55761456,
            0.65178413,
            0.4061564,
            0.4353402,
            0.02365408,
            0.52291863,
            0.185921,
        ]
    ).reshape(-1, 1)

    output = np.array(
        [
            [-1.1861246],
            [0.01063002],
            [-0.82404988],
            [-0.30077851],
            [-0.28515117],
            [-0.47901921],
            [0.00536996],
            [0.22732148],
            [-0.39637961],
            [-0.15920982],
        ]
    )
    assert_almost_equal(rowhouse(a, b), output)


def test_get_max_lag():
    output1 = 1
    r = RegressorDictionary(
        xlag=1, ylag=1, basis_function=Polynomial(degree=1)
    )._get_max_lag()
    output2 = 3
    r2 = RegressorDictionary(
        xlag=1, ylag=3, basis_function=Polynomial(degree=1)
    )._get_max_lag()
    assert_equal(output1, r)
    assert_equal(output2, r2)


def test_errors():
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=-1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=-2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=-2, ylag=2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=1,
    )
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=1)
        ).create_narmax_code,
        n_inputs=0,
    )


def test_create_narmax_code_ylist():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=2, ylag=[1, 2], basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_create_narmax_code_xlist():
    output1 = np.array([2001, 2002]), ([1001, 1002])
    r1 = RegressorDictionary(
        xlag=[1, 2], ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=1)
    assert_array_equal(output1, r1)


def test_create_narmax_code_miso():
    output1 = np.concatenate(
        np.array(
            [np.array([2001, 2002, 3001, 3002]), np.array([1001, 1002])], dtype=object
        )
    )
    r1 = RegressorDictionary(
        xlag=[[1, 2], [1, 2]], ylag=2, basis_function=Polynomial(degree=1)
    ).create_narmax_code(n_inputs=2)
    assert_array_equal(output1, np.concatenate(r1))


def test_regressor_space_raise():
    assert_raises(
        ValueError,
        RegressorDictionary(
            xlag=2, ylag=2, basis_function=Polynomial(degree=1), model_type="NARARMAX"
        ).regressor_space,
        n_inputs=1,
    )


def test_model_predict():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    print(model.final_model, model.err.sum())
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_nfir():
    model = FROLS(
        n_terms=5,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=2)


def test_model_predict_steps_none():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_3():
    model = FROLS(
        n_terms=5,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_fourier_steps_none():
    model = FROLS(
        order_selection=True,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._basis_function_predict(x=X_test, y_initial=y_test)
    assert_almost_equal(yhat.mean(), y_test[model.max_lag : :].mean(), decimal=6)


def test_model_predict_fourier_steps_1():
    model = FROLS(
        order_selection=True,
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=6)


def test_model_predict_fourier_nar_inputs():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


def test_model_predict_fourier_raises():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )


def test_model_predict_fourier_value_error():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_step_prediction,
        x=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=None,
    )


def test_model_predict_fourier_horizon_error():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "NARRARMAX"
    assert_raises(
        ValueError,
        model._basis_function_n_steps_horizon,
        x=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=10,
    )


def test_basis_function_predict_nfir_branch():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)

    horizon = model.max_lag + 5
    y_segment = y_test[:horizon]
    x_segment = X_test[:horizon]
    result = model._basis_function_predict(
        x=x_segment, y_initial=y_segment, forecast_horizon=horizon
    )

    assert result.shape[0] == horizon - model.max_lag


def test_basis_function_predict_invalid_type():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)
    model.model_type = "UNKNOWN"

    with pytest.raises(ValueError, match="model_type must be NARMAX, NAR or NFIR"):
        model._basis_function_predict(
            x=X_test[: model.max_lag + 5],
            y_initial=y_test[: model.max_lag + 5],
            forecast_horizon=model.max_lag + 5,
        )


def test_basis_function_n_step_prediction_nfir_branch():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)

    horizon = model.max_lag + 6
    y_segment = y_test[:horizon]
    x_segment = X_test[:horizon]
    result = model._basis_function_n_step_prediction(
        x=x_segment,
        y=y_segment,
        steps_ahead=2,
        forecast_horizon=horizon,
    )

    assert result.shape[0] == horizon - model.max_lag


def test_basis_function_n_steps_horizon_adjusts_step_nar():
    model = PredictableMSS(model_type="NAR")
    horizon = model.max_lag + 4
    y_segment = np.arange(horizon, dtype=float).reshape(-1, 1)

    result = model._basis_function_n_steps_horizon(
        x=None,
        y=y_segment,
        steps_ahead=horizon,
        forecast_horizon=horizon,
    )

    assert result.shape == (horizon - model.max_lag, 1)


def test_basis_function_n_steps_horizon_adjusts_step_nfir():
    model = PredictableMSS(model_type="NFIR")
    horizon = model.max_lag + 4
    y_segment = np.arange(horizon, dtype=float).reshape(-1, 1)
    x_segment = np.ones((horizon, 1))

    result = model._basis_function_n_steps_horizon(
        x=x_segment,
        y=y_segment,
        steps_ahead=horizon,
        forecast_horizon=horizon,
    )

    assert result.shape == (horizon - model.max_lag, 1)


def test_nar_step_ahead_insufficient_initial_conditions():
    """Test that _nar_step_ahead raises an error if input is too short."""
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NAR",
    )
    model.fit(y=y_train)

    with pytest.raises(ValueError, match="Insufficient initial condition elements!"):
        model._nar_step_ahead(y[0], steps_ahead=2)


@pytest.mark.parametrize("steps_ahead", [0, -1])
def test_nar_step_ahead_rejects_non_positive_steps(steps_ahead):
    model = RecordingPredictableMSS(model_type="NAR")
    y_segment = np.arange(model.max_lag + 1, dtype=float).reshape(-1, 1)

    with pytest.raises(
        ValueError,
        match="steps_ahead must be integer and > zero",
    ):
        model._nar_step_ahead(y_segment, steps_ahead=steps_ahead)


def test_nar_step_ahead_handles_multiple_segments():
    model = RecordingPredictableMSS(model_type="NAR")
    y_segment = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)

    result = model._nar_step_ahead(y_segment, steps_ahead=2)

    assert result.shape == (y_segment.shape[0] - model.max_lag, 1)
    assert_array_equal(result[:, 0], np.array([0.0, 1.0, 0.0, 1.0, 0.0]))
    assert [call[2] for call in model.prediction_calls] == [2, 2, 1]
    assert [call[0] for call in model.prediction_calls] == [None, None, None]
    for block, call in enumerate(model.prediction_calls):
        start = block * 2
        assert_array_equal(call[1], y_segment[start : start + model.max_lag])


def test_nar_step_ahead_handles_single_segment():
    model = RecordingPredictableMSS(model_type="NAR")
    y_segment = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    result = model._nar_step_ahead(y_segment, steps_ahead=10)

    assert result.shape == (3, 1)
    assert_array_equal(result[:, 0], np.arange(3, dtype=float))
    assert len(model.prediction_calls) == 1
    assert model.prediction_calls[0][2] == 3
    assert_array_equal(model.prediction_calls[0][1], y_segment[: model.max_lag])


def test_nar_step_ahead_with_only_initial_conditions_returns_empty_prediction():
    model = RecordingPredictableMSS(model_type="NAR")
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)

    result = model._nar_step_ahead(y_initial, steps_ahead=3)

    assert result.shape == (0, 1)
    assert model.prediction_calls == []


def test_nar_step_ahead_preserves_fractional_predictions_for_integer_output():
    model = RecordingPredictableMSS(model_type="NAR", prediction_offset=0.5)
    y_segment = np.arange(model.max_lag + 3, dtype=int).reshape(-1, 1)

    result = model._nar_step_ahead(y_segment, steps_ahead=2)

    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result[:, 0], np.array([0.5, 1.5, 0.5]))


def test_narmax_predict_reference_promotes_integer_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    model = PredictableMSS(model_type="NAR")
    model.n_inputs = 0
    model.theta = np.array([[0.6]])
    y_initial = xp.asarray([[1], [2]], dtype=xp.int64)

    with config_context(array_api_dispatch=True):
        result = model._narmax_predict_reference(
            x=None,
            y_initial=y_initial,
            forecast_horizon=5,
        )

    assert result.__array_namespace__() is xp
    assert xp.isdtype(result.dtype, "real floating")
    xp_assert_allclose(result, np.array([[1.2], [0.72], [0.432]]))


def test_narmarx_step_ahead_insufficient_initial_conditions():
    """Test that _narmax_step_ahead raises an error if input is too short."""
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)

    with pytest.raises(ValueError, match="Insufficient initial condition elements!"):
        model.narmax_n_step_ahead(X_train, y[0], steps_ahead=2)


def test_narmax_n_step_ahead_handles_single_segment():
    model = PredictableMSS(model_type="NARMAX")
    horizon = model.max_lag + 4
    x_data = np.ones((horizon, 1))
    y_data = np.arange(horizon, dtype=float).reshape(-1, 1)

    result = model.narmax_n_step_ahead(x_data, y_data, steps_ahead=horizon)

    assert result.shape == (horizon - model.max_lag, 1)


def test_miso_x_lag_list_single_input_int():
    """Test get_miso_x_lag_list with a single input and integer xlag."""
    model = FROLS(
        xlag=[[1, 2, 3], [1, 2, 3]],
        basis_function=Polynomial(degree=1),
    )

    expected_output = np.array([2001, 2002, 2003, 3001, 3002, 3003])
    result = model.get_miso_x_lag_list(n_inputs=2)

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, got {result}"


def test_siso_x_lag_list_single_input_list():
    """Test get_siso_x_lag_list with a single input and xlag as a list."""
    model = FROLS(
        xlag=[1, 3, 6],
        basis_function=Polynomial(degree=1),
    )

    expected_output = np.array([2001, 2003, 2006])
    result = model.get_siso_x_lag_list()

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, got {result}"


def test_miso_x_lag_list_single_input_list():
    """Test get_miso_x_lag_list with a single input and xlag as a list."""
    model = FROLS(
        xlag=[[1, 3, 6], [2]],
        basis_function=Polynomial(degree=1),
    )

    expected_output = np.array([2001, 2003, 2006, 3002])
    result = model.get_miso_x_lag_list(n_inputs=2)

    assert np.array_equal(
        result, expected_output
    ), f"Expected {expected_output}, got {result}"


def test_miso_x_lag_list_accepts_int_entries_per_input():
    """Ensure integer xlag entries are expanded when n_inputs > 1."""
    regressor = RegressorDictionary(
        xlag=[2, 3], ylag=1, basis_function=Polynomial(degree=1)
    )

    expected = np.array([2001, 2002, 3001, 3002, 3003])
    result = regressor.get_miso_x_lag_list(n_inputs=2)

    assert_array_equal(result, expected)


class ConcreteMSS(BaseMSS):
    def __init__(self, model_type="NARMAX"):
        super().__init__()
        self.model_type = model_type

    def some_method(self):
        pass

    def _basis_function_n_step_prediction(
        self,
        X: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        pass

    def _basis_function_predict(
        self,
        X: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        pass

    def _model_prediction(
        self,
        X: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        pass

    def _nfir_predict(self, X: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        pass

    def _n_step_ahead_prediction(self, X, y, steps_ahead):
        return super()._n_step_ahead_prediction(X, y, steps_ahead)

    def narmax_n_step_ahead(self, X, y, steps_ahead):
        """Mock function for NARMAX predictions."""
        return np.array([0.5] * steps_ahead)  # Dummy prediction

    def _nar_step_ahead(self, y, steps_ahead):
        """Mock function for NAR predictions."""
        return np.array([1.0] * steps_ahead)  # Dummy prediction

    def fit(self, *, X, y):
        pass

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        pass


class PredictableMSS(BaseMSS):
    """Minimal implementation to exercise BaseMSS helpers deterministically."""

    def __init__(self, model_type="NAR"):
        super().__init__()
        self.model_type = model_type
        self.max_lag = 2
        self.n_inputs = 1
        self.basis_function = Polynomial(degree=1)
        self.theta = np.ones((1, 1))
        self.final_model = np.array([[1001]])
        self.pivv = np.array([0])
        self.xlag = [1, 2]
        self.ylag = [1, 2]

    def _basis_function_n_step_prediction(
        self,
        x: Optional[np.ndarray],
        y: np.ndarray,
        steps_ahead: int,
        forecast_horizon: int,
    ) -> np.ndarray:
        return super()._basis_function_n_step_prediction(
            x, y, steps_ahead, forecast_horizon
        )

    def _basis_function_predict(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        _ = x
        horizon = int(forecast_horizon or len(y_initial))
        data = np.arange(horizon, dtype=float).reshape(-1, 1)
        return data

    def _model_prediction(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        _ = x
        horizon = int(forecast_horizon or len(y_initial))
        return np.arange(horizon, dtype=float).reshape(-1, 1)

    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        _ = y_initial
        length = max(x.shape[0] - self.max_lag, 0)
        return np.arange(length, dtype=float).reshape(-1, 1)

    def _n_step_ahead_prediction(self, x, y, steps_ahead):
        return super()._n_step_ahead_prediction(x, y, steps_ahead)

    def narmax_n_step_ahead(self, x, y, steps_ahead):
        return super().narmax_n_step_ahead(x, y, steps_ahead)

    def fit(self, *, X, y):
        _ = (X, y)

    def predict(
        self,
        *,
        X: Optional[np.ndarray] = None,
        y: np.ndarray,
        steps_ahead: Optional[int] = None,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        _ = (X, y, steps_ahead, forecast_horizon)
        return np.zeros((1, 1))


class RecordingPredictableMSS(PredictableMSS):
    """Predictable BaseMSS variant that records recursive NAR blocks."""

    def __init__(self, model_type="NAR", prediction_offset=0.0):
        super().__init__(model_type=model_type)
        self.prediction_calls = []
        self.prediction_offset = prediction_offset

    def _model_prediction(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        self.prediction_calls.append((x, y_initial.copy(), forecast_horizon))
        prediction = super()._model_prediction(x, y_initial, forecast_horizon)
        return prediction + self.prediction_offset


class ArrayAPIPredictableMSS(PredictableMSS):
    """Predictable BaseMSS variant that delegates NFIR prediction to the base path."""

    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        return BaseMSS._nfir_predict(self, x, y_initial)


def test_base_mss_initialization():
    """Test if BaseMSS initializes correctly."""
    model = ConcreteMSS()

    assert model.max_lag is None, "max_lag should be initialized as None"
    assert model.n_inputs is None, "n_inputs should be initialized as None"
    assert model.theta is None, "theta should be initialized as None"
    assert model.final_model is None, "final_model should be initialized as None"
    assert model.pivv is None, "pivv should be initialized as None"


def test_base_mss_is_instance_of_regressor_dict():
    """Test if BaseMSS is a subclass of RegressorDictionary."""
    model = ConcreteMSS()
    assert isinstance(model, BaseMSS), "ConcreteMSS should be an instance of BaseMSS"
    assert isinstance(
        model, RegressorDictionary
    ), "ConcreteMSS should inherit from RegressorDictionary"


def test_base_mss_abstract_methods():
    """Test if instantiating BaseMSS directly raises an error."""
    with pytest.raises(TypeError):
        BaseMSS()  # type: ignore[arg-type]


def test_n_step_ahead_prediction_narmax():
    """Test `_n_step_ahead_prediction` for NARMAX model."""
    model = ConcreteMSS(model_type="NARMAX")
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2, 3])
    steps_ahead = 3

    result = model._n_step_ahead_prediction(X, y, steps_ahead)
    expected = np.array([0.5, 0.5, 0.5])

    np.testing.assert_array_almost_equal(
        result, expected, err_msg="NARMAX prediction incorrect"
    )


def test_n_step_ahead_prediction_nar():
    """Test `_n_step_ahead_prediction` for NAR model."""
    model = ConcreteMSS(model_type="NAR")
    y = np.array([1, 2, 3])
    steps_ahead = 2

    result = model._n_step_ahead_prediction(None, y, steps_ahead)
    expected = np.array([1.0, 1.0])

    np.testing.assert_array_almost_equal(
        result, expected, err_msg="NAR prediction incorrect"
    )


def test_one_step_ahead_prediction_preserves_array_api_namespace():
    xp = pytest.importorskip("array_api_strict")
    model = PredictableMSS(model_type="NARMAX")
    model.theta = np.array([[2.0]])
    x_base = xp.asarray(np.array([[1.0], [2.0], [3.0]]))

    with config_context(array_api_dispatch=True):
        result = model._one_step_ahead_prediction(x_base)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_array_equal(result, [[2.0], [4.0], [6.0]])


def test_narmax_predict_preserves_array_api_namespace_with_numpy_theta():
    xp = pytest.importorskip("array_api_strict")
    model = PredictableMSS(model_type="NARMAX")
    model.final_model = np.array([[0]])
    model.theta = np.array([[2.0]])
    x_data = xp.asarray(np.ones((6, 1)))
    y_initial = xp.asarray(np.arange(6.0).reshape(-1, 1))

    with config_context(array_api_dispatch=True):
        result = model._narmax_predict(x_data, y_initial, forecast_horizon=6)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_array_equal(result, np.full((4, 1), 2.0))


def test_polynomial_narmax_predict_fast_matches_reference_with_interactions():
    model = PredictableMSS(model_type="NARMAX")
    model.max_lag = 2
    model.n_inputs = 2
    model.basis_function = Polynomial(degree=2)
    model.final_model = np.array(
        [
            [0, 0],
            [1001, 1001],
            [2002, 1001],
            [3001, 1002],
            [3002, 2001],
        ]
    )
    model.theta = np.array([[0.3], [-0.1], [0.4], [-0.2], [0.15]])
    x_data = np.array(
        [
            [1.0, 0.5],
            [2.0, 1.5],
            [3.0, 2.5],
            [4.0, 3.5],
            [5.0, 4.5],
            [6.0, 5.5],
        ]
    )
    y_initial = np.array([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2]])

    reference = model._narmax_predict_reference(
        x_data,
        y_initial,
        forecast_horizon=x_data.shape[0],
    )
    fast = model._polynomial_narmax_predict_fast(
        x_data,
        y_initial,
        forecast_horizon=x_data.shape[0],
    )

    assert_array_equal(fast.shape, reference.shape)
    assert_almost_equal(fast, reference, decimal=12)


def test_polynomial_narmax_predict_cache_preserves_final_model_order():
    model = PredictableMSS(model_type="NARMAX")
    model.max_lag = 2
    model.n_inputs = 1
    model.basis_function = Polynomial(degree=2)
    model.final_model = np.array(
        [
            [2002, 1001],
            [1001, 1001],
            [0, 0],
        ]
    )

    exponent_matrix = model._get_polynomial_narmax_predict_exponents()
    expected = np.vstack([model._code2exponents(code=row) for row in model.final_model])

    assert_array_equal(exponent_matrix, expected)


def test_polynomial_narmax_fast_path_is_disabled_for_short_initial_conditions():
    model = PredictableMSS(model_type="NARMAX")
    model.max_lag = 2
    model.n_inputs = 1
    model.basis_function = Polynomial(degree=2)
    model.final_model = np.array([[1001]])
    model.theta = np.array([[1.0]])
    x_data = np.ones((4, 1))
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)

    assert not model._should_use_polynomial_narmax_fast_path(
        x_data,
        y_initial,
        forecast_horizon=x_data.shape[0],
    )


def test_nfir_predict_preserves_array_api_namespace_with_numpy_theta():
    xp = pytest.importorskip("array_api_strict")
    model = ArrayAPIPredictableMSS(model_type="NFIR")
    model.final_model = np.array([[0]])
    model.theta = np.array([[2.0]])
    x_data = xp.asarray(np.ones((6, 1)))
    y_initial = xp.asarray(np.arange(6.0).reshape(-1, 1))

    with config_context(array_api_dispatch=True):
        result = model._nfir_predict(x_data, y_initial)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_array_equal(result, np.full((4, 1), 2.0))


def test_n_step_ahead_prediction_invalid_model():
    """Test `_n_step_ahead_prediction` with an invalid model type."""
    model = ConcreteMSS(model_type="NFIR")

    with pytest.raises(
        ValueError,
        match=r"n_steps_ahead prediction will be implemented"
        r" for NFIR models in v0\.4\..*",
    ):
        model._n_step_ahead_prediction(None, np.array([1, 2, 3]), 2)
