# pylint: disable=protected-access,unused-argument,redefined-outer-name,useless-super-delegation,arguments-renamed,abstract-class-instantiated
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy import config_context, get_config
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


def test_model_predict_fourier_nar_preserves_inputs_and_model_state():
    model = FROLS(
        order_selection=True,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(),
        basis_function=Fourier(degree=2, n=1),
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    original_n_inputs = model.n_inputs
    x_before = X_test.copy()
    conflicting_x = np.arange(len(X_test) * 3, dtype=np.int64).reshape(-1, 3)
    conflicting_x_before = conflicting_x.copy()
    y_before = y_test.copy()

    first = model.predict(X=X_test, y=y_test)
    second = model.predict(X=conflicting_x, y=y_test)

    assert_equal(model.n_inputs, original_n_inputs)
    assert_array_equal(X_test, x_before)
    assert_array_equal(conflicting_x, conflicting_x_before)
    assert_array_equal(y_test, y_before)
    assert first.dtype == second.dtype
    np.testing.assert_allclose(first, second, rtol=0, atol=0)


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


@pytest.mark.parametrize("steps_ahead", [0, -1, 1.5, True, np.bool_(False)])
def test_nar_step_ahead_rejects_invalid_steps(steps_ahead):
    model = RecordingPredictableMSS(model_type="NAR")
    y_segment = np.arange(model.max_lag + 1, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="steps_ahead"):
        model._nar_step_ahead(y_segment, steps_ahead=steps_ahead)


def test_nar_step_ahead_accepts_numpy_integer_steps():
    model = RecordingPredictableMSS(model_type="NAR")
    y_segment = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    result = model._nar_step_ahead(y_segment, steps_ahead=np.int64(2))

    assert result.shape == (3, 1)
    assert [call[2] for call in model.prediction_calls] == [2, 1]


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


def test_basis_function_n_step_prediction_uses_shared_nar_blocks():
    model = RecordingPredictableMSS(model_type="NAR")
    model.basis_function = Fourier(degree=1)
    y_segment = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)

    result = model._basis_function_n_step_prediction(
        x=None,
        y=y_segment,
        steps_ahead=3,
        forecast_horizon=1,
    )

    assert result.shape == (y_segment.shape[0] - model.max_lag, 1)
    assert_array_equal(result[:, 0], np.array([0.0, 1.0, 2.0, 0.0, 1.0]))
    assert model.prediction_calls == []
    assert [call[2] for call in model.basis_prediction_calls] == [3, 2]
    assert_array_equal(
        model.basis_prediction_calls[0][1],
        y_segment[: model.max_lag],
    )
    assert_array_equal(
        model.basis_prediction_calls[1][1],
        y_segment[3 : 3 + model.max_lag],
    )


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
        if x is not None:
            prediction_length = x.shape[0] - self.max_lag
        else:
            prediction_length = forecast_horizon
        return np.arange(prediction_length, dtype=float).reshape(-1, 1)

    def _model_prediction(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        if x is not None:
            prediction_length = x.shape[0] - self.max_lag
        else:
            prediction_length = forecast_horizon
        return np.arange(prediction_length, dtype=float).reshape(-1, 1)

    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        _ = y_initial
        length = max(x.shape[0] - self.max_lag, 0)
        return np.arange(length, dtype=float).reshape(-1, 1)

    def _n_step_ahead_prediction(self, x, y, steps_ahead):
        return super()._n_step_ahead_prediction(x, y, steps_ahead)

    def _one_step_ahead_prediction(self, x, y=None):
        prediction_length = y.shape[0] - self.max_lag
        if self.model_type == "NFIR":
            prediction_length = x.shape[0] - self.max_lag
        return np.arange(prediction_length, dtype=float).reshape(-1, 1)

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
        return super().predict(
            X=X,
            y=y,
            steps_ahead=steps_ahead,
            forecast_horizon=forecast_horizon,
        )


class RecordingPredictableMSS(PredictableMSS):
    """Predictable BaseMSS variant that records recursive NAR blocks."""

    def __init__(self, model_type="NAR", prediction_offset=0.0):
        super().__init__(model_type=model_type)
        self.prediction_calls = []
        self.basis_prediction_calls = []
        self.prediction_offset = prediction_offset

    def _basis_function_predict(
        self,
        x: Optional[np.ndarray],
        y_initial: np.ndarray,
        forecast_horizon: int = 1,
    ) -> np.ndarray:
        self.basis_prediction_calls.append((x, y_initial.copy(), forecast_horizon))
        prediction = super()._basis_function_predict(
            x,
            y_initial,
            forecast_horizon,
        )
        return prediction + self.prediction_offset

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

    def _model_prediction(self, x, y_initial, forecast_horizon=1):
        return BaseMSS._model_prediction(self, x, y_initial, forecast_horizon)

    def _nfir_predict(self, x: np.ndarray, y_initial: np.ndarray) -> np.ndarray:
        return BaseMSS._nfir_predict(self, x, y_initial)


def _segmented_public_prediction_reference(model, x, y, steps_ahead):
    """Build an n-step reference from independent public free runs."""
    blocks = [y[: model.max_lag].astype(float, copy=True)]
    block_end = model.max_lag
    while block_end < len(y):
        block_horizon = min(steps_ahead, len(y) - block_end)
        block_start = block_end - model.max_lag
        y_initial = y[block_start:block_end]
        if model.model_type == "NAR":
            block_prediction = model.predict(
                X=None,
                y=y_initial,
                forecast_horizon=block_horizon,
            )
        else:
            block_prediction = model.predict(
                X=x[block_start : block_end + block_horizon],
                y=y_initial,
            )
        blocks.append(block_prediction[-block_horizon:])
        block_end += block_horizon

    return np.concatenate(blocks, axis=0)


@pytest.mark.parametrize(
    ("invalid_y", "error_message"),
    [
        (None, "y cannot be None"),
        (np.ones(4), "y must be a 2D array"),
        (np.ones((4, 2)), "single column"),
    ],
)
def test_predict_validates_output_shape_at_public_boundary(
    invalid_y,
    error_message,
):
    model = PredictableMSS(model_type="NAR")

    with pytest.raises(ValueError, match=error_message):
        model.predict(X=None, y=invalid_y, steps_ahead=1)


def test_predict_validates_optional_input_shape_at_public_boundary():
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 2, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="X must be a 2D array"):
        model.predict(X=np.ones(len(y_data)), y=y_data, steps_ahead=1)


@pytest.mark.parametrize("model_type", ["NARMAX", "NFIR"])
def test_predict_requires_input_for_input_models(model_type):
    model = PredictableMSS(model_type=model_type)
    y_data = np.arange(model.max_lag + 2, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match=f"X cannot be None for {model_type}"):
        model.predict(X=None, y=y_data)


@pytest.mark.parametrize("model_type", ["NARMAX", "NFIR"])
def test_predict_validates_number_of_input_columns(model_type):
    model = PredictableMSS(model_type=model_type)
    y_data = np.ones((model.max_lag + 2, 1))
    x_data = np.ones((len(y_data), 2))

    with pytest.raises(ValueError, match="X must have 1 input column"):
        model.predict(X=x_data, y=y_data)


@pytest.mark.parametrize("model_type", ["NARMAX", "NFIR"])
@pytest.mark.parametrize("steps_ahead", [1, 2])
@pytest.mark.parametrize("input_length_offset", [-1, 1])
def test_predict_requires_equal_lengths_for_conditioned_input_models(
    model_type,
    steps_ahead,
    input_length_offset,
):
    model = PredictableMSS(model_type=model_type)
    y_data = np.ones((model.max_lag + 3, 1))

    with pytest.raises(ValueError, match="same number of samples"):
        model.predict(
            X=np.ones((len(y_data) + input_length_offset, 1)),
            y=y_data,
            steps_ahead=steps_ahead,
        )


@pytest.mark.parametrize("model_type", ["NAR", "NARMAX", "NFIR"])
def test_predict_rejects_free_run_input_shorter_than_max_lag(model_type):
    model = PredictableMSS(model_type=model_type)
    y_data = np.ones((model.max_lag, 1))
    x_data = np.ones((model.max_lag - 1, 1))

    with pytest.raises(ValueError, match="X must contain at least"):
        model.predict(X=x_data, y=y_data)


def test_predict_rejects_unknown_model_type_at_public_boundary():
    model = PredictableMSS(model_type="UNKNOWN")
    y_data = np.ones((model.max_lag + 1, 1))

    with pytest.raises(ValueError, match="model_type must be NARMAX, NAR or NFIR"):
        model.predict(X=None, y=y_data, steps_ahead=1)


@pytest.mark.parametrize("invalid_steps", [0, -1, 1.0, True, np.bool_(True)])
def test_predict_rejects_invalid_steps_ahead(invalid_steps):
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="steps_ahead"):
        model.predict(X=None, y=y_data, steps_ahead=invalid_steps)


def test_predict_accepts_numpy_integer_steps_ahead():
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    result = model.predict(X=None, y=y_data, steps_ahead=np.int64(2))
    expected = model.predict(X=None, y=y_data, steps_ahead=2)

    assert_array_equal(result, expected)


def test_predict_rejects_backend_boolean_prediction_parameters():
    torch = pytest.importorskip("torch")
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="steps_ahead"):
        model.predict(X=None, y=y_data, steps_ahead=torch.tensor(True))

    with pytest.raises(ValueError, match="forecast_horizon"):
        model.predict(
            X=None,
            y=y_data[: model.max_lag],
            forecast_horizon=torch.tensor(True),
        )


def test_predict_accepts_backend_integer_prediction_parameters():
    torch = pytest.importorskip("torch")
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    n_step = model.predict(X=None, y=y_data, steps_ahead=torch.tensor(2))
    expected_n_step = model.predict(X=None, y=y_data, steps_ahead=2)
    free_run = model.predict(
        X=None,
        y=y_data[: model.max_lag],
        forecast_horizon=torch.tensor(3),
    )
    expected_free_run = model.predict(
        X=None,
        y=y_data[: model.max_lag],
        forecast_horizon=3,
    )

    assert_array_equal(n_step, expected_n_step)
    assert_array_equal(free_run, expected_free_run)


@pytest.mark.parametrize(
    "invalid_horizon",
    [None, -1, 1.0, True, np.bool_(False)],
)
def test_nar_free_run_rejects_invalid_active_forecast_horizon(invalid_horizon):
    model = PredictableMSS(model_type="NAR")
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)

    with pytest.raises(ValueError, match="forecast_horizon"):
        model.predict(X=None, y=y_initial, forecast_horizon=invalid_horizon)


def test_nar_free_run_accepts_numpy_integer_and_zero_forecast_horizon():
    model = RecordingPredictableMSS(model_type="NAR")
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)

    prefix_only = model.predict(
        X=None,
        y=y_initial,
        forecast_horizon=np.int64(0),
    )
    prediction = model.predict(
        X=None,
        y=y_initial,
        forecast_horizon=np.int64(3),
    )

    assert_array_equal(prefix_only, y_initial)
    assert not np.shares_memory(prefix_only, y_initial)
    assert prediction.shape == (model.max_lag + 3, 1)
    assert len(model.prediction_calls) == 1


def test_nar_free_run_without_input_uses_horizon_and_only_initial_conditions():
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)
    modified = y_data.copy()
    modified[model.max_lag :] = -999

    full_y = model.predict(X=None, y=y_data, forecast_horizon=4)
    initial_only = model.predict(
        X=None,
        y=y_data[: model.max_lag],
        forecast_horizon=4,
    )
    modified_suffix = model.predict(X=None, y=modified, forecast_horizon=4)

    assert full_y.shape == (model.max_lag + 4, 1)
    assert_array_equal(full_y, initial_only)
    assert_array_equal(full_y, modified_suffix)
    assert_array_equal(full_y[: model.max_lag], y_data[: model.max_lag])


def test_nar_free_run_with_input_uses_input_length_and_ignores_horizon():
    model = PredictableMSS(model_type="NAR")
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)
    x_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)

    result = model.predict(
        X=x_data,
        y=y_initial,
        forecast_horizon="ignored",
    )
    changed_x = model.predict(
        X=-x_data,
        y=y_initial,
        forecast_horizon=999,
    )

    assert result.shape == x_data.shape
    assert_array_equal(result[: model.max_lag], y_initial)
    assert_array_equal(result, changed_x)


@pytest.mark.parametrize("steps_ahead", [1, 2, 3])
def test_conditioned_nar_prediction_uses_y_length_and_ignores_x_and_horizon(
    steps_ahead,
):
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)
    irrelevant_x = np.arange(
        3 * (model.max_lag + 1),
        dtype=np.int64,
    ).reshape(-1, 3)
    x_before = irrelevant_x.copy()

    expected = model.predict(X=None, y=y_data, steps_ahead=steps_ahead)
    result = model.predict(
        X=irrelevant_x,
        y=y_data,
        steps_ahead=steps_ahead,
        forecast_horizon="ignored",
    )

    assert result.shape == y_data.shape
    assert result.dtype == expected.dtype
    assert_array_equal(result, expected)
    assert_array_equal(irrelevant_x, x_before)


@pytest.mark.parametrize("steps_ahead", [1, 3])
def test_conditioned_nar_discards_x_before_namespace_dispatch(steps_ahead):
    xp = pytest.importorskip("array_api_strict")
    model = PredictableMSS(model_type="NAR")
    y_data = np.arange(model.max_lag + 4, dtype=float).reshape(-1, 1)
    irrelevant_x = xp.asarray(np.ones((model.max_lag + 1, 2)))
    expected = model.predict(X=None, y=y_data, steps_ahead=steps_ahead)

    with config_context(array_api_dispatch=True):
        result = model.predict(
            X=irrelevant_x,
            y=y_data,
            steps_ahead=steps_ahead,
        )

    assert isinstance(result, np.ndarray)
    assert_array_equal(result, expected)


def test_narmax_free_run_uses_x_length_and_only_initial_conditions():
    model = PredictableMSS(model_type="NARMAX")
    x_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)
    y_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)

    full_y = model.predict(
        X=x_data,
        y=y_data,
        forecast_horizon="ignored",
    )
    initial_only = model.predict(
        X=x_data,
        y=y_data[: model.max_lag],
        forecast_horizon="ignored",
    )

    assert full_y.shape == x_data.shape
    assert_array_equal(full_y, initial_only)


def test_narmax_one_step_uses_aligned_y_interval_and_ignores_horizon():
    model = PredictableMSS(model_type="NARMAX")
    x_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)
    y_data = np.arange(len(x_data), dtype=float).reshape(-1, 1)

    result = model.predict(
        X=x_data,
        y=y_data,
        steps_ahead=1,
        forecast_horizon="ignored",
    )

    assert result.shape == y_data.shape
    assert_array_equal(result[: model.max_lag], y_data[: model.max_lag])


@pytest.mark.parametrize("basis_function", [Polynomial(degree=1), Fourier(degree=1)])
def test_nfir_prediction_modes_are_feed_forward_aliases(basis_function):
    model = PredictableMSS(model_type="NFIR")
    model.basis_function = basis_function
    x_data = np.arange(model.max_lag + 5, dtype=float).reshape(-1, 1)
    y_data = np.arange(len(x_data), dtype=float).reshape(-1, 1)
    changed_suffix = y_data.copy()
    changed_suffix[model.max_lag :] = -999

    free_run = model.predict(X=x_data, y=y_data, forecast_horizon="ignored")
    initial_only = model.predict(
        X=x_data,
        y=y_data[: model.max_lag],
        forecast_horizon="ignored",
    )
    one_step = model.predict(
        X=x_data,
        y=y_data,
        steps_ahead=1,
        forecast_horizon="ignored",
    )
    n_step = model.predict(
        X=x_data,
        y=y_data,
        steps_ahead=3,
        forecast_horizon="ignored",
    )
    changed = model.predict(X=x_data, y=changed_suffix, steps_ahead=2)

    assert free_run.shape == x_data.shape
    assert_array_equal(free_run, initial_only)
    assert_array_equal(free_run, one_step)
    assert_array_equal(free_run, n_step)
    assert_array_equal(free_run, changed)
    assert_array_equal(free_run[: model.max_lag], y_data[: model.max_lag])


@pytest.mark.parametrize(
    ("model_type", "steps_ahead", "use_x", "forecast_horizon"),
    [
        ("NAR", None, False, 0),
        ("NAR", 1, False, 1),
        ("NAR", 3, False, 1),
        ("NARMAX", None, True, None),
        ("NARMAX", 1, True, None),
        ("NARMAX", 3, True, None),
        ("NFIR", None, True, None),
        ("NFIR", 1, True, None),
        ("NFIR", 3, True, None),
    ],
)
def test_predict_returns_prefix_without_evaluating_empty_suffix(
    model_type,
    steps_ahead,
    use_x,
    forecast_horizon,
):
    model = RecordingPredictableMSS(model_type=model_type)
    y_initial = np.arange(model.max_lag, dtype=float).reshape(-1, 1)
    x_data = np.ones((model.max_lag, 1)) if use_x else None

    result = model.predict(
        X=x_data,
        y=y_initial,
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )

    assert_array_equal(result, y_initial)
    assert not np.shares_memory(result, y_initial)
    assert result.dtype == y_initial.dtype
    assert model.prediction_calls == []
    assert model.basis_prediction_calls == []


@pytest.mark.parametrize("model_type", ["NAR", "NARMAX", "NFIR"])
def test_predict_rejects_fewer_than_max_lag_initial_conditions(model_type):
    model = PredictableMSS(model_type=model_type)
    y_short = np.ones((model.max_lag - 1, 1))
    x_data = None if model_type == "NAR" else np.ones((model.max_lag, 1))

    with pytest.raises(ValueError, match="Insufficient initial condition"):
        model.predict(
            X=x_data,
            y=y_short,
            steps_ahead=1,
            forecast_horizon=1,
        )


@pytest.mark.parametrize("basis_function", [Polynomial(degree=1), Fourier(degree=1)])
@pytest.mark.parametrize("steps_ahead", [2, 3, 20])
def test_public_narmax_n_step_matches_segmented_free_runs(
    basis_function,
    steps_ahead,
):
    model = PredictableMSS(model_type="NARMAX")
    model.basis_function = basis_function
    x_data = np.arange(18, dtype=float).reshape(9, 2)
    y_data = np.arange(9, dtype=float).reshape(-1, 1)
    model.n_inputs = x_data.shape[1]
    expected = _segmented_public_prediction_reference(
        model,
        x_data,
        y_data,
        steps_ahead,
    )

    result = model.predict(
        X=x_data,
        y=y_data,
        steps_ahead=steps_ahead,
        forecast_horizon=1,
    )

    assert result.shape == y_data.shape
    assert_array_equal(result, expected)


@pytest.mark.parametrize("basis_function", [Polynomial(degree=1), Fourier(degree=1)])
def test_narmax_scheduler_uses_exact_windows_and_partial_final_block(
    basis_function,
):
    model = RecordingPredictableMSS(model_type="NARMAX", prediction_offset=0.5)
    model.basis_function = basis_function
    model.n_inputs = 2
    x_data = np.arange(18, dtype=int).reshape(9, 2)
    y_data = np.arange(9, dtype=int).reshape(-1, 1)
    x_before = x_data.copy()
    y_before = y_data.copy()

    result = model.narmax_n_step_ahead(
        x_data,
        y_data,
        steps_ahead=np.int64(3),
    )
    calls = (
        model.prediction_calls
        if isinstance(basis_function, Polynomial)
        else model.basis_prediction_calls
    )

    assert result.shape == (len(y_data) - model.max_lag, 1)
    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(
        result[:, 0],
        np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5]),
    )
    assert [call[2] for call in calls] == [3, 3, 1]
    for call, block_start, block_horizon in zip(
        calls,
        [0, 3, 6],
        [3, 3, 1],
        strict=True,
    ):
        assert_array_equal(
            call[0],
            x_data[block_start : block_start + model.max_lag + block_horizon],
        )
        assert_array_equal(
            call[1],
            y_data[block_start : block_start + model.max_lag],
        )
    assert_array_equal(x_data, x_before)
    assert_array_equal(y_data, y_before)
    assert model.n_inputs == 2


def test_narmax_scheduler_validates_required_and_aligned_input():
    model = PredictableMSS(model_type="NARMAX")
    y_data = np.ones((model.max_lag + 2, 1))

    with pytest.raises(ValueError, match="X cannot be None"):
        model.narmax_n_step_ahead(None, y_data, steps_ahead=2)

    with pytest.raises(ValueError, match="same number of samples"):
        model.narmax_n_step_ahead(
            np.ones((len(y_data) - 1, 1)),
            y_data,
            steps_ahead=2,
        )


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
    X = np.array([[1], [2], [3]])
    y = np.array([[1], [2], [3]])
    steps_ahead = 3

    result = model._n_step_ahead_prediction(X, y, steps_ahead)
    expected = np.array([0.5, 0.5, 0.5])

    np.testing.assert_array_almost_equal(
        result, expected, err_msg="NARMAX prediction incorrect"
    )


def test_n_step_ahead_prediction_nar():
    """Test `_n_step_ahead_prediction` for NAR model."""
    model = ConcreteMSS(model_type="NAR")
    y = np.array([[1], [2], [3]])
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
        result = BaseMSS._one_step_ahead_prediction(model, x_base)

    assert result.__array_namespace__().__name__ == xp.__name__
    xp_assert_array_equal(result, [[2.0], [4.0], [6.0]])


def test_one_step_ahead_prediction_promotes_integer_regressors():
    model = PredictableMSS(model_type="NARMAX")
    model.theta = np.array([[0.5]])
    regressors = np.array([[1], [2], [3]], dtype=np.int64)

    result = BaseMSS._one_step_ahead_prediction(model, regressors)

    assert np.issubdtype(result.dtype, np.floating)
    np.testing.assert_allclose(result, np.array([[0.5], [1.0], [1.5]]))


def test_prediction_dtype_preserves_floats_and_promotes_integral_data():
    model = PredictableMSS(model_type="NARMAX")

    float32_result = model._prediction_dtype(np, np.dtype(np.float32))
    mixed_float_result = model._prediction_dtype(
        np,
        np.dtype(np.float32),
        np.dtype(np.float64),
    )
    integer_result = model._prediction_dtype(
        np,
        np.dtype(np.int64),
        np.dtype(np.float32),
    )

    assert float32_result == np.dtype(np.float32)
    assert mixed_float_result == np.dtype(np.float64)
    assert np.issubdtype(integer_result, np.floating)


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


def test_prediction_exponents_cache_preserves_order_and_reuses_result():
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

    exponent_matrix = model._get_prediction_exponents()
    cached = model._get_prediction_exponents()
    expected = np.vstack([model._code2exponents(code=row) for row in model.final_model])

    assert_array_equal(exponent_matrix, expected)
    assert cached is exponent_matrix
    assert not exponent_matrix.flags.writeable
    assert model._get_polynomial_narmax_predict_exponents() is exponent_matrix


def test_prediction_exponents_cache_invalidates_after_in_place_model_change():
    model = PredictableMSS(model_type="NARMAX")
    model.final_model = np.array([[1001], [2001]])
    original = model._get_prediction_exponents()

    model.final_model[0, 0] = 2002
    updated = model._get_prediction_exponents()

    assert updated is not original
    assert_array_equal(
        updated,
        np.vstack([model._code2exponents(code=row) for row in model.final_model]),
    )


def test_prediction_exponents_cache_uses_effective_nar_input_count():
    model = PredictableMSS(model_type="NAR")
    model.n_inputs = 7
    original = model._get_prediction_exponents()

    model.n_inputs = 3
    cached = model._get_prediction_exponents()

    assert cached is original
    assert cached.shape == (len(model.final_model), model.max_lag)
    assert model.n_inputs == 3


def test_prediction_exponents_cache_handles_empty_model():
    model = PredictableMSS(model_type="NARMAX")
    model.final_model = np.empty((0, 1), dtype=int)

    result = model._get_prediction_exponents()

    assert result.shape == (0, model.max_lag * (1 + model.n_inputs))
    assert not result.flags.writeable


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


def test_public_array_api_fallback_preserves_model_state_and_inputs():
    xp = pytest.importorskip("array_api_strict")
    model = ArrayAPIPredictableMSS(model_type="NAR")
    model.final_model = np.array([[1001]])
    model.theta = np.array([[0.5]], dtype=np.float32)
    y_numpy = np.arange(model.max_lag, dtype=np.int64).reshape(-1, 1)
    expected_model = ArrayAPIPredictableMSS(model_type="NAR")
    expected_model.final_model = model.final_model.copy()
    expected_model.theta = model.theta.copy()
    expected = expected_model.predict(X=None, y=y_numpy, forecast_horizon=4)
    y_data = xp.asarray(y_numpy)
    original_theta = model.theta

    with config_context(array_api_dispatch=True):
        result = model.predict(X=None, y=y_data, forecast_horizon=4)

    assert result.__array_namespace__() is xp
    assert xp.isdtype(result.dtype, "real floating")
    xp_assert_allclose(result, expected)
    xp_assert_array_equal(y_data, y_numpy)
    assert model.theta is original_theta
    assert model._prediction_exponents_cache is not None
    assert model._prediction_exponents_cache_key is not None
    assert model.n_inputs == 1


def test_empty_array_api_prediction_skips_cpu_fallback_and_cache_warmup():
    xp = pytest.importorskip("array_api_strict")
    model = ArrayAPIPredictableMSS(model_type="NAR")
    y_initial = xp.asarray(np.arange(model.max_lag, dtype=np.float32).reshape(-1, 1))

    with config_context(array_api_dispatch=True):
        prediction = model.predict(
            X=None,
            y=y_initial,
            forecast_horizon=0,
        )

    assert prediction.__array_namespace__() is xp
    xp_assert_array_equal(prediction, y_initial)
    assert model._prediction_exponents_cache is None
    assert model._prediction_exponents_cache_key is None


def test_public_array_api_fallback_restores_nested_config_after_error():
    xp = pytest.importorskip("array_api_strict")
    model = ArrayAPIPredictableMSS(model_type="NAR")
    model.final_model = np.array([[1001]])
    model.theta = np.array([[0.5]])
    model._prediction_dispatch = MagicMock(side_effect=RuntimeError("failed"))
    y_data = xp.asarray(np.arange(model.max_lag, dtype=float).reshape(-1, 1))
    original_dispatch = get_config()["array_api_dispatch"]

    with config_context(array_api_dispatch=True):
        with pytest.raises(RuntimeError, match="failed"):
            model.predict(X=None, y=y_data, forecast_horizon=3)
        assert get_config()["array_api_dispatch"] is True

    assert get_config()["array_api_dispatch"] is original_dispatch


def test_n_step_ahead_prediction_nfir_delegates_to_feed_forward_kernel():
    model = PredictableMSS(model_type="NFIR")
    x_data = np.arange(model.max_lag + 3, dtype=float).reshape(-1, 1)
    y_data = np.arange(len(x_data), dtype=float).reshape(-1, 1)

    result = model._n_step_ahead_prediction(x_data, y_data, steps_ahead=2)
    expected = model._model_prediction(x_data, y_data)

    assert_array_equal(result, expected)
