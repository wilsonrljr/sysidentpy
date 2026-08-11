# pylint: disable=protected-access
# pyright: reportMissingTypeStubs=false
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)
from sklearn.dummy import DummyRegressor  # type: ignore[reportMissingTypeStubs]
from sklearn.linear_model import LinearRegression  # type: ignore[reportMissingTypeStubs]

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.general_estimators import NARX
from sysidentpy.tests._array_api_asserts import assert_allclose as xp_assert_allclose
from sysidentpy.tests.test_narmax_base import create_test_data
from sysidentpy.utils.information_matrix import build_lagged_matrix

base_estimator = LinearRegression()

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


_SENTINEL = object()


class _LargeFloatRegressor:
    def fit(self, x_data, y_data, **fit_params):
        _ = x_data, y_data, fit_params
        return self

    def predict(self, x_data):
        return np.full(x_data.shape[0], 1e40, dtype=np.float64)


def fit_narx_model(
    *,
    model_type="NARMAX",
    basis_function=None,
    x_data=_SENTINEL,
    y_data=_SENTINEL,
    xlag=2,
    ylag=2,
):
    if x_data is _SENTINEL:
        x_data = X_train
    if y_data is _SENTINEL:
        y_data = y_train
    bf = basis_function if basis_function is not None else Polynomial(degree=2)
    model = NARX(
        xlag=xlag,
        ylag=ylag,
        model_type=model_type,
        basis_function=bf,
        base_estimator=LinearRegression(),
    )
    model.fit(X=x_data, y=y_data)
    return model


def segmented_nar_prediction(model, y_data, steps_ahead):
    """Build an n-step NAR reference from independent free-run segments."""
    reference = np.empty_like(y_data)
    reference[: model.max_lag] = y_data[: model.max_lag]

    for block_start in range(model.max_lag, len(y_data), steps_ahead):
        block_horizon = min(steps_ahead, len(y_data) - block_start)
        y_initial = y_data[block_start - model.max_lag : block_start]
        block_prediction = model.predict(
            X=None,
            y=y_initial,
            steps_ahead=None,
            forecast_horizon=block_horizon,
        )
        reference[block_start : block_start + block_horizon] = block_prediction[
            -block_horizon:
        ]

    return reference


def segmented_narmax_prediction(model, x_data, y_data, steps_ahead):
    """Build an n-step NARMAX reference from independent free-run segments."""
    reference = np.empty(y_data.shape, dtype=float)
    reference[: model.max_lag] = y_data[: model.max_lag]

    for block_start in range(model.max_lag, len(y_data), steps_ahead):
        block_horizon = min(steps_ahead, len(y_data) - block_start)
        context_start = block_start - model.max_lag
        block_prediction = model.predict(
            X=x_data[context_start : block_start + block_horizon],
            y=y_data[context_start:block_start],
            steps_ahead=None,
        )
        reference[block_start : block_start + block_horizon] = block_prediction[
            -block_horizon:
        ]

    return reference


def one_step_nar_reference(model, y_data):
    """Build a one-step NAR reference directly from the fitted estimator."""
    lagged_data = build_lagged_matrix(
        None,
        y_data,
        model.xlag,
        model.ylag,
        model.model_type,
    )
    regressor_matrix = model.basis_function.transform(
        lagged_data,
        model.max_lag,
        model.ylag,
        model.xlag,
        model.model_type,
    )
    prediction = model.base_estimator.predict(regressor_matrix).reshape(-1, 1)
    return np.concatenate([y_data[: model.max_lag], prediction], axis=0)


def free_run_nar_reference(model, y_initial, forecast_horizon):
    """Build a free-run NAR reference from repeated one-step predictions."""
    reference = np.empty((model.max_lag + forecast_horizon, 1), dtype=float)
    reference[: model.max_lag] = y_initial[: model.max_lag]

    for index in range(model.max_lag, len(reference)):
        one_step_context = np.concatenate(
            [reference[index - model.max_lag : index], np.zeros((1, 1))],
            axis=0,
        )
        reference[index] = one_step_nar_reference(model, one_step_context)[-1]

    return reference


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
    }
    model = NARX(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
    ]
    assert list(default.values()) == model_values


def test_model_nfir():
    basis_function = Polynomial(degree=2)
    model = NARX(
        xlag=2,
        basis_function=basis_function,
        model_type="NFIR",
        base_estimator=base_estimator,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat.mean(), y_test[model.max_lag : :].mean(), decimal=1)


def test_validate():
    assert_raises(ValueError, NARX, ylag=-1, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, ylag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, xlag=1.3, basis_function=Polynomial(degree=1))
    assert_raises(ValueError, NARX, xlag=-1, basis_function=Polynomial(degree=1))


def test_fit_raise():
    assert_raises(
        ValueError,
        NARX,
        base_estimator=LinearRegression(),
        basis_function=Polynomial(degree=1),
        model_type="NARARMAX",
    )


def test_fit_raise_y():
    model = NARX(basis_function=Polynomial(degree=2), base_estimator=base_estimator)
    assert_raises(ValueError, model.fit, X=X_train, y=None)


def test_fit_lag_nar():
    model = NARX(
        basis_function=Polynomial(degree=2),
        model_type="NAR",
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_nfir():
    model = NARX(
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_fit_lag_narmax():
    model = NARX(
        basis_function=Polynomial(degree=2),
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_polynomial_without_bias_aligns_codes_estimator_and_predictions():
    model = NARX(
        basis_function=Polynomial(degree=2, include_bias=False),
        base_estimator=LinearRegression(fit_intercept=False),
        xlag=2,
        ylag=2,
    )

    model.fit(X=X_train, y=y_train)
    prediction = model.predict(X=X_test, y=y_test, steps_ahead=1)

    assert not np.any(np.all(model.regressor_code == 0, axis=1))
    assert model.regressor_code.shape[0] == model.base_estimator.coef_.size
    assert_equal(prediction.shape, y_test.shape)


def test_fit_lag_narmax_fourier():
    model = NARX(
        basis_function=Fourier(degree=2), base_estimator=base_estimator, xlag=2, ylag=2
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 2)


def test_model_predict():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=base_estimator,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_none():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_steps_3():
    basis_function = Polynomial(degree=2)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_predict_fourier_steps_1():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), 0.0016457328739105236, decimal=6)


def test_model_predict_fourier_nar_preserves_fitted_input_count():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    n_inputs = model.n_inputs
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, n_inputs)
    assert_equal(model._prediction_n_inputs(), 0)


def test_fourier_nar_one_step_without_input_matches_estimator():
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )
    y_evaluation = y_test[:10]

    expected = one_step_nar_reference(model, y_evaluation)
    yhat = model.predict(X=None, y=y_evaluation, steps_ahead=1)

    assert_equal(yhat.shape, y_evaluation.shape)
    assert_allclose(yhat[: model.max_lag], y_evaluation[: model.max_lag])
    assert_allclose(yhat, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(yhat).all()


def test_fourier_nar_free_run_without_input_matches_iterated_one_step():
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )
    forecast_horizon = 7
    y_initial = y_test[: model.max_lag]

    expected = free_run_nar_reference(model, y_initial, forecast_horizon)
    yhat = model.predict(
        X=None,
        y=y_initial,
        steps_ahead=None,
        forecast_horizon=forecast_horizon,
    )

    assert_equal(yhat.shape, (model.max_lag + forecast_horizon, 1))
    assert_allclose(yhat[: model.max_lag], y_initial)
    assert_allclose(yhat, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(yhat).all()


@pytest.mark.parametrize("steps_ahead", [2, 3, 20])
def test_fourier_nar_n_step_without_input_matches_segmented_free_runs(steps_ahead):
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )
    y_evaluation = y_test[:10]

    expected = segmented_nar_prediction(model, y_evaluation, steps_ahead)
    yhat = model.predict(X=None, y=y_evaluation, steps_ahead=steps_ahead)

    assert_equal(yhat.shape, y_evaluation.shape)
    assert_allclose(yhat[: model.max_lag], y_evaluation[: model.max_lag])
    assert_allclose(yhat, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(yhat).all()


@pytest.mark.parametrize("steps_ahead", [2, 3, 20])
@pytest.mark.parametrize("forecast_horizon", [0, 1, 100])
def test_fourier_nar_n_step_ignores_conflicting_forecast_horizon(
    steps_ahead, forecast_horizon
):
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )
    y_evaluation = y_test[:10]

    expected = segmented_nar_prediction(model, y_evaluation, steps_ahead)
    yhat = model.predict(
        X=None,
        y=y_evaluation,
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )

    assert_equal(yhat.shape, y_evaluation.shape)
    assert_allclose(yhat[: model.max_lag], y_evaluation[: model.max_lag])
    assert_allclose(yhat, expected, rtol=1e-12, atol=1e-12)
    assert np.isfinite(yhat).all()


@pytest.mark.parametrize("steps_ahead", [0, -1, 1.5])
def test_fourier_nar_n_step_rejects_invalid_steps(steps_ahead):
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )

    with pytest.raises(ValueError, match="steps_ahead must be"):
        model.predict(X=None, y=y_test[:10], steps_ahead=steps_ahead)


def test_fourier_nar_n_step_with_only_initial_conditions_returns_prefix():
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Fourier(degree=1, n=1),
        x_data=None,
    )
    y_initial = y_test[: model.max_lag]

    yhat = model.predict(X=None, y=y_initial, steps_ahead=3)

    assert_equal(yhat.shape, y_initial.shape)
    assert_allclose(yhat, y_initial)


def test_model_predict_fourier_raises():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )


def test_model_predict_fourier_value_error():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
        base_estimator=LinearRegression(),
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


def test_model_prediction_rejects_unknown_model_type():
    model = fit_narx_model()
    model.model_type = "UNKNOWN"
    assert_raises(ValueError, model._model_prediction, X_test, y_test)


def test_model_prediction_invokes_nfir_branch():
    model = fit_narx_model(model_type="NFIR")
    yhat = model._model_prediction(X_test, y_test)
    assert_equal(yhat.shape[0], X_test.shape[0] - model.max_lag)


def test_basis_function_predict_uses_forecast_horizon_when_input_missing():
    model = fit_narx_model(basis_function=Fourier(degree=1))
    yhat = model._basis_function_predict(
        x=None,
        y_initial=y_test,
        forecast_horizon=0,
    )
    assert_equal(yhat.size, 0)


def test_model_predict_nfir_cat():
    basis_function = Polynomial(degree=2)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NFIR",
    )

    model.fit(X=X_train, y=y_train)
    # yhat = model.predict(x=x_valid, y=y_valid)
    assert_equal(model.max_lag, 10)


def test_model_predict_steps_1():
    basis_function = Polynomial(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=2,
        ylag=2,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    # yhat = model.predict(x=x_valid, y=y_valid, steps_ahead=1)
    assert_equal(model.max_lag, 2)


def test_model_predict_fourier_none():
    basis_function = Fourier(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 10)


def test_model_predict_fourier_n():
    basis_function = Fourier(degree=1)
    model = NARX(
        base_estimator=base_estimator,
        xlag=10,
        ylag=10,
        basis_function=basis_function,
        model_type="NARMAX",
    )

    model.fit(X=X_train, y=y_train)
    assert_equal(model.max_lag, 10)


def test_fit_without_input_sets_single_input_space():
    model = fit_narx_model(model_type="NAR", x_data=None)
    assert_equal(model.n_inputs, 1)


def test_predict_fourier_multi_step():
    basis_function = Fourier(degree=2, n=1)
    model = fit_narx_model(basis_function=basis_function)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=2)
    assert_equal(yhat.shape, y_test.shape)


def test_one_step_ahead_requires_initial_output():
    model = fit_narx_model()
    assert_raises(ValueError, model._one_step_ahead_prediction, X_test, None)


def test_nar_step_ahead_requires_initial_conditions():
    model = fit_narx_model(model_type="NAR", x_data=None)
    short_y = y_test[: model.max_lag - 1]
    assert_raises(ValueError, model._nar_step_ahead, short_y, 2)


def test_nar_step_ahead_multi_segment_prediction():
    model = fit_narx_model(model_type="NAR", x_data=None)
    model._model_prediction = MagicMock(
        return_value=np.arange(100, dtype=float).reshape(-1, 1)
    )
    steps = 3
    yhat = model._nar_step_ahead(y_test, steps_ahead=steps)
    prediction_count = y_test.shape[0] - model.max_lag
    expected_horizons = [
        min(steps, prediction_count - start)
        for start in range(0, prediction_count, steps)
    ]
    expected_values = np.concatenate(
        [np.arange(100, dtype=float)[-horizon:] for horizon in expected_horizons]
    )

    assert_equal(yhat.shape, (prediction_count, 1))
    assert_allclose(yhat[:, 0], expected_values)
    assert [
        call.kwargs["forecast_horizon"]
        for call in model._model_prediction.call_args_list
    ] == expected_horizons
    for block, call in enumerate(model._model_prediction.call_args_list):
        start = block * steps
        assert_allclose(call.kwargs["y_initial"], y_test[start : start + model.max_lag])


def test_nar_step_ahead_single_segment_prediction():
    model = fit_narx_model(model_type="NAR", x_data=None)
    y_small = y_test[: model.max_lag + 1]
    yhat = model._nar_step_ahead(y_small, steps_ahead=4)
    expected = model._model_prediction(
        x=None,
        y_initial=y_small[: model.max_lag],
        forecast_horizon=1,
    )
    assert_equal(yhat.shape, (1, 1))
    assert_allclose(yhat, expected)


@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("steps_ahead", [2, 3])
def test_nar_n_step_prediction_matches_segmented_free_runs(include_bias, steps_ahead):
    model = fit_narx_model(
        model_type="NAR",
        basis_function=Polynomial(degree=2, include_bias=include_bias),
        x_data=None,
    )
    y_evaluation = y_test[:25]

    expected = segmented_nar_prediction(model, y_evaluation, steps_ahead)
    yhat = model.predict(X=None, y=y_evaluation, steps_ahead=steps_ahead)

    assert_equal(yhat.shape, y_evaluation.shape)
    assert_allclose(yhat[: model.max_lag], y_evaluation[: model.max_lag])
    assert_allclose(yhat, expected)


def test_nar_step_larger_than_remaining_horizon_matches_single_free_run():
    model = fit_narx_model(model_type="NAR", x_data=None)
    y_evaluation = y_test[: model.max_lag + 4]
    steps_ahead = len(y_evaluation)

    expected = model.predict(
        X=None,
        y=y_evaluation[: model.max_lag],
        steps_ahead=None,
        forecast_horizon=len(y_evaluation) - model.max_lag,
    )
    yhat = model.predict(X=None, y=y_evaluation, steps_ahead=steps_ahead)

    assert_equal(yhat.shape, y_evaluation.shape)
    assert_allclose(yhat, expected)


def test_narmax_n_step_ahead_requires_initial_conditions():
    model = fit_narx_model()
    short_y = y_test[: model.max_lag - 1]
    assert_raises(ValueError, model.narmax_n_step_ahead, X_test, short_y, 2)


def test_narmax_n_step_ahead_single_segment_prediction():
    model = fit_narx_model()
    x_small = X_test[: model.max_lag + 1]
    y_small = y_test[: model.max_lag + 1]
    yhat = model.narmax_n_step_ahead(x_small, y_small, steps_ahead=3)
    assert_equal(yhat.shape[0], y_small.shape[0] - model.max_lag)


def test_nar_n_step_prediction_path():
    model = fit_narx_model(model_type="NAR", x_data=None)
    model._model_prediction = MagicMock(
        return_value=np.arange(100, dtype=float).reshape(-1, 1)
    )
    steps = 2
    yhat = model._n_step_ahead_prediction(None, y_test, steps_ahead=steps)
    expected = y_test.shape[0] - model.max_lag
    assert_equal(yhat.shape[0], expected)


def test_model_prediction_invalid_type_raises():
    model = fit_narx_model()
    model.model_type = "INVALID"
    assert_raises(
        ValueError,
        model._model_prediction,
        X_test,
        y_test,
        5,
    )


def test_narmax_predict_requires_min_initial_conditions():
    model = fit_narx_model()
    short_y = y_test[: model.max_lag - 1]
    assert_raises(ValueError, model._narmax_predict, X_test, short_y, 5)


def test_narmax_predict_requires_forecast_horizon_without_input():
    model = fit_narx_model()
    assert_raises(ValueError, model._narmax_predict, None, y_test, None)


def test_narmax_predict_preserves_fitted_input_count_for_nar():
    model = fit_narx_model(model_type="NAR", x_data=None)
    n_inputs = model.n_inputs
    y_initial = y_test[: model.max_lag + 5]
    horizon = 6
    yhat = model._narmax_predict(x=None, y_initial=y_initial, forecast_horizon=horizon)
    assert_equal(model.n_inputs, n_inputs)
    assert_equal(model._prediction_n_inputs(), 0)
    assert_equal(yhat.shape[0], horizon)


@pytest.mark.parametrize(
    "basis_function",
    [
        pytest.param(Polynomial(degree=2), id="polynomial"),
        pytest.param(Fourier(degree=1, n=1), id="fourier"),
    ],
)
@pytest.mark.parametrize("steps_ahead", [2, 3, 20])
def test_narmax_n_step_matches_segmented_free_runs(basis_function, steps_ahead):
    model = fit_narx_model(basis_function=basis_function)
    x_evaluation = X_test[:11]
    y_evaluation = y_test[:11]
    expected = segmented_narmax_prediction(
        model,
        x_evaluation,
        y_evaluation,
        steps_ahead,
    )

    prediction = model.predict(
        X=x_evaluation,
        y=y_evaluation,
        steps_ahead=steps_ahead,
    )

    assert_equal(prediction.shape, y_evaluation.shape)
    assert_allclose(prediction[: model.max_lag], y_evaluation[: model.max_lag])
    assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


def test_polynomial_kernel_preserves_fitted_feature_order():
    model = fit_narx_model(basis_function=Polynomial(degree=2))
    x_window = X_test[: model.max_lag + 1]
    y_initial = y_test[: model.max_lag]
    model.base_estimator.predict = MagicMock(wraps=model.base_estimator.predict)

    prediction = model._narmax_predict(x_window, y_initial)

    exponents = model._get_prediction_exponents()
    raw_regressor = np.concatenate(
        [
            y_initial[:, 0],
            x_window[: model.max_lag, 0],
        ]
    )
    expected_features = np.prod(
        np.power(raw_regressor, exponents),
        axis=1,
    ).reshape(1, -1)
    estimator_features = model.base_estimator.predict.call_args.args[0]

    assert_equal(prediction.shape, (1, 1))
    assert_equal(expected_features.shape[1], model.regressor_code.shape[0])
    assert_allclose(estimator_features, expected_features, rtol=0, atol=0)


def test_polynomial_prediction_reuses_and_invalidates_exponent_cache():
    model = fit_narx_model(model_type="NAR", x_data=None)
    code2exponents = model._code2exponents
    model._code2exponents = MagicMock(wraps=code2exponents)
    y_evaluation = y_test[:11]

    first = model.predict(X=None, y=y_evaluation, steps_ahead=3)
    first_call_count = model._code2exponents.call_count
    second = model.predict(X=None, y=y_evaluation, steps_ahead=3)

    assert first_call_count == len(model.final_model)
    assert model._code2exponents.call_count == first_call_count
    assert_allclose(first, second, rtol=1e-12, atol=1e-12)

    model.final_model = model.final_model[::-1].copy()
    reordered_exponents = model._get_prediction_exponents()
    expected = np.vstack([code2exponents(code=term) for term in model.final_model])

    assert model._code2exponents.call_count == first_call_count + len(model.final_model)
    assert_array_equal(reordered_exponents, expected)


def test_integer_nar_predictions_preserve_fractional_estimator_output():
    y_training = np.arange(1, 13, dtype=int).reshape(-1, 1)
    model = NARX(
        ylag=1,
        xlag=1,
        model_type="NAR",
        basis_function=Polynomial(degree=1),
        base_estimator=DummyRegressor(strategy="constant", constant=0.5),
    ).fit(X=None, y=y_training)
    y_evaluation = np.array([[3], [100], [100], [100], [100]])
    n_inputs = model.n_inputs

    one_step = model.predict(X=None, y=y_evaluation, steps_ahead=1)
    n_step = model.predict(X=None, y=y_evaluation, steps_ahead=2)
    free_run = model.predict(
        X=None,
        y=y_evaluation[: model.max_lag],
        forecast_horizon=4,
    )
    expected = np.array([[3.0], [0.5], [0.5], [0.5], [0.5]])

    for prediction in (one_step, n_step, free_run):
        assert np.issubdtype(prediction.dtype, np.floating)
        assert_allclose(prediction, expected, rtol=0, atol=0)
    assert_equal(model.n_inputs, n_inputs)


def test_integer_narmax_predictions_preserve_fractional_estimator_output():
    x_training = np.arange(1, 13, dtype=int).reshape(-1, 1)
    y_training = np.arange(2, 14, dtype=int).reshape(-1, 1)
    model = NARX(
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        basis_function=Polynomial(degree=1),
        base_estimator=DummyRegressor(strategy="constant", constant=0.5),
    ).fit(X=x_training, y=y_training)
    x_evaluation = np.arange(5, dtype=int).reshape(-1, 1)
    y_evaluation = np.array([[3], [100], [100], [100], [100]])

    one_step = model.predict(X=x_evaluation, y=y_evaluation, steps_ahead=1)
    n_step = model.predict(X=x_evaluation, y=y_evaluation, steps_ahead=2)
    free_run = model.predict(
        X=x_evaluation,
        y=y_evaluation[: model.max_lag],
    )
    expected = np.array([[3.0], [0.5], [0.5], [0.5], [0.5]])

    for prediction in (one_step, n_step, free_run):
        assert np.issubdtype(prediction.dtype, np.floating)
        assert_allclose(prediction, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "basis_function",
    [
        pytest.param(Polynomial(degree=2), id="polynomial"),
        pytest.param(Fourier(degree=1, n=1), id="fourier"),
    ],
)
def test_nfir_prediction_modes_share_feed_forward_result(basis_function):
    model = fit_narx_model(model_type="NFIR", basis_function=basis_function)
    x_evaluation = X_test[:11]
    y_evaluation = y_test[:11]

    free_run = model.predict(X=x_evaluation, y=y_evaluation)
    one_step = model.predict(X=x_evaluation, y=y_evaluation, steps_ahead=1)
    n_step = model.predict(
        X=x_evaluation,
        y=y_evaluation,
        steps_ahead=3,
        forecast_horizon=100,
    )
    prefix_only = model.predict(
        X=x_evaluation,
        y=y_evaluation[: model.max_lag],
    )
    altered_y = y_evaluation.copy()
    altered_y[model.max_lag :] += 1_000
    altered_suffix = model.predict(X=x_evaluation, y=altered_y)

    assert_equal(free_run.shape, y_evaluation.shape)
    assert_allclose(one_step, free_run, rtol=1e-12, atol=1e-12)
    assert_allclose(n_step, free_run, rtol=1e-12, atol=1e-12)
    assert_allclose(prefix_only, free_run, rtol=1e-12, atol=1e-12)
    assert_allclose(altered_suffix, free_run, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "basis_function",
    [
        pytest.param(Polynomial(degree=2), id="polynomial"),
        pytest.param(Fourier(degree=1, n=1), id="fourier"),
    ],
)
def test_empty_nar_one_step_returns_prefix_without_estimator_call(basis_function):
    model = fit_narx_model(
        model_type="NAR",
        basis_function=basis_function,
        x_data=None,
    )
    model.base_estimator.predict = MagicMock(wraps=model.base_estimator.predict)
    y_initial = y_test[: model.max_lag]

    prediction = model.predict(X=None, y=y_initial, steps_ahead=1)

    assert_array_equal(prediction, y_initial)
    model.base_estimator.predict.assert_not_called()


def test_nar_n_step_preserves_estimator_prediction_dtype():
    y_data = np.linspace(0.1, 0.8, 8, dtype=np.float32).reshape(-1, 1)
    model = NARX(
        ylag=2,
        model_type="NAR",
        basis_function=Polynomial(degree=1),
        base_estimator=_LargeFloatRegressor(),
    )
    model.fit(X=None, y=y_data)

    prediction = model.predict(X=None, y=y_data, steps_ahead=3)

    assert prediction.dtype == np.float64
    assert np.all(np.isfinite(prediction))
    assert_allclose(prediction[: model.max_lag], y_data[: model.max_lag])
    assert_array_equal(
        prediction[model.max_lag :],
        np.full((len(y_data) - model.max_lag, 1), 1e40),
    )


@pytest.mark.parametrize(
    "basis_function",
    [
        pytest.param(Polynomial(degree=2), id="polynomial"),
        pytest.param(Fourier(degree=1, n=1), id="fourier"),
    ],
)
def test_one_step_prediction_returns_array_api_namespace(basis_function):
    xp = pytest.importorskip("array_api_strict")
    model = fit_narx_model(basis_function=basis_function)
    x_evaluation = X_test[:11].astype(np.float32)
    y_evaluation = y_test[:11].astype(np.float32)
    expected = model.predict(
        X=x_evaluation,
        y=y_evaluation,
        steps_ahead=1,
    )

    with config_context(array_api_dispatch=True):
        prediction = model.predict(
            X=xp.asarray(x_evaluation),
            y=xp.asarray(y_evaluation),
            steps_ahead=1,
        )

    assert prediction.__array_namespace__() is xp
    xp_assert_allclose(prediction, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_empty_narmax_interval_returns_prefix_without_estimator_call(steps_ahead):
    model = fit_narx_model()
    model.base_estimator.predict = MagicMock(wraps=model.base_estimator.predict)
    x_initial = X_test[: model.max_lag]
    y_initial = y_test[: model.max_lag]

    prediction = model.predict(
        X=x_initial,
        y=y_initial,
        steps_ahead=steps_ahead,
    )

    assert_equal(prediction.shape, y_initial.shape)
    assert_allclose(prediction, y_initial, rtol=0, atol=0)
    assert_equal(model.base_estimator.predict.call_count, 0)


def test_general_narx_prediction_contract_rejects_mismatched_samples():
    model = fit_narx_model()

    with pytest.raises(ValueError, match="same number of samples"):
        model.predict(X=X_test[:-1], y=y_test, steps_ahead=3)


@pytest.mark.parametrize(
    ("model_type", "basis_function", "kernel_name"),
    [
        ("NARMAX", Polynomial(degree=2), "_model_prediction"),
        ("NFIR", Polynomial(degree=2), "_model_prediction"),
        ("NARMAX", Fourier(degree=1, n=1), "_basis_function_predict"),
    ],
)
def test_prediction_kernels_return_suffix_only(
    model_type,
    basis_function,
    kernel_name,
):
    model = fit_narx_model(
        model_type=model_type,
        basis_function=basis_function,
    )
    x_window = X_test[: model.max_lag + 3]
    y_initial = y_test[: model.max_lag]
    kernel = getattr(model, kernel_name)

    prediction = kernel(x_window, y_initial)

    assert_equal(prediction.shape, (3, 1))
