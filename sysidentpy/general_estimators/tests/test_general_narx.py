# ruff: noqa: SLF001
# pylint: disable=protected-access
# pyright: reportMissingTypeStubs=false
import numpy as np
from sklearn.linear_model import LinearRegression  # type: ignore[reportMissingTypeStubs]
from unittest.mock import MagicMock
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.general_estimators import NARX
from sysidentpy.tests.test_narmax_base import create_test_data

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


def test_model_predict_fourier_nar_inputs():
    basis_function = Fourier(degree=2, n=1)
    model = NARX(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
        base_estimator=LinearRegression(),
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


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


def test_model_predict_fourier_horizon_error():
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
        model._basis_function_n_steps_horizon,
        x=X_test,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=10,
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
    expected = y_test.shape[0] + steps - model.max_lag
    assert_equal(yhat.shape[0], expected)
    assert_equal(model._model_prediction.called, True)


def test_nar_step_ahead_single_segment_prediction():
    model = fit_narx_model(model_type="NAR", x_data=None)
    y_small = y_test[: model.max_lag + 1]
    yhat = model._nar_step_ahead(y_small, steps_ahead=4)
    expected = y_small.shape[0] + 4 - model.max_lag
    assert_equal(yhat.shape[0], expected)


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
    expected = y_test.shape[0] + steps - model.max_lag
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


def test_narmax_predict_sets_zero_inputs_for_nar():
    model = fit_narx_model(model_type="NAR", x_data=None)
    model.n_inputs = 5
    y_initial = y_test[: model.max_lag + 5]
    horizon = 6
    yhat = model._narmax_predict(x=None, y_initial=y_initial, forecast_horizon=horizon)
    assert_equal(model.n_inputs, 0)
    assert_equal(yhat.shape[0], horizon)


def test_basis_function_n_step_prediction_requires_initial_conditions():
    basis_function = Fourier(degree=2, n=1)
    model = fit_narx_model(basis_function=basis_function)
    short_y = y_test[: model.max_lag - 1]
    assert_raises(
        ValueError,
        model._basis_function_n_step_prediction,
        X_test,
        short_y,
        2,
        X_test.shape[0],
    )


def test_basis_function_n_step_prediction_narmax_flow():
    basis_function = Fourier(degree=2, n=1)
    model = fit_narx_model(basis_function=basis_function)
    x_small = X_test[: model.max_lag + 3]
    y_small = y_test[: model.max_lag + 3]

    def _mock_predict_factory(current_model):
        def _mock_predict(*args, **kwargs):
            horizon = kwargs.get("forecast_horizon")
            if horizon is None and len(args) >= 3:
                horizon = args[2]
            size = horizon + current_model.max_lag
            return np.arange(size, dtype=float).reshape(-1, 1)

        return _mock_predict

    model._basis_function_predict = MagicMock(side_effect=_mock_predict_factory(model))
    yhat = model._basis_function_n_step_prediction(
        x_small, y_small, 4, x_small.shape[0]
    )
    assert_equal(yhat.shape[0], x_small.shape[0] - model.max_lag)


def test_basis_function_n_step_prediction_nar_flow():
    basis_function = Fourier(degree=2, n=1)
    model = fit_narx_model(model_type="NAR", basis_function=basis_function, x_data=None)
    y_small = y_test[: model.max_lag + 3]

    def _mock_predict(*_args, **kwargs):
        horizon = kwargs.get("forecast_horizon")
        if horizon is None and len(_args) >= 3:
            horizon = _args[2]
        size = horizon + model.max_lag
        return np.arange(size, dtype=float).reshape(-1, 1)

    model._basis_function_predict = MagicMock(side_effect=_mock_predict)
    yhat = model._basis_function_n_step_prediction(None, y_small, 4, y_small.shape[0])
    assert_equal(yhat.shape[0], y_small.shape[0])


def test_basis_function_n_step_prediction_nfir_flow():
    basis_function = Fourier(degree=2, n=1)
    model = fit_narx_model(model_type="NFIR", basis_function=basis_function)
    x_small = X_test[: model.max_lag + 3]
    y_small = y_test[: model.max_lag + 3]

    def _mock_predict(*_args, **kwargs):
        horizon = kwargs.get("forecast_horizon")
        if horizon is None and len(_args) >= 3:
            horizon = _args[2]
        size = horizon + model.max_lag
        return np.arange(size, dtype=float).reshape(-1, 1)

    model._basis_function_predict = MagicMock(side_effect=_mock_predict)
    yhat = model._basis_function_n_step_prediction(
        x_small, y_small, 4, x_small.shape[0]
    )
    assert_equal(yhat.shape[0], x_small.shape[0] - model.max_lag)


def test_basis_function_n_steps_horizon_handles_all_model_types():
    basis_function = Fourier(degree=2, n=1)
    cases = [
        ("NARMAX", X_train, X_test[: X_train.shape[0]]),
        ("NAR", None, None),
        ("NFIR", X_train, X_test[: X_train.shape[0]]),
    ]
    for model_type, fit_x, predict_x in cases:
        model = fit_narx_model(
            model_type=model_type,
            basis_function=basis_function,
            x_data=fit_x,
        )
        y_small = y_test[: model.max_lag + 3]
        if predict_x is not None:
            x_small = predict_x[: y_small.shape[0]]
        else:
            x_small = None

        def _mock_predict_factory(current_model):
            def _mock_predict(*args, **kwargs):
                horizon = kwargs.get("forecast_horizon")
                if horizon is None and len(args) >= 3:
                    horizon = args[2]
                size = horizon + current_model.max_lag
                return np.arange(size, dtype=float).reshape(-1, 1)

            return _mock_predict

        model._basis_function_predict = MagicMock(
            side_effect=_mock_predict_factory(model)
        )
        yhat = model._basis_function_n_steps_horizon(
            x_small,
            y_small,
            4,
            y_small.shape[0],
        )
        assert_equal(yhat.shape[0], y_small.shape[0] - model.max_lag)
