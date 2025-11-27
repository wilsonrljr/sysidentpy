# ruff: noqa: SLF001
# pylint: disable=protected-access
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.parameter_estimation.estimators import LeastSquares
from sysidentpy.model_structure_selection.accelerated_orthogonal_least_squares import (
    AOLS,
)
from sysidentpy.tests.test_narmax_base import create_test_data

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


class _NullEstimator:
    def optimize(self, psi_slice, _y_slice):
        return np.zeros((psi_slice.shape[1], 1))


class _OnesEstimator:
    def optimize(self, psi_slice, _y_slice):
        return np.ones((psi_slice.shape[1], 1))


def test_default_values():
    default = {
        "ylag": 2,
        "xlag": 2,
        "k": 1,
        "L": 1,
        "threshold": 10e-10,
        "model_type": "NARMAX",
    }
    model = AOLS(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.k,
        model.L,
        model.threshold,
        model.model_type,
    ]
    assert list(default.values()) == model_values
    assert isinstance(model.estimator, LeastSquares)
    assert isinstance(model.basis_function, Polynomial)


def test_validate_ylag():
    assert_raises(ValueError, AOLS, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, AOLS, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, xlag=1.3, basis_function=Polynomial(degree=2))


def test_k():
    assert_raises(ValueError, AOLS, k=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, k=1.3, basis_function=Polynomial(degree=2))


def test_n_terms():
    assert_raises(ValueError, AOLS, L=1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, L=-1, basis_function=Polynomial(degree=2))


def test_threshold():
    assert_raises(ValueError, AOLS, threshold=-1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, AOLS, threshold=-1, basis_function=Polynomial(degree=2))


def test_model_prediction():
    basis_function = Polynomial(degree=2)
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=basis_function)
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_model_predict_fourier_steps_none():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._basis_function_predict(x=X_test, y_initial=y_test)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=1)


def test_model_predict_fourier_steps_1():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=1)


def test_model_predict_fourier_nar_inputs():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    model.predict(X=X_test, y=y_test)
    assert_equal(model.n_inputs, 0)


def test_model_predict_fourier_raises():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NARMAX",
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(
        Exception, model._basis_function_n_step_prediction, X=X_test, y=y_test[:1]
    )


def test_model_prediction_ell():
    basis_function = Polynomial(degree=2)
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=basis_function, L=2)
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_model_prediction_osa():
    basis_function = Polynomial(degree=2)
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=basis_function)
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1], steps_ahead=1)


def test_nar():
    basis_function = Polynomial(degree=2)
    model = AOLS(ylag=[1, 2], basis_function=basis_function, model_type="NAR")
    model.fit(y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1], forecast_horizon=5)


def test_model_predict_fourier_nsa():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=3)
    assert_almost_equal(yhat.mean(), y_test.mean(), decimal=1)


def test_model_prediction_rejects_unknown_type():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    model.model_type = "UNKNOWN"
    assert_raises(ValueError, model._model_prediction, X_test, y_test)


def test_fit_requires_output_data():
    model = AOLS(basis_function=Polynomial(degree=2))
    assert_raises(ValueError, model.fit, X=X_train, y=None)


def test_predict_polynomial_returns_concatenated_output():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_equal(yhat.shape, y_test.shape)
    assert_equal(yhat[: model.max_lag], y_test[: model.max_lag])


def test_predict_polynomial_multi_step_returns_prediction():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=2)
    assert_equal(yhat.shape, y_test.shape)


def test_predict_polynomial_one_step_returns_prediction():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test, steps_ahead=1)
    assert_equal(yhat.shape, y_test.shape)


def test_predict_rejects_non_positive_steps():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    assert_raises(ValueError, model.predict, X=X_test, y=y_test, steps_ahead=0)


def test_narmax_predict_requires_initial_conditions():
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)
    short_ic = y_test[: model.max_lag - 1]
    assert_raises(ValueError, model._narmax_predict, X_test, short_ic)


def test_basis_function_predict_handles_missing_input():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    horizon = 4
    yhat = model._basis_function_predict(
        x=None, y_initial=y_test, forecast_horizon=horizon
    )
    assert_equal(yhat.shape, (horizon, 1))
    assert_equal(model.n_inputs, 0)


def test_model_prediction_handles_nfir_type():
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._model_prediction(X_test, y_test)
    assert_equal(yhat.shape[0], X_test.shape[0] - model.max_lag)


def test_narmax_predict_with_missing_input_sets_single_input():
    model = AOLS(ylag=[1, 2], basis_function=Polynomial(degree=2), model_type="NAR")
    model.fit(y=y_train)
    horizon = 3
    yhat = model._narmax_predict(x=None, y_initial=y_test, forecast_horizon=horizon)
    assert_equal(model.n_inputs, 0)
    assert_equal(yhat.shape[0], horizon)


def test_nfir_predict_delegates_to_base():
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=Polynomial(degree=2),
        model_type="NFIR",
    )
    model.fit(X=X_train, y=y_train)
    yhat = model._nfir_predict(X_test, y_test)
    assert_equal(yhat.shape[0], X_test.shape[0] - model.max_lag)


def test_basis_function_n_step_prediction_requires_initial_conditions():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    short_y = y_test[: model.max_lag - 1]
    assert_raises(
        ValueError,
        model._basis_function_n_step_prediction,
        X_test,
        short_y,
        2,
        2,
    )


def test_basis_function_n_step_prediction_without_input_uses_horizon():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
        model_type="NAR",
    )
    model.fit(X=X_train, y=y_train)
    horizon = model.max_lag + 2
    yhat = model._basis_function_n_step_prediction(
        x=None,
        y=y_test,
        steps_ahead=1,
        forecast_horizon=horizon,
    )
    assert_equal(yhat.shape[0], horizon)


def test_basis_function_n_steps_horizon_returns_column_vector():
    basis_function = Fourier(degree=2, n=1)
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    window_len = model.max_lag + 3
    steps_ahead = window_len - model.max_lag
    forecast_horizon = window_len
    yhat = model._basis_function_n_steps_horizon(
        x=X_test[:window_len],
        y=y_test[:window_len],
        steps_ahead=steps_ahead,
        forecast_horizon=forecast_horizon,
    )
    assert_equal(tuple(yhat.shape), (forecast_horizon - model.max_lag, 1))


def test_aols_handles_zero_candidate_block():
    model = AOLS(basis_function=Polynomial(degree=1))
    model.max_lag = 1
    model.threshold = 0
    model.k = 1
    model.L = 1
    model.estimator = _NullEstimator()
    psi = np.zeros((4, 0))
    y_local = np.arange(5.0).reshape(-1, 1)
    theta, piv, residual = model.aols(psi, y_local)
    assert_equal(theta.size, 0)
    assert_equal(piv.size, 0)
    assert residual >= 0


def test_aols_skips_degenerate_basis_vectors():
    model = AOLS(basis_function=Polynomial(degree=1))
    model.max_lag = 1
    model.threshold = 0
    model.k = 1
    model.L = 1
    model.estimator = _OnesEstimator()
    psi = np.zeros((4, 1))
    y_local = np.arange(5.0).reshape(-1, 1)
    theta, piv, residual = model.aols(psi, y_local)
    assert_equal(theta.shape, (1, 1))
    assert_equal(piv, np.array([0]))
    assert residual >= 0


def test_aols_runs_multiple_iterations_when_k_exceeds_one():
    model = AOLS(
        ylag=[1, 2],
        xlag=2,
        basis_function=Polynomial(degree=2),
        k=3,
        threshold=0,
    )
    model.fit(X=X_train, y=y_train)
    assert_equal(model.n_terms, 3)
