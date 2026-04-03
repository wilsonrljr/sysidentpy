# pyright: reportPrivateUsage=false
# pylint: disable=protected-access
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_equal, assert_raises

from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy, get_namespace
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


class _AOLSProbe(AOLS):
    def narmax_predict_reference(self, x_data, y_data, forecast_horizon):
        return self._narmax_predict_reference(x_data, y_data, forecast_horizon)

    def polynomial_narmax_predict_fast(self, x_data, y_data, forecast_horizon):
        return self._polynomial_narmax_predict_fast(
            x_data,
            y_data,
            forecast_horizon,
        )


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


def test_predict_polynomial_preserves_array_api_namespace():
    xp = pytest.importorskip("array_api_strict")
    model = AOLS(basis_function=Polynomial(degree=2))
    model.max_lag = 1
    model._model_prediction = lambda _x, _y, forecast_horizon=0: get_namespace(
        _y
    ).asarray(np.full((3, 1), 1.5))
    y_data = xp.asarray(np.arange(4.0).reshape(-1, 1))

    with config_context(array_api_dispatch=True):
        yhat = model.predict(X=None, y=y_data)

    assert yhat.__array_namespace__().__name__ == xp.__name__
    assert_equal(
        _to_numpy(yhat),
        np.array([[0.0], [1.5], [1.5], [1.5]]),
    )


def test_fit_predict_accepts_torch_tensors_under_array_api_dispatch():
    torch = pytest.importorskip("torch")
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    x_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    x_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_t, y=y_train_t)
        yhat = model.predict(X=x_test_t, y=y_test_t)

    assert isinstance(yhat, torch.Tensor)
    assert_equal(tuple(yhat.shape), y_test.shape)
    assert_equal(_to_numpy(yhat[: model.max_lag]), y_test[: model.max_lag])


def test_predict_rejects_mixed_array_api_namespaces_for_aols():
    xp = pytest.importorskip("array_api_strict")
    torch = pytest.importorskip("torch")
    model = AOLS(basis_function=Polynomial(degree=2))

    x_data = torch.tensor(np.arange(4.0).reshape(-1, 1), dtype=torch.float64)
    y_data = xp.asarray(np.arange(4.0).reshape(-1, 1), dtype=xp.float64)

    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="same Array API namespace"):
            model.predict(X=x_data, y=y_data)


def test_fit_predict_accepts_torch_cuda_tensors_under_array_api_dispatch():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    x_train_t = torch.tensor(X_train, dtype=torch.float64, device="cuda")
    y_train_t = torch.tensor(y_train, dtype=torch.float64, device="cuda")
    x_test_t = torch.tensor(X_test, dtype=torch.float64, device="cuda")
    y_test_t = torch.tensor(y_test, dtype=torch.float64, device="cuda")

    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_t, y=y_train_t)
        yhat = model.predict(X=x_test_t, y=y_test_t)

    assert isinstance(yhat, torch.Tensor)
    assert yhat.device.type == "cuda"
    assert_equal(tuple(yhat.shape), y_test.shape)
    assert_equal(_to_numpy(yhat[: model.max_lag]), y_test[: model.max_lag])


def test_polynomial_narmax_fast_path_matches_reference_for_aols_model():
    model = _AOLSProbe(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    model.fit(X=X_train, y=y_train)

    reference = model.narmax_predict_reference(
        X_test,
        y_test,
        forecast_horizon=X_test.shape[0],
    )
    fast = model.polynomial_narmax_predict_fast(
        X_test,
        y_test,
        forecast_horizon=X_test.shape[0],
    )

    np.testing.assert_allclose(fast, reference, rtol=1e-10, atol=1e-12)


def test_predict_repeated_calls_are_stable_under_array_api_dispatch_for_aols():
    torch = pytest.importorskip("torch")
    model = AOLS(ylag=[1, 2], xlag=2, basis_function=Polynomial(degree=2))
    x_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    x_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_t, y=y_train_t)
        yhat_first = model.predict(X=x_test_t, y=y_test_t)
        final_model_first = model.final_model.copy()
        yhat_second = model.predict(X=x_test_t, y=y_test_t)

    np.testing.assert_allclose(
        _to_numpy(yhat_first),
        _to_numpy(yhat_second),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_array_equal(model.final_model, final_model_first)


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
