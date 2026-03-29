# pylint: disable=protected-access,unused-variable
import numpy as np
import pytest
from unittest.mock import MagicMock
from numpy.testing import assert_almost_equal, assert_raises

from sysidentpy._config import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data


def test_simulate():
    _, x_valid, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    yhat = s.simulate(X_test=x_valid, y_test=y_valid, model_code=model, theta=theta)
    assert yhat.shape == (100, 1)


def test_simulate_theta():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=True)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )
    theta = np.array([[0.2, 0.9, 0.1]]).T
    assert_almost_equal(s.theta, theta, decimal=1)


def test_estimate_parameter():
    x_train, _, y_train, _ = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )
    assert_raises(
        TypeError,
        SimulateNARMAX,
        estimate_parameter="False",
        x_train=x_train,
        y_train=y_train,
        basis_function=Polynomial(),
    )


def test_default_values():
    default = {
        "eps": np.finfo(np.float64).eps,
        "model_type": "NARMAX",
        "estimate_parameter": True,
        "calculate_err": False,
    }
    model = SimulateNARMAX(basis_function=Polynomial())
    model_values = [
        model.eps,
        model.model_type,
        model.estimate_parameter,
        model.calculate_err,
    ]
    assert list(default.values()) == model_values
    assert isinstance(model.estimator, RecursiveLeastSquares)
    assert isinstance(model.basis_function, Polynomial)


def test_estimate_parameter_error():
    assert_raises(
        TypeError,
        SimulateNARMAX,
        estimate_parameter=1,
        basis_function=Polynomial(degree=2),
    )


def test_calculate_error():
    assert_raises(
        TypeError, SimulateNARMAX, calculate_err=1, basis_function=Polynomial(degree=2)
    )


def test_model_type_error():
    assert_raises(
        ValueError,
        SimulateNARMAX,
        model_type="NFAR",
        basis_function=Polynomial(degree=2),
    )


def test_model_order_selection():
    _, x_valid, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Fourier(), estimate_parameter=False)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T
    assert_raises(
        NotImplementedError,
        s.simulate,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=theta,
    )


def test_basis_function_error():
    assert_raises(TypeError, SimulateNARMAX, model_type="NFIR", basis_function=None)


def test_raises():
    _, x_valid, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(degree=2), estimate_parameter=False)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T
    assert_raises(
        ValueError,
        s.simulate,
        X_test=x_valid,
        y_test=None,
        model_code=model,
        theta=theta,
    )
    assert_raises(
        TypeError,
        s.simulate,
        X_test=x_valid,
        y_test=y_valid,
        model_code=str(model),
        theta=theta,
    )
    assert_raises(
        ValueError,
        s.simulate,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=theta,
        steps_ahead=0.1,
    )
    assert_raises(
        TypeError,
        s.simulate,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=None,
    )


def test_estimate_parameter_conditions():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=True)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    assert_raises(
        TypeError,
        s.simulate,
        X_train=x_train,
        y_train=str(y_train),
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )


def test_input_dimension():
    _x_train, _x_valid, _y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=False, model_type="NAR"
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    _ = s.simulate(
        X_test=None, y_test=y_valid, model_code=model, theta=theta, forecast_horizon=1
    )
    assert s.n_inputs == 0


def test_miso_dimension():
    _, x_valid, _, y_valid = get_miso_data(
        n=100, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [3002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    _ = s.simulate(X_test=x_valid, y_test=y_valid, model_code=model, theta=theta)
    assert s.xlag == [[1, 2], [1, 2]]


def test_forecast_horizon():
    _, _, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=False, model_type="NAR"
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    _ = s.simulate(
        X_test=None,
        y_test=y_valid,
        model_code=model,
        theta=theta,
        forecast_horizon=None,
    )
    assert (
        s.model_type == "NAR"
    )  # update the code to in SimulateNARMAX to make forecast_horizon global


def test_estimate_parameter_narmax():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=True)

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 2


def test_estimate_parameter_nar():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=True, model_type="NAR"
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 2


def test_estimate_parameter_nfir():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=True, model_type="NFIR"
    )

    model = np.array(
        [
            [2001, 0],  # y(k-1)
            [2001, 2001],  # x1(k-1)y(k-1)
            [2003, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 3


def test_err_narmax():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), calculate_err=True, estimate_parameter=True
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 2


def test_err_nar():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(),
        estimate_parameter=True,
        calculate_err=True,
        model_type="NAR",
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 2


def test_err_nfir():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(),
        estimate_parameter=True,
        calculate_err=True,
        model_type="NFIR",
    )

    model = np.array(
        [
            [2001, 0],  # y(k-1)
            [2001, 2001],  # x1(k-1)y(k-1)
            [2003, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 3


def test_estimate_parameter_els():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(),
        estimate_parameter=True,
    )

    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    _ = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )
    print(s.theta)
    assert_almost_equal(
        s.theta, np.array([[0.19999698], [0.90011667], [0.10080975]]), decimal=3
    )


def _small_siso_dataset():
    return get_siso_data(n=200, colored_noise=False, sigma=0.0, train_percentage=80)


def test_simulate_unbiased_estimator_called_without_err():
    x_train, x_valid, y_train, y_valid = _small_siso_dataset()
    estimator = LeastSquares(unbiased=True, uiter=1)
    estimator.unbiased_estimator = MagicMock(wraps=estimator.unbiased_estimator)
    simulator = SimulateNARMAX(
        basis_function=Polynomial(),
        estimator=estimator,
        estimate_parameter=True,
        calculate_err=False,
    )
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])
    simulator.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )
    assert estimator.unbiased_estimator.called


def test_simulate_unbiased_estimator_called_with_err():
    x_train, x_valid, y_train, y_valid = _small_siso_dataset()
    estimator = LeastSquares(unbiased=True, uiter=1)
    estimator.unbiased_estimator = MagicMock(wraps=estimator.unbiased_estimator)
    simulator = SimulateNARMAX(
        basis_function=Polynomial(),
        estimator=estimator,
        estimate_parameter=True,
        calculate_err=True,
    )
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])
    simulator.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )
    assert estimator.unbiased_estimator.called


def test_predict_paths_cover_all_branches():
    _, x_valid, _, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])
    theta = np.array([[0.2, 0.9, 0.1]]).T
    simulator.simulate(X_test=x_valid, y_test=y_valid, model_code=model, theta=theta)
    free_run = simulator.predict(X=x_valid, y=y_valid)
    one_step = simulator.predict(X=x_valid, y=y_valid, steps_ahead=1)
    multi_step = simulator.predict(X=x_valid, y=y_valid, steps_ahead=3)
    assert free_run.shape == y_valid.shape
    assert one_step.shape == y_valid.shape
    assert multi_step.shape == y_valid.shape


def test_predict_preserves_array_api_namespace_with_numpy_metadata():
    array_api_strict = pytest.importorskip("array_api_strict")
    _, x_valid_np, _, y_valid_np = _small_siso_dataset()
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])
    theta = np.array([[0.2, 0.9, 0.1]]).T

    baseline = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    baseline.simulate(
        X_test=x_valid_np,
        y_test=y_valid_np,
        model_code=model,
        theta=theta,
    )
    expected = baseline.predict(X=x_valid_np, y=y_valid_np)

    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    with config_context(array_api_dispatch=True):
        x_valid = array_api_strict.asarray(x_valid_np, dtype=array_api_strict.float64)
        y_valid = array_api_strict.asarray(y_valid_np, dtype=array_api_strict.float64)
        simulator.simulate(
            X_test=x_valid,
            y_test=y_valid,
            model_code=model,
            theta=theta,
        )
        result = simulator.predict(X=x_valid, y=y_valid)

    assert result.__array_namespace__() is array_api_strict
    np.testing.assert_allclose(_to_numpy(result), expected)


def test_model_prediction_invalid_type_raises():
    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    simulator.model_type = "UNKNOWN"
    simulator.max_lag = 1
    x = np.ones((3, 1))
    y = np.ones((3, 1))
    assert_raises(ValueError, simulator._model_prediction, x, y, 1)


def test_error_reduction_ratio_honors_process_term_limit():
    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    simulator.max_lag = 1
    psi = np.ones((3, 2))
    y = np.ones((4, 1))
    regressor_code = np.array([[1001, 0], [1002, 0]])
    model_code, err, _, psi_orth = simulator.error_reduction_ratio(
        psi, y, 0, regressor_code
    )
    assert model_code.size == 0
    assert err[0] == 0
    assert psi_orth.shape[1] == 0


def test_error_reduction_ratio_preserves_array_api_namespace():
    array_api_strict = pytest.importorskip("array_api_strict")
    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    simulator.max_lag = 1

    psi_np = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    y_np = np.array([[0.0], [2.0], [1.0], [0.0]])
    regressor_code = np.array([[1001, 0], [1002, 0]])

    with config_context(array_api_dispatch=True):
        psi = array_api_strict.asarray(psi_np, dtype=array_api_strict.float64)
        y = array_api_strict.asarray(y_np, dtype=array_api_strict.float64)
        model_code, err, piv, psi_orth = simulator.error_reduction_ratio(
            psi, y, 1, regressor_code
        )

    assert model_code.__array_namespace__() is array_api_strict
    assert err.__array_namespace__() is array_api_strict
    assert piv.__array_namespace__() is array_api_strict
    assert psi_orth.__array_namespace__() is array_api_strict
    np.testing.assert_array_equal(_to_numpy(model_code), regressor_code[:1])
    np.testing.assert_allclose(_to_numpy(err)[0], 0.8)
    np.testing.assert_array_equal(_to_numpy(psi_orth), psi_np[:, :1])


def test_error_reduction_ratio_matches_numpy_for_torch_tensors():
    torch = pytest.importorskip("torch")
    psi_np = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    y_np = np.array([[0.0], [2.0], [1.0], [0.0]])
    regressor_code = np.array([[1001, 0], [1002, 0]])

    baseline = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    baseline.max_lag = 1
    model_code_np, err_np, piv_np, psi_orth_np = baseline.error_reduction_ratio(
        psi_np, y_np, 1, regressor_code
    )

    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    simulator.max_lag = 1
    psi_t = torch.tensor(psi_np, dtype=torch.float64)
    y_t = torch.tensor(y_np, dtype=torch.float64)

    with config_context(array_api_dispatch=True):
        model_code_t, err_t, piv_t, psi_orth_t = simulator.error_reduction_ratio(
            psi_t, y_t, 1, regressor_code
        )

    np.testing.assert_array_equal(_to_numpy(model_code_t), model_code_np)
    np.testing.assert_array_equal(_to_numpy(piv_t), piv_np)
    np.testing.assert_allclose(_to_numpy(err_t), err_np, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        _to_numpy(psi_orth_t), psi_orth_np, rtol=1e-10, atol=1e-12
    )


class _FourierPredictStub(SimulateNARMAX):
    """Stub that bypasses basis function NotImplemented branches."""

    def __init__(self):
        super().__init__(basis_function=Fourier(), estimate_parameter=False)
        self.max_lag = 1
        self.calls = []

    def _basis_function_predict(self, *_args, **_kwargs):
        self.calls.append("predict")
        return np.array([[0.25]])

    def _basis_function_n_step_prediction(self, *_args, **_kwargs):
        self.calls.append("n_step")
        return np.array([[0.75]])

    def _one_step_ahead_prediction(self, *_args, **_kwargs):
        self.calls.append("one_step")
        return np.array([[0.5]])

    def _basis_function_n_steps_horizon(self, *_args, **_kwargs):
        self.calls.append("n_steps_horizon")
        return np.array([[0.0]])

    def fit(self, *, X=None, y=None):
        raise NotImplementedError


def test_predict_with_fourier_basis_raises_not_implemented():
    simulator = SimulateNARMAX(basis_function=Fourier(), estimate_parameter=False)
    simulator.max_lag = 1
    y = np.ones((2, 1))
    assert_raises(NotImplementedError, simulator.predict, X=None, y=y)


def test_non_polynomial_predict_free_run_uses_stub_result():
    simulator = _FourierPredictStub()
    y = np.array([[1.0], [0.0]])
    result = simulator.predict(X=None, y=y)
    expected = np.concatenate([y[: simulator.max_lag], np.array([[0.25]])], axis=0)
    assert np.allclose(result, expected)
    assert simulator.calls == ["predict"]


def test_non_polynomial_predict_one_step_uses_stub_result():
    simulator = _FourierPredictStub()
    y = np.array([[2.0], [0.0]])
    result = simulator.predict(X=None, y=y, steps_ahead=1)
    expected = np.concatenate([y[: simulator.max_lag], np.array([[0.5]])], axis=0)
    assert np.allclose(result, expected)
    assert simulator.calls == ["one_step"]


def test_non_polynomial_predict_n_step_uses_stub_result():
    simulator = _FourierPredictStub()
    y = np.array([[3.0], [0.0]])
    result = simulator.predict(X=None, y=y, steps_ahead=3)
    expected = np.concatenate([y[: simulator.max_lag], np.array([[0.75]])], axis=0)
    assert np.allclose(result, expected)
    assert simulator.calls == ["n_step"]
