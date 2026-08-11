# pylint: disable=protected-access,unused-variable
import numpy as np
import pytest
from unittest.mock import MagicMock
from numpy.testing import assert_almost_equal, assert_raises

from sysidentpy._config import config_context
from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.tests._array_api_asserts import (
    assert_allclose as xp_assert_allclose,
    assert_array_equal as xp_assert_array_equal,
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
    theta = np.array([[0.2, 0.1, 0.9]]).T

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
    theta = np.array([[0.2, 0.1, 0.9]]).T
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


def test_simulate_uses_shared_steps_ahead_validation():
    simulator = SimulateNARMAX(
        model_type="NAR",
        basis_function=Polynomial(degree=1, include_bias=False),
        estimate_parameter=False,
    )
    y_data = np.arange(6.0).reshape(-1, 1)
    model_code = np.array([[1001]])
    theta = np.array([[0.5]])

    prediction = simulator.simulate(
        X_test=None,
        y_test=y_data,
        model_code=model_code,
        theta=theta,
        steps_ahead=np.int64(2),
    )

    assert prediction.shape == y_data.shape
    for invalid_steps in (True, np.bool_(True), 1.5, 0, -1):
        with pytest.raises(ValueError, match="steps_ahead"):
            simulator.simulate(
                X_test=None,
                y_test=y_data,
                model_code=model_code,
                theta=theta,
                steps_ahead=invalid_steps,
            )


@pytest.mark.parametrize(
    "invalid_y",
    [
        pytest.param(np.arange(6.0), id="one-dimensional"),
        pytest.param(np.array(1.0), id="scalar"),
        pytest.param([1.0, 2.0, 3.0], id="list"),
    ],
)
def test_simulate_rejects_invalid_prediction_shapes(invalid_y):
    simulator = SimulateNARMAX(
        model_type="NAR",
        basis_function=Polynomial(degree=1, include_bias=False),
        estimate_parameter=False,
    )
    model_code = np.array([[1001]])
    theta = np.array([[0.5]])

    with pytest.raises(ValueError, match="y must be a 2D array"):
        simulator.simulate(
            X_test=None,
            y_test=invalid_y,
            model_code=model_code,
            theta=theta,
        )

    with pytest.raises(ValueError, match="X must be a 2D array"):
        simulator.simulate(
            X_test=np.arange(6.0),
            y_test=np.arange(6.0).reshape(-1, 1),
            model_code=model_code,
            theta=theta,
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


def test_nar_prediction_does_not_mutate_input_dimension():
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
    n_inputs = s.n_inputs

    _ = s.predict(X=None, y=y_valid, forecast_horizon=1)

    assert n_inputs == 1
    assert s.n_inputs == n_inputs


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
        s.theta, np.array([[0.19999698], [0.10080975], [0.90011667]]), decimal=3
    )


def _small_siso_dataset():
    return get_siso_data(n=200, colored_noise=False, sigma=0.0, train_percentage=80)


def _nar_observed_series(n_samples=25):
    rng = np.random.default_rng(193)
    y = np.zeros((n_samples, 1))
    y[:2, 0] = [0.2, -0.1]

    for k in range(2, n_samples):
        y[k, 0] = 0.55 * y[k - 1, 0] - 0.1 * y[k - 2, 0] + 0.05 + 0.03 * rng.normal()

    return y


def _nar_model_and_theta(include_bias):
    model = np.array([[1001, 0], [1002, 0]])
    theta = np.array([[0.55], [-0.1]])
    if include_bias:
        model = np.concatenate([np.array([[0, 0]]), model], axis=0)
        theta = np.concatenate([np.array([[0.05]]), theta], axis=0)

    return model, theta


def _segmented_nar_free_run_reference(simulator, y, steps_ahead):
    reference = np.full_like(y, np.nan)
    reference[: simulator.max_lag] = y[: simulator.max_lag]

    for block_start in range(simulator.max_lag, len(y), steps_ahead):
        block_horizon = min(steps_ahead, len(y) - block_start)
        initial_condition = y[block_start - simulator.max_lag : block_start]
        free_run = simulator.predict(
            X=None,
            y=initial_condition,
            steps_ahead=None,
            forecast_horizon=block_horizon,
        )
        reference[block_start : block_start + block_horizon] = free_run[-block_horizon:]

    return reference


def _segmented_narmax_free_run_reference(simulator, x, y, steps_ahead):
    reference = np.full_like(y, np.nan)
    reference[: simulator.max_lag] = y[: simulator.max_lag]

    for block_start in range(simulator.max_lag, len(y), steps_ahead):
        block_horizon = min(steps_ahead, len(y) - block_start)
        window_start = block_start - simulator.max_lag
        free_run = simulator.predict(
            X=x[window_start : block_start + block_horizon],
            y=y[window_start:block_start],
            steps_ahead=None,
        )
        reference[block_start : block_start + block_horizon] = free_run[-block_horizon:]

    return reference


def test_simulate_polynomial_without_bias_rejects_bias_model_code():
    _, x_valid, _, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )
    model = np.array([[0, 0], [1001, 0]])
    theta = np.array([[0.2], [0.9]])

    with pytest.raises(ValueError, match="not available"):
        simulator.simulate(
            X_test=x_valid,
            y_test=y_valid,
            model_code=model,
            theta=theta,
        )


def test_simulate_validates_theta_shape_against_model_code():
    _, x_valid, _, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )
    model = np.array([[1001, 0], [2001, 0]])

    with pytest.raises(ValueError, match=r"shape \(n_terms, 1\)"):
        simulator.simulate(
            X_test=x_valid,
            y_test=y_valid,
            model_code=model,
            theta=np.array([[0.2, 0.9]]),
        )


def test_simulate_preserves_requested_model_code_and_theta_order():
    x_data = np.array([[2.0], [4.0], [8.0]])
    y_data = np.array([[3.0], [0.0], [0.0]])
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )
    model = np.array([[2001, 0], [1001, 0]])
    theta = np.array([[10.0], [1.0]])

    yhat = simulator.simulate(
        X_test=x_data,
        y_test=y_data,
        model_code=model,
        theta=theta,
    )

    np.testing.assert_array_equal(simulator.final_model, model)
    np.testing.assert_array_equal(simulator.pivv, np.array([1, 0]))
    assert yhat[1, 0] == pytest.approx(10 * x_data[0, 0] + y_data[0, 0])


def test_simulate_rejects_model_code_width_before_matching_rows():
    x_data = np.arange(1, 5, dtype=float).reshape(-1, 1)
    y_data = np.zeros_like(x_data)
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )

    with pytest.raises(ValueError, match="same number of columns"):
        simulator.simulate(
            X_test=x_data,
            y_test=y_data,
            model_code=np.array([[1001, 0, 0]]),
            theta=np.array([[0.2]]),
        )


@pytest.mark.parametrize(
    ("regressor_code", "model_code", "error_message"),
    [
        (
            np.array([[1001, 0], [1001, 0], [2001, 0]]),
            np.array([[1001, 0]]),
            "ambiguous regressor code space",
        ),
        (
            np.array([[1001, 0], [2001, 0]]),
            np.array([[1001, 0], [1001, 0]]),
            "unique regressors",
        ),
    ],
)
def test_model_code_resolution_rejects_ambiguous_and_duplicate_terms(
    regressor_code,
    model_code,
    error_message,
):
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2),
        estimate_parameter=False,
    )

    with pytest.raises(ValueError, match=error_message):
        simulator._resolve_model_code_indices(regressor_code, model_code)


def test_simulate_normalizes_historical_one_dimensional_theta():
    x_data = np.arange(1, 5, dtype=float).reshape(-1, 1)
    y_data = np.zeros_like(x_data)
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )
    model = np.array([[1001, 0], [2001, 0]])

    yhat = simulator.simulate(
        X_test=x_data,
        y_test=y_data,
        model_code=model,
        theta=np.array([0.2, 0.9]),
    )

    assert simulator.theta.shape == (2, 1)
    assert yhat.shape == y_data.shape


def test_simulate_polynomial_without_bias_accepts_available_model():
    _, x_valid, _, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=False,
    )
    model = np.array([[1001, 0], [2001, 0]])
    theta = np.array([[0.2], [0.9]])

    yhat = simulator.simulate(
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=theta,
    )

    assert yhat.shape == y_valid.shape
    assert not np.any(np.all(simulator.final_model == 0, axis=1))


def test_simulate_estimates_parameters_without_bias_in_model_order():
    x_train, x_valid, y_train, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(
        basis_function=Polynomial(degree=2, include_bias=False),
        estimate_parameter=True,
        calculate_err=False,
    )
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])

    yhat = simulator.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    np.testing.assert_array_equal(simulator.final_model, model)
    np.testing.assert_allclose(
        simulator.theta,
        np.array([[0.2], [0.1], [0.9]]),
        rtol=3e-4,
        atol=3e-5,
    )
    assert not np.any(np.all(simulator.final_model == 0, axis=1))
    assert yhat.shape == y_valid.shape


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


def test_simulate_narmax_n_step_matches_segmented_free_runs():
    _, x_valid, _, y_valid = _small_siso_dataset()
    simulator = SimulateNARMAX(
        basis_function=Polynomial(),
        estimate_parameter=False,
    )
    model = np.array([[1001, 0], [2001, 1001], [2002, 0]])
    theta = np.array([[0.2, 0.9, 0.1]]).T
    simulator.simulate(
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=theta,
    )

    prediction = simulator.predict(X=x_valid, y=y_valid, steps_ahead=3)
    expected = _segmented_narmax_free_run_reference(
        simulator,
        x_valid,
        y_valid,
        3,
    )

    assert prediction.shape == y_valid.shape
    np.testing.assert_array_equal(
        prediction[: simulator.max_lag],
        y_valid[: simulator.max_lag],
    )
    np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


def test_predict_empty_interval_returns_initial_conditions_without_state_change():
    y = np.array([[3], [2]], dtype=np.int64)
    simulator = SimulateNARMAX(
        model_type="NAR",
        basis_function=Polynomial(degree=1, include_bias=False),
        estimate_parameter=False,
    )
    simulator.max_lag = len(y)
    simulator.n_inputs = 1

    prediction = simulator.predict(X=None, y=y, steps_ahead=3)

    np.testing.assert_array_equal(prediction, y)
    assert prediction is not y
    assert prediction.dtype == y.dtype
    assert simulator.n_inputs == 1


@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_simulate_nfir_prediction_is_independent_of_steps_ahead(steps_ahead):
    x = np.arange(1, 7, dtype=np.int64).reshape(-1, 1)
    y = np.full((6, 1), 9, dtype=np.int64)
    simulator = SimulateNARMAX(
        model_type="NFIR",
        basis_function=Polynomial(degree=1, include_bias=False),
        estimate_parameter=False,
    )
    expected = np.array([[9.0], [0.5], [1.0], [1.5], [2.0], [2.5]])
    simulator.simulate(
        X_test=x,
        y_test=y,
        model_code=np.array([[2001]]),
        theta=np.array([[0.5]]),
    )

    prediction = simulator.predict(X=x, y=y, steps_ahead=steps_ahead)

    assert prediction.shape == y.shape
    assert np.issubdtype(prediction.dtype, np.floating)
    np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("steps_ahead", [0, -1, 1.5, True])
def test_predict_uses_shared_steps_ahead_validation(steps_ahead):
    simulator = SimulateNARMAX(
        model_type="NAR",
        basis_function=Polynomial(),
        estimate_parameter=False,
    )
    simulator.max_lag = 1

    with pytest.raises(ValueError, match="steps_ahead must"):
        simulator.predict(
            X=None,
            y=np.ones((2, 1)),
            steps_ahead=steps_ahead,
        )


@pytest.mark.parametrize("include_bias", [True, False])
def test_simulate_nar_n_step_matches_segmented_free_runs(include_bias):
    y = _nar_observed_series()
    model_code, theta = _nar_model_and_theta(include_bias)
    simulator = SimulateNARMAX(
        model_type="NAR",
        estimate_parameter=False,
        basis_function=Polynomial(degree=2, include_bias=include_bias),
    )

    prediction = simulator.simulate(
        X_test=None,
        y_test=y,
        model_code=model_code,
        theta=theta,
        steps_ahead=3,
    )
    expected = _segmented_nar_free_run_reference(simulator, y, 3)

    assert prediction.shape == y.shape
    assert np.all(np.isfinite(prediction))
    np.testing.assert_array_equal(
        prediction[: simulator.max_lag],
        y[: simulator.max_lag],
    )
    np.testing.assert_allclose(prediction, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(prediction[-2:], expected[-2:], rtol=1e-12, atol=1e-12)

    one_step = simulator.predict(X=None, y=y, steps_ahead=1)
    one_step_reference = _segmented_nar_free_run_reference(simulator, y, 1)
    remaining_horizon = int(len(y) - simulator.max_lag)
    free_run = simulator.predict(
        X=None,
        y=y[: simulator.max_lag],
        forecast_horizon=remaining_horizon,
    )

    assert one_step.shape == y.shape
    assert free_run.shape == y.shape
    assert np.all(np.isfinite(one_step))
    assert np.all(np.isfinite(free_run))
    np.testing.assert_array_equal(
        one_step[: simulator.max_lag],
        y[: simulator.max_lag],
    )
    np.testing.assert_array_equal(
        free_run[: simulator.max_lag],
        y[: simulator.max_lag],
    )
    np.testing.assert_allclose(
        one_step,
        one_step_reference,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        free_run,
        _segmented_nar_free_run_reference(
            simulator,
            y,
            remaining_horizon + 1,
        ),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("include_bias", [True, False])
def test_simulate_nar_recursive_predictions_promote_integer_output(include_bias):
    y_integer = np.array([[3], [100], [100], [100], [100]])
    model_code = np.array([[1001]])
    theta = np.array([[0.5]])
    if include_bias:
        model_code = np.array([[0], [1001]])
        theta = np.array([[0.25], [0.5]])
        expected_n_step = np.array([[3], [1.75], [1.125], [50.25], [25.375]])
        expected_free_run = np.array([[3], [1.75], [1.125], [0.8125], [0.65625]])
    else:
        expected_n_step = np.array([[3], [1.5], [0.75], [50], [25]])
        expected_free_run = np.array([[3], [1.5], [0.75], [0.375], [0.1875]])

    simulator = SimulateNARMAX(
        model_type="NAR",
        estimate_parameter=False,
        basis_function=Polynomial(degree=1, include_bias=include_bias),
    )
    n_step = simulator.simulate(
        X_test=None,
        y_test=y_integer,
        model_code=model_code,
        theta=theta,
        steps_ahead=2,
    )
    free_run = simulator.predict(
        X=None,
        y=y_integer[:1],
        forecast_horizon=4,
    )

    assert np.issubdtype(n_step.dtype, np.floating)
    assert np.issubdtype(free_run.dtype, np.floating)
    np.testing.assert_allclose(n_step, expected_n_step, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(free_run, expected_free_run, rtol=1e-12, atol=1e-12)


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
    xp_assert_allclose(result, expected)


def test_predict_n_step_preserves_array_api_namespace_via_cpu_fallback():
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
    expected = baseline.predict(X=x_valid_np, y=y_valid_np, steps_ahead=3)

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
        result = simulator.predict(X=x_valid, y=y_valid, steps_ahead=3)

    assert result.__array_namespace__() is array_api_strict
    xp_assert_allclose(result, expected)


def test_nar_integer_n_step_promotes_array_api_output_via_cpu_fallback():
    array_api_strict = pytest.importorskip("array_api_strict")
    y_integer_np = np.array([[3], [100], [100], [100], [100]])
    expected = np.array([[3], [1.5], [0.75], [50], [25]])
    simulator = SimulateNARMAX(
        model_type="NAR",
        estimate_parameter=False,
        basis_function=Polynomial(degree=1, include_bias=False),
    )
    simulator.simulate(
        X_test=None,
        y_test=y_integer_np,
        model_code=np.array([[1001]]),
        theta=np.array([[0.5]]),
        steps_ahead=2,
    )

    with config_context(array_api_dispatch=True):
        y_integer = array_api_strict.asarray(
            y_integer_np,
            dtype=array_api_strict.int64,
        )
        result = simulator.predict(X=None, y=y_integer, steps_ahead=2)

    assert result.__array_namespace__() is array_api_strict
    assert array_api_strict.isdtype(result.dtype, "real floating")
    xp_assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_predict_rejects_mixed_array_api_namespaces():
    array_api_strict = pytest.importorskip("array_api_strict")
    torch = pytest.importorskip("torch")
    simulator = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)
    simulator.max_lag = 1
    simulator.n_inputs = 1

    x_data = torch.tensor(np.arange(4.0).reshape(-1, 1), dtype=torch.float64)
    y_data = array_api_strict.asarray(
        np.arange(4.0).reshape(-1, 1), dtype=array_api_strict.float64
    )

    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="same Array API namespace"):
            simulator.predict(X=x_data, y=y_data)


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
    xp_assert_array_equal(model_code, regressor_code[:1])
    xp_assert_allclose(err[0], 0.8)
    xp_assert_array_equal(psi_orth, psi_np[:, :1])


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

    xp_assert_array_equal(model_code_t, model_code_np)
    xp_assert_array_equal(piv_t, piv_np)
    xp_assert_allclose(err_t, err_np, rtol=1e-10, atol=1e-12)
    xp_assert_allclose(psi_orth_t, psi_orth_np, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("steps_ahead", [None, 1, 3])
def test_predict_with_fourier_basis_raises_not_implemented(steps_ahead):
    simulator = SimulateNARMAX(basis_function=Fourier(), estimate_parameter=False)
    simulator.max_lag = 1
    simulator.n_inputs = 1
    x = np.ones((2, 1))
    y = np.ones((2, 1))

    with pytest.raises(NotImplementedError, match="Polynomial Basis Function"):
        simulator.predict(X=x, y=y, steps_ahead=steps_ahead)
