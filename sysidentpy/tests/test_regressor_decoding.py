"""Analytical checks of shared regressor codes across prediction consumers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS, MetaMSS
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.utils.lags import get_lag_from_regressor_code
from sysidentpy.utils.simulation import (
    list_input_regressor_code,
    list_output_regressor_code,
)


@pytest.mark.parametrize("lag", [99, 100, 499, 500, 501, 600, 999])
def test_codes_round_trip_variable_and_lag(lag):
    codes = np.array([[1000 + lag, 10000 + lag], [11000 + lag, 0]])
    assert_array_equal(list_output_regressor_code(codes), [1000 + lag])
    assert_array_equal(list_input_regressor_code(codes), [10000 + lag, 11000 + lag])
    assert get_lag_from_regressor_code(codes[codes != 0]) == lag


def test_decoding_preserves_duplicates_and_empty_codes():
    codes = np.array([[1600, 1600], [10001, 10001], [0, 0]])
    assert_array_equal(list_output_regressor_code(codes), [1600, 1600])
    assert_array_equal(list_input_regressor_code(codes), [10001, 10001])
    assert list_output_regressor_code(np.zeros((1, 2), dtype=int)).size == 0
    assert list_input_regressor_code(np.zeros((1, 2), dtype=int)).size == 0
    assert get_lag_from_regressor_code(np.array([], dtype=int)) == 1


def _long_lag_series(lag):
    rng = np.random.default_rng(2026)
    y = rng.normal(size=(lag + 16, 1))
    for k in range(lag, len(y)):
        y[k] = 0.5 * y[k - lag]
    return y


@pytest.mark.parametrize("lag", [99, 100, 499, 500, 501, 600, 999])
def test_fitted_nar_long_lag_predictions_match_recurrence(lag):
    y = _long_lag_series(lag)
    model = FROLS(
        model_type="NAR",
        ylag=[lag],
        xlag=1,
        n_terms=1,
        order_selection=False,
        basis_function=Polynomial(degree=1, include_bias=False),
        estimator=LeastSquares(),
    ).fit(y=y)
    assert_array_equal(model.final_model, [[1000 + lag]])
    assert_allclose(model.theta, [[0.5]], rtol=1e-12, atol=1e-12)
    for steps in (None, 1, 7):
        result = model.predict(y=y, steps_ahead=steps, forecast_horizon=16)
        assert_allclose(result, y, rtol=1e-12, atol=1e-12)
        assert_array_equal(result[:lag], y[:lag])


@pytest.mark.parametrize("simulator_class", [SimulateNARMAX, MetaMSS])
def test_simulation_consumers_decode_long_lag(simulator_class):
    lag = 100
    y = _long_lag_series(lag)
    model = simulator_class(model_type="NAR", basis_function=Polynomial(degree=1))
    model.estimate_parameter = False
    prediction = model.simulate(
        y_test=y, model_code=np.array([[1100]]), theta=np.array([[0.5]])
    )
    assert model.max_lag == lag
    assert_allclose(prediction, y, rtol=1e-12, atol=1e-12)


def test_simulator_preserves_high_input_codes_and_parameter_order():
    rng = np.random.default_rng(80)
    x = rng.normal(size=(12, 10))
    y = np.zeros((12, 1))
    y[1:, 0] = 0.3 + 2 * x[:-1, 8] - x[:-1, 9]
    codes = np.array([[11001], [0], [10001]])
    model = SimulateNARMAX(
        model_type="NFIR", estimate_parameter=False, basis_function=Polynomial(degree=1)
    )
    result = model.simulate(
        X_test=x, y_test=y, model_code=codes, theta=np.array([[-1.0], [0.3], [2.0]])
    )
    assert_array_equal(model.final_model, codes)
    assert_allclose(result, y, rtol=1e-12, atol=1e-12)


def test_long_lag_products_and_bias_match_analytical_model():
    lag = 600
    rng = np.random.default_rng(80)
    y = rng.normal(size=(lag + 12, 1))
    for k in range(lag, len(y)):
        y[k] = 0.1 + 0.2 * y[k - lag] + 0.01 * y[k - lag] ** 2
    model = FROLS(
        model_type="NAR",
        ylag=[lag],
        xlag=1,
        n_terms=3,
        order_selection=False,
        basis_function=Polynomial(degree=2),
        estimator=LeastSquares(),
    ).fit(y=y)
    assert_allclose(model.predict(y=y, forecast_horizon=12), y, rtol=1e-12, atol=1e-12)
    assert_allclose(model.predict(y=y, steps_ahead=1), y, rtol=1e-12, atol=1e-12)


def test_narmax_long_input_lag_matches_recurrence():
    lag = 501
    rng = np.random.default_rng(83)
    x = rng.normal(size=(lag + 30, 1))
    y = rng.normal(size=x.shape)
    for k in range(lag, len(y)):
        y[k] = 0.2 * y[k - 1] + 0.3 * x[k - lag]
    model = FROLS(
        model_type="NARMAX",
        ylag=[1],
        xlag=[lag],
        n_terms=2,
        order_selection=False,
        basis_function=Polynomial(degree=1, include_bias=False),
        estimator=LeastSquares(),
    ).fit(X=x, y=y)
    expected_coefficients = {1001: 0.2, 2501: 0.3}
    assert set(model.final_model[:, 0]) == set(expected_coefficients)
    assert_allclose(
        model.theta[:, 0],
        [expected_coefficients[code] for code in model.final_model[:, 0]],
        rtol=1e-12,
        atol=1e-12,
    )
    for steps in (None, 1, 7):
        prediction = model.predict(X=x, y=y, steps_ahead=steps)
        assert_allclose(prediction, y, rtol=1e-12, atol=1e-12)


def test_general_estimator_uses_long_lag_exponents():
    linear_model = pytest.importorskip("sklearn.linear_model")
    from sysidentpy.general_estimators import NARX

    y = _long_lag_series(600)
    model = NARX(
        model_type="NAR",
        ylag=[600],
        xlag=1,
        basis_function=Polynomial(degree=1, include_bias=False),
        base_estimator=linear_model.LinearRegression(fit_intercept=False),
    ).fit(y=y)
    assert_array_equal(model.final_model, [[1600]])
    for steps in (None, 1, 7):
        assert_allclose(
            model.predict(y=y, steps_ahead=steps, forecast_horizon=16),
            y,
            rtol=1e-12,
            atol=1e-12,
        )


def test_neural_estimator_uses_long_lag_exponents_without_training():
    torch = pytest.importorskip("torch")
    from sysidentpy.neural_network import NARXNN

    net = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        net.weight.fill_(0.5)
    y = _long_lag_series(600).astype(np.float32)
    model = NARXNN(
        model_type="NAR",
        ylag=[600],
        xlag=1,
        basis_function=Polynomial(degree=1, include_bias=False),
        net=net,
    )
    matrix, target = model.split_data(None, y)
    assert_allclose(matrix, y[:-600], rtol=0, atol=0)
    assert_allclose(target, 0.5 * matrix, rtol=0, atol=0)
    for steps in (None, 1, 7):
        assert_allclose(
            model.predict(y=y, steps_ahead=steps, forecast_horizon=16),
            y,
            rtol=1e-6,
            atol=1e-7,
        )


@pytest.mark.parametrize(("n_inputs", "lag"), [(1, 100), (10, 1)])
def test_ails_dynamic_matrix_decodes_fixed_model(n_inputs, lag):
    from sysidentpy.multiobjective_parameter_estimation import AILS

    rng = np.random.default_rng(54)
    x = rng.normal(size=(lag + 8, n_inputs))
    y = rng.normal(size=(lag + 8, 1))
    codes = np.array([[1000 + lag, 0], [(n_inputs + 1) * 1000 + 1, 0]])
    model = AILS(final_model=codes, static_gain=False, static_function=False)
    matrix = model.build_psi(x, y)
    assert model.max_lag == lag
    assert_array_equal(model.final_model, codes)
    expected = np.column_stack([y[:-lag, 0], x[lag - 1 : -1, -1]])
    assert_allclose(matrix, expected, rtol=0, atol=0)
