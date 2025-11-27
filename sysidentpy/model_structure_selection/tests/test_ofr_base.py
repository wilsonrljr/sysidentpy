"""Tests for OFRBase internals."""

# pylint: disable=protected-access,redefined-outer-name

from typing import Tuple
import types

import pytest
import numpy as np

from sysidentpy.model_structure_selection.ofr_base import (
    OFRBase,
    _compute_err_slice,
    get_min_info_value,
)
from sysidentpy.parameter_estimation import (
    LeastSquares,
    RecursiveLeastSquares,
    RidgeRegression,
)
from sysidentpy.basis_function import Polynomial
from sysidentpy.narmax_base import BaseMSS


# Create a subclass to instantiate the abstract class
class TestOFRBase(OFRBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_mss_algorithm(
        self, psi: np.ndarray, y: np.ndarray, process_term_number: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


def test_ofrbase_initialization_default():
    """Test initialization of OFRBase with default parameters."""
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=None,
        n_info_values=15,
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0,
        err_tol=None,
    )

    # Check if the attributes are correctly initialized
    assert model.ylag == 2
    assert model.xlag == 2
    assert model.elag == 2
    assert model.order_selection is True
    assert model.info_criteria == "aic"
    assert model.n_info_values == 15
    assert model.n_terms is None
    assert isinstance(model.estimator, RecursiveLeastSquares)
    assert isinstance(model.basis_function, Polynomial)
    assert model.model_type == "NARMAX"
    assert model.eps == np.finfo(np.float64).eps
    assert model.alpha == 0
    assert model.err_tol is None


def test_ofrbase_initialization_with_ridge():
    """Test initialization of OFRBase with RidgeRegression estimator."""
    ridge_estimator = RidgeRegression(alpha=0.5)
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=None,
        n_info_values=15,
        estimator=ridge_estimator,
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0.5,
        err_tol=None,
    )

    # Check if the alpha value from the estimator is correctly set
    assert model.alpha == 0.5


def test_ofrbase_initialization_invalid_info_criteria():
    """Test initialization of OFRBase with an invalid info_criteria."""
    with pytest.raises(
        ValueError,
        match="info_criteria must be aic, bic, fpe or lilc. Got invalid_criteria",
    ):
        TestOFRBase(
            ylag=2,
            xlag=2,
            elag=2,
            order_selection=True,
            info_criteria="invalid_criteria",  # Invalid criteria
            n_terms=None,
            n_info_values=15,
            estimator=RecursiveLeastSquares(),
            basis_function=Polynomial(),
            model_type="NARMAX",
            eps=np.finfo(np.float64).eps,
            alpha=0,
            err_tol=None,
        )


def test_ofrbase_initialization_with_n_terms():
    """Test initialization of OFRBase with a defined n_terms."""
    model = TestOFRBase(
        ylag=2,
        xlag=2,
        elag=2,
        order_selection=True,
        info_criteria="aic",
        n_terms=10,
        n_info_values=15,
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(),
        model_type="NARMAX",
        eps=np.finfo(np.float64).eps,
        alpha=0,
        err_tol=None,
    )

    # Assert that the model's n_terms is correctly set
    assert model.n_terms == 10


def test_ofrbase_invalid_eps():
    """Test invalid eps value for OFRBase initialization."""
    with pytest.raises(ValueError, match="eps must be float and > zero. Got -1"):
        TestOFRBase(
            ylag=2,
            xlag=2,
            elag=2,
            order_selection=True,
            info_criteria="aic",
            n_terms=None,
            n_info_values=15,
            estimator=RecursiveLeastSquares(),
            basis_function=Polynomial(),
            model_type="NARMAX",
            eps=-1,  # Invalid eps
            alpha=0,
            err_tol=None,
        )


# Create a mock of the class to test the method
class MockOFRBase(OFRBase):
    def __init__(self):
        # Initialize necessary parameters for testing
        self.max_lag = 2
        self.alpha = 0.01
        self.eps = np.finfo(np.float64).eps
        self.err_tol = None  # Set to None or some value depending on the test case
        self.n_terms = 10

    def run_mss_algorithm(self, psi, y, process_term_number):
        return self.error_reduction_ratio(psi, y, process_term_number)


@pytest.fixture
def setup_data():
    # Create mock data for psi and y to simulate real-world input
    n_samples = 200
    n_features = 10
    max_lag = 2
    y = np.random.randn(n_samples + max_lag, 1)
    psi = np.random.randn(n_samples, n_features)
    process_term_number = 5

    return y, psi, process_term_number


def test_error_reduction_ratio_basic(setup_data):
    """Test basic functionality of the error_reduction_ratio method."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure that the output is as expected
    assert isinstance(err, np.ndarray)
    assert isinstance(piv, np.ndarray)
    assert isinstance(psi_orthogonal, np.ndarray)

    # Check that the lengths of the returned arrays match the expected number
    # of regressors (n_features)
    assert err.shape[0] == psi.shape[1]  # Number of model elements (n_features)
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number


def test_error_reduction_ratio_with_large_alpha(setup_data):
    """Test the ERR with a large alpha to observe the effect of regularization."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Set a very large alpha to test its effect on the error reduction ratio
    model.alpha = 1000

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure that the returned values are not empty and match the expected dimensions
    assert err.shape[0] == psi.shape[1]
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number

    # Additional check: Assert that the error values are reasonably
    # affected by a large alpha
    assert np.all(
        err < 1
    )  # Assuming that large alpha reduces the magnitude of the errors


def test_error_reduction_ratio_with_small_eps(setup_data):
    """Test ERR with a very small epsilon to see its effect on stability."""
    y, psi, process_term_number = setup_data
    model = MockOFRBase()

    # Set a very small epsilon value to test numerical stability
    model.eps = 1e-12

    # Call the method
    err, piv, psi_orthogonal = model.error_reduction_ratio(psi, y, process_term_number)

    # Ensure the returned values are valid
    assert err.shape[0] == psi.shape[1]
    assert piv.shape[0] == process_term_number
    assert psi_orthogonal.shape[1] == process_term_number


class SimpleOFR(OFRBase):
    def __init__(self, **kwargs):
        estimator = kwargs.pop("estimator", LeastSquares())
        basis = kwargs.pop("basis_function", Polynomial(degree=1))
        super().__init__(estimator=estimator, basis_function=basis, **kwargs)

    def run_mss_algorithm(self, psi, y, process_term_number):
        return self.error_reduction_ratio(psi, y, process_term_number)


class _FakeBasis:
    degree = 1
    ensemble = False

    def fit(self, lagged_data, *args, **kwargs):
        _ = args
        _ = kwargs
        return np.ones((lagged_data.shape[0], 1), dtype=np.float64)

    def transform(self, lagged_data, *args, **kwargs):
        _ = args
        _ = kwargs
        return np.ones((lagged_data.shape[0], 1), dtype=np.float64)


@pytest.fixture
def series_data():
    x = np.arange(10, dtype=np.float64).reshape(-1, 1)
    y = (np.arange(10, dtype=np.float64) * 0.5).reshape(-1, 1)
    return x, y


def test_get_min_info_value_returns_length_when_no_increase():
    assert get_min_info_value(np.array([3.0, 2.0, 2.0])) == 3


def test_compute_err_slice_handles_empty_block():
    psi = np.ones((3, 3), dtype=np.float64)
    y = np.ones((3, 1), dtype=np.float64)
    result = _compute_err_slice(psi, y, start_idx=3, squared_y=1.0, alpha=0.0, eps=0.0)
    assert result.size == 0


def test_validate_params_rejects_invalid_model_type():
    with pytest.raises(ValueError, match="model_type"):
        SimpleOFR(model_type="UNKNOWN")


def test_error_reduction_ratio_updates_n_terms_with_err_tol():
    model = SimpleOFR(err_tol=0.0)
    model.max_lag = 0
    psi = np.eye(3, dtype=np.float64)
    y = np.arange(3, dtype=np.float64).reshape(-1, 1)
    _, piv, _ = model.error_reduction_ratio(psi, y, process_term_number=2)
    assert model.n_terms == 1
    assert piv.shape[0] == 1


def test_fit_requires_y_argument():
    model = SimpleOFR()
    with pytest.raises(ValueError, match="y cannot be None"):
        model.fit(X=None, y=None)


def test_fit_requires_n_terms_when_order_selection_disabled(series_data):
    model = SimpleOFR(order_selection=False, n_terms=None)
    x, y = series_data
    with pytest.raises(ValueError, match="define n_terms value"):
        model.fit(X=x, y=y)


def test_predict_non_polynomial_uses_basis_function_paths():
    model = SimpleOFR(basis_function=_FakeBasis())
    model.max_lag = 1
    model.final_model = np.array([[0]])
    model.pivv = np.array([0])

    calls = {}

    def fake_n_step(self, _x, _y, steps_ahead, forecast_horizon):
        _ = self
        calls["args"] = (steps_ahead, forecast_horizon)
        return np.ones((4, 1), dtype=np.float64)

    model._basis_function_n_step_prediction = types.MethodType(fake_n_step, model)
    y_data = np.ones((4, 1), dtype=np.float64)
    result = model.predict(X=None, y=y_data, steps_ahead=2, forecast_horizon=1)
    assert calls["args"] == (2, 1)
    assert result.shape == (y_data.shape[0] + 1, 1)


def test_model_prediction_raises_for_invalid_type():
    model = SimpleOFR()
    model.model_type = "UNKNOWN"
    with pytest.raises(ValueError, match="model_type must be"):
        model._model_prediction(np.ones((3, 1)), np.ones((3, 1)))


def test_narmax_predict_sets_inputs_to_zero(monkeypatch):
    model = SimpleOFR(model_type="NAR")
    model.max_lag = 1
    model.n_inputs = 2

    def fake_super(self, _x, _y, horizon):
        _ = self
        return np.ones((horizon, 1), dtype=np.float64)

    monkeypatch.setattr(BaseMSS, "_narmax_predict", fake_super)
    result = model._narmax_predict(
        x=None, y_initial=np.ones((1, 1)), forecast_horizon=2
    )
    assert result.shape == (3, 1)
    assert model.n_inputs == 0


def test_basis_function_predict_extends_horizon(monkeypatch):
    model = SimpleOFR(model_type="NAR")
    model.max_lag = 2
    model.n_inputs = 3

    def fake_super(self, _x, _y, horizon):
        _ = self
        return np.ones((horizon, 1), dtype=np.float64)

    monkeypatch.setattr(BaseMSS, "_basis_function_predict", fake_super)
    result = model._basis_function_predict(
        x=None, y_initial=np.ones((2, 1)), forecast_horizon=1
    )
    assert result.shape == (3, 1)
    assert model.n_inputs == 0


def test_basis_function_n_step_prediction_requires_initial_conditions():
    model = SimpleOFR()
    model.max_lag = 3
    with pytest.raises(ValueError, match="Insufficient initial condition"):
        model._basis_function_n_step_prediction(
            None, np.ones((2, 1)), steps_ahead=1, forecast_horizon=1
        )


def test_basis_function_n_step_prediction_extends_horizon(monkeypatch):
    model = SimpleOFR()
    model.max_lag = 1

    def fake_super(self, _x, _y, _steps, horizon):
        _ = self
        return np.ones((horizon, 1), dtype=np.float64)

    monkeypatch.setattr(BaseMSS, "_basis_function_n_step_prediction", fake_super)
    result = model._basis_function_n_step_prediction(
        None, np.ones((4, 1)), steps_ahead=1, forecast_horizon=2
    )
    assert result.shape == (3, 1)


def test_basis_function_n_steps_horizon_returns_column(monkeypatch):
    model = SimpleOFR()

    def fake_super(self, *_args, **_kwargs):
        _ = self
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(BaseMSS, "_basis_function_n_steps_horizon", fake_super)
    result = model._basis_function_n_steps_horizon(
        None, None, steps_ahead=1, forecast_horizon=1
    )
    assert result.shape == (3, 1)


def test_information_criterion_clamps_n_info_values(monkeypatch):
    model = SimpleOFR(n_info_values=5)
    model.max_lag = 0
    x = np.ones((4, 3), dtype=np.float64)
    y = np.ones((4, 1), dtype=np.float64)

    def fake_run(self, psi, y_vals, process_term_number):
        _ = self
        _ = y_vals
        cols = min(process_term_number, psi.shape[1])
        reg = np.ones((psi.shape[0], cols), dtype=np.float64)
        err = np.zeros(psi.shape[1], dtype=np.float64)
        piv = np.arange(cols)
        return err, piv, reg

    monkeypatch.setattr(SimpleOFR, "run_mss_algorithm", fake_run)
    with pytest.warns(UserWarning):
        output = model.information_criterion(x, y)
    assert len(output) == 3
    assert model.n_info_values == 3


def test_basis_function_branch_in_predict_uses_n_step(monkeypatch):
    model = SimpleOFR(basis_function=_FakeBasis())
    model.max_lag = 1
    model.final_model = np.array([[0]])
    model.pivv = np.array([0])
    called = {}

    def fake_predict(self, *_args, **_kwargs):
        _ = self
        called["mode"] = "basis_n_step"
        return np.ones((4, 1), dtype=np.float64)

    monkeypatch.setattr(
        model,
        "_basis_function_n_step_prediction",
        types.MethodType(fake_predict, model),
        raising=False,
    )
    y_data = np.ones((4, 1))
    res = model.predict(X=None, y=y_data, steps_ahead=3, forecast_horizon=1)
    assert called["mode"] == "basis_n_step"
    assert res.shape == (y_data.shape[0] + 1, 1)
