from typing import Optional

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_equal,
    assert_equal,
    assert_raises,
)

from sysidentpy.basis_function import Polynomial, Fourier
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

    # Load dataset from external source
    url = "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/testing/data_for_testing.txt"
    data = np.loadtxt(url)

    # Extract input (x) and output (y)
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
        # extended_least_squares=False,
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
        # extended_least_squares=False,
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
        # extended_least_squares=False,
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
        # extended_least_squares=False,
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
        # extended_least_squares=False,
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


def test_nar_step_ahead_insufficient_initial_conditions():
    """Test that _nar_step_ahead raises an error if input is too short."""
    model = FROLS(
        order_selection=True,
        # extended_least_squares=False,
        ylag=[1, 2],
        estimator=RecursiveLeastSquares(),
        basis_function=Polynomial(degree=2),
        model_type="NAR",
    )
    model.fit(y=y_train)

    with pytest.raises(ValueError, match="Insufficient initial condition elements!"):
        model._nar_step_ahead(y[0], steps_ahead=2)


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
        BaseMSS()  # Should fail because it's an abstract class


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


def test_n_step_ahead_prediction_invalid_model():
    """Test `_n_step_ahead_prediction` with an invalid model type."""
    model = ConcreteMSS(model_type="NFIR")

    with pytest.raises(
        ValueError,
        match="n_steps_ahead prediction will be implemented for NFIR models in v0.4.*",
    ):
        model._n_step_ahead_prediction(None, np.array([1, 2, 3]), 2)
