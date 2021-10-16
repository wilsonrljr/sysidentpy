from numpy.testing._private.utils import assert_allclose
from sysidentpy.polynomial_basis import PolynomialNarmax
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises
from sysidentpy.polynomial_basis import SimulatePolynomialNarmax


def test_get_index_from_regressor_code():
    s = SimulatePolynomialNarmax()
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    regressor_space = np.array(
        [
            [0, 0],
            [1001, 0],
            [2001, 0],
            [2002, 0],
            [1001, 1001],
            [2001, 1001],
            [2002, 1001],
            [2001, 2001],
            [2002, 2001],
            [2002, 2002],
        ]
    )
    index = s._get_index_from_regressor_code(
        regressor_code=regressor_space, model_code=model
    )

    assert (index == np.array([1, 3, 5])).all()


def test_list_output_regressor():
    s = SimulatePolynomialNarmax()
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    y_code = s._list_output_regressor_code(model)
    assert (y_code == np.array([1001, 1001])).all()


def test_list_input_regressor():
    s = SimulatePolynomialNarmax()
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    x_code = s._list_input_regressor_code(model)
    assert (x_code == np.array([2001, 2002])).all()


def test_get_lag_from_regressor_code():
    s = SimulatePolynomialNarmax()
    list_regressor1 = np.array([2001, 2002])
    list_regressor2 = np.array([1004, 1002])
    max_lag1 = s._get_lag_from_regressor_code(list_regressor1)
    max_lag2 = s._get_lag_from_regressor_code(list_regressor2)

    assert max_lag1 == 2
    assert max_lag2 == 4


def test_simulate():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulatePolynomialNarmax()

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

    yhat, results = s.simulate(
        X_test=x_valid, y_test=y_valid, model_code=model, theta=theta, plot=False
    )
    assert yhat.shape == (100, 1)
    assert len(results) == 3


def test_simulate_theta():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulatePolynomialNarmax(estimate_parameter=True)

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    yhat, results = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        plot=False,
    )
    theta = np.array([[0.2, 0.9, 0.1]]).T
    assert_almost_equal(s.theta, theta, decimal=1)


def test_estimate_parameter():
    assert_raises(TypeError, SimulatePolynomialNarmax, estimmate_parameter=1)