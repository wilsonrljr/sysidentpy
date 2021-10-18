import numpy as np
from numpy.testing import assert_almost_equal
from numpy.testing import assert_raises

# from sysidentpy.model_structure_selection import FROLS
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.simulation import SimulateNARMAX
from sysidentpy.basis_function import Polynomial, Fourier


def test_simulate():
    x_train, x_valid, y_train, y_valid = get_siso_data(
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

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )

    yhat = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )
    theta = np.array([[0.2, 0.9, 0.1]]).T
    assert_almost_equal(s.theta, theta, decimal=1)


def test_estimate_parameter():
    x_train, x_valid, y_train, y_valid = get_siso_data(
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
        "estimator": "recursive_least_squares",
        "extended_least_squares": False,
        "lam": 0.98,
        "delta": 0.01,
        "offset_covariance": 0.2,
        "mu": 0.01,
        "eps": np.finfo(np.float64).eps,
        "gama": 0.2,
        "weight": 0.02,
        "model_type": "NARMAX",
        "estimate_parameter": True,
        "calculate_err": False,
    }
    model = SimulateNARMAX(basis_function=Polynomial())
    model_values = [
        model.estimator,
        model._extended_least_squares,
        model._lam,
        model._delta,
        model._offset_covariance,
        model._mu,
        model._eps,
        model._gama,
        model._weight,
        model.model_type,
        model.estimate_parameter,
        model.calculate_err,
    ]
    assert list(default.values()) == model_values


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
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Fourier(), estimate_parameter=False)

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
    assert_raises(
        NotImplementedError,
        s.simulate,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
        theta=theta,
    )
