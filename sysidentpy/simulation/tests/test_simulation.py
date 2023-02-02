import numpy as np
from numpy.testing import assert_almost_equal, assert_raises

from sysidentpy.basis_function import Fourier, Polynomial
from sysidentpy.simulation import SimulateNARMAX
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
        model.extended_least_squares,
        model.lam,
        model.delta,
        model.offset_covariance,
        model.mu,
        model.eps,
        model.gama,
        model.weight,
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
    _, x_valid, _, y_valid = get_siso_data(
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


def test_basis_function_error():
    assert_raises(TypeError, SimulateNARMAX, model_type="NFIR", basis_function=None)


def test_raises():
    _, x_valid, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(degree=2), estimate_parameter=False)

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

    # the model must be a numpy array
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
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=False, model_type="NAR"
    )

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    yhat = s.simulate(
        X_test=None, y_test=y_valid, model_code=model, theta=theta, forecast_horizon=1
    )
    assert s.n_inputs == 0


def test_miso_dimension():
    _, x_valid, _, y_valid = get_miso_data(
        n=100, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(basis_function=Polynomial(), estimate_parameter=False)

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [3002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    yhat = s.simulate(X_test=x_valid, y_test=y_valid, model_code=model, theta=theta)
    assert s.xlag == [[1, 2], [1, 2]]


def test_forecast_horizon():
    _, _, _, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=False, model_type="NAR"
    )

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )
    # theta must be a numpy array of shape (n, 1) where n is the number of regressors
    theta = np.array([[0.2, 0.9, 0.1]]).T

    yhat = s.simulate(
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

    assert s.max_lag == 2


def test_estimate_parameter_nar():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), estimate_parameter=True, model_type="NAR"
    )

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )

    yhat = s.simulate(
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

    # the model must be a numpy array
    model = np.array(
        [
            [2001, 0],  # y(k-1)
            [2001, 2001],  # x1(k-1)y(k-1)
            [2003, 0],  # x1(k-2)
        ]
    )

    yhat = s.simulate(
        X_train=x_train,
        y_train=y_train,
        X_test=x_valid,
        y_test=y_valid,
        model_code=model,
    )

    assert s.max_lag == 3


##############


def test_err_narmax():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )

    s = SimulateNARMAX(
        basis_function=Polynomial(), calculate_err=True, estimate_parameter=True
    )

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

    # the model must be a numpy array
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [1001, 1001],  # x1(k-1)y(k-1)
            [1002, 0],  # x1(k-2)
        ]
    )

    yhat = s.simulate(
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

    # the model must be a numpy array
    model = np.array(
        [
            [2001, 0],  # y(k-1)
            [2001, 2001],  # x1(k-1)y(k-1)
            [2003, 0],  # x1(k-2)
        ]
    )

    yhat = s.simulate(
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
        extended_least_squares=True,
    )

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
    print(s.theta)
    assert_almost_equal(
        s.theta, np.array([[0.19999698], [0.90011667], [0.10080975]]), decimal=3
    )
