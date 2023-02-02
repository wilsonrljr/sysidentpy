import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises
from sysidentpy.model_structure_selection import MetaMSS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data


def create_test_data(n=1000):
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])
    data = np.loadtxt("examples/datasets/data_for_testing.txt")
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    return x, y, theta


def test_metamss():
    # piv = np.array([4, 2, 7, 11, 5])
    model_code = np.array(
        [[1001, 0], [2002, 0], [2001, 1001]]  # y(k-1)  # x1(k-2)  # x1(k-1)y(k-1)
    )
    basis_function = Polynomial(degree=2)
    X_train, X_test, y_train, y_test = get_siso_data(
        n=1000, colored_noise=False, sigma=0.0001, train_percentage=90
    )

    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=20,
        basis_function=basis_function,
        random_state=42,
    )
    model.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    assert_array_equal(model.final_model, model_code)


def test_default_values():
    default = {
        "ylag": 1,
        "xlag": 1,
        "model_type": "NARMAX",
        "maxiter": 30,
        "alpha": 23,
        "g_zero": 100,
        "k_agents_percent": 2,
        "norm": -2,
        "power": 2,
        "n_agents": 10,
        "p_zeros": 0.5,
        "p_ones": 0.5,
        "p_value": 0.05,
        "estimator": "least_squares",
        "extended_least_squares": False,
        "lam": 0.98,
        "delta": 0.01,
        "offset_covariance": 0.2,
        "mu": 0.01,
        "eps": np.finfo(np.float64).eps,
        "gama": 0.2,
        "weight": 0.02,
        "steps_ahead": None,
        "estimate_parameter": True,
        "loss_func": "metamss_loss",
        "random_state": None,
    }
    model = MetaMSS(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.model_type,
        model.maxiter,
        model.alpha,
        model.g_zero,
        model.k_agents_percent,
        model._norm,
        model._power,
        model.n_agents,
        model.p_zeros,
        model.p_ones,
        model.p_value,
        model.estimator,
        model.extended_least_squares,
        model.lam,
        model.delta,
        model.offset_covariance,
        model.mu,
        model.eps,
        model.gama,
        model.weight,
        model.steps_ahead,
        model.estimate_parameter,
        model.loss_func,
        model.random_state,
    ]
    assert list(default.values()) == model_values


def test_validate_ylag():
    assert_raises(ValueError, MetaMSS, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, MetaMSS, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, MetaMSS, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, MetaMSS, xlag=1.3, basis_function=Polynomial(degree=2))


def test_predict():
    X_train, X_test, y_train, y_test = get_siso_data(
        n=1000, colored_noise=False, sigma=0.0001, train_percentage=90
    )
    basis_function = Polynomial(degree=2)
    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=10,
        basis_function=basis_function,
        random_state=42,
    )
    model.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=2)


def test_model_prediction():
    x, y, _ = create_test_data()
    basis_function = Polynomial(degree=2)
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
    model = MetaMSS(
        ylag=[1, 2],
        xlag=2,
        maxiter=30,
        n_agents=20,
        basis_function=basis_function,
        random_state=42,
    )
    model.fit(X=X_train, y=y_train, X_test=X_test, y_test=y_test)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])
