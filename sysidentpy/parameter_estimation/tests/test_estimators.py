from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial

import numpy as np
from numpy.testing import assert_almost_equal, assert_raises
from .. import Estimators


def create_test_data(n=1000):
    np.random.seed(42)
    x = np.random.uniform(-1, 1, n).T
    y = np.zeros((n, 1))
    theta = np.array([[0.6], [-0.5], [0.7], [-0.7], [0.2]])
    lag = 2
    for k in range(lag, len(x)):
        y[k] = (
            theta[4] * y[k - 1] ** 2
            + theta[2] * y[k - 1] * x[k - 1]
            + theta[0] * x[k - 2]
            + theta[3] * y[k - 2] * x[k - 2]
            + theta[1] * y[k - 2]
        )

    y = np.reshape(y, (len(y), 1))
    x = np.reshape(x, (len(x), 1))

    return x, y, theta


def test_least_squares():
    x, y, theta = create_test_data()
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_squares",
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_total_least_squares():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="total_least_squares",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    print(model.theta)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_recursive_least_squares():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        delta=0.00001,
        lam=0.99,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="recursive_least_squares",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_affine_least_mean_squares():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        mu=0.01,
        offset_covariance=0.2,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="affine_least_mean_squares",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        mu=0.1,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_error():
    x, y, theta = create_test_data(n=5000)
    model = FROLS(
        n_terms=5,
        mu=0.01,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_sign_error",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_normalized_least_mean_squares():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        mu=0.1,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="normalized_least_mean_squares",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_error():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        n_info_values=6,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        info_criteria="aic",
        estimator="least_mean_squares_normalized_sign_error",
        mu=0.005,
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_regressor():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        mu=0.1,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_sign_regressor",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_regressor():
    x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        mu=0.1,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_normalized_sign_regressor",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_sign():
    x, y, theta = create_test_data(n=5000)
    model = FROLS(
        n_terms=5,
        mu=0.001,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_sign_sign",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_sign():
    x, y, theta = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        mu=0.0001,
        # eps=0.05,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_normalized_sign_sign",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_mixed_norm():
    x, y, theta = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        mu=0.05,
        extended_least_squares=False,
        ylag=[1, 2],
        xlag=2,
        estimator="least_mean_squares_mixed_norm",
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_model_order_selection():
    assert_raises(ValueError, Estimators, max_lag=-1)
