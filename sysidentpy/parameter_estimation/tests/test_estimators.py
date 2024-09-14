from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.narmax_base import InformationMatrix

from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RidgeRegression,
    RecursiveLeastSquares,
    TotalLeastSquares,
    LeastMeanSquareMixedNorm,
    LeastMeanSquares,
    LeastMeanSquaresFourth,
    LeastMeanSquaresLeaky,
    LeastMeanSquaresNormalizedLeaky,
    LeastMeanSquaresNormalizedSignRegressor,
    LeastMeanSquaresNormalizedSignSign,
    LeastMeanSquaresSignError,
    LeastMeanSquaresSignSign,
    AffineLeastMeanSquares,
    NormalizedLeastMeanSquares,
    NormalizedLeastMeanSquaresSignError,
    LeastMeanSquaresSignRegressor,
)


import numpy as np
from numpy.testing import assert_almost_equal, assert_raises


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


x, y, theta = create_test_data()
max_lag = 2
lagged_data = InformationMatrix(xlag=2, ylag=2).build_input_output_matrix(X=x, y=y)[
    :, :
]

psi = Polynomial(degree=2).fit(lagged_data, max_lag, predefined_regressors=None)
# optimize(psi, y_train[max_lag:, :], 0.01)


def test_least_squares():
    # x, y, theta = create_test_data()
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_ridge_regression():
    # x, y, theta = create_test_data()
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=RidgeRegression(alpha=np.finfo(np.float64).eps),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_ridge_regression_classic():
    # x, y, theta = create_test_data()
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=RidgeRegression(alpha=np.finfo(np.float64).eps, solver="classic"),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_raise_ridge_regression():
    assert_raises(ValueError, RidgeRegression, alpha=-0.3)


def test_raise():
    assert_raises(ValueError, RecursiveLeastSquares, lam="0.97")


def test_lam_raise():
    assert_raises(ValueError, RecursiveLeastSquares, lam=1.01)


def test_total_least_squares():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=TotalLeastSquares(),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_recursive_least_squares():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=RecursiveLeastSquares(delta=0.00001, lam=0.99),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_affine_least_mean_squares():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=AffineLeastMeanSquares(mu=0.01, offset_covariance=0.2),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquares(mu=0.1),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_error():
    xl, yl, _ = create_test_data(n=5000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresSignError(mu=0.01),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    print(model.theta.shape, theta.shape)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_normalized_least_mean_squares():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=NormalizedLeastMeanSquares(mu=0.1),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_error():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        n_info_values=6,
        ylag=[1, 2],
        xlag=2,
        info_criteria="aic",
        estimator=NormalizedLeastMeanSquaresSignError(mu=0.005),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_regressor():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresSignRegressor(mu=0.1),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_regressor():
    # x, y, theta = create_test_data()
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresNormalizedSignRegressor(mu=0.1),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_sign_sign():
    xl, yl, _ = create_test_data(n=5000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresSignSign(mu=0.001),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_normalized_sign_sign():
    xl, yl, _ = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresNormalizedSignSign(mu=0.0001),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_least_mean_squares_mixed_norm():
    xl, yl, _ = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquareMixedNorm(mu=0.05),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_nlmsl():
    xl, yl, _ = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresNormalizedLeaky(mu=0.05, gama=0.001),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_lmsl():
    xl, yl, _ = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresLeaky(mu=0.05, gama=0.001),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_lmsf():
    xl, yl, _ = create_test_data(n=30000)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastMeanSquaresFourth(mu=0.5),
        basis_function=Polynomial(degree=2),
        err_tol=None,
    )
    model.fit(X=xl, y=yl)
    assert_almost_equal(model.theta, theta, decimal=2)


# def test_model_order_selection():
#     assert_raises(ValueError, Estimators, max_lag=-1)
