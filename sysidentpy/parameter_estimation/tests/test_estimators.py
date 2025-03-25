from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.information_matrix import build_input_output_matrix


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
    BoundedVariableLeastSquares,
    LeastSquaresMinimalResidual,
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
lagged_data = build_input_output_matrix(x=x, y=y, xlag=2, ylag=2)[:, :]

psi = Polynomial(degree=2).fit(lagged_data, max_lag, predefined_regressors=None)


def test_least_squares():
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


def test_unbiased_least_squares():
    # x, y, theta = create_test_data()
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(unbiased=True),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta.shape[0], theta.shape[0], decimal=2)


def test_bvls_init():
    """Test initialization of BoundedVariableLeastSquares."""
    unbiased = True
    uiter = 50
    bounds = (-1.0, 1.0)
    method = "dogbox"
    tol = 1e-8
    lsq_solver = "exact"
    lsmr_tol = 1e-5
    max_iter = 500
    verbose = 1
    lsmr_maxiter = 1000

    model = BoundedVariableLeastSquares(
        unbiased=unbiased,
        uiter=uiter,
        bounds=bounds,
        method=method,
        tol=tol,
        lsq_solver=lsq_solver,
        lsmr_tol=lsmr_tol,
        max_iter=max_iter,
        verbose=verbose,
        lsmr_maxiter=lsmr_maxiter,
    )

    assert model.unbiased == unbiased
    assert model.uiter == uiter
    assert model.bounds == bounds
    assert model.method == method
    assert model.tol == tol
    assert model.lsq_solver == lsq_solver
    assert model.lsmr_tol == lsmr_tol
    assert model.max_iter == max_iter
    assert model.verbose == verbose
    assert model.lsmr_maxiter == lsmr_maxiter

    default_model = BoundedVariableLeastSquares()
    assert default_model.unbiased is False
    assert default_model.uiter == 30
    assert default_model.bounds == (-np.inf, np.inf)
    assert default_model.method == "trf"
    assert default_model.tol == 1e-10
    assert default_model.lsq_solver is None
    assert default_model.lsmr_tol is None
    assert default_model.max_iter is None
    assert default_model.verbose == 0
    assert default_model.lsmr_maxiter is None


def test_bvls_results():
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=BoundedVariableLeastSquares(),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_lsmr_results():
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquaresMinimalResidual(),
        basis_function=basis_function,
        err_tol=None,
    )
    model.fit(X=x, y=y)
    assert_almost_equal(model.theta, theta, decimal=2)


def test_lsqr_init():
    """Test initialization of LeastSquaresMinimalResidual."""
    unbiased = True
    uiter = 50
    damp = 0.1
    atol = 1e-5
    btol = 1e-5
    conlim = 1e7
    maxiter = 100
    show = True
    x0 = [0.5, -0.2, 1.0]

    model = LeastSquaresMinimalResidual(
        unbiased=unbiased,
        uiter=uiter,
        damp=damp,
        atol=atol,
        btol=btol,
        conlim=conlim,
        maxiter=maxiter,
        show=show,
        x0=x0,
    )

    assert model.unbiased == unbiased
    assert model.uiter == uiter
    assert model.damp == damp
    assert model.atol == atol
    assert model.btol == btol
    assert model.conlim == conlim
    assert model.maxiter == maxiter
    assert model.show == show
    assert model.x0 == x0

    default_model = LeastSquaresMinimalResidual()
    assert default_model.unbiased is False
    assert default_model.uiter == 30
    assert default_model.damp == 0.0
    assert default_model.atol == 1e-6
    assert default_model.btol == 1e-6
    assert default_model.conlim == 1e8
    assert default_model.maxiter is None
    assert default_model.show is False
    assert default_model.x0 is None
