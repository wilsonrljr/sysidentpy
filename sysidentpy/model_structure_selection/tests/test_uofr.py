import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_raises

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import UOFR
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
    RidgeRegression,
)
from sysidentpy.tests.test_narmax_base import create_test_data

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


def test_error_reduction_ratio():
    # piv = np.array([4, 2, 7, 11, 5])
    model_code = np.array(
        [[2002, 0], [1002, 0], [1001, 1001], [2001, 1001], [2002, 1002]]
    )
    basis_function = Polynomial(degree=2)
    x, y, _ = create_test_data()
    model = UOFR(
        n_terms=5,
        order_selection=True,
        n_info_values=5,
        info_criteria="aic",
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    assert_array_equal(model.final_model, model_code)


def test_fit_with_information_criteria():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=15,
        order_selection=True,
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    assert "info_values" in dir(model)


def test_fit_without_information_criteria():
    basis_function = Polynomial(degree=2)
    model = UOFR(n_terms=15, basis_function=basis_function, order_selection=False)
    model.fit(X=x, y=y)
    assert model.info_values is None


def test_default_values():
    default = {
        "ylag": 2,
        "xlag": 2,
        "order_selection": True,
        "info_criteria": "aic",
        "n_terms": None,
        "n_info_values": 15,
        "eps": np.finfo(np.float64).eps,
        "alpha": 0,
        "model_type": "NARMAX",
        "err_tol": None,
    }
    model = UOFR(basis_function=Polynomial(degree=2))
    model_values = [
        model.ylag,
        model.xlag,
        model.order_selection,
        model.info_criteria,
        model.n_terms,
        model.n_info_values,
        model.eps,
        model.alpha,
        model.model_type,
        model.err_tol,
    ]
    assert list(default.values()) == model_values
    assert isinstance(model.estimator, RecursiveLeastSquares)
    assert isinstance(model.basis_function, Polynomial)


def test_validate_ylag():
    assert_raises(ValueError, UOFR, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, UOFR, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, UOFR, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, UOFR, xlag=1.3, basis_function=Polynomial(degree=2))


def test_model_order_selection():
    assert_raises(
        TypeError, UOFR, order_selection=1, basis_function=Polynomial(degree=2)
    )
    assert_raises(
        TypeError, UOFR, order_selection="True", basis_function=Polynomial(degree=2)
    )
    assert_raises(
        TypeError, UOFR, order_selection=None, basis_function=Polynomial(degree=2)
    )


def test_n_terms():
    assert_raises(ValueError, UOFR, n_terms=1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, UOFR, n_terms=-1, basis_function=Polynomial(degree=2))


def test_n_info_values():
    assert_raises(
        ValueError, UOFR, n_info_values=1.2, basis_function=Polynomial(degree=2)
    )
    assert_raises(
        ValueError, UOFR, n_info_values=-1, basis_function=Polynomial(degree=2)
    )


def test_info_criteria():
    assert_raises(
        ValueError, UOFR, info_criteria="AIC", basis_function=Polynomial(degree=2)
    )


def test_predict():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        err_tol=None,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    assert_almost_equal(yhat, y_test, decimal=10)


def test_model_prediction():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=X_train, y=y_train)
    assert_raises(Exception, model.predict, X=X_test, y=y_test[:1])


def test_information_criteria_bic():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=5,
        order_selection=True,
        info_criteria="bic",
        n_info_values=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    info_values = np.array([896.976044, 55.273645, -619.342972, -887.017557])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_aicc():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=5,
        order_selection=True,
        info_criteria="aicc",
        n_info_values=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    info_values = np.array([892.074090, 45.475397, -634.036991, -906.600820])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_fpe():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=5,
        order_selection=True,
        info_criteria="fpe",
        n_info_values=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    info_values = np.array([892.069887, 45.462543, -634.060665, -906.640463])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_lilc():
    basis_function = Polynomial(degree=2)
    model = UOFR(
        n_terms=5,
        order_selection=True,
        info_criteria="lilc",
        n_info_values=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    info_values = np.array([893.935255, 49.192165, -628.466752, -899.181576])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_ridge_regression_sets_alpha():
    estimator = RidgeRegression(alpha=0.42)
    model = UOFR(estimator=estimator, basis_function=Polynomial(degree=1))
    assert model.alpha == estimator.alpha


def test_validate_sobolev_order_non_negative():
    assert_raises(
        ValueError, UOFR, sobolev_order=-1, basis_function=Polynomial(degree=1)
    )


def test_validate_test_support_minimum():
    assert_raises(ValueError, UOFR, test_support=1, basis_function=Polynomial(degree=1))


def test_validate_test_support_odd():
    assert_raises(ValueError, UOFR, test_support=4, basis_function=Polynomial(degree=1))


def test_validate_modulating_function_string():
    assert_raises(
        ValueError,
        UOFR,
        modulating_function="invalid",
        basis_function=Polynomial(degree=1),
    )


def test_validate_modulating_function_callable_required():
    assert_raises(
        TypeError, UOFR, modulating_function=5, basis_function=Polynomial(degree=1)
    )


def test_validate_gaussian_sigma_positive():
    assert_raises(
        ValueError,
        UOFR,
        modulating_function="gaussian",
        gaussian_sigma=0,
        basis_function=Polynomial(degree=1),
    )


def test_test_function_grid_gaussian_span_matches_sigma():
    model = UOFR(
        modulating_function="gaussian",
        gaussian_sigma=2.0,
        basis_function=Polynomial(degree=1),
    )
    grid = model._test_function_grid()
    assert_almost_equal(grid[0], -6.0)
    assert_almost_equal(grid[-1], 6.0)


def test_test_function_grid_with_callable_uses_unit_span():
    def custom_phi(t, order):
        return np.ones_like(t)

    model = UOFR(modulating_function=custom_phi, basis_function=Polynomial(degree=1))
    grid = model._test_function_grid()
    assert_almost_equal(grid[0], -1.0)
    assert_almost_equal(grid[-1], 1.0)


def test_evaluate_test_function_with_callable_uses_custom_function():
    def custom_phi(t, order):
        return np.full_like(t, fill_value=float(order))

    model = UOFR(modulating_function=custom_phi, basis_function=Polynomial(degree=1))
    grid = model._test_function_grid()
    values = model._evaluate_test_function(grid, 2)
    assert_array_equal(values, np.full_like(grid, 2.0))


def test_evaluate_test_function_returns_gaussian_branch():
    model = UOFR(
        modulating_function="gaussian",
        gaussian_sigma=1.0,
        basis_function=Polynomial(degree=1),
    )
    grid = model._test_function_grid()
    expected = model.gaussian_test_function(grid, 1)
    result = model._evaluate_test_function(grid, 1)
    assert_array_equal(result, expected)


def test_bspline_kernel_covers_orders_zero_and_three():
    model = UOFR(basis_function=Polynomial(degree=1))
    t = np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    order0 = model._bspline_kernel(t, 0)
    assert_almost_equal(order0[2], 2.0 / 3.0)
    order3 = model._bspline_kernel(t, 3)
    assert_almost_equal(order3[1], -3.0)
    assert_almost_equal(order3[3], 3.0)


def test_bspline_kernel_returns_zero_for_higher_orders():
    model = UOFR(basis_function=Polynomial(degree=1))
    t = np.linspace(-2, 2, 9)
    assert_array_equal(model._bspline_kernel(t, 5), np.zeros_like(t))


def test_gaussian_test_function_higher_order_uses_gradients():
    model = UOFR(
        modulating_function="gaussian",
        gaussian_sigma=1.5,
        basis_function=Polynomial(degree=1),
    )
    t = np.linspace(-1, 1, 7)
    second = model.gaussian_test_function(t, 2)
    assert second.shape == t.shape
    assert not np.allclose(second, 0)


def test_gaussian_test_function_order_zero_returns_gaussian():
    model = UOFR(
        modulating_function="gaussian",
        gaussian_sigma=1.5,
        basis_function=Polynomial(degree=1),
    )
    t = np.linspace(-1, 1, 7)
    gaussian = np.exp(-(t**2) / (2 * model.gaussian_sigma**2))
    gaussian[np.abs(t) >= np.max(np.abs(t))] = 0.0
    assert_array_equal(model.gaussian_test_function(t, 0), gaussian)


def test_augment_uls_terms_returns_original_when_order_zero():
    model = UOFR(sobolev_order=0, basis_function=Polynomial(degree=1))
    y_sample = np.array([[1.0], [2.0], [3.0]])
    psi_sample = np.ones((3, 2))
    y_aug, psi_aug = model.augment_uls_terms(y_sample, psi_sample)
    assert_array_equal(y_aug, y_sample)
    assert_array_equal(psi_aug, psi_sample)


def test_augment_uls_terms_with_default_order_appends_rows():
    model = UOFR(sobolev_order=1, basis_function=Polynomial(degree=1))
    y_sample = np.arange(11.0).reshape(-1, 1)
    psi_sample = np.column_stack((np.arange(11.0), np.arange(11.0)))
    y_aug, psi_aug = model.augment_uls_terms(y_sample, psi_sample)
    expected_rows = y_sample.shape[0] + (y_sample.shape[0] - model.test_support + 1)
    assert y_aug.shape[0] == expected_rows
    assert psi_aug.shape[0] == expected_rows


def test_sobolev_error_reduction_ratio_respects_err_tol():
    model = UOFR(
        sobolev_order=0,
        ylag=1,
        xlag=1,
        elag=1,
        err_tol=0.0,
        basis_function=Polynomial(degree=1),
    )
    y_sample = np.arange(6.0).reshape(-1, 1)
    psi_sample = np.column_stack(
        (
            np.linspace(0.1, 0.5, 5),
            np.linspace(0.6, 1.0, 5),
        )
    )
    _err, piv, psi_ortho, _y_aug = model.sobolev_error_reduction_ratio(
        psi_sample,
        y_sample,
        process_term_number=2,
    )
    assert model.n_terms == 1
    assert len(piv) == 1
    assert psi_ortho.shape[1] == 1


def test_uofr_rejects_array_api_dispatch_with_clear_error():
    xp = pytest.importorskip("array_api_strict")
    model = UOFR(
        ylag=1,
        xlag=1,
        n_terms=2,
        order_selection=False,
        basis_function=Polynomial(degree=1),
        estimator=LeastSquares(),
    )

    with config_context(array_api_dispatch=True):
        with pytest.raises(NotImplementedError, match=r"UOFR.*requires NumPy"):
            model.fit(X=xp.asarray(X_train[:20]), y=xp.asarray(y_train[:20]))
