"""Behavioral regressions for the term that reaches the ERR threshold."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sysidentpy import config_context
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS, UOFR
from sysidentpy.parameter_estimation import LeastSquares


@pytest.mark.parametrize("selector", [FROLS, UOFR])
@pytest.mark.parametrize("order_selection", [False, True])
@pytest.mark.parametrize("n_terms", [1, 2])
def test_threshold_keeps_winning_regressor(selector, order_selection, n_terms):
    rng = np.random.default_rng(901)
    x = rng.normal(size=(300, 2))
    coefficients = np.array([[3.0], [0.5 if n_terms == 2 else 0.0]])
    y = np.zeros((300, 1))
    y[1:] = x[:-1] @ coefficients
    options = {"sobolev_order": 0} if selector is UOFR else {}
    model = selector(
        xlag=[[1], [1]],
        ylag=1,
        n_terms=n_terms,
        n_info_values=n_terms,
        order_selection=order_selection,
        model_type="NFIR",
        err_tol=0.999,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
        **options,
    ).fit(X=x, y=y)

    assert_array_equal(model.final_model, np.array([[2001], [3001]])[:n_terms])
    assert model.n_terms == n_terms
    assert_allclose(model.theta, coefficients[:n_terms], rtol=1e-12, atol=1e-12)
    assert_allclose(model.predict(X=x, y=y), y, rtol=1e-12, atol=1e-12)
    first_projection = x[:-1, :1] @ np.linalg.lstsq(x[:-1, :1], y[1:], rcond=None)[0]
    first_err = np.sum(first_projection**2) / np.sum(y[1:] ** 2)
    expected_err = [first_err] if n_terms == 1 else [first_err, 1 - first_err]
    assert_allclose(model.err[:n_terms], expected_err, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("selector", [FROLS, UOFR])
def test_unreached_threshold_preserves_selection(selector):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(80, 1))
    y = rng.normal(size=(80, 1))
    options = {"sobolev_order": 1} if selector is UOFR else {}
    models = [
        selector(
            xlag=1,
            ylag=1,
            n_terms=2,
            order_selection=False,
            err_tol=threshold,
            estimator=LeastSquares(),
            basis_function=Polynomial(degree=1),
            **options,
        ).fit(X=x, y=y)
        for threshold in (None, 1.0)
    ]
    assert_array_equal(models[0].final_model, models[1].final_model)
    assert_array_equal(models[0].pivv, models[1].pivv)
    assert_array_equal(models[0].theta, models[1].theta)
    assert_array_equal(models[0].err, models[1].err)


def test_sobolev_threshold_keeps_winner_with_augmented_target():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(100, 1))
    y = np.zeros_like(x)
    y[1:] = 2 * x[:-1]
    model = UOFR(
        xlag=1,
        ylag=1,
        n_terms=1,
        order_selection=False,
        err_tol=0.9,
        sobolev_order=1,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=1),
    ).fit(X=x, y=y)
    assert_array_equal(model.final_model, [[2001]])
    assert_allclose(model.theta, [[2.0]], rtol=1e-12, atol=1e-12)
    assert_allclose(model.err[0], 1, atol=1e-12)
    assert_allclose(model.predict(X=x, y=y), y, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend_name", ["array_api_strict", "torch"])
@pytest.mark.parametrize("n_terms", [1, 2])
def test_threshold_preserves_backend_pivots(backend_name, n_terms):
    from sysidentpy.tests._array_api_asserts import assert_allclose as backend_allclose

    xp = pytest.importorskip(backend_name)
    rng = np.random.default_rng(901)
    x = rng.normal(size=(100, 2))
    coefficients = np.array([[3.0], [0.5 if n_terms == 2 else 0.0]])
    y = np.zeros((100, 1))
    y[1:] = x[:-1] @ coefficients
    x = xp.asarray(x, dtype=xp.float64)
    y = xp.asarray(y, dtype=xp.float64)
    with config_context(array_api_dispatch=True):
        model = FROLS(
            xlag=[[1], [1]],
            ylag=1,
            n_terms=n_terms,
            order_selection=False,
            model_type="NFIR",
            err_tol=0.999,
            estimator=LeastSquares(),
            basis_function=Polynomial(degree=1),
        ).fit(X=x, y=y)
        prediction = model.predict(X=x, y=y, steps_ahead=1)
    assert_array_equal(model.final_model, np.array([[2001], [3001]])[:n_terms])
    backend_allclose(prediction, y, rtol=1e-12, atol=1e-12)
    backend_allclose(
        model.theta, xp.asarray(coefficients[:n_terms]), rtol=1e-12, atol=1e-12
    )
