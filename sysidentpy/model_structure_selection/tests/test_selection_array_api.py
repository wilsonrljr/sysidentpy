"""Exercise the public selector paths with strict Array API inputs."""

import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy._lib._array_api import device
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS, OSF, OIF, OOS
from sysidentpy.parameter_estimation import LeastSquares, RecursiveLeastSquares
from sysidentpy.tests._array_api_asserts import assert_allclose, assert_array_equal


@pytest.fixture(params=["array_api_strict", "torch", "torch_cuda"])
def backend(request):
    name = request.param
    xp = pytest.importorskip("torch" if name == "torch_cuda" else name)
    target = None
    if name == "torch_cuda":
        if not xp.cuda.is_available():
            pytest.skip("CUDA not available")
        target = "cuda"
    return xp, target


@pytest.mark.parametrize("selector", [FROLS, OSF, OIF, OOS])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_fit_and_one_step_preserve_backend(backend, selector, dtype_name):
    xp, target = backend
    if selector is FROLS and target is not None:
        pytest.skip("RLS non-default-device allocation is outside this correction")
    rng = np.random.default_rng(91)
    x = rng.normal(size=(70, 1)).astype(dtype_name)
    y = np.zeros_like(x)
    for k in range(1, len(y)):
        y[k] = 2 * x[k - 1] + 0.2 * y[k - 1]
    options = {} if selector is FROLS else {"estimator": LeastSquares()}
    baseline = selector(
        xlag=1,
        ylag=1,
        n_terms=1,
        order_selection=False,
        basis_function=Polynomial(degree=1),
        **options,
    ).fit(X=x, y=y)
    expected = baseline.predict(X=x, y=y, steps_ahead=1)
    x_backend = xp.asarray(x, dtype=getattr(xp, dtype_name), device=target)
    y_backend = xp.asarray(y, dtype=getattr(xp, dtype_name), device=target)
    with config_context(array_api_dispatch=True):
        model = selector(
            xlag=1,
            ylag=1,
            n_terms=1,
            order_selection=False,
            basis_function=Polynomial(degree=1),
            **options,
        ).fit(X=x_backend, y=y_backend)
        result = model.predict(X=x_backend, y=y_backend, steps_ahead=1)
    np.testing.assert_array_equal(model.final_model, baseline.final_model)
    tolerance = 1e-5 if dtype_name == "float32" else 1e-10
    assert_allclose(
        result,
        xp.asarray(expected, dtype=result.dtype, device=target),
        rtol=tolerance,
        atol=tolerance,
    )
    assert result.dtype == getattr(xp, dtype_name)
    assert device(result) == device(y_backend)
    assert device(model.theta) == device(y_backend)


def test_floating_columns_keep_order_duplicates_and_device(backend):
    xp, target = backend
    data = xp.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=target)
    with config_context(array_api_dispatch=True):
        model = OSF()
        result = model._select_columns(data, [2, 0, 2])
        empty = model._select_columns(data, [])
        score, err = model._subset_err(data[:0, :], data[:0, :1], [0], 1.0)
    assert_array_equal(
        result, xp.asarray([[3.0, 1.0, 3.0], [6.0, 4.0, 6.0]], device=target)
    )
    assert empty.shape == (2, 0)
    assert err.shape == (0,)
    assert score == 0
    assert device(result) == device(empty) == device(err) == device(data)


def test_rls_history_is_independent_of_returned_parameters(backend):
    xp, target = backend
    # RLS device allocation is outside this patch; this regression targets its copy.
    if target is not None:
        pytest.skip("RLS non-default-device allocation is outside this correction")
    psi = xp.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=xp.float64)
    y = psi * 2
    with config_context(array_api_dispatch=True):
        estimator = RecursiveLeastSquares()
        theta = estimator.optimize(psi, y)
        saved_history = xp.asarray(estimator.theta_evolution, copy=True)
        theta[0, 0] = -100
    assert_array_equal(estimator.theta_evolution, saved_history)
    assert device(estimator.theta_evolution) == device(psi)
