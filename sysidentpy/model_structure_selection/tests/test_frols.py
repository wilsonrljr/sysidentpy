# pyright: reportPrivateUsage=false
# pylint: disable=protected-access,redefined-outer-name
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_allclose, assert_array_equal
from numpy.testing import assert_raises
from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation.estimators import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.tests.test_narmax_base import create_test_data
from sysidentpy.utils.generate_data import get_siso_data

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


class _FROLSProbe(FROLS):
    def narmax_predict_reference(self, x_data, y_data, forecast_horizon):
        return self._narmax_predict_reference(x_data, y_data, forecast_horizon)

    def polynomial_narmax_predict_fast(self, x_data, y_data, forecast_horizon):
        return self._polynomial_narmax_predict_fast(
            x_data,
            y_data,
            forecast_horizon,
        )


def _build_order_selection_model(info_criteria="aic"):
    return FROLS(
        ylag=[1, 2],
        xlag=2,
        n_info_values=5,
        info_criteria=info_criteria,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )


def _fit_predict_with_dispatch(
    model, x_train_data, y_train_data, x_test_data, y_test_data
):
    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_data, y=y_train_data)
        return model.predict(X=x_test_data, y=y_test_data)


def _fit_numpy_order_selection_baseline(info_criteria="aic"):
    model = _build_order_selection_model(info_criteria=info_criteria)
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)
    return model, yhat


def _assert_order_selection_matches_numpy(
    model, yhat, baseline_model, baseline_yhat, expected_y
):
    yhat_backend = _to_numpy(yhat)
    rrse_backend = root_relative_squared_error(expected_y, yhat_backend)
    rrse_numpy = root_relative_squared_error(expected_y, baseline_yhat)

    assert model.n_terms == baseline_model.n_terms
    assert_array_equal(
        np.asarray(_to_numpy(model.pivv), dtype=np.intp),
        np.asarray(baseline_model.pivv, dtype=np.intp),
    )
    assert_array_equal(model.final_model, baseline_model.final_model)
    assert_array_equal(yhat_backend[: model.max_lag], expected_y[: model.max_lag])
    assert_allclose(yhat_backend, baseline_yhat, rtol=1e-10, atol=1e-12)
    assert_allclose(rrse_backend, rrse_numpy, rtol=1e-10, atol=1e-12)
    assert_allclose(
        np.max(np.abs(yhat_backend - baseline_yhat)),
        0.0,
        rtol=0.0,
        atol=1e-10,
    )


def test_error_reduction_ratio():
    # piv = np.array([4, 2, 7, 11, 5])
    model_code = np.array(
        [[2002, 0], [1002, 0], [2001, 1001], [2002, 1002], [1001, 1001]]
    )
    basis_function = Polynomial(degree=2)
    x_data, y_data, _ = create_test_data()
    model = FROLS(
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
    model.fit(X=x_data, y=y_data)
    assert_array_equal(model.final_model, model_code)


def test_fit_with_information_criteria():
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=15,
        order_selection=True,
        basis_function=basis_function,
    )
    model.fit(X=x, y=y)
    assert "info_values" in dir(model)


def test_fit_without_information_criteria():
    basis_function = Polynomial(degree=2)
    model = FROLS(n_terms=15, basis_function=basis_function, order_selection=False)
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
    model = FROLS(basis_function=Polynomial(degree=2))
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
    assert_raises(ValueError, FROLS, ylag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, FROLS, ylag=1.3, basis_function=Polynomial(degree=2))


def test_validate_xlag():
    assert_raises(ValueError, FROLS, xlag=-1, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, FROLS, xlag=1.3, basis_function=Polynomial(degree=2))


def test_model_order_selection():
    assert_raises(
        TypeError, FROLS, order_selection=1, basis_function=Polynomial(degree=2)
    )
    assert_raises(
        TypeError, FROLS, order_selection="True", basis_function=Polynomial(degree=2)
    )
    assert_raises(
        TypeError, FROLS, order_selection=None, basis_function=Polynomial(degree=2)
    )


def test_n_terms():
    assert_raises(ValueError, FROLS, n_terms=1.2, basis_function=Polynomial(degree=2))
    assert_raises(ValueError, FROLS, n_terms=-1, basis_function=Polynomial(degree=2))


def test_n_info_values():
    assert_raises(
        ValueError, FROLS, n_info_values=1.2, basis_function=Polynomial(degree=2)
    )
    assert_raises(
        ValueError, FROLS, n_info_values=-1, basis_function=Polynomial(degree=2)
    )


def test_info_criteria():
    assert_raises(
        ValueError, FROLS, info_criteria="AIC", basis_function=Polynomial(degree=2)
    )


def test_predict():
    basis_function = Polynomial(degree=2)
    model = FROLS(
        n_terms=5,
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
    model = FROLS(
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
    model = FROLS(
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
    info_values = np.array([-1764.885, -2320.101, -2976.391, -4461.908, -72845.768])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_aicc():
    basis_function = Polynomial(degree=2)
    model = FROLS(
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
    info_values = np.array([-1769.787, -2329.901, -2991.084, -4481.490])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_fpe():
    basis_function = Polynomial(degree=2)
    model = FROLS(
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
    info_values = np.array(
        [-1769.7907932, -2329.9129013, -2991.1078281, -4481.5306067, -72870.296884]
    )
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


def test_information_criteria_lilc():
    basis_function = Polynomial(degree=2)
    model = FROLS(
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
    info_values = np.array([-1767.926, -2326.183, -2985.514, -4474.072, -72860.973])
    assert_almost_equal(model.info_values[:4], info_values[:4], decimal=3)


@pytest.mark.parametrize("info_criteria", ["aic", "apress"])
def test_fit_with_order_selection_preserves_array_api_dispatch(info_criteria):
    xp = pytest.importorskip("array_api_strict")
    baseline_model, baseline_yhat = _fit_numpy_order_selection_baseline(
        info_criteria=info_criteria
    )

    model = _build_order_selection_model(info_criteria=info_criteria)
    x_train_xp = xp.asarray(X_train)
    y_train_xp = xp.asarray(y_train)
    x_test_xp = xp.asarray(X_test)
    y_test_xp = xp.asarray(y_test)

    yhat = _fit_predict_with_dispatch(
        model, x_train_xp, y_train_xp, x_test_xp, y_test_xp
    )

    assert yhat.__array_namespace__().__name__ == xp.__name__
    assert model.n_terms is not None
    _assert_order_selection_matches_numpy(
        model, yhat, baseline_model, baseline_yhat, y_test
    )


@pytest.mark.parametrize("info_criteria", ["aic", "apress"])
def test_fit_with_order_selection_accepts_torch_tensors_under_array_api_dispatch(
    info_criteria,
):
    torch = pytest.importorskip("torch")
    baseline_model, baseline_yhat = _fit_numpy_order_selection_baseline(
        info_criteria=info_criteria
    )

    model = _build_order_selection_model(info_criteria=info_criteria)
    x_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    x_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    yhat = _fit_predict_with_dispatch(model, x_train_t, y_train_t, x_test_t, y_test_t)

    assert model.n_terms is not None
    assert isinstance(model.info_values, torch.Tensor)
    assert model.info_values.device.type == "cpu"
    assert isinstance(yhat, torch.Tensor)
    _assert_order_selection_matches_numpy(
        model, yhat, baseline_model, baseline_yhat, y_test
    )


def test_fit_predict_accepts_torch_cuda_tensors_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = FROLS(
        ylag=[1, 2],
        xlag=2,
        n_terms=5,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    x_train_t = torch.tensor(X_train, dtype=torch.float64, device="cuda")
    y_train_t = torch.tensor(y_train, dtype=torch.float64, device="cuda")
    x_test_t = torch.tensor(X_test, dtype=torch.float64, device="cuda")
    y_test_t = torch.tensor(y_test, dtype=torch.float64, device="cuda")

    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_t, y=y_train_t)
        yhat = model.predict(X=x_test_t, y=y_test_t)

    assert isinstance(yhat, torch.Tensor)
    assert yhat.device.type == "cuda"
    assert yhat.shape == y_test_t.shape
    assert_almost_equal(_to_numpy(yhat[: model.max_lag]), y_test[: model.max_lag])


def test_polynomial_narmax_fast_path_matches_reference_for_frols_model():
    model = _FROLSProbe(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)

    reference = model.narmax_predict_reference(
        X_test,
        y_test,
        forecast_horizon=X_test.shape[0],
    )
    fast = model.polynomial_narmax_predict_fast(
        X_test,
        y_test,
        forecast_horizon=X_test.shape[0],
    )

    assert_allclose(fast, reference, rtol=1e-10, atol=1e-12)


def test_predict_repeated_calls_are_stable_under_array_api_dispatch_for_frols():
    torch = pytest.importorskip("torch")
    model = FROLS(
        n_terms=5,
        ylag=[1, 2],
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    x_train_t = torch.tensor(X_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    x_test_t = torch.tensor(X_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    with config_context(array_api_dispatch=True):
        model.fit(X=x_train_t, y=y_train_t)
        yhat_first = model.predict(X=x_test_t, y=y_test_t)
        final_model_first = model.final_model.copy()
        yhat_second = model.predict(X=x_test_t, y=y_test_t)

    assert_allclose(
        _to_numpy(yhat_first),
        _to_numpy(yhat_second),
        rtol=1e-10,
        atol=1e-12,
    )
    assert_array_equal(model.final_model, final_model_first)


def test_fit_with_order_selection_accepts_torch_cuda_tensors_when_available():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    baseline_model, baseline_yhat = _fit_numpy_order_selection_baseline()

    model = _build_order_selection_model()
    x_train_t = torch.tensor(X_train, dtype=torch.float64, device="cuda")
    y_train_t = torch.tensor(y_train, dtype=torch.float64, device="cuda")
    x_test_t = torch.tensor(X_test, dtype=torch.float64, device="cuda")
    y_test_t = torch.tensor(y_test, dtype=torch.float64, device="cuda")

    yhat = _fit_predict_with_dispatch(model, x_train_t, y_train_t, x_test_t, y_test_t)

    assert model.n_terms is not None
    assert isinstance(model.info_values, torch.Tensor)
    assert model.info_values.device.type == "cuda"
    assert isinstance(yhat, torch.Tensor)
    assert yhat.device.type == "cuda"
    assert yhat.shape == y_test_t.shape
    _assert_order_selection_matches_numpy(
        model, yhat, baseline_model, baseline_yhat, y_test
    )


def test_notebook_order_selection_matches_numpy_for_torch_tensors():
    torch = pytest.importorskip("torch")
    x_train_np, x_valid_np, y_train_np, y_valid_np = get_siso_data(
        n=5_000, colored_noise=False, sigma=0.001, train_percentage=80
    )

    baseline_model = FROLS(
        ylag=2,
        xlag=2,
        order_selection=True,
        n_info_values=10,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    baseline_model.fit(X=x_train_np, y=y_train_np)
    baseline_yhat = baseline_model.predict(X=x_valid_np, y=y_valid_np)

    model = FROLS(
        ylag=2,
        xlag=2,
        order_selection=True,
        n_info_values=10,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    x_train_t = torch.tensor(x_train_np, dtype=torch.float64)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float64)
    x_valid_t = torch.tensor(x_valid_np, dtype=torch.float64)
    y_valid_t = torch.tensor(y_valid_np, dtype=torch.float64)

    yhat = _fit_predict_with_dispatch(model, x_train_t, y_train_t, x_valid_t, y_valid_t)

    _assert_order_selection_matches_numpy(
        model, yhat, baseline_model, baseline_yhat, y_valid_np
    )
