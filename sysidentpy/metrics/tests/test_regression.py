import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.metrics import (
    forecast_error,
    mean_forecast_error,
    mean_squared_error,
    root_mean_squared_error,
    normalized_root_mean_squared_error,
    root_relative_squared_error,
    mean_absolute_error,
    mean_squared_log_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
    symmetric_mean_absolute_percentage_error,
)


def test_mean_forecast_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = -0.25
    assert_array_equal(metric, mean_forecast_error(y, y_predicted))


def test_forecast_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = [0.5, -0.5, 0, -1]
    assert_array_equal(metric, forecast_error(y, y_predicted))


def test_mean_squared_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.375
    assert_array_equal(metric, mean_squared_error(y, y_predicted))


def test_root_mean_squared_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.612372
    result = root_mean_squared_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_normalized_root_mean_squared_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.081649
    result = normalized_root_mean_squared_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_root_relative_squared_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.226697
    result = root_relative_squared_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_mean_absolute_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.500000
    result = mean_absolute_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_mean_squared_log_error():
    y = np.array([3, 5, 2.5, 7])
    y_predicted = np.array([2.5, 5, 4, 8])
    metric = 0.039730
    result = mean_squared_log_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_median_absolute_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.500000
    result = median_absolute_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_explained_variance_score():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 0.957173
    result = explained_variance_score(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_r2_score():
    y = np.array([3, -0.5, 2, 7]).reshape(-1, 1)
    y_predicted = np.array([2.5, 0.0, 2, 8]).reshape(-1, 1)
    metric = 0.948608
    result = r2_score(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


def test_symmetric_mean_absolute_percentage_error():
    y = np.array([3, -0.5, 2, 7])
    y_predicted = np.array([2.5, 0.0, 2, 8])
    metric = 57.878787
    result = symmetric_mean_absolute_percentage_error(y, y_predicted)
    assert_almost_equal(metric, result, decimal=6)


@pytest.mark.parametrize(
    "metric",
    [
        mean_forecast_error,
        mean_squared_error,
        root_mean_squared_error,
        normalized_root_mean_squared_error,
        root_relative_squared_error,
        mean_absolute_error,
        mean_squared_log_error,
        median_absolute_error,
        explained_variance_score,
        r2_score,
        symmetric_mean_absolute_percentage_error,
    ],
)
def test_scalar_regression_metrics_accept_array_api_strict(metric):
    xp = pytest.importorskip("array_api_strict")
    y = np.array([3.0, 5.0, 2.5, 7.0]).reshape(-1, 1)
    yhat = np.array([2.5, 5.0, 4.0, 8.0]).reshape(-1, 1)

    expected = metric(y, yhat)

    with config_context(array_api_dispatch=True):
        result = metric(xp.asarray(y), xp.asarray(yhat))

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_forecast_error_accepts_array_api_strict():
    xp = pytest.importorskip("array_api_strict")
    y = np.array([3.0, -0.5, 2.0, 7.0])
    yhat = np.array([2.5, 0.0, 2.0, 8.0])

    with config_context(array_api_dispatch=True):
        result = forecast_error(xp.asarray(y), xp.asarray(yhat))

    assert_allclose(_to_numpy(result), forecast_error(y, yhat))


@pytest.mark.parametrize("metric", [explained_variance_score, r2_score])
def test_variance_metrics_do_not_require_boolean_indexing(metric):
    xp = pytest.importorskip("array_api_strict")
    y = np.array([3.0, -0.5, 2.0, 7.0]).reshape(-1, 1)
    yhat = np.array([2.5, 0.0, 2.0, 8.0]).reshape(-1, 1)
    expected = metric(y, yhat)

    with config_context(array_api_dispatch=True):
        with xp.ArrayAPIStrictFlags(boolean_indexing=False):
            result = metric(xp.asarray(y), xp.asarray(yhat))

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("metric", [explained_variance_score, r2_score])
def test_variance_metrics_handle_constant_targets_with_array_api_strict(metric):
    xp = pytest.importorskip("array_api_strict")
    y = np.ones((4, 1))
    perfect = np.ones((4, 1))
    imperfect = np.array([0.0, 1.0, 2.0, 1.0]).reshape(-1, 1)

    with config_context(array_api_dispatch=True):
        assert_allclose(metric(xp.asarray(y), xp.asarray(perfect)), 1.0)
        assert_allclose(metric(xp.asarray(y), xp.asarray(imperfect)), 0.0)
