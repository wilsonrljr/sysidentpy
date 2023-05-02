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

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal


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
    metric = 0.205737
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
