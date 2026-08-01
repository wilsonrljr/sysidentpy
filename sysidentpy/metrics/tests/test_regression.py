import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_almost_equal, assert_array_equal

from sysidentpy import config_context
from sysidentpy.metrics import (
    forecast_error,
    mean_forecast_error,
    mean_squared_error,
    root_mean_squared_error,
    normalized_root_mean_squared_error,
    root_relative_squared_error,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_squared_log_error,
    median_absolute_error,
    explained_variance_score,
    r2_score,
    symmetric_mean_absolute_percentage_error,
)
from sysidentpy.tests._array_api_asserts import assert_allclose as xp_assert_allclose


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


def test_mean_absolute_scaled_error_matches_analytical_result():
    y_train = np.array([10.0, 12.0, 13.0, 12.0, 15.0])
    y = np.array([16.0, 14.0, 18.0])
    yhat = np.array([15.0, 16.0, 17.0])

    result = mean_absolute_scaled_error(y, yhat, y_train)

    # MAE = 4/3 and the one-step naive in-sample MAE = 7/4.
    assert_allclose(result, 16 / 21)


def test_mean_absolute_scaled_error_supports_seasonal_period():
    y_train = np.array([10.0, 30.0, 12.0, 32.0, 14.0, 34.0])
    y = np.array([36.0, 16.0])
    yhat = np.array([35.0, 19.0])

    result = mean_absolute_scaled_error(
        y, yhat, y_train, seasonal_period=np.int64(2)
    )

    # Both the forecast MAE and the two-step naive in-sample MAE equal 2.
    assert_allclose(result, 1.0)


def test_mean_absolute_scaled_error_is_zero_for_perfect_prediction():
    y_train = np.array([-3.0, -1.0, 2.0, 4.0])
    y = np.array([-2.0, 3.0])

    assert mean_absolute_scaled_error(y, y.copy(), y_train) == 0.0


def test_mean_absolute_scaled_error_is_scale_and_translation_invariant():
    y_train = np.array([-2.0, 1.0, 4.0, 8.0])
    y = np.array([5.0, -1.0, 3.0])
    yhat = np.array([4.0, 1.0, 2.5])
    expected = mean_absolute_scaled_error(y, yhat, y_train)
    scale = -3.5
    offset = 7.0

    result = mean_absolute_scaled_error(
        scale * y + offset,
        scale * yhat + offset,
        scale * y_train + offset,
    )

    assert_allclose(result, expected)


def test_mean_absolute_scaled_error_promotes_unsigned_integer_inputs():
    y_train = np.array([0, 1, 2], dtype=np.uint8)
    y = np.array([1], dtype=np.uint8)
    yhat = np.array([2], dtype=np.uint8)

    # Promotion must happen before subtraction so that 1 - 2 does not wrap.
    assert_allclose(mean_absolute_scaled_error(y, yhat, y_train), 1.0)


def test_mean_absolute_scaled_error_handles_zero_naive_scale():
    y_train = np.ones(4)
    y = np.array([1.0, 1.0])

    assert mean_absolute_scaled_error(y, y.copy(), y_train) == 0.0
    assert np.isinf(mean_absolute_scaled_error(y, np.zeros_like(y), y_train))


def test_mean_absolute_scaled_error_preserves_nan_with_zero_naive_scale():
    y_train = np.ones(4)
    y = np.array([1.0, 1.0])
    yhat = np.array([np.nan, 1.0])

    assert np.isnan(mean_absolute_scaled_error(y, yhat, y_train))


@pytest.mark.parametrize(
    "seasonal_period",
    [True, np.bool_(False), 1.5, "1", None],
)
def test_mean_absolute_scaled_error_rejects_non_integer_period(seasonal_period):
    values = np.arange(4.0)

    with pytest.raises(TypeError, match="seasonal_period must be an integer"):
        mean_absolute_scaled_error(values, values, values, seasonal_period)


@pytest.mark.parametrize("seasonal_period", [0, -1])
def test_mean_absolute_scaled_error_rejects_non_positive_period(seasonal_period):
    values = np.arange(4.0)

    with pytest.raises(ValueError, match="seasonal_period must be a positive"):
        mean_absolute_scaled_error(values, values, values, seasonal_period)


@pytest.mark.parametrize(
    ("y_train", "seasonal_period"),
    [
        (np.array([]), 1),
        (np.array([1.0, 2.0]), 2),
        (np.array(1.0), 1),
    ],
)
def test_mean_absolute_scaled_error_rejects_insufficient_training_data(
    y_train, seasonal_period
):
    y = np.array([1.0])

    with pytest.raises(ValueError, match="more samples than seasonal_period"):
        mean_absolute_scaled_error(y, y, y_train, seasonal_period)


def test_mean_absolute_scaled_error_rejects_empty_forecast_arrays():
    empty = np.array([])

    with pytest.raises(ValueError, match="at least one sample"):
        mean_absolute_scaled_error(empty, empty, np.arange(3.0))


def test_mean_absolute_scaled_error_rejects_mismatched_shapes():
    y = np.array([1.0, 2.0])
    yhat = y.reshape(-1, 1)

    with pytest.raises(ValueError, match="must have the same shape"):
        mean_absolute_scaled_error(y, yhat, np.arange(3.0))


@pytest.mark.parametrize(
    ("y", "yhat", "y_train"),
    [
        (np.ones((2, 2)), np.ones((2, 2)), np.arange(3.0)),
        (np.ones(2), np.ones(2), np.ones((3, 2))),
    ],
)
def test_mean_absolute_scaled_error_rejects_multiple_outputs(y, yhat, y_train):
    with pytest.raises(ValueError, match="single output"):
        mean_absolute_scaled_error(y, yhat, y_train)


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
    [normalized_root_mean_squared_error, root_relative_squared_error],
)
def test_normalized_metrics_handle_constant_targets(metric):
    y = np.ones((4, 1))

    assert metric(y, y.copy()) == 0.0
    assert np.isinf(metric(y, np.zeros_like(y)))


@pytest.mark.parametrize(
    "metric",
    [normalized_root_mean_squared_error, root_relative_squared_error],
)
def test_normalized_metrics_do_not_mask_nan_predictions(metric):
    y = np.ones((4, 1))
    yhat = y.copy()
    yhat[0] = np.nan

    assert np.isnan(metric(y, yhat))


def test_symmetric_mean_absolute_percentage_error_handles_zero_pairs():
    y = np.array([0.0, 1.0])
    yhat = np.array([0.0, 2.0])

    result = symmetric_mean_absolute_percentage_error(y, yhat)

    assert_allclose(result, 100 / 3)


@pytest.mark.parametrize(
    ("y", "yhat"),
    [
        (np.array([-1.0, 1.0]), np.array([0.0, 1.0])),
        (np.array([0.0, 1.0]), np.array([-2.0, 1.0])),
    ],
)
def test_mean_squared_log_error_rejects_invalid_domain(y, yhat):
    with pytest.raises(ValueError, match="less than or equal to -1"):
        mean_squared_log_error(y, yhat)


def test_mean_squared_log_error_accepts_values_above_negative_one():
    y = np.array([-0.5, 1.0])
    yhat = np.array([-0.25, 1.5])

    result = mean_squared_log_error(y, yhat)

    assert np.isfinite(result)


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

    xp_assert_allclose(result, forecast_error(y, yhat))


def test_mean_absolute_scaled_error_accepts_array_api_strict():
    xp = pytest.importorskip("array_api_strict")
    y_train = np.array([10.0, 30.0, 12.0, 32.0, 14.0, 34.0])
    y = np.array([36.0, 16.0])
    yhat = np.array([35.0, 19.0])
    expected = mean_absolute_scaled_error(y, yhat, y_train, seasonal_period=2)

    with config_context(array_api_dispatch=True):
        result = mean_absolute_scaled_error(
            xp.asarray(y),
            xp.asarray(yhat),
            xp.asarray(y_train),
            seasonal_period=2,
        )

    assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_mean_absolute_scaled_error_handles_zero_scale_with_array_api_strict():
    xp = pytest.importorskip("array_api_strict")
    y_train = np.ones(4)
    y = np.ones(2)
    yhat = np.zeros(2)

    with config_context(array_api_dispatch=True):
        result = mean_absolute_scaled_error(
            xp.asarray(y), xp.asarray(yhat), xp.asarray(y_train)
        )

    assert np.isinf(result)


def test_mean_absolute_scaled_error_accepts_integer_array_api_inputs():
    xp = pytest.importorskip("array_api_strict")
    y_train = xp.asarray([0, 1, 3])
    y = xp.asarray([1, 2])
    yhat = xp.asarray([1, 3])

    with config_context(array_api_dispatch=True):
        result = mean_absolute_scaled_error(y, yhat, y_train)

    assert_allclose(result, 1 / 3)


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


@pytest.mark.parametrize(
    "metric",
    [normalized_root_mean_squared_error, root_relative_squared_error],
)
def test_normalized_metrics_handle_constant_array_api_targets(metric):
    xp = pytest.importorskip("array_api_strict")
    y = np.ones((4, 1))

    with config_context(array_api_dispatch=True):
        assert metric(xp.asarray(y), xp.asarray(y)) == 0.0
        assert np.isinf(metric(xp.asarray(y), xp.asarray(np.zeros_like(y))))


def test_smape_handles_zero_pairs_with_array_api_strict():
    xp = pytest.importorskip("array_api_strict")
    y = np.array([0.0, 1.0])
    yhat = np.array([0.0, 2.0])

    with config_context(array_api_dispatch=True):
        result = symmetric_mean_absolute_percentage_error(
            xp.asarray(y), xp.asarray(yhat)
        )

    assert_allclose(result, 100 / 3)


def test_msle_domain_contract_with_array_api_strict():
    xp = pytest.importorskip("array_api_strict")

    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match="less than or equal to -1"):
            mean_squared_log_error(
                xp.asarray(np.array([-1.0, 1.0])),
                xp.asarray(np.array([0.0, 1.0])),
            )

        result = mean_squared_log_error(
            xp.asarray(np.array([-0.5, 1.0])),
            xp.asarray(np.array([-0.25, 1.5])),
        )

    assert np.isfinite(result)
