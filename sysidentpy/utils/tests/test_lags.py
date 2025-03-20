import pytest
import numpy as np
from sysidentpy.utils.lags import (
    get_max_ylag,
    get_max_xlag,
    get_lag_from_regressor_code,
    get_max_lag_from_model_code,
    _process_xlag,
    _process_ylag,
)
from sysidentpy.utils.simulation import (
    list_input_regressor_code,
    list_output_regressor_code,
)


@pytest.mark.parametrize(
    ("ylag", "expected"),
    [
        (1, 1),
        (5, 5),
        (10, 10),
    ],
)
def test_get_max_ylag(ylag, expected):
    assert get_max_ylag(ylag) == expected


@pytest.mark.parametrize(
    ("xlag", "expected"),
    [
        (1, 1),  # Single integer
        ([1, 2, 3], 3),  # Flat list
        ([[1, 2], [3, 4]], 4),  # Nested list
    ],
)
def test_get_max_xlag(xlag, expected):
    assert get_max_xlag(xlag) == expected


def test_get_max_xlag_invalid():
    with pytest.raises(ValueError, match="Unsupported data type for xlag"):
        get_max_xlag("invalid")  # Should raise an error


def test_get_lag_from_regressor_code():
    regressors = np.array([2001, 2002])
    expected = 2
    assert get_lag_from_regressor_code(regressors) == expected


def test_get_max_lag_from_model_code():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    assert get_max_lag_from_model_code(model) == 2  # Max sum of digits


def test_process_xlag():
    X = np.array([[1, 2], [3, 4]])
    assert _process_xlag(X, [[1, 2], [1, 2]]) == (2, [[1, 2], [1, 2]])
    assert _process_xlag(X, [[1, 2], [1, 2], [1, 2]]) == (2, [[1, 2], [1, 2], [1, 2]])


def test_process_xlag_invalid():
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(
        ValueError, match="If n_inputs > 1, xlag must be a nested list. Got 3"
    ):
        _process_xlag(X, 3)  # Should raise an error for multiple inputs


# ---- TESTS FOR _process_ylag ----
@pytest.mark.parametrize(
    ("ylag", "expected"),
    [
        (2, [1, 2]),
        ([3, 5], [3, 5]),
    ],
)
def test_process_ylag(ylag, expected):
    assert _process_ylag(ylag) == expected
