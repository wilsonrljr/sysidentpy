import pytest
import numpy as np
from numpy.testing import (
    assert_equal,
    assert_raises,
)

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
from sysidentpy.utils import get_siso_data, get_miso_data


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


def test_process_xlag_data():
    a = np.array(
        [
            0.42544384,
            0.39365905,
            0.22209413,
            0.69760074,
            0.88183369,
            0.24818225,
            0.78482346,
            0.26967285,
            0.53987842,
            0.17367185,
        ]
    ).reshape(-1, 1)

    n_inputs, xlag = _process_xlag(a.reshape(-1, 1), xlag=2)
    output1 = 1
    output2 = list(range(1, 3))
    assert_equal(output1, n_inputs)
    assert_equal(output2, xlag)


def test_process_ylag_data():
    ylag = _process_ylag(ylag=2)
    output1 = list(range(1, 3))
    assert_equal(output1, ylag)


def test_process_lag():
    x_train, _, _, _ = get_miso_data(
        n=10, colored_noise=False, sigma=0.001, train_percentage=90
    )
    assert_raises(ValueError, _process_xlag, X=x_train, xlag=2)


def test_process_lag_n1():
    x_train, _, _, _ = get_siso_data(
        n=10, colored_noise=False, sigma=0.001, train_percentage=90
    )
    n_inputs, xlag = _process_xlag(X=x_train, xlag=2)
    assert n_inputs == 1
    assert list(xlag) == [1, 2]


def test_get_lag_from_regressor_code_list():
    list_regressor1 = np.array([2001, 2002])
    list_regressor2 = np.array([1004, 1002])
    max_lag1 = get_lag_from_regressor_code(list_regressor1)
    max_lag2 = get_lag_from_regressor_code(list_regressor2)

    assert max_lag1 == 2
    assert max_lag2 == 4


def test_model_information_get_lag():
    laglist = np.array([2001, 2002, 3001, 3002, 1001, 1002])
    output = 2
    r1 = get_lag_from_regressor_code(laglist)
    assert r1 == output


def test_model_information_empty_list():
    laglist = np.array([])
    output = 1
    r1 = get_lag_from_regressor_code(laglist)
    assert r1 == output


def test_get_max_lag_from_model_code_gd():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    assert get_max_lag_from_model_code(model) == 2
