import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
)

from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.information_matrix import (
    shift_column,
    _build_sliding_windows,
    _create_lagged_x,
    _create_lagged_y,
    _normalize_lag_input,
    count_model_regressors,
    initial_lagged_matrix,
    build_input_matrix,
    build_input_output_matrix,
    build_output_matrix,
    get_build_io_method,
    build_lagged_matrix,
)


def test_create_lagged_y():
    """Test lagged matrix creation for output variable."""
    y = np.array([[1], [2], [3], [4], [5]])
    ylag = [1, 2]

    y_lagged = _create_lagged_y(y, ylag)

    expected = np.array([[0, 0], [1, 0], [2, 1], [3, 2], [4, 3]])
    np.testing.assert_array_equal(y_lagged, expected)


def test_initial_lagged_matrix():
    """Test initial lagged matrix construction."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[10], [20], [30], [40]])
    xlag = [[1], [1]]
    ylag = [1]

    result = initial_lagged_matrix(X, y, xlag, ylag)

    expected = np.array([[0, 0, 0], [10, 1, 2], [20, 3, 4], [30, 5, 6]])
    np.testing.assert_array_equal(result, expected)


def test_build_output_matrix():
    """Test output matrix construction."""
    y = np.array([[1], [2], [3], [4]])
    ylag = [1]
    result = build_output_matrix(y, ylag)

    expected = np.array([[1, 0], [1, 1], [1, 2], [1, 3]])
    np.testing.assert_array_equal(result, expected)


def test_build_input_matrix():
    """Test input matrix construction."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    xlag = [[1], [1]]

    result = build_input_matrix(X, xlag)

    expected = np.array([[1, 0, 0], [1, 1, 2], [1, 3, 4], [1, 5, 6]])
    np.testing.assert_array_equal(result, expected)


def test_build_input_output_matrix():
    """Test input-output information matrix construction."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([[10], [20], [30], [40]])
    xlag = [[1], [1]]
    ylag = [1]

    result = build_input_output_matrix(X, y, xlag, ylag)

    expected = np.array([[1, 0, 0, 0], [1, 10, 1, 2], [1, 20, 3, 4], [1, 30, 5, 6]])
    np.testing.assert_array_equal(result, expected)


def test_get_build_io_method():
    """Test method retrieval based on model type."""
    assert get_build_io_method("NARMAX") is build_input_output_matrix
    assert get_build_io_method("NFIR") is build_input_matrix
    assert get_build_io_method("NAR") is build_output_matrix
    assert get_build_io_method("UNKNOWN") is None


def test_prepare_data_narmax():
    """Test data preparation for NARMAX models."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[10], [20], [30]])
    xlag = [[1], [1]]
    ylag = [1]
    result = build_lagged_matrix(X, y, xlag, ylag, "NARMAX")
    expected = np.array([[1, 0, 0, 0], [1, 10, 1, 2], [1, 20, 3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_prepare_data_nfir():
    """Test data preparation for NFIR models."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    xlag = [[1], [1]]
    result = build_lagged_matrix(X, None, xlag, None, "NFIR")
    expected = np.array([[1, 0, 0], [1, 1, 2], [1, 3, 4]])
    np.testing.assert_array_equal(result, expected)


def test_prepare_data_nar():
    """Test data preparation for NAR models."""
    y = np.array([[10], [20], [30]])
    ylag = [1]
    result = build_lagged_matrix(None, y, None, ylag, "NAR")
    expected = np.array([[1, 0], [1, 10], [1, 20]])
    np.testing.assert_array_equal(result, expected)


def test_shift_column():
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

    output = np.array(
        [
            [0.0],
            [0.0],
            [0.42544384],
            [0.39365905],
            [0.22209413],
            [0.69760074],
            [0.88183369],
            [0.24818225],
            [0.78482346],
            [0.26967285],
        ]
    )
    r = shift_column(a, 2)
    assert_almost_equal(output, r)


def test_shift_column_rejects_negative_lag():
    data = np.arange(5.0).reshape(-1, 1)
    with pytest.raises(ValueError, match="lag must be non-negative"):
        shift_column(data, -1)


def test_create_lagged_x():
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    r = _create_lagged_x(x=X, n_inputs=1, xlag=[1, 2])
    assert_equal(
        r,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]]
        ),
    )


def test_create_lagged_x_accepts_one_dimensional_input():
    X = np.array([1.0, 2.0, 3.0, 4.0])
    result = _create_lagged_x(x=X, n_inputs=1, xlag=[1])
    assert_equal(result[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))


def test_create_lagged_x_miso():
    X = np.array(range(1, 13)).reshape(-1, 2)
    r = _create_lagged_x(x=X, n_inputs=2, xlag=[[1, 2], [1, 2]])
    assert_equal(
        r,
        np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 2.0, 0.0],
                [3.0, 1.0, 4.0, 2.0],
                [5.0, 3.0, 6.0, 4.0],
                [7.0, 5.0, 8.0, 6.0],
                [9.0, 7.0, 10.0, 8.0],
            ]
        ),
    )


def test_normalize_lag_input_handles_iterables():
    lag_array = _normalize_lag_input([3, 1, 4])
    assert_equal(lag_array, np.array([3, 1, 4]))


def test_normalize_lag_input_scalar():
    lag_array = _normalize_lag_input(2)
    assert_equal(lag_array, np.array([2]))


def test_create_lagged_y_accepts_one_dimensional_input():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    result = _create_lagged_y(y, ylag=[1])
    assert_equal(result[:, 0], np.array([0.0, 1.0, 2.0, 3.0]))


def test_build_sliding_windows_requires_positive_lag():
    data = np.ones((3, 1))
    with pytest.raises(ValueError, match="max_lag must be >= 1"):
        _build_sliding_windows(data, 0)


def test_build_sliding_windows_shapes_and_values():
    data = np.arange(6.0).reshape(-1, 1)
    windows = _build_sliding_windows(data, 2)
    assert windows.shape == (6, 3, 1)
    # Third row should reference the last three padded values (0,1,2)
    assert_almost_equal(windows[2, :, 0], np.array([0.0, 1.0, 2.0]))


def test_shift_column_zero_lag_returns_copy():
    column = np.arange(5.0).reshape(-1, 1)
    shifted = shift_column(column, 0)
    assert_almost_equal(column, shifted)
    assert shifted is not column


def test_count_model_regressors_neural_adjustment():
    x = np.arange(12.0).reshape(-1, 1)
    y = np.arange(12.0).reshape(-1, 1)
    poly_features = count_model_regressors(
        x=x,
        y=y,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        basis_function=Polynomial(degree=2),
        is_neural_narx=True,
    )
    total_features = count_model_regressors(
        x=x,
        y=y,
        xlag=2,
        ylag=2,
        model_type="NARMAX",
        basis_function=Polynomial(degree=2),
    )
    assert poly_features == total_features - 1


def test_create_lagged_x_handles_lag_larger_than_series():
    X = np.arange(5.0).reshape(-1, 1)
    lagged = _create_lagged_x(x=X, n_inputs=1, xlag=[4])
    # First four entries must be zeros due to insufficient history
    assert_equal(lagged[:4, 0], np.zeros(4))
