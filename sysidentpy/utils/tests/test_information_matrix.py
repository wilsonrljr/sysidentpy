import numpy as np
from unittest.mock import patch
from numpy.testing import (
    assert_almost_equal,
    assert_equal,
)

from sysidentpy.utils.information_matrix import (
    shift_column,
    _create_lagged_x,
    _create_lagged_y,
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
    assert get_build_io_method("NARMAX") == build_input_output_matrix
    assert get_build_io_method("NFIR") == build_input_matrix
    assert get_build_io_method("NAR") == build_output_matrix
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


def test_create_lagged_x():
    X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    r = _create_lagged_x(x=X, n_inputs=1, xlag=[1, 2])
    assert_equal(
        r,
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]]
        ),
    )


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
