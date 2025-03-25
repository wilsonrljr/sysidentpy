import numpy as np
from numpy.testing import assert_equal, assert_raises
from sysidentpy.utils.check_arrays import (
    check_positive_int,
    num_features,
    check_dimension,
    check_infinity,
    check_length,
    check_nan,
    check_x_y,
)
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
from sysidentpy.utils.save_load import load_model, save_model
from sysidentpy.utils.display_results import results


def test_check_positive_int():
    assert_raises(ValueError, check_positive_int, -1, "name")


def test_num_features():
    X = np.random.rand(10, 2)
    Y = np.random.rand(10, 1)
    assert_equal(num_features(X), 2)
    assert_equal(num_features(Y), 1)


def test_check_infinity():
    X = np.array([1, 2, 3, 4, 4, 5, 0, 6, np.inf]).reshape((-1, 1))
    y = np.array([1, 2, 3, 4, 2, 5, 5, 6, 9]).reshape((-1, 1))
    X2 = np.array([1, 2, 3, 4, 4, 5, 0, 6, 2]).reshape((-1, 1))
    y2 = np.array([1, 2, 3, 4, 2, np.inf, 5, 6, 9]).reshape((-1, 1))
    assert_raises(ValueError, check_infinity, X, y)
    assert_raises(ValueError, check_infinity, X2, y2)


def test_check_nan():
    X = np.array([1, 2, 3, 4, 4, 5, 0, 6, np.nan]).reshape((-1, 1))
    y = np.array([1, 2, 3, 4, 2, 5, 5, 6, 9]).reshape((-1, 1))
    X2 = np.array([1, 2, 3, 4, 4, 5, 0, 6, 2]).reshape((-1, 1))
    y2 = np.array([1, 2, 3, 4, 2, np.nan, 5, 6, 9]).reshape((-1, 1))
    assert_raises(ValueError, check_nan, X, y)
    assert_raises(ValueError, check_nan, X2, y2)


def test_check_length():
    X = np.ones([10, 2])
    y = np.ones([8, 1])
    assert_raises(ValueError, check_length, X, y)


def test_check_dimension():
    X = np.ones(10)
    y = np.ones(10)
    assert_raises(ValueError, check_dimension, X, y)


def test_check_X_y():
    X = np.ones([10, 2])
    y = np.ones([8, 1])
    assert_raises(ValueError, check_x_y, X, y)


def test_get_siso_data():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )
    assert len(x_train) == 900
    assert len(x_valid) == 100
    assert len(y_train) == 900
    assert len(y_valid) == 100
    assert x_train.shape[1] == 1
    assert y_train.shape[1] == 1


def test_get_miso_data():
    x_train, x_valid, y_train, y_valid = get_miso_data(
        n=1000, colored_noise=False, sigma=0.001, train_percentage=90
    )
    assert len(x_train) == 900
    assert len(x_valid) == 100
    assert len(y_train) == 900
    assert len(y_valid) == 100
    assert x_train.shape[1] == 2
    assert y_train.shape[1] == 1


def test_save_model():
    assert_raises(TypeError, save_model, path=1)
    assert_raises(TypeError, save_model, path=False)
    assert_raises(TypeError, save_model, model=None)


def test_load_model():
    assert_raises(TypeError, load_model, path=1)
    assert_raises(TypeError, load_model, path=False)
    assert_raises(TypeError, load_model, model=None)


def test_results():
    model = np.array(
        [
            [1001, 0],  # y(k-1)
            [2001, 1001],  # x1(k-1)y(k-1)
            [2002, 0],  # x1(k-2)
        ]
    )
    theta = np.array([[0.19999698], [0.90011667], [0.10080975]])
    err = np.array([0.98, 0.01, 0.01])
    table = results(
        final_model=model,
        theta=theta,
        err=err,
        n_terms=3,
        theta_precision=4,
        err_precision=8,
        dtype="sci",
    )
    assert_equal(
        table,
        [
            ["y(k-1)", "2.0000E-01", "9.80000000E-01"],
            ["x1(k-1)y(k-1)", "9.0012E-01", "1.00000000E-02"],
            ["x1(k-2)", "1.0081E-01", "1.00000000E-02"],
        ],
    )
