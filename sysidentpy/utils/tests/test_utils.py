from sysidentpy.utils._check_arrays import (
    check_length,
    check_dimension,
    check_infinity,
    check_nan,
    check_X_y,
)
from sysidentpy.utils.generate_data import get_miso_data, get_siso_data
import numpy as np
from numpy.testing import assert_raises


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
    assert_raises(ValueError, check_X_y, X, y)
    
def test_get_miso_data():
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=1000,
        colored_noise=False,
        sigma=0.001,
        train_percentage=90)
    assert len(x_train) == 900
    assert len(x_valid) == 100
    assert len(y_train) == 900
    assert len(y_valid) == 100
    assert x_train.shape[1] == 1
    assert y_train.shape[1] == 1

def test_get_miso_data():
    x_train, x_valid, y_train, y_valid = get_miso_data(
        n=1000,
        colored_noise=False,
        sigma=0.001,
        train_percentage=90)
    assert len(x_train) == 900
    assert len(x_valid) == 100
    assert len(y_train) == 900
    assert len(y_valid) == 100
    assert x_train.shape[1] == 2
    assert y_train.shape[1] == 1
