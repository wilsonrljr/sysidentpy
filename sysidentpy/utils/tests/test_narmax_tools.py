import pytest
import numpy as np
from sysidentpy.utils.narmax_tools import regressor_code, set_weights, train_test_split
from sysidentpy.basis_function import Polynomial
from sysidentpy.narmax_base import RegressorDictionary


def test_regressor_code():
    """Test regressor_code function with default parameters."""
    encoding = regressor_code(xlag=2, ylag=2, model_type="NARMAX")
    assert isinstance(encoding, np.ndarray), "Output should be a numpy array"
    assert encoding.shape[1] > 0, "Encoding should have at least one column"


def test_regressor_code_invalid_basis_function():
    """Test regressor_code function with an invalid basis function."""

    class FakeBasisFunction:
        ensemble = True
        n = 3
        degree = 2

    encoding = regressor_code(basis_function=FakeBasisFunction())
    assert isinstance(encoding, np.ndarray), "Output should be a numpy array"


def test_set_weights_default():
    """Test set_weights function with default parameters."""
    weights = set_weights()
    assert isinstance(weights, np.ndarray), "Output should be a numpy array"
    assert weights.shape[0] in [2, 3], "Weights should have 2 or 3 rows"


def test_train_test_split():
    """Test train_test_split with valid inputs."""
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    assert X_train.shape[0] == 80, "X_train should have 80 samples"
    assert X_test.shape[0] == 20, "X_test should have 20 samples"
    assert y_train.shape[0] == 80, "y_train should have 80 samples"
    assert y_test.shape[0] == 20, "y_test should have 20 samples"


def test_train_test_split_invalid_test_size():
    """Test train_test_split with an invalid test_size."""
    with pytest.raises(ValueError, match="test_size should be between 0 and 1"):
        train_test_split(np.random.rand(100, 5), np.random.rand(100), test_size=1.5)
