import pytest
import numpy as np
from sysidentpy.utils.narmax_tools import regressor_code, set_weights, train_test_split


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


def test_regressor_code_without_ensemble():
    """Ensure non-polynomial basis without ensemble still returns encoding."""

    class FakeBasisFunction:
        ensemble = False
        n = 2
        degree = 1

    encoding = regressor_code(basis_function=FakeBasisFunction())
    assert encoding.size > 0


def test_set_weights_default():
    """Test set_weights function with default parameters."""
    weights = set_weights()
    assert isinstance(weights, np.ndarray), "Output should be a numpy array"
    assert weights.shape[0] in [2, 3], "Weights should have 2 or 3 rows"


def test_set_weights_without_static_function():
    """Weights should collapse to two objectives when static data absent."""
    weights = set_weights(static_function=False)
    assert weights.shape[0] == 2
    np.testing.assert_allclose(weights.sum(axis=0), np.ones(weights.shape[1]))


def test_train_test_split():
    """Test train_test_split with valid inputs."""
    x = np.random.rand(100, 5)
    y = np.random.rand(100)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    assert x_train.shape[0] == 80, "x_train should have 80 samples"
    assert x_test.shape[0] == 20, "x_test should have 20 samples"
    assert y_train.shape[0] == 80, "y_train should have 80 samples"
    assert y_test.shape[0] == 20, "y_test should have 20 samples"


def test_train_test_split_without_inputs():
    """When X is None the splitter must return None for both feature sets."""
    X_train, X_test, y_train, y_test = train_test_split(None, np.arange(10), 0.3)
    assert X_train is None and X_test is None
    assert y_train.shape[0] == 7 and y_test.shape[0] == 3


def test_train_test_split_invalid_test_size():
    """Test train_test_split with an invalid test_size."""
    with pytest.raises(ValueError, match="test_size should be between 0 and 1"):
        train_test_split(np.random.rand(100, 5), np.random.rand(100), test_size=1.5)
