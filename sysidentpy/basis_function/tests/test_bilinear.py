import pytest
import numpy as np
from sysidentpy.basis_function import Bilinear  # Replace with actual import path


def test_bilinear_init():
    """Test that Bilinear class initializes correctly with the specified degree."""
    b = Bilinear(degree=3)
    assert b.degree == 3

    b_default = Bilinear()
    assert b_default.degree == 2  # Default degree is 2


def test_bilinear_fit():
    """Test that fit correctly generates the information matrix."""
    b = Bilinear(degree=2)
    data = np.random.rand(10, 3)  # 10 samples, 3 features
    max_lag = 2

    transformed = b.fit(data, max_lag=max_lag)

    assert transformed.shape[0] == data.shape[0] - max_lag
    assert transformed.shape[1] > 0  # Ensure non-empty feature matrix


def test_bilinear_transform():
    """Test that transform behaves identically to fit."""
    b = Bilinear(degree=2)
    data = np.random.rand(10, 3)

    transformed_fit = b.fit(data)
    transformed_transform = b.transform(data)

    np.testing.assert_array_equal(transformed_fit, transformed_transform)


def test_bilinear_fit_predefined_regressors():
    """Test fit with predefined regressors filtering output features correctly."""
    b = Bilinear(degree=2)
    data = np.random.rand(10, 3)
    predefined_regressors = [0, 2]  # Selecting only certain regressors

    transformed = b.fit(data, predefined_regressors=predefined_regressors)

    assert transformed.shape[1] == len(
        predefined_regressors
    )  # Ensure correct feature selection


def test_bilinear_degree_warning():
    """Test that a warning is raised when degree=1 is chosen."""
    with pytest.warns(UserWarning, match="linear polynomial model"):
        b = Bilinear(degree=1)
        data = np.random.rand(10, 3)
        b.fit(data)
