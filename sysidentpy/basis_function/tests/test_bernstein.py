import pytest
import numpy as np
from sysidentpy.basis_function import Bernstein


def test_bernstein_init():
    """Test that Bernstein class initializes correctly."""
    b1 = Bernstein()
    assert b1.degree == 1
    assert b1.include_bias is True
    assert b1.ensemble is False

    b2 = Bernstein(degree=3, include_bias=False, ensemble=True)
    assert b2.degree == 3
    assert b2.include_bias is False
    assert b2.ensemble is True

    b3 = Bernstein(n=4)  # `n` should override `degree`
    assert b3.degree == 4

    b4 = Bernstein(bias=False)  # `bias` should override `include_bias`
    assert b4.include_bias is False


def test_bernstein_fit():
    """Test that fit transforms input data correctly."""
    b = Bernstein(degree=2, include_bias=True)
    data = np.random.rand(10, 3)  # 10 samples, 3 features (including intercept)
    max_lag = 2

    transformed = b.fit(data, max_lag=max_lag)

    expected_rows = data.shape[0] - max_lag
    assert transformed.shape[0] == expected_rows

    if b.include_bias:
        assert np.all(transformed[:, 0] == 1), "Bias column not added correctly"


def test_bernstein_fit_ensemble():
    """Test that ensemble mode includes original data."""
    b = Bernstein(degree=2, include_bias=True, ensemble=True)
    data = np.random.rand(10, 3)
    max_lag = 2

    transformed = b.fit(data, max_lag=max_lag)

    expected_features = (data.shape[1] - 1) * 2  # 2 basis functions per feature
    assert (
        transformed.shape[1] == expected_features + (data.shape[1] - 1) + 1
    )  # +1 for bias


def test_bernstein_transform():
    """Test that transform calls fit and produces the same result."""
    b = Bernstein(degree=2, include_bias=True)
    data = np.random.rand(10, 3)

    transformed_fit = b.fit(data)
    transformed_transform = b.transform(data)

    np.testing.assert_array_equal(transformed_fit, transformed_transform)
