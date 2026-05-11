import numpy as np
from sysidentpy.basis_function import Polynomial


def test_polynomial_init():
    """Test that Polynomial class initializes correctly."""
    p = Polynomial(degree=3, include_bias=False)
    assert p.degree == 3
    assert not p.include_bias

    p_default = Polynomial()
    assert p_default.degree == 2
    assert not p_default.include_bias  # Default should be False for backward compatibility


def test_polynomial_fit():
    """Test that fit correctly generates the polynomial feature matrix."""
    p = Polynomial(degree=2)
    data = np.random.rand(10, 3)  # 10 samples, 3 features
    max_lag = 2

    transformed = p.fit(data, max_lag=max_lag)

    assert transformed.shape[0] == data.shape[0] - max_lag
    assert transformed.shape[1] > 0  # Ensure non-empty feature matrix


def test_polynomial_transform():
    """Test that transform behaves identically to fit."""
    p = Polynomial(degree=2)
    data = np.random.rand(10, 3)

    transformed_fit = p.fit(data)
    transformed_transform = p.transform(data)

    np.testing.assert_array_equal(transformed_fit, transformed_transform)


def test_polynomial_include_bias():
    """Test that bias is included when requested."""
    p = Polynomial(degree=2, include_bias=True)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    max_lag = 1

    transformed = p.fit(data, max_lag=max_lag)
    # First column should be all ones (bias term)
    assert np.all(transformed[:, 0] == 1)

    p_no_bias = Polynomial(degree=2, include_bias=False)
    transformed_no_bias = p_no_bias.fit(data, max_lag=max_lag)
    # Should not have a bias column (no column of all ones)
    has_bias_column = any(np.all(transformed_no_bias[:, i] == 1)
                          for i in range(transformed_no_bias.shape[1]))
    assert not has_bias_column


def test_polynomial_include_bias_shapes():
    """Test that include_bias affects the output shape correctly."""
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    max_lag = 1

    p_with_bias = Polynomial(degree=2, include_bias=True)
    p_without_bias = Polynomial(degree=2, include_bias=False)

    transformed_with = p_with_bias.fit(data, max_lag=max_lag)
    transformed_without = p_without_bias.fit(data, max_lag=max_lag)

    # With bias should have one more column
    assert transformed_with.shape[1] == transformed_without.shape[1] + 1
    # Same number of rows
    assert transformed_with.shape[0] == transformed_without.shape[0]


def test_polynomial_fit_predefined_regressors_with_bias():
    """Test fit with predefined regressors when include_bias=True."""
    p = Polynomial(degree=2, include_bias=True)
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    max_lag = 1
    predefined_regressors = np.array([0, 2])  # Selecting bias and one other regressor

    transformed = p.fit(data, max_lag=max_lag, predefined_regressors=predefined_regressors)

    assert transformed.shape[1] == len(predefined_regressors)
    # First selected regressor should be bias (all ones)
    assert np.all(transformed[:, 0] == 1)
