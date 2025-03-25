import numpy as np
from sysidentpy.basis_function import HermiteNormalized


def test_hermite_normalized_init():
    """Test that HermiteNormalized class initializes correctly."""
    h = HermiteNormalized(degree=3, include_bias=False, ensemble=True)
    assert h.degree == 3
    assert not h.include_bias
    assert h.ensemble

    h_default = HermiteNormalized()
    assert h_default.degree == 1
    assert h_default.include_bias
    assert not h_default.ensemble


def test_hermite_normalized_fit():
    """Test that fit correctly generates the Hermite feature matrix."""
    h = HermiteNormalized(degree=2)
    data = np.random.rand(10, 3)  # 10 samples, 3 features
    max_lag = 2

    transformed = h.fit(data, max_lag=max_lag)

    assert transformed.shape[0] == data.shape[0] - max_lag
    assert transformed.shape[1] > 0  # Ensure non-empty feature matrix


def test_hermite_normalized_transform():
    """Test that transform behaves identically to fit."""
    h = HermiteNormalized(degree=2)
    data = np.random.rand(10, 3)

    transformed_fit = h.fit(data)
    transformed_transform = h.transform(data)

    np.testing.assert_array_equal(transformed_fit, transformed_transform)


def test_hermite_normalized_include_bias():
    """Test that bias is included when requested."""
    h = HermiteNormalized(degree=2, include_bias=True)
    data = np.random.rand(10, 3)

    transformed = h.fit(data)
    assert np.all(transformed[:, 0] == 1)  # First column should be ones

    h_no_bias = HermiteNormalized(degree=2, include_bias=False)
    transformed_no_bias = h_no_bias.fit(data)
    assert not np.all(transformed_no_bias[:, 0] == 1)  # No bias column


def test_hermite_normalized_ensemble():
    """Test ensemble mode (concatenation of original data)."""
    h = HermiteNormalized(degree=2, ensemble=True)
    data = np.random.rand(10, 3)

    transformed = h.fit(data)
    assert transformed.shape[1] > data.shape[1]  # Must include original features


def test_hermite_normalized_fit_predefined_regressors():
    """Test fit with predefined regressors filtering output features correctly."""
    h = HermiteNormalized(degree=2)
    data = np.random.rand(10, 3)
    predefined_regressors = [0, 2]  # Selecting only certain regressors

    transformed = h.fit(data, predefined_regressors=predefined_regressors)

    assert transformed.shape[1] == len(
        predefined_regressors
    )  # Ensure correct feature selection
