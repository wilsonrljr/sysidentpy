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


def test_bilinear_include_bias_removes_only_the_constant_combination():
    data = np.array(
        [
            [1.0, 1.0, 2.0],
            [1.0, 3.0, 4.0],
            [1.0, 5.0, 6.0],
        ]
    )
    default_basis = Bilinear(degree=2)
    explicit_basis = Bilinear(degree=2, include_bias=True)
    without_bias_basis = Bilinear(degree=2, include_bias=False)

    default = default_basis.fit(data, max_lag=1)
    explicit = explicit_basis.fit(data, max_lag=1)
    without_bias = without_bias_basis.fit(data, max_lag=1)
    bias_columns = np.flatnonzero(np.all(explicit == 1, axis=0))

    np.testing.assert_array_equal(default, explicit)
    assert bias_columns.size == 1
    np.testing.assert_array_equal(
        without_bias,
        np.delete(explicit, bias_columns, axis=1),
    )
    assert without_bias.shape[1] == 3

    selected = without_bias_basis.fit(
        data,
        max_lag=1,
        predefined_regressors=np.array([0, 2]),
    )
    np.testing.assert_array_equal(selected, without_bias[:, [0, 2]])


def test_bilinear_degree_one_without_bias_rejects_empty_feature_space():
    with pytest.raises(ValueError, match="generates no regressors"):
        Bilinear(degree=1, include_bias=False)


def test_bilinear_degree_warning():
    """Test that a warning is raised when degree=1 is chosen."""
    b = Bilinear(degree=1)
    data = np.random.rand(10, 3)
    with pytest.warns(UserWarning, match="linear polynomial model"):
        b.fit(data)
