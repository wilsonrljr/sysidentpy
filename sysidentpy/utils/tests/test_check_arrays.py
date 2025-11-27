import numpy as np
import pytest

from sysidentpy.utils.check_arrays import check_random_state, check_dimension


def test_check_random_state_invalid_seed():
    """Invalid types should raise when seeding random state."""
    with pytest.raises(ValueError):
        check_random_state(object())


def test_check_dimension_scalar_input():
    """Scalar inputs must be rejected."""
    x = np.array(1.0)
    y = np.array([[1.0]])
    with pytest.raises(ValueError, match="Input must be a 2d array"):
        check_dimension(x, y)


def test_check_dimension_scalar_output():
    """Scalar outputs should also trigger validation errors."""
    x = np.array([[1.0]])
    y = np.array(1.0)
    with pytest.raises(ValueError, match="Output must be a 2d array"):
        check_dimension(x, y)


def test_check_dimension_output_vector():
    """One-dimensional outputs must be reshaped by the caller."""
    x = np.array([[1.0], [2.0]])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Output must be a 2d array"):
        check_dimension(x, y)
