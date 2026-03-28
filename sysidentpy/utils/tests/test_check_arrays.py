import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy.utils.check_arrays import check_random_state, check_dimension
from sysidentpy.utils.check_arrays import check_infinity, check_nan


def test_check_random_state_invalid_seed():
    """Invalid types should raise when seeding random state."""
    with pytest.raises(ValueError):
        check_random_state(object())


def test_check_random_state_none_returns_generator():
    """None should create a fresh NumPy Generator instance."""
    random_state = check_random_state(None)
    assert isinstance(random_state, np.random.Generator)


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


def test_check_infinity_accepts_array_api_inputs():
    """Array API inputs should preserve validation behavior for Inf checks."""
    xp = pytest.importorskip("array_api_strict")
    x = xp.asarray(np.array([[1.0], [np.inf], [3.0]]))
    y = xp.asarray(np.array([[1.0], [2.0], [3.0]]))

    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match=r"index \[\[1 0\]\]"):
            check_infinity(x, y)


def test_check_nan_accepts_array_api_inputs():
    """Array API inputs should preserve validation behavior for NaN checks."""
    xp = pytest.importorskip("array_api_strict")
    x = xp.asarray(np.array([[1.0], [np.nan], [3.0]]))
    y = xp.asarray(np.array([[1.0], [2.0], [3.0]]))

    with config_context(array_api_dispatch=True):
        with pytest.raises(ValueError, match=r"index \[\[1 0\]\]"):
            check_nan(x, y)
