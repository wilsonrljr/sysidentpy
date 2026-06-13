import warnings

import numpy as np
import pytest

from sysidentpy import config_context
from sysidentpy.utils.check_arrays import check_dimension, check_linear_dependence_rows
from sysidentpy.utils.check_arrays import check_infinity, check_nan, check_random_state


def test_check_random_state_invalid_seed():
    """Invalid types should raise when seeding random state."""
    with pytest.raises(ValueError, match="seed"):
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


def test_check_linear_dependence_rows_accepts_array_api_full_rank():
    """Array API inputs should use the backend SVD path for full-rank matrices."""
    xp = pytest.importorskip("array_api_strict")
    psi = xp.asarray(np.eye(3, dtype=float))

    with config_context(array_api_dispatch=True):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            check_linear_dependence_rows(psi)

    assert recorded == []


def test_check_linear_dependence_rows_warns_for_array_api_rank_deficient_matrix():
    """Array API inputs should preserve rank-deficiency warnings."""
    xp = pytest.importorskip("array_api_strict")
    psi = xp.asarray(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [1.0, 0.0, 1.0],
            ]
        )
    )

    with config_context(array_api_dispatch=True):
        with pytest.warns(UserWarning, match="linearly dependent rows"):
            check_linear_dependence_rows(psi)
