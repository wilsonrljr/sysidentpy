import numpy as np
import pytest

from sysidentpy.tests._array_api_asserts import (
    assert_allclose,
    assert_array_equal,
    assert_array_less,
)


def test_assert_array_equal_accepts_numpy_array_and_literal_expected():
    actual = np.array([1.0, 2.0, 3.0])

    assert_array_equal(actual, [1.0, 2.0, 3.0])


def test_assert_allclose_accepts_array_api_strict_with_numpy_expected():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    desired = np.array([1.0, 2.0, 3.0])

    assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def test_assert_array_less_accepts_array_api_strict_with_literal_expected():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    assert_array_less(actual, [2.0, 3.0, 4.0])


def test_assert_array_equal_checks_dtype_by_default():
    actual = np.array([1.0, 2.0, 3.0])
    desired = np.array([1, 2, 3])

    with pytest.raises(AssertionError, match="dtypes do not match"):
        assert_array_equal(actual, desired)


def test_assert_array_equal_allows_dtype_check_opt_out():
    actual = np.array([1.0, 2.0, 3.0])
    desired = np.array([1, 2, 3])

    assert_array_equal(actual, desired, check_dtype=False)


def test_assert_array_equal_rejects_actual_namespace_mismatch():
    xp = pytest.importorskip("array_api_strict")
    actual = np.array([1.0, 2.0, 3.0])

    with pytest.raises(AssertionError, match="does not match the `xp` argument"):
        assert_array_equal(actual, [1.0, 2.0, 3.0], xp=xp)


def test_assert_allclose_accepts_zero_dimensional_tolerances():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)
    desired = xp.asarray([1.0, 2.0, 3.0], dtype=xp.float64)

    assert_allclose(
        actual,
        desired,
        rtol=xp.asarray(1e-12, dtype=xp.float64),
        atol=xp.asarray(1e-12, dtype=xp.float64),
    )


@pytest.mark.xfail(
    reason=(
        "array_api_extra.testing reaches array_api_strict.to_device(CPU_DEVICE) "
        "for non-default devices; that currently fails in this environment "
        "with NumPy's copy keyword"
    ),
)
def test_assert_array_equal_documents_array_api_strict_non_default_device_edge_case():
    xp = pytest.importorskip("array_api_strict")
    actual = xp.asarray([1.0, 2.0, 3.0], device=xp.Device("device1"))

    assert_array_equal(actual, [1.0, 2.0, 3.0])
