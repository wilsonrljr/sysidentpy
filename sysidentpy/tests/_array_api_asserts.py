"""Array API-aware assertions for SysIdentPy tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sysidentpy._lib._vendor.array_api_compat import array_namespace
from sysidentpy._lib._vendor.array_api_extra import testing as xpx_testing

if TYPE_CHECKING:
    from types import ModuleType


_PYTHON_SCALAR_TYPES = (bool, int, float, complex)


def _unsupported_numpy_subclass(value: Any) -> bool:
    """Return whether value is an ndarray subclass with additional semantics."""
    return isinstance(value, np.ndarray) and type(value) is not np.ndarray


def _namespace_from_actual(actual: Any) -> ModuleType:
    """Infer the namespace of an array result, rejecting scalar-like inputs."""
    if _unsupported_numpy_subclass(actual):
        msg = (
            "Array API assertions do not support NumPy ndarray subclasses. "
            "Use numpy.testing for NumPy-only types such as masked arrays."
        )
        raise AssertionError(msg)

    if np.isscalar(actual) or isinstance(actual, (list, tuple)):
        msg = (
            "Array API assertions require 'actual' to be an array. Use "
            "pytest.approx or math.isclose for scalar comparisons."
        )
        raise AssertionError(msg)

    try:
        return array_namespace(actual)
    except TypeError as exc:
        msg = (
            "Array API assertions require 'actual' to be an Array API-compatible "
            "array."
        )
        raise AssertionError(msg) from exc


def _normalize_expected(desired: Any, xp: ModuleType) -> Any:
    """Normalize only the expected-value forms allowed by SysIdentPy tests."""
    if _unsupported_numpy_subclass(desired):
        msg = (
            "Array API assertions do not support NumPy ndarray subclasses as "
            "expected values. Use numpy.testing for NumPy-only types such as "
            "masked arrays."
        )
        raise AssertionError(msg)

    if (
        type(desired) in _PYTHON_SCALAR_TYPES
        or isinstance(desired, np.generic)
        or type(desired) is np.ndarray
        or isinstance(desired, (list, tuple))
    ):
        return xp.asarray(desired)

    try:
        array_namespace(desired)
    except TypeError as exc:
        msg = (
            "Expected values must be an Array API-compatible array, a NumPy "
            "array or scalar, a numeric Python scalar, or a list/tuple."
        )
        raise AssertionError(msg) from exc

    return desired


def assert_allclose(
    actual: Any,
    desired: Any,
    *,
    rtol: float | Any | None = None,
    atol: float | Any = 0,
    equal_nan: bool = True,
    check_dtype: bool = True,
    check_shape: bool = True,
) -> None:
    """Assert close values under SysIdentPy's expected-value policy."""
    __tracebackhide__ = True
    xp = _namespace_from_actual(actual)
    desired = _normalize_expected(desired, xp)

    xpx_testing.assert_close(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_dtype=check_dtype,
        check_shape=check_shape,
    )


def assert_array_equal(
    actual: Any,
    desired: Any,
    *,
    check_dtype: bool = True,
    check_shape: bool = True,
) -> None:
    """Assert exact equality under SysIdentPy's expected-value policy."""
    __tracebackhide__ = True
    xp = _namespace_from_actual(actual)
    desired = _normalize_expected(desired, xp)

    xpx_testing.assert_equal(
        actual,
        desired,
        check_dtype=check_dtype,
        check_shape=check_shape,
    )
