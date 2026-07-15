"""Array API-aware assertions for SysIdentPy tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from sysidentpy._lib._vendor.array_api_compat import array_namespace
from sysidentpy._lib._vendor.array_api_extra import testing as xpx_testing

if TYPE_CHECKING:
    from types import ModuleType


def _namespace_from_actual(actual: Any, xp: ModuleType | None) -> ModuleType:
    """Infer the namespace that the actual result is expected to preserve."""
    if xp is not None:
        return xp

    try:
        return array_namespace(actual)
    except TypeError as exc:
        msg = (
            "Array API assertions require the actual value to be an array. "
            "Use pytest.approx or math.isclose for pure scalar comparisons."
        )
        raise AssertionError(msg) from exc


def _normalize_expected(desired: Any, xp: ModuleType) -> Any:
    """Convert test literals, but leave array objects to xpx.testing."""
    if isinstance(desired, (list, tuple)) or np.isscalar(desired):
        return xp.asarray(desired)

    return desired


def assert_allclose(
    actual: Any,
    desired: Any,
    *,
    rtol: float | Any | None = None,
    atol: float | Any = 0,
    equal_nan: bool = True,
    err_msg: str = "",
    verbose: bool = True,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
    xp: ModuleType | None = None,
) -> None:
    """Assert close while preserving Array API namespace checks."""
    __tracebackhide__ = True
    namespace = _namespace_from_actual(actual, xp)
    desired = _normalize_expected(desired, namespace)

    xpx_testing.assert_close(
        actual,
        desired,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        err_msg=err_msg,
        verbose=verbose,
        check_dtype=check_dtype,
        check_shape=check_shape,
        check_scalar=check_scalar,
        xp=namespace,
    )


def assert_array_equal(
    actual: Any,
    desired: Any,
    *,
    err_msg: str = "",
    verbose: bool = True,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
    xp: ModuleType | None = None,
) -> None:
    """Assert exact equality while preserving Array API namespace checks."""
    __tracebackhide__ = True
    namespace = _namespace_from_actual(actual, xp)
    desired = _normalize_expected(desired, namespace)

    xpx_testing.assert_equal(
        actual,
        desired,
        err_msg=err_msg,
        verbose=verbose,
        check_dtype=check_dtype,
        check_shape=check_shape,
        check_scalar=check_scalar,
        xp=namespace,
    )


def assert_array_less(
    x: Any,
    y: Any,
    *,
    err_msg: str = "",
    verbose: bool = True,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
    xp: ModuleType | None = None,
) -> None:
    """Assert elementwise ordering while preserving Array API namespace checks."""
    __tracebackhide__ = True
    namespace = _namespace_from_actual(x, xp)
    y = _normalize_expected(y, namespace)

    xpx_testing.assert_less(
        x,
        y,
        err_msg=err_msg,
        verbose=verbose,
        check_dtype=check_dtype,
        check_shape=check_shape,
        check_scalar=check_scalar,
        xp=namespace,
    )


def assert_close_nulp(
    actual: Any,
    desired: Any,
    *,
    nulp: int = 1,
    check_dtype: bool = True,
    check_shape: bool = True,
    check_scalar: bool = False,
    xp: ModuleType | None = None,
) -> None:
    """Assert closeness by units in the last place."""
    __tracebackhide__ = True
    namespace = _namespace_from_actual(actual, xp)
    desired = _normalize_expected(desired, namespace)

    xpx_testing.assert_close_nulp(
        actual,
        desired,
        nulp=nulp,
        check_dtype=check_dtype,
        check_shape=check_shape,
        check_scalar=check_scalar,
        xp=namespace,
    )
