"""Array API compatibility utilities for SysIdentPy.

This module provides helpers that allow SysIdentPy to work with any
array library that implements the Array API standard (NumPy, CuPy,
PyTorch, JAX, etc.).

The approach follows scikit-learn and SciPy: use ``get_namespace()``
to obtain the correct array namespace, then call standard operations
through that namespace.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._config import get_config
from ._vendor.array_api_extra import at as _at
from ._vendor.array_api_compat import (
    array_namespace as _array_namespace,
    device as _compat_device,
    is_array_api_obj,
    is_cupy_namespace as _is_cupy_ns,
    is_jax_namespace as _is_jax_ns,
    is_numpy_namespace as _is_numpy_ns,
    is_torch_namespace as _is_torch_ns,
    size,
    to_device,
)


def _array_api_arrays(*arrays):
    """Return the Array API objects from *arrays*."""
    return [a for a in arrays if a is not None and is_array_api_obj(a)]


def _namespace_key(xp) -> str:
    """Return a normalized backend identifier for *xp*."""
    if _is_numpy_namespace(xp):
        return "numpy"
    if _is_torch_ns(xp):
        return "torch"
    if _is_cupy_ns(xp):
        return "cupy"
    if _is_jax_ns(xp):
        return "jax"

    return getattr(xp, "__name__", xp.__class__.__name__)


def _validate_namespace_consistency(arrays) -> None:
    """Ensure all Array API objects use the same backend namespace."""
    namespace_keys = sorted(
        {_namespace_key(_array_namespace(array)) for array in arrays}
    )
    if len(namespace_keys) <= 1:
        return

    raise ValueError(
        "Input arrays must use the same Array API namespace when "
        f"array_api_dispatch=True. Got {namespace_keys!r}."
    )


def get_namespace(*arrays):
    """Get the array API namespace for the given arrays.

    When ``array_api_dispatch`` is disabled (the default), this always
    returns :mod:`numpy`, regardless of the input types.

    When enabled, the namespace is inferred from the input arrays via
    :func:`array_api_compat.array_namespace`.

    Parameters
    ----------
    *arrays : array-like
        Input arrays.  Non-array objects (``None``, scalars) are ignored.

    Returns
    -------
    xp : module
        The array API namespace.

    Examples
    --------
    >>> import numpy as np
    >>> from sysidentpy._lib._array_api import get_namespace
    >>> xp = get_namespace(np.array([1, 2, 3]))
    >>> xp is np
    True
    """
    if not get_config()["array_api_dispatch"]:
        return np

    arrays = _array_api_arrays(*arrays)
    if not arrays:
        return np

    _validate_namespace_consistency(arrays)
    return _array_namespace(*arrays)


def _get_namespace_and_device(*arrays):
    """Return a validated namespace/device pair for the given inputs.

    When Array API dispatch is enabled, all Array API inputs must belong to the
    same namespace and live on the same device.
    """
    xp = get_namespace(*arrays)
    if not get_config()["array_api_dispatch"]:
        return xp, None

    arrays = _array_api_arrays(*arrays)
    if not arrays:
        return xp, None

    return xp, device(*arrays)


def _is_numpy_namespace(xp) -> bool:
    """Check if *xp* is a NumPy namespace.

    This is used to gate fast-path branches that use NumPy- or
    SciPy-specific features not available in the Array API standard.
    """
    return _is_numpy_ns(xp)


def _require_numpy_namespace(xp, *, feature: str, dependency: str = "SciPy") -> None:
    """Raise when *feature* is used with a non-NumPy Array API namespace.

    This is used for code paths that still rely on external libraries with
    NumPy-only semantics and therefore cannot preserve backend/device under
    ``array_api_dispatch=True``.
    """
    if _is_numpy_namespace(xp):
        return

    namespace_name = getattr(xp, "__name__", xp.__class__.__name__)
    raise NotImplementedError(
        f"{feature} does not support Array API dispatch with namespace "
        f"{namespace_name!r}. This path currently relies on {dependency} "
        "and requires NumPy inputs. Use NumPy arrays or choose an estimator "
        "implemented with Array API operations."
    )


def device(*arrays) -> Any:
    """Return the device shared by *arrays*.

    Raises
    ------
    ValueError
        If arrays live on different devices.
    """
    arrays = [a for a in arrays if a is not None and is_array_api_obj(a)]
    if not arrays:
        return None

    devices = [_compat_device(a) for a in arrays]
    unique = {str(d) for d in devices}
    if len(unique) > 1:
        raise ValueError(
            f"Input arrays reside on different devices: {unique}. "
            "All arrays must be on the same device."
        )
    return devices[0]


def _to_numpy(arr) -> np.ndarray:
    """Convert any array to a NumPy ndarray on the CPU.

    Handles GPU-resident arrays from PyTorch, CuPy, JAX, and other
    backends by using library-specific transfer methods before falling
    back to the ``__array__`` or DLPack protocols.

    Used when SysIdentPy must call NumPy/SciPy/matplotlib functions
    that do not support the Array API.
    """
    if isinstance(arr, np.ndarray):
        return arr

    # Library-specific fast paths for GPU arrays
    xp = _array_namespace(arr)
    if _is_torch_ns(xp):
        return arr.detach().cpu().numpy()

    if _is_cupy_ns(xp):
        return arr.get()

    # array_api_strict: move to CPU device before converting
    if hasattr(xp, "Device") and hasattr(arr, "device"):
        try:
            arr = xp.asarray(arr, device=xp.Device("CPU_DEVICE"))
        except (TypeError, ValueError):
            pass

    # Try DLPack (works for JAX and other backends)
    if hasattr(arr, "__dlpack__"):
        try:
            return np.from_dlpack(arr)
        except (TypeError, BufferError):
            pass

    # Fallback via __array__ protocol
    if hasattr(arr, "__array__"):
        return np.asarray(arr)

    return np.asarray(arr)


def _asarray(data, *, xp, dtype=None, target_device=None):
    """Convert *data* to an array in the given namespace.

    Parameters
    ----------
    data : array-like
        Data to convert.
    xp : module
        Target array API namespace.
    dtype : dtype, optional
        Desired dtype.
    target_device : device, optional
        Target device.  If ``None``, inferred from *data* when possible.
    """
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if target_device is not None and not _is_numpy_namespace(xp):
        kwargs["device"] = target_device
    return xp.asarray(data, **kwargs)


def _zeros(xp, shape, *, dtype=None, target_device=None):
    """Create a zero-filled array while preserving the target device."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if target_device is not None and not _is_numpy_namespace(xp):
        kwargs["device"] = target_device
    return xp.zeros(shape, **kwargs)


def _ones(xp, shape, *, dtype=None, target_device=None):
    """Create a one-filled array while preserving the target device."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if target_device is not None and not _is_numpy_namespace(xp):
        kwargs["device"] = target_device
    return xp.ones(shape, **kwargs)


def _full(xp, shape, fill_value, *, dtype=None, target_device=None):
    """Create a filled array while preserving the target device."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if target_device is not None and not _is_numpy_namespace(xp):
        kwargs["device"] = target_device
    return xp.full(shape, fill_value, **kwargs)


def _vector_norm(xp, x):
    """Return a vector norm for namespaces with or without ``vector_norm``."""
    if hasattr(xp.linalg, "vector_norm"):
        return xp.linalg.vector_norm(x)

    return xp.sqrt(xp.sum(x * x))


def _pow(xp, x1, x2):
    """Raise arrays elementwise for namespaces with ``pow`` or ``power``."""
    if hasattr(xp, "pow"):
        return xp.pow(x1, x2)

    return xp.power(x1, x2)


# ---------------------------------------------------------------------------
# Fallback implementations for operations not in the Array API standard
# ---------------------------------------------------------------------------


def _lstsq(xp, A, b):
    """Least-squares solve ``A @ x ≈ b``.

    For NumPy, delegates to :func:`numpy.linalg.lstsq`.
    For other backends, uses a truncated SVD approach that is robust
    to rank-deficient (singular) matrices.
    """
    if _is_numpy_namespace(xp):
        result = np.linalg.lstsq(A, b, rcond=None)
        return result[0]

    # SVD-based fallback — robust to singular / rank-deficient A
    U, S, Vh = xp.linalg.svd(A, full_matrices=False)
    # Threshold based on machine epsilon, matching NumPy's lstsq
    eps = xp.finfo(A.dtype).eps
    cutoff = eps * float(max(A.shape)) * float(xp.max(S))
    mask = S > cutoff
    S_inv = xp.where(mask, 1.0 / S, xp.zeros_like(S))
    # x = V @ diag(1/S) @ U^T @ b
    return (Vh.T * S_inv[None, :]) @ (U.T @ b)


def _concat(xp, arrays, axis=0):
    """Equivalent of ``np.concatenate`` using Array API-compatible backends."""
    if _is_numpy_namespace(xp):
        return np.concatenate(arrays, axis=axis)

    arrays = list(arrays)
    target_device = None
    for arr in arrays:
        if is_array_api_obj(arr):
            target_device = _compat_device(arr)
            break

    if target_device is not None:
        arrays = [
            (
                _asarray(arr, xp=xp, target_device=target_device)
                if not is_array_api_obj(arr)
                or str(_compat_device(arr)) != str(target_device)
                else arr
            )
            for arr in arrays
        ]

    return xp.concat(arrays, axis=axis)


def _column_stack(xp, arrays):
    """Equivalent of ``np.column_stack`` using Array API.

    Each 1-D array is reshaped to a column, then all are concatenated
    along ``axis=1``.
    """
    result = []
    for a in arrays:
        col = xp.reshape(a, (-1, 1)) if a.ndim == 1 else a
        result.append(col)
    return _concat(xp, result, axis=1)


def _vstack(xp, arrays):
    """Equivalent of ``np.vstack`` using Array API."""
    result = []
    for a in arrays:
        row = xp.reshape(a, (1, -1)) if a.ndim == 1 else a
        result.append(row)
    return _concat(xp, result, axis=0)


def _hstack(xp, arrays):
    """Equivalent of ``np.hstack`` using Array API."""
    if arrays[0].ndim == 1:
        return _concat(xp, arrays, axis=0)
    return _concat(xp, arrays, axis=1)


def _nanargmin(xp, a, axis=None):
    """``np.nanargmin`` equivalent using Array API.

    Replaces NaNs with +inf before calling ``argmin``.
    """
    if _is_numpy_namespace(xp):
        return np.nanargmin(a, axis=axis)

    mask = xp.isnan(a)
    filled = xp.where(mask, xp.asarray(float("inf"), dtype=a.dtype), a)
    return xp.argmin(filled, axis=axis)


def _median(xp, a, axis=None):
    """``np.median`` equivalent using Array API (sort-based)."""
    if _is_numpy_namespace(xp):
        return np.median(a, axis=axis)

    sorted_a = xp.sort(a, axis=axis if axis is not None else 0)
    n = a.shape[axis] if axis is not None else size(a)
    mid = n // 2
    if axis is not None:
        slices = [slice(None)] * a.ndim
        if n % 2 == 1:
            slices[axis] = mid
            return sorted_a[tuple(slices)]
        slices_lo = list(slices)
        slices_hi = list(slices)
        slices_lo[axis] = mid - 1
        slices_hi[axis] = mid
        return (sorted_a[tuple(slices_lo)] + sorted_a[tuple(slices_hi)]) / 2
    # No axis: flatten
    flat = xp.reshape(sorted_a, (-1,))
    if n % 2 == 1:
        return flat[mid]
    return (flat[mid - 1] + flat[mid]) / 2


def _einsum_ij_ij_j(xp, A, B):
    """``np.einsum('ij,ij->j', A, B)`` using Array API.

    Equivalent to ``(A * B).sum(axis=0)``.
    """
    return xp.sum(A * B, axis=0)


def _diag(xp, v):
    """Create a diagonal matrix from a 1-D array or extract a diagonal.

    For 1-D input, creates a 2-D diagonal matrix.
    For 2-D input, extracts the diagonal.
    """
    if _is_numpy_namespace(xp):
        return np.diag(v)

    if v.ndim == 1:
        n = v.shape[0]
        out = xp.zeros((n, n), dtype=v.dtype)
        for i in range(n):
            out = _set_element(xp, out, (i, i), v[i])
        return out
    elif v.ndim == 2:
        n = min(v.shape)
        indices = xp.arange(n)
        return v[indices, indices]
    raise ValueError(f"Input must be 1-D or 2-D, got {v.ndim}-D")


def _set_element(xp, arr, idx, val):
    """Set a single element of an array.

    Handles both mutable (NumPy, CuPy, PyTorch) and immutable (JAX) arrays.
    """
    try:
        arr[idx] = val
        return arr
    except (RuntimeError, TypeError, ValueError):
        if hasattr(arr, "at"):
            return arr.at[idx].set(val)

        return _at(arr, idx).set(val, xp=xp)


def _copy(xp, arr):
    """Create a copy of an array."""
    if _is_numpy_namespace(xp):
        return arr.copy()
    # Array API 2023.12+ has xp.asarray with copy=True
    try:
        return xp.asarray(arr, copy=True)
    except TypeError:
        # Fallback for older namespaces
        return arr * xp.ones_like(arr)


__all__ = [
    "_asarray",
    "_column_stack",
    "_concat",
    "_copy",
    "_diag",
    "_einsum_ij_ij_j",
    "_full",
    "_get_namespace_and_device",
    "_hstack",
    "_is_numpy_namespace",
    "_lstsq",
    "_median",
    "_nanargmin",
    "_ones",
    "_pow",
    "_require_numpy_namespace",
    "_set_element",
    "_to_numpy",
    "_vector_norm",
    "_vstack",
    "_zeros",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "size",
    "to_device",
]
