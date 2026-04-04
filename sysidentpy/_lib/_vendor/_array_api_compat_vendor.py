"""Hook for array-api-extra to find the vendored array-api-compat."""

from .array_api_compat import (
    array_namespace,
    device,
    is_array_api_obj,
    is_array_api_strict_namespace,
    is_cupy_array,
    is_cupy_namespace,
    is_dask_array,
    is_dask_namespace,
    is_jax_array,
    is_jax_namespace,
    is_lazy_array,
    is_numpy_array,
    is_numpy_namespace,
    is_pydata_sparse_array,
    is_pydata_sparse_namespace,
    is_torch_array,
    is_torch_namespace,
    is_writeable_array,
    size,
    to_device,
)
