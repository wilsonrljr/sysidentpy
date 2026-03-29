# Array API Standard Support

SysIdentPy implements support for the [Python Array API standard](https://data-apis.org/array-api/latest/), enabling users to run system identification workflows with different array backends, including PyTorch (CPU and CUDA), CuPy, JAX, and any other library that conforms to the standard, without modifying model code.

This page explains what the Array API standard is, why SysIdentPy adopted it, how the implementation works, which modules and algorithms are supported, and how to use it in practice.

## What is the Array API Standard?

The Python Array API standard is a specification that defines a common set of operations for array/tensor libraries. It was created by the [Consortium for Python Data API Standards](https://data-apis.org/) to address a practical problem in the scientific Python ecosystem: libraries like NumPy, PyTorch, CuPy, and JAX all provide multidimensional arrays with overlapping functionality, but their APIs differ in subtle ways — function names, argument ordering, return types, device handling, and mutability semantics are not the same across libraries.

Before the Array API standard, a scientific library that wanted to work with multiple backends had two choices: either duplicate the implementation for each backend (leading to maintenance burden and divergent behavior), or write adapter layers with case-by-case compatibility code. Neither approach scaled well.

The Array API standard solves this by defining a minimal, well-specified set of array operations: creation, manipulation, linear algebra, statistical reductions, and element-wise math. That all conforming libraries must implement with identical signatures and semantics. A library like SysIdentPy can then be written against this standard interface and work with any backend that implements it.

The standard is backed by the core teams of NumPy, PyTorch, CuPy, JAX, and Dask. Libraries like scikit-learn and SciPy have already adopted it. SysIdentPy follows the same path.

## Why Array API Support Matters for System Identification

System identification involves fitting mathematical models (typically NARMAX-family models) to observed input/output data. The computational core of these algorithms consists of matrix operations: building regressor matrices, computing orthogonal projections, solving least-squares problems, evaluating error reduction ratios, and running simulations.

These are the same types of operations that benefit most from GPU acceleration and hardware-aware computation. With Array API support, SysIdentPy gains several capabilities:

**GPU Acceleration**

PyTorch tensors on CUDA or CuPy arrays reside on GPU memory. When SysIdentPy receives these arrays, all matrix multiplications, SVD decompositions, element-wise operations, and reductions run on the GPU. For large-scale identification problems (many regressors, long time series, high polynomial degrees), this can provide substantial speedups compared to CPU-only NumPy.

**Backend Flexibility**

Researchers working in different ecosystems (PyTorch for deep learning pipelines, JAX for automatic differentiation, CuPy for GPU-accelerated scientific computing) can now use SysIdentPy within their existing workflows without converting data back and forth to NumPy. The model receives the data in the backend's native format and returns results in the same format.

**Device Preservation**

When the input data lives on a specific device (e.g., `cuda:0` for PyTorch, or a specific GPU for CuPy), SysIdentPy preserves that device throughout the entire computation. Intermediate arrays, results, and predictions are created on the same device as the inputs. There is no implicit data transfer between CPU and GPU, which avoids the performance penalty of memory copies.

**Ecosystem Alignment**

The Scientific Python ecosystem is converging on the Array API standard. scikit-learn, SciPy, and other major libraries have adopted it. By following the same approach, SysIdentPy ensures that its users benefit from the same patterns, tools, and ecosystem improvements.

**Zero Changes to Existing Code**

Array API support is opt-in. The default behavior (NumPy-only) is completely unchanged. Existing code, notebooks, and pipelines continue to work identically. Users who want to use other backends simply enable dispatch and pass their data (no API changes, no new classes, no different function signatures).

## How It Works

### Architecture Overview

SysIdentPy's Array API implementation consists of three layers:

1. **Configuration layer** (`sysidentpy._config`): A global, thread-safe flag (`array_api_dispatch`) that controls whether dispatch is active. When disabled, everything uses NumPy. When enabled, the namespace is inferred from input arrays.

2. **Compatibility layer** (`sysidentpy._lib._array_api`): A module that provides backend-agnostic wrappers for operations that differ across backends or are not part of the Array API standard. This includes `get_namespace()` for namespace detection, `_lstsq()` for least-squares solving, `_diag()` for diagonal matrix creation, `_set_element()` for mutable/immutable array element assignment, and many others.

3. **Vendored dependencies**: Copies of [`array-api-compat`](https://github.com/data-apis/array-api-compat) (v1.14.0) and [`array-api-extra`](https://github.com/data-apis/array-api-extra) (v0.10.1) bundled inside SysIdentPy. These provide the foundational `array_namespace()` function for backend detection and additional utility functions not in the core standard.

### The `get_namespace()` Pattern

The core pattern used throughout SysIdentPy is:

```python
from sysidentpy._lib._array_api import get_namespace

def some_operation(X, y):
    xp = get_namespace(X, y)
    # xp is now the correct module: numpy, torch, cupy, jax.numpy, etc.
    result = xp.zeros((n, m), dtype=X.dtype)
    result = xp.linalg.svd(X)
    # ...
```

When `array_api_dispatch` is `False` (the default), `get_namespace()` always returns NumPy, regardless of what the input arrays are. When `True`, it inspects the input arrays and returns the corresponding namespace module. If the inputs are PyTorch tensors, it returns the `torch` module; if they are CuPy arrays, it returns the `cupy` module; and so on.

The function filters out `None` values and scalar arguments, so it is safe to call with optional parameters. If no valid array is found and dispatch is enabled, it falls back to NumPy.

### Device Management

When creating new arrays during computation, SysIdentPy preserves the device of the input arrays:

```python
from sysidentpy._lib._array_api import get_namespace, device, _zeros, _asarray

def some_operation(X, y):
    xp = get_namespace(X, y)
    target_device = device(X, y)  # e.g., cuda:0, cpu

    # New arrays are created on the same device
    buffer = _zeros(xp, (n, m), dtype=X.dtype, target_device=target_device)

    # Scalar constants are elevated to the correct device and dtype
    eps = _asarray(1e-10, xp=xp, dtype=X.dtype, target_device=target_device)
```

The `device()` function extracts the device from input arrays and verifies that all inputs share the same device. If arrays reside on different devices (e.g., one on CPU and another on `cuda:0`), it raises a `ValueError`.

### Operations Not in the Standard

The Array API standard intentionally defines a minimal set of operations. Several operations used in system identification algorithms are not part of the standard. SysIdentPy provides its own implementations:

| Operation | Standard? | SysIdentPy Implementation |
|-----------|-----------|---------------------------|
| `lstsq` (least squares solve) | No | SVD-based: $\theta = V_h^T \cdot \text{diag}(S^{-1}) \cdot U^T \cdot b$, with truncation for numerical stability |
| `diag` (create diagonal matrix) | No | Loop-based construction using `_set_element()` |
| `median` | No | Sort-based: `sort()` then index middle element |
| `nanargmin` | No | Replace NaN with infinity, then `argmin()` |
| `diff` | No | Manual `array[1:] - array[:-1]` |
| `set_element` (mutable assignment) | Varies | Direct indexing for mutable backends (NumPy, PyTorch, CuPy); `.at[idx].set()` for immutable backends (JAX) |
| `concat` / `column_stack` / `vstack` | Partial | Wraps `xp.concat()` with device preservation |
| `copy` | Partial | Uses `xp.asarray(arr, copy=True)` or `arr.copy()` depending on backend |
| `vector_norm` | Partial | Uses `xp.linalg.vector_norm` when available, falls back to `sqrt(sum(x²))` |

The SVD-based `_lstsq()` implementation deserves special attention. NumPy provides `np.linalg.lstsq()` directly, but this function is not part of the Array API standard. For non-NumPy backends, SysIdentPy decomposes the problem via SVD ($A = U \Sigma V^T$), applies a singular-value cutoff for numerical stability, and computes $\theta = V^T \cdot \text{diag}(\sigma_i^{-1}) \cdot U^T \cdot b$ where $\sigma_i^{-1}$ is set to zero for singular values below the cutoff. This approach is robust to rank-deficient regressor matrices and works with any backend that provides SVD.

### NumPy-Only Fallbacks

Some SysIdentPy algorithms depend on SciPy operations that do not support the Array API standard. These algorithms use `_require_numpy_namespace()` to raise a clear `NotImplementedError` when called with non-NumPy backends:

```python
from sysidentpy._lib._array_api import get_namespace, _require_numpy_namespace

class RMSS(OFRBase):
    def fit(self, ...):
        xp = get_namespace(y)
        _require_numpy_namespace(xp, feature="RMSS", dependency="SciPy")
        # ... SciPy-dependent code ...
```

This pattern ensures that users get an informative error message ("RMSS does not support Array API dispatch with namespace 'torch'. This path currently relies on SciPy and requires NumPy inputs.") rather than a cryptic backend error.

### Scalar Index Extraction

A subtle but important pattern in the implementation is how scalar indices are extracted from backend arrays. Many model structure selection algorithms involve finding the best regressor at each step — e.g., `argmax(err_values)` — and using that index to re-order arrays. In NumPy, the result of `argmax()` is a scalar that can be used directly as an index. In PyTorch, it is a 0-dimensional tensor that cannot always be used for Python indexing.

SysIdentPy normalizes these values through `_to_numpy()`:

```python
# Convert backend scalar to Python int for indexing
piv_index = int(_to_numpy(xp.argmax(tmp_err[i:]))) + i
```

The `_to_numpy()` function handles the conversion for each backend:
- NumPy arrays: returned as-is
- PyTorch tensors: `.detach().cpu().numpy()`
- CuPy arrays: `.get()`
- JAX arrays: via DLPack protocol or `__array__` interface
- Other backends: `np.asarray(arr)` fallback

## Enabling Array API Dispatch

### Global Configuration

To enable Array API dispatch for the entire session:

```python
from sysidentpy import set_config

set_config(array_api_dispatch=True)
```

After this call, all SysIdentPy operations will detect the input backend automatically. To check the current configuration:

```python
from sysidentpy import get_config

print(get_config())
# {'array_api_dispatch': True}
```

To disable it:

```python
set_config(array_api_dispatch=False)
```

### Context Manager

For temporary dispatch within a specific block of code:

```python
from sysidentpy import config_context

with config_context(array_api_dispatch=True):
    model.fit(X=X_torch, y=y_torch)
    yhat = model.predict(X=X_test_torch, y=y_test_torch)

# Outside the context, dispatch is back to its previous state
```

The `config_context` context manager saves the current configuration, applies the new one, and restores the original on exit (even if an exception occurs). This is useful for benchmarking, testing, or mixing NumPy and GPU code in the same script.

### Thread Safety

The configuration is stored in thread-local storage, ensuring that each thread can have its own `array_api_dispatch` setting. This is important for applications that use threads for concurrent model fitting.

## Usage Examples

### Basic Usage with PyTorch

```python
import torch
from sysidentpy import config_context
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

# Prepare data as PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float64)
y_train = torch.tensor(y_train_np, dtype=torch.float64)
X_test = torch.tensor(X_test_np, dtype=torch.float64)
y_test = torch.tensor(y_test_np, dtype=torch.float64)

# Fit and predict with Array API dispatch
with config_context(array_api_dispatch=True):
    model = FROLS(
        order_selection=True,
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)

# yhat is a PyTorch tensor
print(type(yhat))  # <class 'torch.Tensor'>
```

### GPU Acceleration with PyTorch CUDA

```python
import torch
from sysidentpy import set_config
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

set_config(array_api_dispatch=True)

# Move data to GPU
device = torch.device("cuda:0")
X_train = torch.tensor(X_train_np, dtype=torch.float64, device=device)
y_train = torch.tensor(y_train_np, dtype=torch.float64, device=device)
X_test = torch.tensor(X_test_np, dtype=torch.float64, device=device)
y_test = torch.tensor(y_test_np, dtype=torch.float64, device=device)

# All computation happens on the GPU
model = FROLS(
    order_selection=True,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(),
    basis_function=Polynomial(degree=2),
)
model.fit(X=X_train, y=y_train)
yhat = model.predict(X=X_test, y=y_test)

# Result is on the same GPU device
print(yhat.device)  # cuda:0
```

### GPU Acceleration with CuPy

```python
import cupy as cp
from sysidentpy import set_config
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

set_config(array_api_dispatch=True)

X_train = cp.asarray(X_train_np)
y_train = cp.asarray(y_train_np)
X_test = cp.asarray(X_test_np)
y_test = cp.asarray(y_test_np)

model = FROLS(
    order_selection=True,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(),
    basis_function=Polynomial(degree=2),
)
model.fit(X=X_train, y=y_train)
yhat = model.predict(X=X_test, y=y_test)

# Result is a CuPy array on the GPU
print(type(yhat))  # <class 'cupy.ndarray'>
```

### Comparing Results Across Backends

```python
import numpy as np
import torch
from sysidentpy import config_context
from sysidentpy._lib._array_api import _to_numpy
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

# Fit with NumPy (baseline)
model_np = FROLS(
    order_selection=True,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(),
    basis_function=Polynomial(degree=2),
)
model_np.fit(X=X_train_np, y=y_train_np)
yhat_np = model_np.predict(X=X_test_np, y=y_test_np)

# Fit with PyTorch
X_train_torch = torch.tensor(X_train_np, dtype=torch.float64)
y_train_torch = torch.tensor(y_train_np, dtype=torch.float64)
X_test_torch = torch.tensor(X_test_np, dtype=torch.float64)
y_test_torch = torch.tensor(y_test_np, dtype=torch.float64)

with config_context(array_api_dispatch=True):
    model_torch = FROLS(
        order_selection=True,
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model_torch.fit(X=X_train_torch, y=y_train_torch)
    yhat_torch = model_torch.predict(X=X_test_torch, y=y_test_torch)

# Convert PyTorch result to NumPy for comparison
yhat_torch_np = _to_numpy(yhat_torch)

# Results should be numerically equivalent (within floating-point tolerance)
np.testing.assert_allclose(yhat_np, yhat_torch_np, rtol=1e-7)
```

### Using with Existing PyTorch Pipelines

```python
import torch
from sysidentpy import config_context
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

# Data already in a PyTorch pipeline (e.g., from a DataLoader)
X_train = some_pytorch_pipeline.get_inputs()  # torch.Tensor on cuda:0
y_train = some_pytorch_pipeline.get_targets()  # torch.Tensor on cuda:0

with config_context(array_api_dispatch=True):
    model = FROLS(
        order_selection=True,
        ylag=2,
        xlag=2,
        estimator=LeastSquares(),
        basis_function=Polynomial(degree=2),
    )
    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_test, y=y_test)

    # yhat is a torch.Tensor on cuda:0 — can be used directly in the pipeline
    loss = torch.nn.functional.mse_loss(yhat, y_target)
```

## Supported Modules and Algorithms

### Model Structure Selection

| Algorithm | Array API Support | Notes |
|-----------|:-:|-------|
| FROLS | ✅ | Full support for all backends |
| AOLS | ✅ | Full support for all backends |
| OFRBase | ✅ | Base class for ERR-based algorithms |
| UOFR | ✅ | Full support for all backends |
| OSF | ✅ | Orthogonal Sequential Floating Forward |
| OIF | ✅ | Orthogonal Insertion-removal Floating |
| OOS / O2S | ✅ | Orthogonal Oscillating Search |
| MetaMSS | ❌ | Requires NumPy (SciPy dependency) |
| Entropic Regression (ER) | ❌ | Requires NumPy (SciPy dependency) |
| RMSS | ❌ | Requires NumPy (SciPy dependency) |

Algorithms marked with ❌ raise `NotImplementedError` with a descriptive message when called with non-NumPy backends. They continue to work normally with NumPy arrays regardless of the `array_api_dispatch` setting.

### Parameter Estimation

| Estimator | Array API Support | Notes |
|-----------|:-:|-------|
| LeastSquares | ✅ | SVD-based fallback for non-NumPy |
| RidgeRegression | ✅ | SVD path for all backends |
| TotalLeastSquares | ✅ | Full SVD support |
| RecursiveLeastSquares | ✅ | Full support |
| AffineLeastMeanSquares | ✅ | Full support |
| LeastMeanSquares | ✅ | Full support |
| NormalizedLeastMeanSquares | ✅ | Full support |
| LeastMeanSquaresSignError | ✅ | Full support |
| NormalizedLeastMeanSquaresSignError | ✅ | Full support |
| LeastMeanSquaresSignRegressor | ✅ | Full support |
| LeastMeanSquaresNormalizedSignRegressor | ✅ | Full support |
| LeastMeanSquaresSignSign | ✅ | Full support |
| LeastMeanSquaresNormalizedSignSign | ✅ | Full support |
| LeastMeanSquaresNormalizedLeaky | ✅ | Full support |
| LeastMeanSquaresLeaky | ✅ | Full support |
| LeastMeanSquaresFourth | ✅ | Full support |
| LeastMeanSquareMixedNorm | ✅ | Full support |
| NonNegativeLeastSquares | ❌ | Requires NumPy (`scipy.optimize.nnls`) |
| BoundedVariableLeastSquares | ❌ | Requires NumPy (`scipy.optimize.lsq_linear`) |
| LeastSquaresMinimalResidual | ❌ | Requires NumPy (`scipy.sparse.linalg.lsmr`) |

### Basis Functions

| Basis Function | Array API Support |
|----------------|:-:|
| Polynomial | ✅ |
| Fourier | ✅ |
| Bilinear | ✅ |
| Bernstein | ✅ |
| Legendre | ✅ |
| Hermite | ✅ |
| Hermite Normalized | ✅ |
| Laguerre | ✅ |

### Other Modules

| Module | Array API Support |
|--------|:-:|
| Simulation (`SimulateNARMAX`) | ✅ |
| Metrics (all regression metrics) | ✅ |
| Utilities (`check_arrays`, `information_matrix`) | ✅ |
| Residual Analysis | ✅ |
| Neural NARX | ❌ (requires PyTorch/NumPy directly) |

## Supported Backends

SysIdentPy's Array API dispatch works with any library that implements the Array API standard. The following backends are explicitly supported and tested through the vendored `array-api-compat` (v1.14.0) layer:

| Backend | Status | Devices | Notes |
|---------|--------|---------|-------|
| **NumPy** | ✅ Fully tested | CPU | Default backend. Always works regardless of dispatch setting. |
| **PyTorch** | ✅ Fully tested | CPU, CUDA (all GPU devices) | Tested in CI and benchmarks. Recommended for GPU acceleration. |
| **CuPy** | ✅ Supported | GPU (NVIDIA) | Requires CUDA-enabled GPU and CuPy installation. |
| **JAX** | ✅ Supported | CPU, GPU, TPU | Supports immutable arrays via `.at[idx].set()` pattern. |
| **array_api_strict** | ✅ Tested in CI | CPU | Reference implementation for spec conformance testing. |
| **Dask** | Experimental | Distributed | Via `array-api-compat` support. Not yet tested with SysIdentPy. |

## Implementation Design Decisions

### Vendoring vs. Runtime Dependency

SysIdentPy vendors `array-api-compat` and `array-api-extra` instead of declaring them as runtime dependencies. This decision follows the approach used by scikit-learn and ensures:

- **No new runtime dependency**: Users who do not use Array API dispatch have zero additional packages to install.
- **Version stability**: The vendored version is tested with SysIdentPy and will not break due to upstream updates.
- **Self-contained distribution**: The package works in environments where installing additional packages is restricted.

The vendored versions are tracked in `sysidentpy/_lib/_vendor/VENDORED_VERSIONS`.

### Opt-in Dispatch

Array API dispatch is disabled by default (`array_api_dispatch=False`). This design decision guarantees:

- **Backward compatibility**: Existing code that passes NumPy arrays continues to work identically. No behavior change, no performance change.
- **Explicit control**: Users must consciously enable dispatch. This avoids surprises when, for example, a function receives a PyTorch tensor accidentally.
- **NumPy fast-paths**: When dispatch is disabled, `get_namespace()` returns NumPy immediately without inspecting array types, preserving zero-overhead for NumPy-only usage.

### Thread-Local Configuration

The `array_api_dispatch` flag is stored in Python's `threading.local()` storage. Each thread has its own copy of the configuration, which enables:

- **Parallel model fitting**: Different threads can independently enable or disable dispatch.
- **Worker isolation**: A web server or parallel processing system can run NumPy-based and GPU-based model fitting concurrently without interference.

### Metadata as NumPy Arrays

SysIdentPy's model structure selection algorithms produce metadata (regressor codes, pivot indices, information criteria values) that are used for model interpretation, equation formatting, and indexing into Python data structures. This metadata is always stored as NumPy arrays, even when the computational backend is different:

```python
# At the end of fit(), regardless of backend:
self.pivv = np.asarray(_to_numpy(self.pivv), dtype=np.intp)
self.final_model = self.regressor_code[self.pivv]
```

This ensures that model attributes like `final_model`, `pivv`, `err`, and `info_values` are always NumPy arrays, which simplifies post-fitting analysis, equation formatting, and serialization.

## Performance Considerations

### When GPU Acceleration Helps

GPU acceleration is most beneficial for:

- **Large regressor matrices**: When the number of candidate regressors is large (high polynomial degrees, many lags), the matrix multiplication and SVD operations in ERR computation and parameter estimation benefit significantly from GPU parallelism.
- **Long time series**: Building and manipulating large lagged matrices transfers more computation to parallel hardware.
- **Batch operations**: When fitting multiple models or running cross-validation, GPU batch processing amortizes the overhead of data transfer.

### When GPU Acceleration Does Not Help

For small problems (few lags, low degree, short time series), the overhead of GPU kernel launches and CPU-GPU memory copies may outweigh the computational benefit. SysIdentPy includes an internal fast-path for Polynomial NARMAX prediction that automatically decides whether to use the GPU or fall back to CPU based on the problem size and backend characteristics.

### Numerical Equivalence

The Array API dispatch is designed to produce numerically equivalent results to the NumPy path. Small floating-point differences (on the order of $10^{-7}$ to $10^{-8}$) are expected due to:

- Different operation ordering and fused multiply-add (FMA) behavior across backends
- SVD-based least-squares implementation vs. QR-based NumPy implementation
- Different floating-point precision handling (especially when using `float32`)

For the standard `float64` case, the cross-backend differences are comparable to the numerical noise inherent in the algorithms themselves and do not affect model selection or interpretation.

## Limitations and Known Constraints

### SciPy-Dependent Algorithms

Algorithms that rely on SciPy operations (sparse solvers, constrained optimization, mutual information computation) currently require NumPy inputs:

- **MetaMSS**: Uses SciPy for random split generation and model evaluation.
- **Entropic Regression (ER)**: Uses SciPy for mutual information computation.
- **RMSS**: Uses SciPy for statistical aggregation with resampling.
- **NonNegativeLeastSquares**: Wraps `scipy.optimize.nnls`.
- **BoundedVariableLeastSquares**: Wraps `scipy.optimize.lsq_linear`.
- **LeastSquaresMinimalResidual**: Wraps `scipy.sparse.linalg.lsmr`.
- **Neural NARX**: Uses PyTorch directly (not through Array API).
- **Multiobjective Parameter Estimation (AILS)**: Uses SciPy for the augmented inverse least-squares computation.

These algorithms raise `NotImplementedError` with specific messages indicating which dependency prevents Array API support. As SciPy expands its own Array API support, these restrictions will be progressively removed.

### All Input Arrays Must Share the Same Backend and Device

SysIdentPy requires that all input arrays (`X`, `y`) use the same backend and reside on the same device. Mixing backends (e.g., NumPy array for `X` and PyTorch tensor for `y`) or devices (e.g., `X` on `cuda:0` and `y` on `cuda:1`) raises a `ValueError`:

```python
# This will raise ValueError:
model.fit(X=numpy_array, y=torch_tensor)

# This will also raise ValueError:
model.fit(X=tensor_on_cuda0, y=tensor_on_cuda1)
```

### Immutable Arrays (JAX)

JAX arrays are immutable (they do not support in-place modification like `arr[idx] = val`). SysIdentPy handles this through the `_set_element()` function, which detects immutable backends and uses the JAX `.at[idx].set()` pattern to create modified copies:

```python
def _set_element(xp, arr, idx, val):
    try:
        arr[idx] = val  # Works for NumPy, PyTorch, CuPy
        return arr
    except (RuntimeError, TypeError, ValueError):
        # JAX and other immutable backends
        return arr.at[idx].set(val)
```

This is transparent to the user but has a minor performance implication: each element assignment in JAX creates a new array copy. For algorithms with many element-level operations per iteration, this overhead can add up.

## Technical Reference

### Configuration API

```python
sysidentpy.set_config(*, array_api_dispatch=None)
```
Set the global `array_api_dispatch` flag. Pass `True` to enable, `False` to disable.

```python
sysidentpy.get_config() -> dict
```
Returns a dictionary with the current configuration: `{'array_api_dispatch': bool}`.

```python
sysidentpy.config_context(*, array_api_dispatch=None)
```
Context manager that temporarily changes the configuration. Restores the previous setting on exit.

### Internal Utilities (for developers)

The following functions are available in `sysidentpy._lib._array_api` for internal use:

| Function | Purpose |
|----------|---------|
| `get_namespace(*arrays)` | Returns the array namespace (`xp`) for the given arrays |
| `_is_numpy_namespace(xp)` | Returns `True` if `xp` is NumPy |
| `_require_numpy_namespace(xp, *, feature, dependency)` | Raises `NotImplementedError` if `xp` is not NumPy |
| `device(*arrays)` | Returns the shared device of the input arrays |
| `_to_numpy(arr)` | Converts any array to NumPy on CPU |
| `_asarray(data, *, xp, dtype, target_device)` | Converts data to the given namespace with device preservation |
| `_zeros(xp, shape, *, dtype, target_device)` | Creates a zero array with device preservation |
| `_ones(xp, shape, *, dtype, target_device)` | Creates a ones array with device preservation |
| `_full(xp, shape, fill_value, *, dtype, target_device)` | Creates a filled array with device preservation |
| `_lstsq(xp, A, b)` | Solves $A\theta \approx b$ via SVD (backend-agnostic) |
| `_diag(xp, v)` | Creates a diagonal matrix from a vector |
| `_set_element(xp, arr, idx, val)` | Sets element at index, handling immutable arrays |
| `_copy(xp, arr)` | Creates a copy with device preservation |
| `_concat(xp, arrays, axis)` | Concatenates arrays with device preservation |
| `_column_stack(xp, arrays)` | Stacks arrays as columns |
| `_vstack(xp, arrays)` | Vertical stack with device preservation |
| `_hstack(xp, arrays)` | Horizontal stack with device preservation |
| `_median(xp, a, axis)` | Computes median (sort-based for non-NumPy) |
| `_nanargmin(xp, a, axis)` | Argmin ignoring NaN (mask-based for non-NumPy) |
| `_vector_norm(xp, x)` | Computes vector L2 norm |
| `_pow(xp, x1, x2)` | Element-wise power |
| `_einsum_ij_ij_j(xp, A, B)` | Optimized einsum pattern |

## FAQ

**Q: Do I need to install `array-api-compat` or `array-api-extra`?**

No. These packages are vendored (bundled) inside SysIdentPy. You do not need to install them separately.

**Q: Do I need to install PyTorch/CuPy/JAX to use SysIdentPy?**

No. These are optional. SysIdentPy works with NumPy only (its only required array dependency). You only need to install other backends if you want to use them.

**Q: What happens if I enable dispatch but pass NumPy arrays?**

SysIdentPy will detect the NumPy namespace and use NumPy operations. The overhead of namespace detection is negligible.

**Q: Can I use `float32` arrays?**

Yes. SysIdentPy preserves the dtype of input arrays. However, system identification algorithms can be numerically sensitive. Using `float32` may reduce accuracy, especially for model structure selection algorithms that rely on numerical precision in the ERR computation. We recommend `float64` for most applications.

**Q: How do I get my results back as NumPy arrays?**

Use `_to_numpy()` from the internal API, or simply call `.numpy()` (PyTorch), `.get()` (CuPy), or `np.asarray()` depending on your backend:

```python
from sysidentpy._lib._array_api import _to_numpy

yhat_numpy = _to_numpy(yhat)
```

**Q: Why are some algorithms NumPy-only?**

These algorithms depend on SciPy functions (constrained optimization, sparse linear algebra, nearest-neighbor statistics) that do not yet support the Array API standard. As SciPy adds Array API support for these functions, SysIdentPy will progressively remove the restrictions.

**Q: Is the numerical result identical between backends?**

It is numerically equivalent, not bitwise identical. Small floating-point differences (typically $< 10^{-7}$ in relative terms for `float64`) are expected due to different backend implementations of the same mathematical operations. These differences do not affect model correctness.

**Q: Can I mix NumPy and GPU arrays in the same model?**

No. All input arrays to a single `fit()` or `predict()` call must use the same backend and device. Convert them to the same backend before calling SysIdentPy functions.
