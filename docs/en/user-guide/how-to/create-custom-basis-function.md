# Create a Custom Basis Function

> This walkthrough mirrors the example provided in `examples/custom-basis-function.ipynb` and shows how easy it is to plug your own feature generator into SysIdentPy.

In this how-to we extend `BaseBasisFunction` to create a harmonic feature map powered only by NumPy. The new class works exactly like the built-in basis functions, so it can be reused across any estimator that expects the same interface.

## Requirements

You can reuse the project environment or install a minimal set of packages:

```
sysidentpy
numpy
matplotlib
```

```bash
pip install -r requirements.txt
```

- The example runs entirely on CPU.
- No additional datasets are required.

## Generate a synthetic dataset

We build a simple SISO system with a strong sinusoidal component driven by the input. The first 1600 samples are used for training and the remainder for validation.

```python
import numpy as np
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.plotting import plot_results
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function.basis_function_base import BaseBasisFunction

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=False, sigma=0.0001, train_percentage=50
)
```

## Implement the custom basis function

The new `HarmonicBasis` class only needs to implement `fit` and `transform`. Internally we create a matrix that contains the raw signals plus sine/cosine transforms for the requested harmonics. Because the class inherits from `BaseBasisFunction`, SysIdentPy can use it just like any built-in option.

```python
class HarmonicBasis(BaseBasisFunction):
    """Map lagged regressors to sine/cosine features."""

    def __init__(self, harmonics=(1,), include_linear=True, scale=np.pi):
        super().__init__(degree=1)
        self.harmonics = tuple(harmonics)
        self.include_linear = include_linear
        self.scale = scale

    def _build_matrix(self, data, predefined_regressors):
        features = []
        if self.include_linear:
            features.append(data)
        for harmonic in self.harmonics:
            scaled = self.scale * harmonic * data
            features.append(np.sin(scaled))
            features.append(np.cos(scaled))
        if not features:
            raise ValueError("The basis needs at least one active transformation.")
        psi = np.hstack(features)
        if predefined_regressors is not None:
            idx = np.asarray(predefined_regressors, dtype=int)
            psi = psi[:, idx]
        return psi

    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        psi = self._build_matrix(data, predefined_regressors)
        return psi[max_lag:, :]

    def transform(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        return self.fit(data, max_lag, ylag, xlag, model_type, predefined_regressors)
```

## Train with the custom basis

The workflow is identical to every other example. We simply pass an instance of `HarmonicBasis` to `FROLS` and proceed with training, evaluation, and plotting.

```python
basis_function = HarmonicBasis(harmonics=(1, 2, 3), include_linear=True, scale=np.pi)

model = FROLS(
    ylag=2,
    xlag=2,
    order_selection=True,
    n_info_values=20,
    info_criteria="aic",
    estimator=LeastSquares(),
    basis_function=basis_function,
    model_type="NARX",
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

rrse = root_relative_squared_error(y_valid[model.max_lag:], yhat[model.max_lag:])
print(f"RRSE (validation): {rrse:.4f}")
```

```python
plot_results(
    y=y_valid[model.max_lag:],
    yhat=yhat[model.max_lag:],
    n=400,
    figsize=(12, 4),
    title="Validation results with HarmonicBasis",
)
```

## Wrap up

With only a few lines of code we built a drop-in replacement for the stock basis functions. Any NumPy/SciPy/Scikit-Learn transformation can be exported to a class like `HarmonicBasis`, enabling you to reuse bespoke feature maps across every SysIdentPy estimator without touching the rest of your workflow.
