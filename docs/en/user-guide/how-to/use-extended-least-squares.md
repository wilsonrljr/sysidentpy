# Use Extended Least Squares

Example created by Wilson Rocha Lacerda Junior

> **Looking for more details on NARMAX models?**
> For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in SysIdentPy, check out our book:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> This book provides in-depth guidance to support your work with SysIdentPy.

To use the **Extended Least Squares (ELS)** algorithm, set the `unbiased` parameter to `True` when defining the parameter estimator algorithm.


```python
from sysidentpy.parameter_estimation import LeastSquares

estimator = LeastSquares(unbiased=True)
```


The `unbiased` hyperparameter is available in all parameter estimation algorithms, with a default value of `False`.

Additionally, the **Extended Least Squares** algorithm is iterative. In **SysIdentPy**, the default number of iterations is set to 20 (`uiter=20`), as studies in the literature indicate that the algorithm typically converges within 10 to 20 iterations. However, you can adjust this value to any number of iterations you prefer.



```python
from sysidentpy.parameter_estimation import LeastSquares

estimator = LeastSquares(unbiased=True, uiter=40)
```

A simple yet complete code example demonstrating parameter estimation using the **Extended Least Squares (ELS)** algorithm is shown below.

*(Simulated data is used for illustrative purposes.)*



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.generate_data import get_siso_data

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=True, sigma=0.2, train_percentage=90
)

basis_function = Polynomial(degree=2)
estimator = LeastSquares(unbiased=True)
parameters = np.zeros([3, 50])

for i in range(50):
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=3000, colored_noise=True, train_percentage=90
    )

    model = FROLS(
        order_selection=False,
        n_terms=3,
        ylag=2,
        xlag=2,
        elag=2,
        info_criteria="aic",
        estimator=estimator,
        basis_function=basis_function,
    )

    model.fit(X=x_train, y=y_train)
    parameters[:, i] = model.theta.flatten()

plt.figure(figsize=(14, 4))

# Compute and plot KDE for each parameter using scipy's gaussian_kde
x_grid = np.linspace(np.min(parameters), np.max(parameters), 1000)

for i, label in enumerate(["Parameter 1", "Parameter 2", "Parameter 3"]):
    kde = gaussian_kde(parameters[i, :])
    plt.plot(x_grid, kde(x_grid), label=label)

# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color="k", linestyle="--", label="Real Value 0.1")
plt.axvline(x=0.2, color="k", linestyle="--", label="Real Value 0.2")
plt.axvline(x=0.9, color="k", linestyle="--", label="Real Value 0.9")

plt.xlabel("Parameter Value")
plt.ylabel("Density")
plt.title("Kernel Density Estimate of Parameters (Matplotlib only)")
plt.legend()
plt.show()
```
