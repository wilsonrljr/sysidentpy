# Set Specific Lags

Example created by Wilson Rocha Lacerda Junior

> **Looking for more details on NARMAX models?**
> For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in SysIdentPy, check out our book:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> This book provides in-depth guidance to support your work with SysIdentPy.

Different ways to set the maximum lag for input and output


```python
pip install sysidentpy
```


```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
```

## Setting lags using a range of values

If you pass int values for *ylag* and *xlag*, the lags are defined as a range from 1-*ylag* and 1-*xlag*. 

For example: if *ylag=4* then the candidate regressors are $y_{k-1}, y_{k-2}, y_{k-3}, y_{k-4}$


```python
basis_function = Polynomial(degree=1)

model = FROLS(
    order_selection=True,
    ylag=4,
    xlag=4,
    info_criteria="aic",
    basis_function=basis_function,
)
```

## Setting specific lags using lists

If you pass the *ylag* and *xlag* as a list, only the lags related to values in the list will be created.
$y_{k-1}, y_{k-4}$,  $x_{k-1}, x_{k-4}$


```python
model = FROLS(
    order_selection=True,
    ylag=[1, 4],
    xlag=[1, 4],
    info_criteria="aic",
    basis_function=basis_function,
)
```

## Setting lags for Multiple Input Single Output (MISO) models

The following example shows how to define specific lags for each input. One should notice that we have to use a nested list in that case.


```python
# The example considers a model with 2 inputs, but you can use the same for any amount of inputs.

model = FROLS(
    order_selection=True,
    ylag=[1, 4],
    xlag=[[1, 2, 3, 4], [1, 7]],
    info_criteria="aic",
    basis_function=basis_function,
)
# The lags defined are:
# x1(k-1), x1(k-2), x(k-3), x(k-4)
# x2(k-1), x1(k-7)
```


```python

```
