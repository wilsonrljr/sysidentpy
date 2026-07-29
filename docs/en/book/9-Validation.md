## The `predict` Method in SysIdentPy

Before discussing validation, it is important to understand what the `predict` method computes. In dynamic systems, the predicted output may depend on past values of the output itself. Therefore, two predictions made by the same model can be very different depending on which values are fed back into the prediction loop.

A typical use of `predict` in SysIdentPy is

```python
yhat = model.predict(X=x_test, y=y_test)
```

The explanations below use FROLS with a polynomial basis, which is also the configuration adopted throughout the rest of the chapter. This path provides free run, one-step-ahead, and $n$-step-ahead predictions for NAR and NARMAX models. NFIR models do not feed the output back into the model; the distinction between these modes therefore does not have the same meaning, and $n$-step-ahead prediction is not available through this path.

Two questions come up frequently:

1. Why do we need to pass `y_test` to predict the output?
2. Why are the first values of `yhat` equal to the first values of `y_test`?

The answer lies in the initial conditions. Consider the model

$$
y_k = y_{k-1} + 2x_{k-1}.
\tag{9.1}
$$

To calculate $\hat{y}_1$, the model must know $y_0$. If the largest model lag is $n_{\text{lag}}$, at least $n_{\text{lag}}$ output samples are required to start the recursion. In SysIdentPy, this number is available as `model.max_lag` once the model has been configured.

```python
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=False,
    n_terms=15,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.max_lag
```

For the NARMAX model in this example, `max_lag` is obtained from the lags configured in `xlag` and `ylag`, not only from the terms retained in the final model. If `xlag=ylag=10`, for example, ten initial conditions are still required even if the largest lag among the selected regressors is smaller.

### Infinity-Step-Ahead Prediction

Infinity-step-ahead prediction, also called *free run simulation*, uses previously **predicted** outputs to continue the recursion. In `predict`, this is the behavior obtained when `steps_ahead=None`, which is the default.

Consider

$$
x_{test} = [1, 2, 3, 4, 5, 6, 7]
$$

and

$$
y_{test} = [8, 9, 10, 11, 12, 13, 14].
$$

For the model in Equation 9.1, with a maximum lag of 1, the simulation starts with $y_0=8$:

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*14 + 2*3 = 20
yhat(4) = 1*20 + 2*4 = 28
```

After the initial condition, no other observed value from `y_test` is used. An error made at one sample influences the samples that follow. This propagation is exactly what makes free run simulation an important test for dynamic models.

When `X` is provided, its number of samples defines the simulation horizon. Consequently, passing the complete output or only the initial conditions produces the same result:

```python
yhat = model.predict(X=x_test, y=y_test)
yhat_from_initial_conditions = model.predict(
    X=x_test,
    y=y_test[: model.max_lag],
)
```

In this case, a value supplied through `forecast_horizon` does not replace the length of `X`: `X.shape[0]` determines the number of samples returned, including the initial-condition prefix.

For NAR models, which have no input `X`, the horizon must be provided through `forecast_horizon`:

```python
yhat = model.predict(
    X=None,
    y=y_initial,
    forecast_horizon=100,
)
```

In this example, the result contains the initial conditions followed by 100 predicted values. Thus, `forecast_horizon` controls prediction length when `X` is absent; it does not change which outputs are fed back. The latter choice is controlled by `steps_ahead`.

### One-Step-Ahead Prediction

In one-step-ahead prediction, the model uses the previous **observed** output for every new prediction. For the same example,

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*9 + 2*2 = 13
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*11 + 2*4 = 19
```

The error is not propagated indefinitely because the recursion is corrected by the observed output at every sample. In SysIdentPy, this option is selected with `steps_ahead=1`:

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=1)
```

In this case, `y_test` must contain the complete interval to be evaluated. One-step-ahead results are usually better than free run results, but they answer a less demanding question: how well does the model predict the next sample when the true output history is known?

### n-Step-Ahead Prediction

An $n$-step-ahead prediction lies between the two previous cases. Within each block, the model uses its own predictions. After `steps_ahead` samples, the recursion is restarted with observed values and a new block is calculated.

With `steps_ahead=2`, we have

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*16 + 2*4 = 24
```

In SysIdentPy:

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=2)
```

`steps_ahead` must be a positive built-in Python `int`; NumPy integer scalars are not accepted by the current validator. As its value increases, the test approaches free run simulation and gives errors more opportunity to propagate.

### Aligning the Output Before Evaluation

SysIdentPy copies the first `model.max_lag` samples from `y` to the beginning of `yhat` because they are initial conditions, not predictions. Including them in a metric artificially reduces the error. The comparison should start after this prefix:

```python
start = model.max_lag
y_eval = y_test[start:]
yhat_eval = yhat[start:]
```

The same alignment must be used for metrics and residual correlations. This rule is particularly important when comparing models with different `max_lag` values.

## Model Performance and Validation

A metric summarizes the distance between the observed and predicted outputs. This is necessary, but it is not sufficient to validate a dynamic model. Two models may have similar numerical errors while representing very different dynamics.

Whenever possible, performance should be measured on data that was not used for parameter estimation or structure selection. Error on the estimation set shows how well the model fitted the data used to build it; error on a separate validation set provides evidence about its ability to generalize. This separation, however, does not replace residual analysis or the need to identify the prediction mode being used.

Define the residual as

$$
e_k = y_k - \hat{y}_k.
\tag{9.2}
$$

If the model has explained all deterministic information available in the data, it should not be possible to predict $e_k$ from previous residuals or past inputs. In practical terms, we look for residuals that are small and free of systematic patterns. A low RRSE accompanied by strongly correlated residuals indicates that some structure remains unexplained. This combination of numerical performance and residual analysis is central to NARMAX model validity tests ([Billings and Voon, 1983](https://doi.org/10.1049/ip-d.1983.0034)).

This distinction also explains why the prediction mode must be reported alongside any metric. An error computed for a one-step-ahead prediction is not directly comparable to the same error computed for a free run simulation.

### Residual Correlation in SysIdentPy

SysIdentPy provides two public functions for this analysis:

```python
from sysidentpy.residues.residues_correlation import (
    compute_cross_correlation,
    compute_residues_autocorrelation,
)
from sysidentpy.utils.plotting import plot_residues_correlation
```

The arguments must already be aligned and represent one signal per call. If `X`
contains several input columns, call `compute_cross_correlation` separately for
each input instead of passing the complete input matrix as `arr`.

`compute_residues_autocorrelation(y, yhat)` computes the normalized autocorrelation of the residuals and returns the $N$ non-negative lags. The implementation correlates $e$ directly with itself, without subtracting the residual mean, and normalizes the result by its value at lag zero. When residual energy is nonzero, this first value is 1; the remaining lags are the ones to inspect. Relevant values away from zero suggest that previous residuals still carry information about the current residual.

`compute_cross_correlation(y, yhat, arr)` computes the cross-correlation between the residuals and another aligned signal, usually the input used to excite the system, and returns the first $\lfloor N/2\rfloor$ non-negative lags. Relevant correlations suggest that part of the input effect has not been captured by the model. These two checks look for specific linear dependencies; they do not exhaust every pattern that may remain in the residuals.

Both functions return a tuple in the order `correlation, upper_bound, lower_bound`. For a residual series with $N$ samples, the implementation uses the approximate bounds

$$
\pm\frac{1.96}{\sqrt{2N-1}}.
\tag{9.3}
$$

These bounds are the nominal approximate 95% visual reference returned by the implementation. A few points may fall outside the interval by chance, particularly when many lags are examined. The plots are therefore diagnostic tools: by themselves, they do not prove that a model is valid or invalid.

`plot_residues_correlation(data=...)` accepts the tuple returned by either function directly, plots the correlation, and shades the bounds. By default, it displays the first 100 lags; the `n` argument changes that number. If every residual is zero, autocorrelation normalization involves a $0/0$ division and the implementation returns a vector containing `NaN`. Cross-correlation is also undefined when either normalized signal has zero energy, but in this case the current implementation raises `ZeroDivisionError`. These outcomes represent a normalized correlation that cannot be calculated, not the detection of correlation.

## Metrics Available in SysIdentPy

The `sysidentpy.metrics` module provides 13 public functions for regression and forecasting errors. All of them compare observed values, `y`, with predicted values, `yhat`. `forecast_error` returns the error at every sample; the remaining functions return a scalar. MASE is the only metric in this module that also requires training data and, optionally, a seasonal period.

| Group | Functions | Unit or scale | Best value | Main characteristic |
|---|---|---|---:|---|
| Signed error | `forecast_error`, `mean_forecast_error` | Same unit as $y$ | 0 | Show error direction and average bias |
| Squared error | `mean_squared_error`, `root_mean_squared_error` | $y^2$ for MSE; unit of $y$ for RMSE | 0 | Penalize large errors more heavily |
| Normalized squared error | `normalized_root_mean_squared_error`, `root_relative_squared_error` | Dimensionless | 0 | Facilitate scale comparisons, but use different references |
| Absolute error | `mean_absolute_error`, `median_absolute_error` | Same unit as $y$ | 0 | Have direct interpretations and less influence from extreme errors |
| Scaled absolute error | `mean_absolute_scaled_error` | Dimensionless | 0 | Compares MAE with a naive forecast computed on training data |
| Logarithmic and percentage error | `mean_squared_log_error`, `symmetric_mean_absolute_percentage_error` | Squared logarithmic difference; percentage | 0 | Assess relative differences under their own restrictions |
| Goodness of fit | `explained_variance_score`, `r2_score` | Dimensionless | 1 | Compare the error with output variability |

### Forecast Error and Mean Forecast Error

`forecast_error` implements Equation 9.2 directly. For a single output, it returns the vector $[e_1,\ldots,e_N]$; more generally, the `y - yhat` operation preserves the resulting array shape. Because SysIdentPy defines $e_k=y_k-\hat{y}_k$, a positive error indicates underprediction and a negative error indicates overprediction.

`mean_forecast_error` calculates

$$
\mathrm{MFE}=\frac{1}{N}\sum_{k=1}^{N}e_k.
\tag{9.4}
$$

MFE, expressed in the same unit as $y$, is useful for detecting average bias, but positive and negative errors can cancel each other. An MFE close to zero does not, by itself, imply accurate predictions.

### MSE and RMSE

The mean squared error is

$$
\mathrm{MSE}=\frac{1}{N}\sum_{k=1}^{N}(y_k-\hat{y}_k)^2,
\tag{9.5}
$$

and its square root is

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\tag{9.6}
$$

MSE is expressed in the squared unit of $y$. RMSE returns to the original output unit and is usually easier to interpret. Because both rely on squared errors, large errors receive substantial weight.

### NRMSE and RRSE

In SysIdentPy, `normalized_root_mean_squared_error` normalizes RMSE by the observed output range:

$$
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\max(y)-\min(y)}.
\tag{9.7}
$$

This definition makes the result dimensionless, but also makes it sensitive to extreme values that widen the range. If `y` is constant, the denominator is zero. The implementation returns 0 for a perfect prediction and `inf` for an imperfect one; an error containing `NaN` remains `NaN`.

`root_relative_squared_error` uses the output mean as its reference:

$$
\mathrm{RRSE}=
\sqrt{
\frac{\sum_{k=1}^{N}(y_k-\hat{y}_k)^2}
{\sum_{k=1}^{N}(y_k-\bar{y})^2}
}.
\tag{9.8}
$$

For a non-constant output, an RRSE below 1 means that the model outperforms the constant prediction $\hat{y}_k=\bar{y}$ on the same evaluation set. An RRSE equal to 1 indicates equivalent performance, and an RRSE above 1 indicates worse performance. For a constant output, the behavior is the same as NRMSE: 0 for a perfect prediction, `inf` for an imperfect one, and `NaN` when the error is not finite.

Although both are called normalized metrics, NRMSE and RRSE are not interchangeable. The former divides by the output range; the latter compares the squared error sum with variability around the mean.

### MAE, Median Absolute Error, and MASE

The mean absolute error is

$$
\mathrm{MAE}=\frac{1}{N}\sum_{k=1}^{N}|y_k-\hat{y}_k|.
\tag{9.9}
$$

`median_absolute_error` replaces the mean with the median of the absolute errors. MAE and median absolute error retain the unit of `y`, but the median is less affected by a small number of extreme errors. This robustness concerns aggregation: an extreme value still produces a large absolute error, but it has less influence on the sample median.

`mean_absolute_scaled_error` normalizes MAE using the error of a naive forecast computed on the training data. The implementation generalizes the scale proposed by [Hyndman and Koehler (2006)](https://doi.org/10.1016/j.ijforecast.2006.03.001) to a configurable seasonal period:

$$
\mathrm{MASE}=
\frac{\frac{1}{N}\sum_{k=1}^{N}|y_k-\hat{y}_k|}
{\frac{1}{T-m}\sum_{t=m+1}^{T}|y^{\mathrm{train}}_t-y^{\mathrm{train}}_{t-m}|},
\tag{9.10}
$$

where $m$ is `seasonal_period`. The default is $m=1$, corresponding to a one-step naive forecast. A MASE below 1 means that the model MAE is lower than the average naive error within the training set; a MASE equal to 1 indicates equivalent performance.

Its signature differs from the other metrics:

```python
mean_absolute_scaled_error(
    y,
    yhat,
    y_train,
    seasonal_period=1,
)
```

MASE accepts a single output only, as a one-dimensional vector or a matrix with one column. `y` and `yhat` must have the same shape, `seasonal_period` must be a positive integer, and `y_train` must contain more samples than the period. If the naive error is zero, the implementation returns 0 for a perfect prediction, `inf` for an imperfect one, and preserves `NaN`.

### MSLE and SMAPE

The mean squared logarithmic error is

$$
\mathrm{MSLE}=\frac{1}{N}\sum_{k=1}^{N}
\left[\log(1+y_k)-\log(1+\hat{y}_k)\right]^2.
\tag{9.11}
$$

MSLE reduces the influence of absolute differences at large values and emphasizes relative differences. In the SysIdentPy implementation, every value in `y` and `yhat` must be strictly greater than -1. Otherwise, the function raises `ValueError` because `log1p` is not defined in the real domain.

The symmetric mean absolute percentage error is calculated as

$$
\mathrm{SMAPE}=\frac{100}{N}\sum_{k=1}^{N}
\frac{2|y_k-\hat{y}_k|}{|y_k|+|\hat{y}_k|}.
\tag{9.12}
$$

For a single output and finite values, the result ranges from 0% to 200%. When $y_k=\hat{y}_k=0$, that sample contributes zero. Despite its name, overpredictions and underpredictions of the same magnitude are not necessarily penalized equally.

### Explained Variance and R²

`explained_variance_score` calculates

$$
\mathrm{EVS}=1-\frac{\mathrm{Var}(y-\hat{y})}{\mathrm{Var}(y)},
\tag{9.13}
$$

whereas the coefficient of determination is

$$
R^2=1-
\frac{\sum_{k=1}^{N}(y_k-\hat{y}_k)^2}
{\sum_{k=1}^{N}(y_k-\bar{y})^2}.
\tag{9.14}
$$

The best value for both is 1, and negative results are possible. Their main difference appears in the presence of a constant offset. If $\hat{y}=y+c$, the residual variance is zero and EVS can be 1 even when $c\neq0$. $R^2$ includes this bias in its squared error sum and will be lower than 1.

For a constant output, $R^2$ returns 1 when the prediction is perfect and 0 when there is an error. With EVS, a constant-offset prediction still returns 1 because the residual variance is zero; if the residuals vary, the result is 0. This behavior follows directly from the difference between the two definitions, but reinforces why EVS should not be used alone to detect bias.

### Shapes, Aggregation, and Missing Values

Except for MASE, the metrics do not perform common shape validation. In practice, `y` and `yhat` should have the same shape and temporal alignment; otherwise, backend broadcasting rules may produce an unintended result or raise an error. Scalar metrics generally aggregate all supplied elements. Two exceptions deserve attention: `r2_score` computes $R^2$ for each output column and returns their mean; SMAPE sums all columns but divides only by the number of samples, `y.shape[0]`. For $q$ outputs, its theoretical range therefore becomes 0% to $200q$%. MASE explicitly rejects multiple outputs.

The functions do not remove or impute missing values either. For a finite, non-constant target, a `NaN` in `yhat` appears at the corresponding position in `forecast_error` and propagates as `NaN` through the scalar metrics. NRMSE, RRSE, and MASE also preserve `NaN` when their respective divisors are zero. EVS and $R^2$ have an exception defined by the internal function used for constant targets: if target variance is zero and the numerator is nonzero or `NaN`, the returned result is 0. This does not represent missing-value handling; it is only the result of the implemented zero-denominator safety rule.

### Usage Example

```python
import numpy as np

from sysidentpy.metrics import (
    explained_variance_score,
    forecast_error,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_forecast_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    normalized_root_mean_squared_error,
    r2_score,
    root_mean_squared_error,
    root_relative_squared_error,
    symmetric_mean_absolute_percentage_error,
)

y = np.array([3.0, -0.5, 2.0, 7.0])
yhat = np.array([2.5, 0.0, 2.0, 8.0])
y_train = np.array([1.0, 2.0, 3.0, 4.0])

errors = forecast_error(y, yhat)
bias = mean_forecast_error(y, yhat)
mse = mean_squared_error(y, yhat)
rmse = root_mean_squared_error(y, yhat)
nrmse = normalized_root_mean_squared_error(y, yhat)
rrse = root_relative_squared_error(y, yhat)
mae = mean_absolute_error(y, yhat)
median_ae = median_absolute_error(y, yhat)
mase = mean_absolute_scaled_error(y, yhat, y_train)
msle = mean_squared_log_error(y, yhat)
smape = symmetric_mean_absolute_percentage_error(y, yhat)
evs = explained_variance_score(y, yhat)
r2 = r2_score(y, yhat)
```

These metrics also participate in SysIdentPy's Array API support when dispatch is enabled. Regardless of the backend, metric selection should be guided by the evaluation question, not merely by the convenience of obtaining a single number.

## Case Study: Electromechanical System

Let us revisit the electromechanical system introduced in Chapter 4. The goal here is not to find an optimal configuration, but to show how a conclusion changes when we examine free run simulation, one-step-ahead prediction, and residual correlations.

The data are loaded from a specific commit in the `sysidentpy-data` repository, ensuring that the example always uses the same version of the signals.

```python
import numpy as np
import pandas as pd

from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.residues.residues_correlation import (
    compute_cross_correlation,
    compute_residues_autocorrelation,
)
from sysidentpy.utils.plotting import (
    plot_residues_correlation,
    plot_results,
)

data_url = (
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/"
    "4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/generator/"
)
df1 = pd.read_csv(f"{data_url}x_cc.csv")
df2 = pd.read_csv(f"{data_url}y_cc.csv")

x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)
```

To avoid repeating the alignment, we define a small evaluation function:

```python
def evaluate(model, *, steps_ahead=None):
    yhat = model.predict(
        X=x_valid,
        y=y_valid,
        steps_ahead=steps_ahead,
    )
    start = model.max_lag
    y_eval = y_valid[start:]
    yhat_eval = yhat[start:]
    x_eval = x_valid[start:]

    rrse = root_relative_squared_error(y_eval, yhat_eval)
    ee = compute_residues_autocorrelation(y_eval, yhat_eval)
    x1e = compute_cross_correlation(y_eval, yhat_eval, x_eval)
    return y_eval, yhat_eval, rrse, ee, x1e
```

First, we use Least Squares, two lags, and BIC order selection:

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)

y_eval, yhat_eval, rrse, ee, x1e = evaluate(model)
print(rrse)

plot_results(y=y_eval, yhat=yhat_eval, n=100)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
671.3206640454391
```

The RRSE far above 1 shows that this configuration is inadequate in free run simulation. The simulated output quickly moves away from the observed output, and residual autocorrelation remains high across many lags. A plot restricted to the first samples may hide the extent of this divergence; the metric and plot should therefore be examined together.

We now increase the lags and use Recursive Least Squares:

```python
model = FROLS(
    order_selection=True,
    n_info_values=50,
    ylag=5,
    xlag=5,
    info_criteria="bic",
    estimator=RecursiveLeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)

y_eval, yhat_eval, rrse, ee, x1e = evaluate(model)
print(rrse)

plot_results(y=y_eval, yhat=yhat_eval, n=100)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
255.83144074914404
```

The value decreases relative to the first model, but remains far above 1. It would therefore be incorrect to call this a good result simply because it represents a relative improvement. The free run simulation is still unstable and the residuals remain temporally dependent.

Finally, we evaluate exactly the same model using one-step-ahead prediction:

```python
y_eval, yhat_eval, rrse, ee, x1e = evaluate(
    model,
    steps_ahead=1,
)
print(rrse)

plot_results(y=y_eval, yhat=yhat_eval, n=100)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
0.02061510242049919
```

The RRSE now appears excellent, even though this is the same model that produced an RRSE above 255 in free run simulation. The difference comes from feeding the observed output back at every step, which prevents error propagation. Residual autocorrelation decreases sharply, but the input-residual correlation still has lags outside the approximate bounds. These results should be inspected rather than summarized by RRSE.

This example shows why the statement “the model has an RRSE of 0.02” is incomplete. We must report the dataset, the alignment, and, most importantly, the prediction mode. For applications in which the model must evolve without continuous access to the true output, the free run result is the decisive diagnosis in this case.
