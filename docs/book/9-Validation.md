## The `predict` Method in SysIdentPy

Before getting into the validation process in System Identification, it's essential to understand how the `predict` method works in SysIdentPy.

### Using the `predict` Method

A typical usage of the `predict` method in SysIdentPy looks like this:

```python
yhat = model.predict(X=x_test, y=y_test)
```

SysIdentPy users often have two common questions about this method:

1. Why do we need to pass the test data, `y_test`, as an argument in the `predict` method?
2. Why are the initial predicted values identical to the values in the test data?

To address these questions, let’s first explain the concepts of infinity-step-ahead prediction, n-step-ahead prediction, and one-step-ahead prediction in dynamic systems.

### Infinity-Step-Ahead Prediction

Infinity-step-ahead prediction, also known as *free run simulation*, refers to making predictions using previously **predicted** values, $\hat{y}_{k-n_y}$, in the prediction loop.

For example, consider the following test input and output data:

$$
x_{test} = [1, 2, 3, 4, 5, 6, 7]
$$

$$
y_{test} = [8, 9, 10, 11, 12, 13, 14]
$$

Suppose we want to validate a model $m$ defined by:

$$
m \rightarrow y_k = 1*y_{k-1} + 2*x_{k-1}
$$

To predict the first value, we need access to both $y_{k-1}$ and $x_{k-1}$. This requirement explains why you need to pass `y_test` as an argument in the `predict` method. It also answers the second question: SysIdentPy requires the user to provide the initial conditions explicitly. The `y_test` data passed in the `predict` method is not used entirely; only the initial values needed for the model’s lag structure are used.

In this example, the model's maximum lag is 1, so we need only 1 initial condition. The predicted values, `yhat`, are then calculated as follows:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*14 + 2*3 = 20
yhat(4) = 1*20 + 2*4 = 28
```

As shown, the first value of `yhat` matches the first value of `y_test` because it serves as the initial condition. Another key point is that the prediction loop uses the previously **predicted** values, not the actual `y_test` values, which is why it's called infinity-step-ahead or free run simulation.

In system identification, we often aim for models that perform well in infinity-step-ahead predictions. Since the prediction error propagates over time, a model that shows good performance in free run simulation is considered a robust model.

In SysIdentPy, users only need to pass the initial conditions when performing an infinity-step-ahead prediction. If you pass only the initial conditions, the results will be the same! Therefore

```python
yhat = model.predict(X=x_test, y=y_test)
```

is actually the same as

```python
yhat = model.predict(X=x_test, y=y_test[:model.max_lag].reshape(-1, 1))
```

> `model.max_lag` can be accessed after we fit the model using the code below.

```python
model = FROLS(
	order_selection=False,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
    e_tol=0.9999
    n_terms=15
)
model.fit(X=x, y=y)
model.max_lag
```

> Its important to mention that, in current version of SysIdentPy, the maximum lag considered is actually the maximum lag between `xlag` and `ylag` definition. This is important because you can pass `ylag = xlag = 10` and the final model, after the model structure selection, select terms where the maximum lag is 3. You have to pass 10 initial conditions, but internally the calculations are done using the correct regressors. This is necessary due the way the regressors are created after that the model is fitted. Therefore, is recommended to use the `model.max_lag` to be sure.

### 1-step ahead prediction

The difference between 1 step-ahead prediction and infinity-steps ahead prediction is that the model take the previous real `y_test` test values in the loop instead of the predicted `yhat` values. And that is a huge and important difference. Let's do prediction using 1-step ahead method:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*9 + 2*2 = 13
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*11 + 2*4 = 19
and so on
```

The model uses real values in the loop and only predict the next value. The prediction error, in this case, is always corrected because we are not propagating the error using the predicted values in the loop.

SysIdentPy's `predict` method allow the user to perform a 1-step ahead prediction by setting `steps_ahead=1`

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=1)
```

In this case, as you can imagine, we need to pass the all the `y_test` data because the method have to access the real values at each iteration. If you pass only the initial conditions, `yhat` will have only the initial conditions plus 1 more sample, that is the 1-step ahead prediction. To predict another point, you would need to pass the new initial conditions again and so on. SysIdentPy already to everything for you, so just pass all the data you want to validate using the 1-step ahead method.

### n-steps ahead prediction

The n-steps ahead prediction is almost the same as the 1-step ahead, but here you can define the number of steps ahead you want to test your model. If you set `steps_ahead=5`, for example, it means that the first 5 values will be predicted using `yhat` in the loop, but then the process is *restarted* by feeding the real values in `y_test` in the next iteration, then performing other 5 predictions using the `yhat` and so on. Let's check the example considering `steps_ahead=2`:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*16 + 2*4 = 24
and so on
```

## Model Performance

Model validation is one of the most crucial part in system identification. As we mentioned before, in system identification we are trying the model the dynamic of the process without for task like control design. In such cases, we can not only rely on regression metrics, but also ensuring that residuals are unpredictable across various combinations of past inputs and outputs ([Billings, S. A. and Voon, W. S. F., "Structure detection and model validity tests in the identification of nonlinear systems"](https://digital-library.theiet.org/content/journals/10.1049/ip-d.1983.0034)). One often used statistical tests is the normalized RMSE, called RRSE, which can be expressed by

$$
\begin{equation}
        \textrm{RRSE}= \frac{\sqrt{\sum\limits_{k=1}^{n}(y_k-\hat{y}_k)^2}}{\sqrt{\sum\limits_{k=1}^{n}(y_k-\bar{y})^2}},
\end{equation}
\tag{1}
$$

where $\hat{y}_k \in \mathbb{R}$ the model predicted output and $\bar{y} \in \mathbb{R}$ the mean of the measured output $y_k$. The RRSE gives some indication regarding the quality of the model, but concluding about the best model by evaluating only this quantity may lead to an incorrect interpretation, as shown in following example.

Consider the models

$$
y_{{_a}k} = 0.7077y_{{_a}k-1} + 0.1642u_{k-1} + 0.1280u_{k-2}
$$

and

$$y_{{_b}k}=0.7103y_{{_b}k-1} + 0.1458u_{k-1} + 0.1631u_{k-2} -1467y^3_{{_b}k-1} + 0.0710y^3_{{_b}k-2} +0.0554y^2_{{_b}k-3}u_{k-3}$$

defined in the [Meta Model Structure Selection: An Algorithm For Building Polynomial NARX Models For Regression And Classification](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). The former results in a $RRSE = 0.1202$ while the latter gives $RRSE~=0.0857$. Although the model $y_{{_b}k}$ fits the data better, it is only a biased representation to one piece of data and not a good description of the entire system.

The RRSE (or any other metric) shows that validations test might be performed carefully. Another traditional practice is split the data set in two parts. In this respect, one can test the models obtained from the estimation part of the data using a specific data for validation. However, the one-step-ahead performance of NARX models generally results in misleading interpretations because even strongly biased models may fit the data well. Therefore, a free run simulation approach usually allows a better interpretation if the model is adequate or not ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)).

Statistical tests for SISO models based on the correlation functions were proposed in ([Billings, S. A. and Voon, W. S. F., "A prediction-error and stepwise-regression estimation algorithm for non-linear systems"](https://www.tandfonline.com/doi/abs/10.1080/00207178608933633)), ([Model validity tests for non-linear signal processing applications](https://www.tandfonline.com/doi/abs/10.1080/00207179108934155)). The tests are:

$$
\begin{align}
    \phi_{_{\xi \xi}\tau} &= E\{\xi_k \xi_{k-\tau}\} = \delta_{\tau}, \\
    \phi_{_{\xi x}\tau} &= E\{\xi_k x_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{\xi \xi x}\tau} &= E\{\xi_k \xi_{k-\tau} x_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{x^2 \xi}\tau} &= E\{(u^2_k - E\{x^2_k\})\xi_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{x^2 \xi^2}\tau} &= E\{(u^2_k - E\{x^2_k\})\xi^2_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{(y\xi) x^2}\tau} &= E\{(y_k\xi_k - E\{y_k\xi_k\})(x^2_{k-\tau} - E\{x^2_k\})\} = 0 \forall \tau,
\end{align}
\tag{2}
$$


where $\delta$ is the Dirac delta function and the cross-correlation function $\phi$ is denoted by ([Billings, S. A. and Voon, W. S. F.](https://digital-library.theiet.org/content/journals/10.1049/ip-d.1983.0034)):

$$
\begin{equation}
\phi_{{_{ab}}\tau} = \frac{\frac{1}{n}\sum\limits_{k=1}^{n-\tau}(a_k - \hat{a})(b_{k+\tau}-\hat{b})}{\sqrt{\frac{1}{n}\sum\limits_{k=1}^{n}(a_k-\hat{a})^2} \sqrt{\frac{1}{n}\sum\limits_{k=1}^{n}(b_k-\hat{b})^2}} = \frac{\sum\limits_{k=1}^{n-\tau}(a_k - \hat{a})(b_{k+\tau}-\hat{b})}{\sqrt{\sum\limits_{k=1}^{n}(a_k-\hat{a})^2} \sqrt{\sum\limits_{k=1}^{n}(b_k-\hat{b})^2}},
\end{equation}
\tag{3}
$$

where $a$ and $b$ are two signal sequences. If the tests are true, then the model residues can be considered as white noise.

### Metrics Available in SysIdentPy

SysIdentPy provides the following regression metrics out of the box:

- forecast_error
- mean_forecast_error
- mean_squared_error
- root_mean_squared_error
- normalized_root_mean_squared_error
- root_relative_squared_error
- mean_absolute_error
- mean_squared_log_error
- median_absolute_error
- explained_variance_score
- r2_score
- symmetric_mean_absolute_percentage_error

To use them, the user only need to import the desired metric using, for example

```python
from sysidentpy.metrics import root_relative_squared_error
```

SysIdentPy also provides methods for calculate and analyse the residues correlation

```python

from sysidentpy.utils.plotting import plot_residues_correlation
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
```

Lets check the metrics of the eletro mechanical system modeled in Chapter 4.

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.metrics import root_relative_squared_error

df1 = pd.read_csv("examples/datasets/x_cc.csv")
df2 = pd.read_csv("examples/datasets/y_cc.csv")

x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)

basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)

ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c9_dc_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_1.png?raw=true)

The RRSE is 0.0800, which is a very good metric. However, we can see that the residues have somo high auto-correlations anda with the input. This mean that our model maybe is not good enough as it could be.

Lets check what happens if we increase `xlag`, `ylag` and change the parameter estimation algorithm from Least Squares to the Recursive Least Squares

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=50,
    ylag=5,
    xlag=5,
    info_criteria="bic",
    estimator=RecursiveLeastSquares(unbiased=False),
    basis_function=basis_function
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

Now the RRSE is 0.0568 and we have a better residual correlation!

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_dc_2.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_2.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_2.png?raw=true)

In the end of day, the best model will be the model that satisfy the user needs. However, it's important to understand how to analyse the models so you can have an idea if you can get some improvements without too much work.

For the sake of curiosity, lets check how the model perform if we run a 1-step ahead prediction. We don't need to fit the model again, just make another prediction using the 1-step option.

```python
yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

The same model, but evaluating the 1-step ahead prediction, now return a RRSE$= 0.02044$ and the residues are even better. But remember, that is expected, as explained in the previous section.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_dc_3.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_3.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_03.png?raw=true)
