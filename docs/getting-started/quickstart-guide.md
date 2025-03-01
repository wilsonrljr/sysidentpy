---
template: overrides/main.html
title: Basic Usage
---

## 1. Prerequisites

You’ll need to know a bit of Python.

To work the examples, you’ll need `pandas` installed in addition to NumPy.

```bash
pip install sysidentpy pandas
# Optional: For neural networks and advanced features
pip install sysidentpy["all"]
```

## 2. Key Features

SysIdentPy provides a flexible framework for building, predicting, validating, and visualizing nonlinear time series models. The modeling process involves several key decisions: defining the mathematical representation of the model, choosing the parameter estimation algorithm, selecting the appropriate model structure, and determining the prediction approach.

The following features are available in SysIdentPy:

### Model Classes
- NARMAX, NARX, NARMA, NAR, NFIR, ARMAX, ARX, AR, and their variants.

### Mathematical Representations
- Polynomial
- Neural
- Fourier
- Laguerre
- Bernstein
- Bilinear
- Legendre
- Hermite
- HermiteNormalized

You can also define advanced NARX models such as Bayesian and Gradient Boosting models using the GeneralNARX class, which provides seamless integration with various machine learning algorithms.

### Model Structure Selection Algorithms
- Forward Regression Orthogonal Least Squares (FROLS)
- Meta-model Structure Selection (MeMoSS)
- Accelerated Orthogonal Least Squares (AOLS)
- Entropic Regression

### Parameter Estimation Methods
- Least Squares (LS)
- Total Least Squares (TLS)
- Recursive Least Squares (RLS)
- Ridge Regression
- Non-Negative Least Squares (NNLS)
- Least Squares Minimal Residues (LSMR)
- Bounded Variable Least Squares (BVLS)
- Least Mean Squares (LMS) and its variants:
  - Affine LMS
  - LMS with Sign Error
  - Normalized LMS
  - LMS with Normalized Sign Error
  - LMS with Sign Regressor
  - Normalized LMS with Sign Sign
  - Leaky LMS
  - Fourth-Order LMS
  - Mixed Norm LMS

### Order Selection Criteria
- Akaike Information Criterion (AIC)
- Corrected Akaike Information Criterion (AICc)
- Bayesian Information Criterion (BIC)
- Final Prediction Error (FPE)
- Khundrin's Law of Iterated Logarithm Criterion

### Prediction Methods
- One-step ahead
- n-steps ahead
- Infinity-steps ahead

### Visualization Tools
- Prediction plots
- Residual analysis
- Model structure visualization
- Parameter visualization

---

As you can see, SysIdentPy supports numerous model combinations, each tailored to different use cases. But don’t worry about picking the perfect combination right away—let’s start with the default settings to get you up and running quickly.



<div class="custom-collapsible-card">
  <input type="checkbox" id="toggle-info">
  <label for="toggle-info">
    📚 <strong>Looking for more details on NARMAX models?</strong>
    <span class="arrow">▼</span>
  </label>
  <div class="collapsible-content">
    <p>
      For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in <strong>SysIdentPy</strong>, check out our book:
    </p>
    <a href="https://sysidentpy.org/book/0-Preface/" target="_blank">
      <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
    </a>
    <p>
      This book provides in-depth guidance to support your work with <strong>SysIdentPy</strong>.
    </p>
    <p>
      🛠️ You can also explore the <a href="https://sysidentpy.org/user-guide/overview/" target="_blank"><strong>tutorials in the documentation</strong></a> for practical, hands-on examples.
    </p>
  </div>
</div>

## 3. Quickstart

To keep things simple, let's load some simulated data for the examples.

``` py
from sysidentpy.utils.generate_data import get_siso_data

# Generate a dataset from a simulated dynamic system.
x_train, x_valid, y_train, y_valid = get_siso_data(
	n=300,
	colored_noise=False,
	sigma=0.0001,
	train_percentage=80
)
```

### Build your first NARX model

With the loaded dataset, let's build a Polynomial NARX model. Using SysIdentPy's default options, you need to define at least the model structure selection method and the mathematical representation of the model (specified here by the basis function).

``` python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = FROLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)
```

The model structure selection (MSS) method enables the model's fit and predict operations.

While different MSS algorithms come with various hyperparameters, they are not the focus here. In this guide, we will show how to modify these hyperparameters but will not discuss the best configurations.

``` python
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

To evaluate the model's performance, you can use any of the native metric functions available in SysIdentPy. For example, the Root Relative Squared Error (RRSE) metric can be used as follows:

``` python
from sysidentpy.metrics import root_relative_squared_error

rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
```

``` console
0.00014
```

To view the final mathematical equation of the Polynomial NARX model, use the results function. This function requires:

- `final_model`: The selected regressors after fitting
- `theta`: The estimated parameters
- `err`: The error reduction ratio (ERR)

Here’s how to display the results:

``` python
from sysidentpy.utils.display_results import results

r = pd.DataFrame(
	results(
		model.final_model, model.theta, model.err,
		model.n_terms, err_precision=8, dtype='sci'
		),
	columns=['Regressors', 'Parameters', 'ERR'])
print(r)
```
This output shows the selected regressors, their corresponding estimated parameters, and the contribution of each regressor to the model’s performance (ERR).

``` console
Regressors     Parameters        ERR
0        x1(k-2)     0.9000  0.95556574
1         y(k-1)     0.1999  0.04107943
2  x1(k-1)y(k-1)     0.1000  0.00335113
```

To visualize the model's performance, you can use the `plot_results` function. This method plots the predicted values against the actual data, allowing you to see how well the model fits the dataset.

``` python
from sysidentpy.utils.plotting import plot_results

plot_results(y=y_valid, yhat=yhat, n=1000)
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-polynomial-narx.png?raw=true)

Residual analysis is essential to check if the model has captured all the relevant dynamics of the system. You can analyze the residuals by computing their autocorrelation and the cross-correlation between the residuals and one of the model inputs.

``` python
from sysidentpy.utils.plotting import plot_residues_correlation
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)

# Compute and plot autocorrelation of the residuals
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")

# Compute and plot cross-correlation between residuals and an input
x1e = compute_cross_correlation(y_valid, yhat, x2_val)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

<div style="display: flex; justify-content: space-between;">
  <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-ee.png?raw=true" width="400" alt="Quickstart EE" />
  <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-xe.png?raw=true" width="400" alt="Quickstart XE" />
</div>


Here’s the full code example for your reference:

```python
import pandas as pd

from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.utils.plotting import plot_residues_correlation
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)

# Generate a dataset from a simulated dynamic system.
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=300,
    colored_noise=False,
    sigma=0.0001,
    train_percentage=80
)


basis_function = Polynomial(degree=2)
model = FROLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)

r = pd.DataFrame(
    results(
        model.final_model, model.theta, model.err,
        model.n_terms, err_precision=8, dtype='sci'
        ),
    columns=['Regressors', 'Parameters', 'ERR'])
print(r)

plot_results(y=y_valid, yhat=yhat, n=1000, figsize=(15, 4))
# Compute and plot autocorrelation of the residuals
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
# Compute and plot cross-correlation between residuals and an input
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

### Customizing your model configuration

In the previous section, we showed how easy it is to fit a Polynomial NARX model with SysIdentPy using the default configuration. But what if you want to experiment with different combinations of algorithms for model structure selection, parameter estimation, and other settings?

#### Model Structure Selection

SysIdentPy makes this process simple. For instance, if you want to use the **Accelerated Orthogonal Least Squares (AOLS)** algorithm instead of the default `FROLS`, you only need to import and use it when defining your model.

``` python
import pandas as pd

from sysidentpy.model_structure_selection import AOLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = AOLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

The evaluation, residual analysis, and plots remain the same as before, so they are not shown here.

With just a small change in the import statement, you can explore different algorithms. Similarly, you can customize the parameter estimation methods, prediction strategies, and mathematical representations to suit your specific needs.

Similarly, you can use the **Meta-model Structure Selection**

``` python
import pandas as pd

from sysidentpy.model_structure_selection import MetaMSS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = MetaMSS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

and the **Entropic Regression**

``` python
import pandas as pd

from sysidentpy.model_structure_selection import ER
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = ER(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

#### Parameter Estimation

Changing the parameter estimation algorithm in SysIdentPy is also straightforward. You can check the list of available algorithms with the following code:


``` python
from sysidentpy import parameter_estimation

print("Parameter Estimation Algorithms available:", parameter_estimation.__all__)
```

``` console
Parameter Estimation Algorithms available: ['LeastSquares', 'RidgeRegression', 'RecursiveLeastSquares', 'TotalLeastSquares', 'LeastMeanSquareMixedNorm', 'LeastMeanSquares', 'LeastMeanSquaresFourth', 'LeastMeanSquaresLeaky', 'LeastMeanSquaresNormalizedLeaky', 'LeastMeanSquaresNormalizedSignRegressor', 'LeastMeanSquaresNormalizedSignSign', 'LeastMeanSquaresSignError', 'LeastMeanSquaresSignSign', 'AffineLeastMeanSquares', 'NormalizedLeastMeanSquares', 'NormalizedLeastMeanSquaresSignError', 'LeastMeanSquaresSignRegressor', 'NonNegativeLeastSquares', 'LeastSquaresMinimalResidual', 'BoundedVariableLeastSquares']
```

Although the default estimator may change depending on the model structure selection method, it is usually `LeastSquares` or `RecursiveLeastSquares`. To define a specific estimator, simply import the desired method and set it using the estimator hyperparameter:

``` python
import pandas as pd

from sysidentpy.model_structure_selection import ER
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquaresMinimalResidual


basis_function = Polynomial(degree=2)
model = ER(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
	estimator=LeastSquaresMinimalResidual(),
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

You can apply this approach to any model structure selection algorithm. It’s that simple to change the parameter estimation method.

#### Customizing the Mathematical Representation - Basis Function

Changing the mathematical representation (basis function) in SysIdentPy is as simple as customizing the parameter estimation method. For example, to build a **Fourier NARX model**, you just need to import the desired basis function and set it in the model structure selection algorithm.

To check all available basis functions, use:

``` python

from sysidentpy import basis_function

print("Basis Function Available:", basis_function.__all__)
```

``` console
Basis Function Available: ['Bersntein', 'Bilinear', 'Fourier', 'Legendre', 'Laguerre', 'Hermite', 'HermiteNormalized', 'Polynomial']
```

After choosing the basis function, you can define it in your model as shown below:

``` python
import pandas as pd

from sysidentpy.model_structure_selection import ER
from sysidentpy.basis_function import Fourier
from sysidentpy.parameter_estimation import LeastSquaresMinimalResidual


basis_function = Fourier(degree=2)
model = AOLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
	estimator=LeastSquaresMinimalResidual(),
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

With this approach, you can easily explore different mathematical representations by simply switching the basis function. No complex changes required.

!!! note

    The `results` method, which returns the mathematical equation of the fitted model, currently supports only the **Polynomial** basis function. Support for all basis functions is planned for version 1.0.

#### Customizing the Model Type

The key difference between a **NARX** and an **ARX** model lies in the presence of nonlinear relationships between the regressors. For instance, if you set the degree of the basis function to 2 for Polynomial basis function, as shown in previous examples, you'll have a **NARX** model. If the degree is set to 1, it results in an **ARX** model.

However, the distinction isn't purely based on the degree of the basis function. It ultimately depends on the final model equation. Even with a degree set to 2, the fitted model might be linear if the model structure selection algorithm removes the nonlinear terms. This means that while setting the degree to 2 gives the algorithm an opportunity to explore nonlinear relationships, the final model might still be linear.

Always check the final model to confirm whether it is linear or nonlinear, regardless of the degree you set for the basis function.

```python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

# basis_function = Polynomial(degree=2) for NARX (and maybe ARX) or
basis_function = Polynomial(degree=1)  # ARX model
model = FROLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

The difference between **NARX**, **NAR**, and **NFIR** models lies in the type of regressors used. **NARX** models involve both input and output regressors, while **NAR** models use only output regressors, and **NFIR** models use only input regressors.

To create a **NAR** model, you simply need to specify the `model_type` argument as `"NAR"`. In this case, you don't need to define the lags of the inputs since you're working with output-only regressors. You also don't need to pass input data in the `fit` and `predict` methods. Only the output data is required.

Because **NAR** models do not include input variables to define the forecasting horizon, you must set the `forecast_horizon` parameter to specify how many periods ahead you want to predict.

```python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

# basis_function = Polynomial(degree=2) for NARX (and maybe ARX) or
basis_function = Polynomial(degree=1)  # ARX model
model = FROLS(
    ylag=2,
    basis_function=basis_function,
    model_type="NAR",
)

model.fit(y=y_train)
yhat = model.predict(y=y_valid, forecast_horizon=23)
```

For the **NFIR** model, however, you still need to pass the output array because autoregressive models require initial conditions to operate.

```python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

# basis_function = Polynomial(degree=2) for NARX (and maybe ARX) or
basis_function = Polynomial(degree=1)  # ARX model
model = FROLS(
    xlag=2,
    basis_function=basis_function,
    model_type="NFIR",  # Specify NFIR model type
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

<div class="custom-collapsible-card">
  <input type="checkbox" id="initial-info">
  <label for="initial-info">
    📚 <strong>Looking for more details on what are initial conditions?</strong>
    <span class="arrow">▼</span>
  </label>
  <div class="collapsible-content">
    <p>
      Check chapter 9 of our companion book for more information on why autoregressive models need initial conditions to operate:
    </p>
    <a href="https://sysidentpy.org/book/9-Validation/" target="_blank">
      <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
    </a>
  </div>
</div>


#### Prediction and Forecasting Horizon

By default, when you call `model.predict(X=x_valid, y=y_valid)`, it performs an infinity-steps ahead prediction, also known as a free run simulation. However, if you need to make a specific number of steps ahead prediction, such as a one-step ahead or n-steps ahead forecast, you can simply pass the `steps_ahead` hyperparameter in the `predict` method:

```python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

# basis_function = Polynomial(degree=2) for NARX (and maybe ARX) or
basis_function = Polynomial(degree=2)  # ARX model
model = FROLS(
    ylag=2,
	xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)

# one-step ahead
yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=1)

# 4-steps ahead
yhat_4_steps = model.predict(X=x_valid, y=y_valid, steps_ahead=4)
```

<div class="custom-collapsible-card">
  <input type="checkbox" id="steps-info">
  <label for="steps-info">
    📚 <strong>Looking for more details about how steps-ahead prediction works?</strong>
    <span class="arrow">▼</span>
  </label>
  <div class="collapsible-content">
    <p>
      Check chapter 9 of our companion book for more information on how infinity-steps, n-steps and one-step ahead prediction works:
    </p>
    <a href="https://sysidentpy.org/book/9-Validation/" target="_blank">
      <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
    </a>
  </div>
</div>



#### Order Selection

Order selection is a classical approach to automatically determine the optimal model order when using the **FROLS** algorithm. It helps in identifying the best combination of lags and regressors by evaluating different models based on an information criterion.

!!! Important

    Information criteria are *only applicable* when using the **FROLS** algorithm. Other algorithms employ alternative methods for model order selection, each developed to their specific approach.

To enable order selection, simply:
1. Set `order_selection=True`.
2. Specify the desired `info_criteria` (e.g., `"aic"`, `"aicc"`, `"bic"`, `"fpe"`, or `"lilc"`).

``` python
import pandas as pd

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = FROLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
	order_selection=True,
	info_criteria="bic"
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

You can control how many regressors are tested during order selection using the `n_info_values` hyperparameter. The default is `15`, but you might want to increase it when working with high lag orders or multiple input variables.

``` python
model = FROLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
	order_selection=True,
	info_criteria="bic",
	n_info_values=50
)
```

!!! Important

    Increasing `n_info_values` can improve accuracy but will also increase computational time.

#### NARX Neural Network

You can create a **Neural NARX Network** with SysIdentPy, thanks to its seamless integration with PyTorch. This flexibility allows you to design various Neural NARX architectures by customizing not only the hidden layer configurations and other neural network parameters but also by selecting any available basis function, just like in other NARX representations.


``` python
from torch import nn
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_results


basis_function=Polynomial(degree=1)

class NARX(nn.Module):
	def __init__(self):
		super().__init__()
		self.lin = nn.Linear(4, 10)
		self.lin2 = nn.Linear(10, 10)
		self.lin3 = nn.Linear(10, 1)
		self.tanh = nn.Tanh()

	def forward(self, xb):
		z = self.lin(xb)
		z = self.tanh(z)
		z = self.lin2(z)
		z = self.tanh(z)
		z = self.lin3(z)
		return z


narx_net = NARXNN(
	net=NARX(),
	ylag=2,
	xlag=2,
	basis_function=basis_function,
	model_type="NARMAX",
	loss_func='mse_loss',
	optimizer='Adam',
	epochs=200,
	verbose=False,
	optim_params={'betas': (0.9, 0.999), 'eps': 1e-05} # optional parameters of the optimizer
)

narx_net.fit(X=x_train, y=y_train)
yhat = narx_net.predict(X=x_valid, y=y_valid)
plot_results(y=y_valid, yhat=yhat, n=1000, figsize=(15, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-neural-narx.png?raw=true)

#### General Estimators

SysIdentPy also offers the flexibility to integrate any regression method from popular packages like `scikit-learn`, `xgboost`, `catboost`, and many others. To make it work, the estimator simply needs to follow the standard `fit` and `predict` API.

This significantly expands the range of possible NARX model representations, enabling diverse analyses to help you build the best model for your specific use case.

The following example demonstrates how to use a `catboost` model. Ensure you have `catboost` installed before running the example.


``` python
from sysidentpy.general_estimators import NARX
from catboost import CatBoostRegressor
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_results

catboost_narx = NARX(
	base_estimator=CatBoostRegressor(
		iterations=300,
		learning_rate=0.1,
		depth=6),
	xlag=2,
	ylag=2,
	basis_function=basis_function,
	model_type="NARMAX",
	fit_params={'verbose': False}
)

catboost_narx.fit(X=x_train, y=y_train)
yhat = catboost_narx.predict(X=x_valid, y=y_valid)
plot_results(y=y_valid, yhat=yhat, n=200)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-catboost-narx.png?raw=true)

To highlight the importance of transforming `catboost` into a NARX model, the following example shows the performance of `catboost` *without* the NARX configuration.

``` python
catboost = CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6
)

catboost.fit(x_train, y_train, verbose=False)
plot_results(y=y_valid, yhat=catboost.predict(x_valid), figsize=(15, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-catboost-without-narx.png?raw=true)

Note that you can still explore various combinations to better fit your use case. For example, you can create a CatBoost NARX model using a Fourier basis function and perform an n-steps ahead prediction. This flexibility allows you to capture complex seasonality patterns while leveraging CatBoost's powerful gradient boosting capabilities. The same approach applies to any other regression model you choose, enabling you to experiment with different basis functions, prediction horizons, and estimators to find the best configuration for your specific problem.

This is just a quickstart guide to **SysIdentPy**. For more comprehensive tutorials, step-by-step guides, detailed explanations, and advanced use cases, be sure to check out our full [documentation](https://sysidentpy.org/) and [companion book](https://sysidentpy.org/book/0%20-%20Preface/).
They provide in-depth content to help you get the most out of SysIdentPy for your system identification and forecasting tasks.

