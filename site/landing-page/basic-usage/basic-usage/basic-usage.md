---
template: overrides/main.html
title: Basic Usage
---

The NARMAX model is described as:

$$
	y_k= F[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
$$

where $n_y\in \mathbb{N}$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$, are the maximum lags for the system output and input respectively; $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$ is the system output at discrete time $k \in \mathbb{N}^n$;
$e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise at discrete time $k$. In this case, $\mathcal{F}$ is some nonlinear function of the input and output regressors and $d$ is a time delay typically set to $d=1$.

``` py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import mean_squared_error
from sysidentpy.utils.generate_data import get_siso_data


# Generate a dataset of a simulated dynamical system
x_train, x_valid, y_train, y_valid = get_siso_data(
	n=1000,
	colored_noise=False,
	sigma=0.001,
	train_percentage=80
)
```

### Build a Polynomial NARX model

``` py
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

basis_function = Polynomial(degree=2)
model = FROLS(
	order_selection=True,
	n_info_values=10,
	extended_least_squares=False,
	ylag=2,
	xlag=2,
	info_criteria='aic',
	estimator='least_squares',
	basis_function=basis_function
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
mse = mean_squared_error(y_valid, yhat)
print(mse)
r = pd.DataFrame(
	results(
		model.final_model, model.theta, model.err,
		model.n_terms, err_precision=8, dtype='sci'
		),
	columns=['Regressors', 'Parameters', 'ERR'])
print(r)

Regressors     Parameters        ERR
0        x1(k-2)     0.9000  0.95556574
1         y(k-1)     0.1999  0.04107943
2  x1(k-1)y(k-1)     0.1000  0.00335113

plot_results(y=y_valid, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x2_val)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

<figure markdown>
  ![Polynomial NARMAX](\C:\Users\wilso\Desktop\projects\GitHub\sysidentpy\examples\figures\polynomial_narmax.png)
  <figcaption>Validation of the Polynomial NARMAX model</figcaption>
</figure>


### NARX Neural Network

``` py
from torch import nn
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

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

basis_function=Polynomial(degree=1)

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
plot_results(y=y_valid, yhat=yhat, n=200)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

<figure markdown>
  ![Polynomial NARMAX](\C:\Users\wilso\Desktop\projects\GitHub\sysidentpy\examples\figures\narx_network.png)
  <figcaption>Validation of the Neural NARX model</figcaption>
</figure>

### Catboost-narx

``` py
from sysidentpy.general_estimators import NARX
from catboost import CatBoostRegressor
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation, compute_cross_correlation

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
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

<figure markdown>
  ![Polynomial NARMAX](\C:\Users\wilso\Desktop\projects\GitHub\sysidentpy\examples\figures\catboost_narx.png)
  <figcaption>Validation of the Catboost NARX model</figcaption>
</figure>

#### Catboost without NARX configuration

The following is the Catboost performance *without* the NARX configuration.

``` py
def plot_results_tmp(y_valid, yhat):
	_, ax = plt.subplots(figsize=(14, 8))
	ax.plot(y_valid[:200], label='Data', marker='o')
	ax.plot(yhat[:200], label='Prediction', marker='*')
	ax.set_xlabel("$n$", fontsize=18)
	ax.set_ylabel("$y[n]$", fontsize=18)
	ax.grid()
	ax.legend(fontsize=18)
	plt.show()

catboost = CatBoostRegressor(
	iterations=300,
	learning_rate=0.1,
	depth=6
)

catboost.fit(x_train, y_train, verbose=False)
plot_results(y_valid, catboost.predict(x_valid))
```

<figure markdown>
  ![Polynomial NARMAX](\C:\Users\wilso\Desktop\projects\GitHub\sysidentpy\examples\figures\catboost.png)
  <figcaption>Validation of the Catboost model without NARX configuration</figcaption>
</figure>

