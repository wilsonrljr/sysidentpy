<div align="center">
<img src="images/sysidentpy-logo.svg" width="640" height="320" />

[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.02384-%23FF7800)](https://joss.theoj.org/papers/10.21105/joss.02384)
[![PyPI version](https://img.shields.io/pypi/v/sysidentpy?color=%23ff7800)](https://pypi.org/project/sysidentpy/)
[![License](https://img.shields.io/pypi/l/sysidentpy?color=%23FF7800)](https://opensource.org/licenses/BSD-3-Clause)
[![openissues](https://img.shields.io/github/issues/wilsonrljr/sysidentpy?color=%23FF7800)](https://github.com/wilsonrljr/sysidentpy/issues)
[![issuesclosed](https://img.shields.io/github/issues-closed-raw/wilsonrljr/sysidentpy?color=%23FF7800)](https://github.com/wilsonrljr/sysidentpy/issues)
[![downloads](https://img.shields.io/pypi/dm/sysidentpy?color=%23FF7800)](https://pypi.org/project/sysidentpy/)
[![python](https://img.shields.io/pypi/pyversions/sysidentpy?color=%23FF7800)](https://pypi.org/project/sysidentpy/)
[![status](https://img.shields.io/pypi/status/sysidentpy?color=%23FF7800)](https://pypi.org/project/sysidentpy/)
[![discord](https://img.shields.io/discord/711610087700955176?color=%23FF7800&label=discord)](https://discord.gg/7afBSzU4)
[![contributors](https://img.shields.io/github/contributors/wilsonrljr/sysidentpy?color=%23FF7800)](https://github.com/wilsonrljr/sysidentpy/graphs/contributors)
[![stars](https://img.shields.io/github/stars/wilsonrljr/sysidentpy?style=social)](https://github.com/wilsonrljr/sysidentpy/stargazers)

<h1 align="center"> NARMAX Methods For System Identification and TimeSeries Forecasting </h1>
<h3 align="center">From Classical Approaches to Neural Networks</h3>

**SysIdentPy** offers State of the Art techniques to build your NARMAX models, including its variants `NARX`, `NARMA`, `NAR`, `NFIR`, `ARMAX`, `ARX`, `ARMA` and others. It also includes tons of interesting examples to help you build nonlinear forecasting models using SysIdentPy.

</div>

## Table of Contents

- [What is SysIdentPy?](#introduction)
- [How do I install SysIdentPy?](#how-do-i-install-sysidentpy)
- [Features](#what-are-the-main-features-of-sysidentpy)
- [Why does SysIdentPy exist?](#why-does-sysidentpy-exist)
- [How do I use SysIdentPy?](#how-do-i-use-SysIdentPy)
- [Examples](#examples)
- [Communication](#communication)
- [Citation](#citation)
- [Inspiration](#inspiration)
- [Contributors](#contributors)
- [Sponsors](#sponsors)


## Introduction

SysIdentPy is an open-source Python module for System Identification using **NARMAX** models built on top of **numpy** and is distributed under the 3-Clause BSD license. SysIdentPy provides an easy-to-use and  flexible framework for building Dynamical Nonlinear Models for time series and dynamic systems.

With **SysIdentPy**, you can:

- Build and customize nonlinear forecasting models.
- Utilize state-of-the-art techniques for model structure selection and parameter estimation.
- Experiment with neural NARX models and other advanced algorithms.

Check our [documentation](https://sysidentpy.org)!

For an in depth documentation, check our companion book:

<a href="https://sysidentpy.org/book/0-Preface/">
  <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/Nonlinear_System_identification.png?raw=true" alt="Nonlinear System Identification" style="width: 200px; height: auto;" />
</a>


## How do I install SysIdentPy?

The easiest way to get SysIdentPy running is to install it using ``pip``
``` console
pip install sysidentpy
```

### Requirements

`SysIdentPy` requires:

- Python (>= 3.7)
- NumPy (>= 1.9.2) for numerical algorithms
- Matplotlib >= 3.3.2 for static plotting and visualizations
- Pytorch (>=1.7.1) for building NARX neural networks
- scipy (>= 1.7.0) for numerical and optimization algorithms

The library is compatible with Linux, Windows, and macOS. Some examples may also require additional packages like pandas.

For more details, check our [installation guide](https://sysidentpy.org/getting-started/getting-started/)

## What are the main features of SysIdentPy?

| Feature | What is this? |
|----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| NARMAX philosophy | You can build variations of NARMAX models like NARX, NAR, NARMA, NFIR, ARMA, ARX, AR, and others. |
| Model Structure Selection | Easy-to-use methods to select the best terms to build your models, including FROLS and MetaMSS and several combinations with parameter estimation techniques to select the model terms. |
| Basis Function | You can use up to 8 different basis functions to build your models. You can set linear and nonlinear basis functions and ensemble them to get custom NARMAX models. |
| Parameter Estimation | More than 15 methods to estimate the model parameters and test different structure selection scenarios. |
| Multiobjective Parameter Estimation | You can use affine information to estimate the model parameters minimizing different objective functions. |
| Model Simulation | You can reproduce results from papers easily with SimulateNARMAX class. Moreover, you can test published models with different parameter estimation methods and compare the performance. |
| Neural NARX | You can use SysIdentPy with Pytorch to create custom neural NARX models architectures which support all the optimizers and loss functions from Pytorch. |
| General Estimators | You can use estimators from packages like scikit-learn, Catboost, and many other compatible interfaces and composition tools to create NARMAX models. |

## Why does SysIdentPy exist?

SysIdentPy aims to be a free and open-source package to help the community to design NARMAX models for System Identification and TimeSeries Forecasting. More than that, be a free and robust alternative to one of the most used tools to build NARMAX models, which is the Matlab's System Identification Toolbox.

The project is actively maintained by Wilson R. L. Junior and looking for contributors.

## How do I use SysIdentPy?

The [SysIdentPy documentation](https://sysidentpy.org) includes more than 20 examples to help get you started:
- Quickstart guide, for an [entry-level description of the main SysIdentPy concepts](https://sysidentpy.org/getting-started/quickstart-guide/)
- A dedicated section focusing on SysIdentPy features, like model structure selection algorithms, basis functions, parameter estimation, and more.
- A dedicated section focusing on use cases using SysIdentPy with real world datasets. Besides, there is some brief comparisons and benchmarks against other time series tools, like Prophet, Neural Prophet, ARIMA, and more.


### Examples
```python
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data


# Generate a dataset of a simulated dynamical system
x_train, x_valid, y_train, y_valid = get_siso_data(
  n=1000,
  colored_noise=False,
  sigma=0.001,
  train_percentage=80
)
```


#### Building Polynomial NARX models with FROLS algorithm

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)

basis_function = Polynomial(degree=2)
estimator = LeastSquares()
model = FROLS(
    order_selection=True,
    n_info_values=3,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    estimator=estimator,
    err_tol=None,
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

```
```console
Regressors     Parameters        ERR
0        x1(k-2)     0.9000  0.95556574
1         y(k-1)     0.1999  0.04107943
2  x1(k-1)y(k-1)     0.1000  0.00335113
````
```python
plot_results(y=y_valid, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```
![polynomial](./examples/figures/polynomial_narmax.png)

#### NARX Neural Network
```python
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation
from sysidentpy.residues.residues_correlation import compute_cross_correlation

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
plot_results(y=y_valid, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```
![neural](/examples/figures/narx_network.png)

#### Catboost-narx
```python
from catboost import CatBoostRegressor
from sysidentpy.general_estimators import NARX
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import compute_residues_autocorrelation
from sysidentpy.residues.residues_correlation import compute_cross_correlation


basis_function=Polynomial(degree=1)

catboost_narx = NARX(
  base_estimator=CatBoostRegressor(
    iterations=300,
    learning_rate=0.1,
    depth=6),
  xlag=2,
  ylag=2,
  basis_function=basis_function,
  fit_params={'verbose': False}
)

catboost_narx.fit(X=x_train, y=y_train)
yhat = catboost_narx.predict(X=x_valid, y=y_valid)
plot_results(y=y_valid, yhat=yhat, n=1000)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```
![catboost](/examples/figures/catboost_narx.png)

#### Catboost without NARX configuration

The following is the Catboost performance without the NARX configuration.


```python

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
plot_results_tmp(y_valid, catboost.predict(x_valid))
```
![catboost](/examples/figures/catboost.png)

The examples directory has several Jupyter notebooks with tutorials of how to use the package and some specific applications of sysidentpy. Try it out!

## Communication

- Discord server: https://discord.gg/8eGE3PQ

  [![discord](https://img.shields.io/discord/711610087700955176?color=%23FF7800&label=discord)](https://discord.gg/8eGE3PQ)


- Website: http://sysidentpy.org

## Citation
[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.02384-%23FF7800)](https://joss.theoj.org/papers/10.21105/joss.02384)

If you use SysIdentPy on your project, please [drop me a line](mailto:wilsonrljr@outlook.com).

If you use SysIdentPy on your scientific publication, we would appreciate citations to the following paper:

- Lacerda et al., (2020). SysIdentPy: A Python package for System Identification using NARMAX models. Journal of Open Source Software, 5(54), 2384, https://doi.org/10.21105/joss.02384

```
@article{Lacerda2020,
  doi = {10.21105/joss.02384},
  url = {https://doi.org/10.21105/joss.02384},
  year = {2020},
  publisher = {The Open Journal},
  volume = {5},
  number = {54},
  pages = {2384},
  author = {Wilson Rocha Lacerda Junior and Luan Pascoal Costa da Andrade and Samuel Carlos Pessoa Oliveira and Samir Angelo Milani Martins},
  title = {SysIdentPy: A Python package for System Identification using NARMAX models},
  journal = {Journal of Open Source Software}
}
```

## Inspiration

The documentation and structure (even this section) is openly inspired by Scikit-learn, EinsteinPy, and many others as we used (and keep using) them to learn.

## Contributors

<a href="https://github.com/wilsonrljr/sysidentpy/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=wilsonrljr/sysidentpy" width = 500/>
</a>

## Sponsors

**Special thanks** to our **sponsors**

### Monthly Sponsors

<a href="https://github.com/statisticallyinsifnificant">
    <img alt="statisticallyinsifnificant" src="https://avatars.githubusercontent.com/u/158107107?v=4" width="90" height="90">
</a>

<hr />

### Individual Sponsors

<a href="https://github.com/nataliakeles">
    <img alt="Nath Keles" src="https://avatars.githubusercontent.com/u/61664158?v=4" width="90" height="90">
</a>

<hr />

  [wilsonrljr's sponsor profile]: https://github.com/sponsors/wilsonrljr


### Powered by
[![PyCharm logo](https://resources.jetbrains.com/storage/products/company/brand/logos/PyCharm.svg)](https://jb.gg/OpenSourceSupport)

<hr />
