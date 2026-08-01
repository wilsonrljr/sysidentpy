# Air Passenger Benchmark

Example created by Wilson Rocha Lacerda Junior

> **Looking for more details on NARMAX models?**
> For comprehensive information on models, methods, and a wide range of examples and benchmarks implemented in SysIdentPy, check out our book:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> This book provides in-depth guidance to support your work with SysIdentPy.

## Reproducibility

This tutorial was verified with SysIdentPy 0.9.0 on Python 3.12.12. Its optional
packages are `sktime==1.0.1`, `neuralprophet==0.9.0`, `prophet==1.3.0`,
`pmdarima==2.1.1`, `tbats==1.1.3`, `statsmodels==0.14.6`,
`scipy==1.15.3` and `torch==2.5.1`. The SciPy pin satisfies the
`scipy<1.16` constraint enforced by the BATS/TBATS adapters in this sktime
version. Randomized models use an explicit seed; numerical results must be
recomputed when the environment or model configuration changes.

## Note

The following example is **not** intended to say that one library is better than another. The main focus of these examples is to show that SysIdentPy can be a good alternative for people looking to model time series.

We compare SysIdentPy with forecasters from **sktime** and with the standalone
**Prophet** and **NeuralProphet** libraries.

From sktime, the following models will be used:

- AutoARIMA

- ARIMA

- BATS

- TBATS

- Exponential Smoothing

- AutoETS

For the sake of brevity, from **SysIdentPy** only the **MetaMSS**, **AOLS**, **FROLS** (with polynomial base function) and **NARXNN** methods will be used. See the SysIdentPy documentation to learn other ways of modeling with the library.

```python
import logging
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from neuralprophet import NeuralProphet, set_random_seed
from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.bats import BATS
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from prophet import Prophet
from sktime.forecasting.tbats import TBATS
from sktime.split import temporal_train_test_split
from torch import nn

from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import mean_squared_error
from sysidentpy.model_structure_selection import AOLS, FROLS, MetaMSS
from sysidentpy.neural_network import NARXNN
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.plotting import plot_results

simplefilter("ignore", FutureWarning)
loss = mean_squared_error


def plot_series(*series, labels):
    for values, label in zip(series, labels):
        index = (
            values.index.to_timestamp()
            if isinstance(values.index, pd.PeriodIndex)
            else values.index
        )
        plt.plot(index, values.to_numpy(), label=label)
    plt.legend()

logging.getLogger("NP").setLevel(logging.ERROR)
```

## Air passengers data

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-01.png?raw=true)

## Reproduced results

Every model is trained on the first 120 observations and forecasts the final 24
months. The table reports mean squared error (MSE); lower is better. Randomized
models use seed 42.

MetaMSS also uses a chronological internal split: the final 17% of the 120-sample
training block guides structure selection. Its selected parameters are fitted on
the preceding internal identification segment; `fit` does not refit them on the
complete block.

| Rank | Method | MSE |
| ---: | --- | ---: |
| 1 | SysIdentPy (AOLS) | 440.9993 |
| 2 | SysIdentPy (MetaMSS) | 510.3495 |
| 3 | NeuralProphet | 514.0477 |
| 4 | Prophet | 910.7187 |
| 5 | Exponential Smoothing | 1055.5128 |
| 6 | SysIdentPy (Neural NARX) | 1621.5225 |
| 7 | SysIdentPy (FROLS) | 1811.9000 |
| 8 | AutoARIMA | 2230.3321 |
| 9 | Manual ARIMA | 2592.7244 |
| 10 | AutoETS | 3128.9366 |
| 11 | TBATS | 8825.0097 |
| 12 | BATS | 9043.4934 |

This is a controlled comparison of the configurations shown below, not a general
ranking of the libraries.

The historical Neural NARX result, MSE 316.5409, is not a result on this split.
It fitted the network to the first 108 observations, used the next 13 measured
outputs only to initialize the recursion, and scored the final 23 months.
Reproducing that protocol with the current stack gives 316.8340. With the common
120/24 split, the previous mini-batch configuration gives 2967.6219. Using only
an internal 96/24 split of the training block, the configuration below selects a
128-sample full batch, learning rate 0.01 and 1,500 epochs; after refitting on all
120 training observations, its free-run MSE is 1621.5225. Its one-step-ahead MSE
is 307.1843, which identifies recursive error accumulation as the main remaining
limitation.

## SysIdentPy FROLS

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

basis_function = Polynomial(degree=1)
sysidentpy = FROLS(
    order_selection=True,
    ylag=13,  # the lags for all models will be 13
    n_info_values=14,
    basis_function=basis_function,
    model_type="NAR",
    estimator=LeastSquares(),
)
sysidentpy.fit(y=y_train)
y_test = np.concatenate([y_train[-sysidentpy.max_lag :], y_test])

yhat = sysidentpy.predict(y=y_test, forecast_horizon=24)
frols_loss = loss(
    y_test[sysidentpy.max_lag :],
    yhat[sysidentpy.max_lag :],
)
print(frols_loss)

plot_results(y=y_test[sysidentpy.max_lag :], yhat=yhat[sysidentpy.max_lag :])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-02.png?raw=true)

## SysIdentPy AOLS

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

df_train, df_test = temporal_train_test_split(y, test_size=24)
df_train = df_train.reset_index()
df_train.columns = ["ds", "y"]
df_train["ds"] = pd.to_datetime(df_train["ds"].astype(str))
df_test = df_test.reset_index()
df_test.columns = ["ds", "y"]
df_test["ds"] = pd.to_datetime(df_test["ds"].astype(str))

sysidentpy_AOLS = AOLS(
    ylag=13, k=2, L=1, model_type="NAR", basis_function=basis_function
)
sysidentpy_AOLS.fit(y=y_train)
y_test = np.concatenate([y_train[-sysidentpy_AOLS.max_lag :], y_test])

yhat = sysidentpy_AOLS.predict(y=y_test, steps_ahead=None, forecast_horizon=24)
aols_loss = loss(
    y_test[sysidentpy_AOLS.max_lag :],
    yhat[sysidentpy_AOLS.max_lag :],
)
print(aols_loss)

plot_results(y=y_test[sysidentpy_AOLS.max_lag :], yhat=yhat[sysidentpy_AOLS.max_lag :])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-03.png?raw=true)

## SysIdentPy MetaMSS

```python
set_random_seed(42)

y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)

sysidentpy_metamss = MetaMSS(
    basis_function=basis_function, ylag=13, model_type="NAR", test_size=0.17, random_state=42
)
sysidentpy_metamss.fit(y=y_train)

y_test = np.concatenate([y_train[-sysidentpy_metamss.max_lag :], y_test])

yhat = sysidentpy_metamss.predict(y=y_test, steps_ahead=None, forecast_horizon=24)
metamss_loss = loss(
    y_test[sysidentpy_metamss.max_lag :],
    yhat[sysidentpy_metamss.max_lag :],
)
print(metamss_loss)

plot_results(
    y=y_test[sysidentpy_metamss.max_lag :], yhat=yhat[sysidentpy_metamss.max_lag :]
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-04.png?raw=true)

## SysIdentPy Neural NARX

```python
import torch

torch.manual_seed(42)

y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
y_train = y_train.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)
x_train = np.zeros_like(y_train)
x_test = np.zeros_like(y_test)


class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(13, 20)
        self.lin2 = nn.Linear(20, 20)
        self.lin3 = nn.Linear(20, 20)
        self.lin4 = nn.Linear(20, 1)
        self.relu = nn.ReLU()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.relu(z)
        z = self.lin3(z)
        z = self.relu(z)
        z = self.lin4(z)
        return z


narx_net = NARXNN(
    net=NARX(),
    ylag=13,
    model_type="NAR",
    basis_function=Polynomial(degree=1),
    batch_size=128,
    epochs=1500,
    verbose=False,
    learning_rate=1e-02,
    optim_params={},  # optional parameters of the optimizer
    random_state=42,
)

narx_net.fit(y=y_train)
y_initial = y_train[-narx_net.max_lag :]
yhat = narx_net.predict(y=y_initial, forecast_horizon=24)
narxnet_loss = loss(y_test, yhat[narx_net.max_lag :])
print(narxnet_loss)
plot_results(y=y_test, yhat=yhat[narx_net.max_lag :])

one_step_context = np.concatenate([y_initial, y_test])
one_step_yhat = narx_net.predict(y=one_step_context, steps_ahead=1)
narxnet_one_step_loss = loss(
    y_test, one_step_yhat[narx_net.max_lag :]
)
print(narxnet_one_step_loss)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-05.png?raw=true)

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-06.png?raw=true)

## Exponential Smoothing

```python
es = ExponentialSmoothing(trend="add", seasonal="multiplicative", sp=12)
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
es.fit(y_train)
y_pred_es = es.predict(fh)

plot_series(y_test, y_pred_es, labels=["y_test", "y_pred"])
es_loss = loss(y_test, y_pred_es)
es_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-07.png?raw=true)

## AutoETS

```python
y = load_airline()

y_train, y_test = temporal_train_test_split(y, test_size=24)
ets = AutoETS(auto=True, sp=12, n_jobs=-1)
ets.fit(y_train)
y_pred_ets = ets.predict(fh)

plot_series(y_test, y_pred_ets, labels=["y_test", "y_pred"])
ets_loss = loss(y_test, y_pred_ets)
ets_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-08.png?raw=true)

## AutoArima

```python
auto_arima = AutoARIMA(sp=12, suppress_warnings=True)
y = load_airline()

y_train, y_test = temporal_train_test_split(y, test_size=24)
auto_arima.fit(y_train)
y_pred_auto_arima = auto_arima.predict(fh)

plot_series(y_test, y_pred_auto_arima, labels=["y_test", "y_pred"])
autoarima_loss = loss(y_test, y_pred_auto_arima)
autoarima_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-09.png?raw=true)

## Arima

```python
y = load_airline()

y_train, y_test = temporal_train_test_split(y, test_size=24)
manual_arima = ARIMA(
    order=(13, 1, 0), suppress_warnings=True
)  # seasonal_order=(0, 1, 0, 12)
manual_arima.fit(y_train)
y_pred_manual_arima = manual_arima.predict(fh)
plot_series(y_test, y_pred_manual_arima, labels=["y_test", "y_pred"])
manualarima_loss = loss(y_test, y_pred_manual_arima)
manualarima_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-10.png?raw=true)

## BATS

```python
y = load_airline()

y_train, y_test = temporal_train_test_split(y, test_size=24)
bats = BATS(sp=12, use_trend=True, use_box_cox=False)
bats.fit(y_train)
y_pred_bats = bats.predict(fh)

plot_series(y_test, y_pred_bats, labels=["y_test", "y_pred"])
bats_loss = loss(y_test, y_pred_bats)
bats_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-11.png?raw=true)

## TBATS

```python
y = load_airline()

y_train, y_test = temporal_train_test_split(y, test_size=24)
tbats = TBATS(sp=12, use_trend=True, use_box_cox=False)
tbats.fit(y_train)
y_pred_tbats = tbats.predict(fh)
plot_series(y_test, y_pred_tbats, labels=["y_test", "y_pred"])
tbats_loss = loss(y_test, y_pred_tbats)
tbats_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-12.png?raw=true)

## Prophet

```python
set_random_seed(42)

y = load_airline().to_timestamp(how="start")
df = y.rename_axis("ds").rename("y").reset_index()
df_train = df.iloc[:-24].copy()
df_test = df.iloc[-24:].copy()

prophet = Prophet(
    seasonality_mode="multiplicative",
    n_changepoints=int(len(df_train) / 12),
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)
prophet.add_country_holidays(country_name="Germany")
prophet.fit(df_train)
forecast_prophet = prophet.predict(df_test[["ds"]])
y_pred_prophet = forecast_prophet["yhat"].to_numpy()

plot_series(
    df_test.set_index("ds")["y"],
    pd.Series(y_pred_prophet, index=df_test["ds"]),
    labels=["y_test", "y_pred"],
)
prophet_loss = loss(df_test["y"].to_numpy(), y_pred_prophet)
prophet_loss
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-13.png?raw=true)

## Neural Prophet

```python
set_random_seed(42)

y = load_airline().to_timestamp(how="start")
df = y.rename_axis("ds").rename("y").reset_index()
df_train = df.iloc[:-24].copy()
df_test = df.iloc[-24:].copy()

m = NeuralProphet(
    seasonality_mode="multiplicative", epochs=100, learning_rate=0.01
)
m.fit(df_train, freq="MS", progress=None)
future = m.make_future_dataframe(
    df_train, periods=24, n_historic_predictions=False
)
forecast = m.predict(future)

neuralprophet_loss = loss(
    df_test["y"].to_numpy(), forecast["yhat1"].to_numpy()
)
print(neuralprophet_loss)
plt.plot(df_test["ds"], df_test["y"], label="observed")
plt.plot(forecast["ds"], forecast["yhat1"], label="predicted")
plt.legend()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-14.png?raw=true)

```python
results = {
    "Exponential Smoothing": es_loss,
    "ETS": ets_loss,
    "AutoArima": autoarima_loss,
    "Manual Arima": manualarima_loss,
    "BATS": bats_loss,
    "TBATS": tbats_loss,
    "Prophet": prophet_loss,
    "SysIdentPy (Polynomial Model)": frols_loss,
    "SysIdentPy (Neural Model)": narxnet_loss,
    "SysIdentPy (AOLS)": aols_loss,
    "SysIdentPy (MetaMSS)": metamss_loss,
    "NeuralProphet": neuralprophet_loss,
}

sorted(results.items(), key=lambda result: result[1])
```
