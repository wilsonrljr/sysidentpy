##  M4 Dataset

The M4 dataset is a well known resource for time series forecasting, offering a wide range of data series used to test and improve forecasting methods. Created for the M4 competition organized by Spyros Makridakis, this dataset has driven many advancements in forecasting techniques.

The M4 dataset includes 100,000 time series from various fields such as demographics, finance, industry, macroeconomics, and microeconomics, which were selected randomly from the ForeDeCk database. The series come in different frequencies (yearly, quarterly, monthly, weekly, daily, and hourly), making it a comprehensive collection for testing forecasting methods.

In this case study, we will focus on the hourly subset of the M4 dataset. This subset consists of time series data recorded hourly, providing a detailed and high-frequency look at changes over time. Hourly data presents unique challenges due to its granularity and the potential for capturing short-term fluctuations and patterns.

The M4 dataset provides a standard benchmark to compare different forecasting methods, allowing researchers and practitioners to evaluate their models consistently. With series from various domains and frequencies, the M4 dataset represents real-world forecasting challenges, making it valuable for developing robust forecasting techniques. The competition and the dataset itself have led to the creation of new algorithms and methods, significantly improving forecasting accuracy and reliability.

We will present a end to end walkthrough using the M4 hourly dataset to demonstrate the capabilities of SysIdentPy. SysIdentPy offers a range of tools and techniques designed to effectively handle the complexities of time series data, but we will focus on fast and easy setup for this case. We will cover model selection and evaluation metrics specific to the hourly dataset.

By the end of this case study, you will have a solid understanding of how to use SysIdentPy for forecasting with the M4 hourly dataset, preparing you to tackle similar forecasting challenges in real-world scenarios.

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12 and
`datasetsforecast==1.0.1`. Install the repository checkout and the M4 loader
explicitly:

```bash
python -m pip install -e .
python -m pip install datasetsforecast==1.0.1
```

Use a virtual environment to isolate these optional dependencies. Randomized
examples use seed 42; numerical results should be recomputed if the environment
or model configuration changes.

### SysIdentPy configuration

In this section, we will demonstrate the application of SysIdentPy to the M4
hourly dataset. The following code guides you through loading the data,
configuring SysIdentPy and building the forecasting models used in this study.

```python
import warnings
import numpy as np
import pandas as pd
from pandas.errors import SettingWithCopyWarning
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS, AOLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import (
    root_relative_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from sysidentpy.utils.plotting import plot_results

from datasetsforecast.m4 import M4, M4Evaluation

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

m4_data, _, _ = M4.load(directory="data", group="Hourly")
test = (
    m4_data.groupby("unique_id", group_keys=False)
    .tail(48)
    .rename(columns={"y": "y_test"})
)
train = m4_data.drop(test.index)
```

The following plots provide a visualization of the training data for a small subset of the time series. The plot shows the raw data, giving you an insight into the patterns and behaviors inherent in each series.

By observing the data, you can get a sense of the variety and complexity of the time series we are working with. The plots can reveal important characteristics such as trends, seasonal patterns, and potential anomalies within the time series. Understanding these elements is crucial for the development of accurate forecasting models.

However, when dealing with a large number of different time series, it is common to start with broad assumptions rather than detailed individual analysis. In this context, we will adopt a similar approach. Instead of going into the specifics of each dataset, we will make some general assumptions and see how SysIdentPy handles them.

This approach provides a practical starting point, demonstrating how SysIdentPy can manage different types of time series data without too much work. As you become more familiar with the tool, you can refine your models with more detailed insights. For now, let's focus on using SysIdentPy to create the forecasts based on these initial assumptions.

Our first assumption is that there is a 24-hour seasonal pattern in the series. By examining the plots below, this seems reasonable. Therefore, we'll begin building our models with `ylag=24`.

```python
for unique_id in ("H10", "H100", "H20", "H150"):
    ax = (
        train[train["unique_id"] == unique_id]
        .reset_index(drop=True)["y"]
        .plot(figsize=(15, 2), title=unique_id)
    )
    for xc in range(24, 24 * 30, 24):
        ax.axvline(x=xc, color="red", linestyle="--", alpha=0.5)
    plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-01.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h100_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h20_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h150_1.png?raw=true)

Let's check build a model for the `H20` group before we extrapolate the settings for every group. Because there are no input features, we will be using a `NAR` model type in SysIdentPy. To keep things simple and fast, we will start with Polynomial basis function with degree $1$.

```python
unique_id = "H20"
y_id = train[train["unique_id"] == unique_id]["y"].values.reshape(-1, 1)
y_val = test[test["unique_id"] == unique_id]["y_test"].values.reshape(-1, 1)

basis_function = Polynomial(degree=1)
model = FROLS(
    order_selection=True,
    ylag=24,
    estimator=LeastSquares(),
    basis_function=basis_function,
    model_type="NAR",
)

model.fit(y=y_id)
y_val = np.concatenate([y_id[-model.max_lag :], y_val])
y_hat = model.predict(y=y_val, forecast_horizon=48)
smape = symmetric_mean_absolute_percentage_error(
    y_val[model.max_lag : :], y_hat[model.max_lag : :]
)

plot_results(
    y=y_val[model.max_lag :],
    yhat=y_hat[model.max_lag :],
    n=30000,
    figsize=(15, 4),
    title=f"Group: {unique_id} - SMAPE {round(smape, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-02.png?raw=true)

Probably, the result are not optimal and will not work for every group. However, let's check how this setting performs against the winner model [M4 time series competition](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward): the Exponential Smoothing with Recurrent Neural Networks ([ESRNN](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153)).

```python
esrnn_url = (
    "https://github.com/Nixtla/m4-forecasts/raw/e3dce409604c55f1f588f02db439b4cbe9a482a3/forecasts/submission-118.zip"
)
esrnn_forecasts = M4Evaluation.load_benchmark("data", "Hourly", esrnn_url)
esrnn_evaluation = M4Evaluation.evaluate("data", "Hourly", esrnn_forecasts)

esrnn_evaluation
```

|        | SMAPE | MASE  | OWA   |
| ------ | ----- | ----- | ----- |
| Hourly | 9.328 | 0.893 | 0.440 |
> Table 1. ESRNN SOTA results

The following code took only 49 seconds to run on my machine (AMD Ryzen 5 5600x processor, 32GB RAM at 3600MHz). Because of its efficiency, I didn't create a parallel version. By the end of this use case, you will see how SysIdentPy can be both fast and effective, delivering good results without too much optimization.

```python
r = []
ds_test = list(range(701, 749))
for u_id, data in train.groupby("unique_id", observed=True):
    y_id = data["y"].values.reshape(-1, 1)
    basis_function = Polynomial(degree=1)
    model = FROLS(
        ylag=24,
        estimator=LeastSquares(),
        basis_function=basis_function,
        model_type="NAR",
        n_info_values=25,
    )
    try:
        model.fit(y=y_id)
        y_val = y_id[-model.max_lag :].reshape(-1, 1)
        y_hat = model.predict(y=y_val, forecast_horizon=48)
        forecast = y_hat[model.max_lag :].ravel()
        if forecast.shape != (48,) or not np.isfinite(forecast).all():
            raise RuntimeError(f"Invalid 48-step forecast for {u_id}.")
        r.append(
            [
                [u_id] * 48,
                ds_test,
                forecast,
            ]
        )
    except Exception as exc:
        raise RuntimeError(f"Forecasting failed for {u_id}.") from exc

results_1 = pd.DataFrame(r, columns=["unique_id", "ds", "NARMAX_1"]).explode(
    ["unique_id", "ds", "NARMAX_1"]
)
results_1["NARMAX_1"] = results_1["NARMAX_1"].astype(float)  # .clip(lower=10)
expected_ids = train["unique_id"].drop_duplicates().tolist()
pivot_df = results_1.pivot(
    index="unique_id", columns="ds", values="NARMAX_1"
).reindex(expected_ids)
results = pivot_df.to_numpy()
if len(expected_ids) != 414 or results.shape != (414, 48):
    raise RuntimeError("The M4 hourly evaluation requires 414 complete forecasts.")
if not np.isfinite(results).all():
    raise RuntimeError("The M4 forecast matrix contains non-finite values.")

daily_evaluation = M4Evaluation.evaluate("data", "Hourly", results)
h147_index = expected_ids.index("H147")
h147_observed = test.loc[test["unique_id"] == "H147", "y_test"].to_numpy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(h147_observed, "o-", label="Observed")
ax.plot(results[h147_index], "*-", label="FROLS, 24 lags")
ax.set_title("H147: 48-step forecast with daily lags")
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("y")
ax.legend()
plt.show()
daily_evaluation
```

|        |    SMAPE   |    MASE    |    OWA     |
|--------|------------|------------|------------|
| Hourly | 16.034196  | 0.958083   | 0.636132   |
Table 2. First test with SysIdentPy

The initial results are reasonable, but they don't quite match the performance of `ESRNN`. These results are based solely on our first assumption. To better understand the performance, let’s examine the groups with the worst results.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-03.png?raw=true)

The following plot illustrates two such groups, `H147` and `H136`. Both exhibit a 24-hour seasonal pattern.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_seasonal_h147_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h136_seasonal_1.png?raw=true)

However, a closer look reveals an additional insight: in addition to the daily pattern, these series also show a weekly pattern. Observe how the data looks like when we split the series into weekly segments.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h147_seasonal_1.png?raw=true)

```python
xcoords = list(range(0, 168 * 5, 168))
filtered_train = train[train["unique_id"] == "H147"].reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 1.5 * len(xcoords[1:])))
for i, start in enumerate(xcoords[:-1]):
    end = xcoords[i + 1]
    ax = fig.add_subplot(len(xcoords[1:]), 1, i + 1)
    filtered_train["y"].iloc[start:end].plot(ax=ax)
    ax.set_title(f"H147 -> Slice {i+1}: Hour {start} to {end-1}")

plt.tight_layout()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-04.png?raw=true)

Therefore, we will build models setting `ylag=168`.

> Note that this is a very high number for lags, so be careful if you want to try it with higher polynomial degrees because the time to run the models can increase significantly. I tried some configurations with polynomial degree equal to 2 and only took $6$ minutes to run (even less, using `AOLS`), without making the code run in parallel. As you can see, SysIdentPy can be very fast, and you can make it faster by applying parallelization.

```python
r = []
ds_test = list(range(701, 749))
for u_id, data in train.groupby("unique_id", observed=True):
    y_id = data["y"].values.reshape(-1, 1)
    basis_function = Polynomial(degree=1)
    model = FROLS(
        ylag=168,
        estimator=LeastSquares(),
        basis_function=basis_function,
        model_type="NAR",
    )
    try:
        model.fit(y=y_id)
        y_val = y_id[-model.max_lag :].reshape(-1, 1)
        y_hat = model.predict(y=y_val, forecast_horizon=48)
        forecast = y_hat[model.max_lag :].ravel()
        if forecast.shape != (48,) or not np.isfinite(forecast).all():
            raise RuntimeError(f"Invalid 48-step forecast for {u_id}.")
        r.append(
            [
                [u_id] * 48,
                ds_test,
                forecast,
            ]
        )
    except Exception as exc:
        raise RuntimeError(f"Forecasting failed for {u_id}.") from exc

results_1 = pd.DataFrame(r, columns=["unique_id", "ds", "NARMAX_1"]).explode(
    ["unique_id", "ds", "NARMAX_1"]
)
results_1["NARMAX_1"] = results_1["NARMAX_1"].astype(float)  # .clip(lower=10)
expected_ids = train["unique_id"].drop_duplicates().tolist()
pivot_df = results_1.pivot(
    index="unique_id", columns="ds", values="NARMAX_1"
).reindex(expected_ids)
results = pivot_df.to_numpy()
if len(expected_ids) != 414 or results.shape != (414, 48):
    raise RuntimeError("The M4 hourly evaluation requires 414 complete forecasts.")
if not np.isfinite(results).all():
    raise RuntimeError("The M4 forecast matrix contains non-finite values.")
weekly_evaluation = M4Evaluation.evaluate("data", "Hourly", results)
h147_index = expected_ids.index("H147")
h147_observed = test.loc[test["unique_id"] == "H147", "y_test"].to_numpy()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(h147_observed, "o-", label="Observed")
ax.plot(results[h147_index], "*-", label="FROLS, 168 lags")
ax.set_title("H147: 48-step forecast with weekly lags")
ax.set_xlabel("Forecast horizon")
ax.set_ylabel("y")
ax.legend()
plt.show()
weekly_evaluation
```

|        |    SMAPE   |    MASE    |    OWA     |
|--------|------------|------------|------------|
| Hourly | 10.475998  | 0.773749   | 0.446471   |
> Table 3. Improved results using SysIdentPy

Now, the results are much closer to those of the `ESRNN` model! While the Symmetric Mean Absolute Percentage Error (`SMAPE`) is slightly worse, the Mean Absolute Scaled Error (`MASE`) is better when comparing against `ESRNN`, leading to a very similar Overall Weighted Average (`OWA`) metric. Remarkably, these results are achieved using only simple `AR` models. Next, let's see if the `AOLS` method can provide even better results.

```python
r = []
ds_test = list(range(701, 749))
for u_id, data in train.groupby("unique_id", observed=True):
    y_id = data["y"].values.reshape(-1, 1)
    basis_function = Polynomial(degree=1)
    model = AOLS(
        ylag=168,
        basis_function=basis_function,
        model_type="NAR",
        # due to high lag settings, k was increased to 6 as an initial guess
        k=6,
    )
    try:
        model.fit(y=y_id)
        y_val = y_id[-model.max_lag :].reshape(-1, 1)
        y_hat = model.predict(y=y_val, forecast_horizon=48)
        forecast = y_hat[model.max_lag :].ravel()
        if forecast.shape != (48,) or not np.isfinite(forecast).all():
            raise RuntimeError(f"Invalid 48-step forecast for {u_id}.")
        r.append(
            [
                [u_id] * 48,
                ds_test,
                forecast,
            ]
        )
    except Exception as exc:
        raise RuntimeError(f"Forecasting failed for {u_id}.") from exc

results_1 = pd.DataFrame(r, columns=["unique_id", "ds", "NARMAX_1"]).explode(
    ["unique_id", "ds", "NARMAX_1"]
)
results_1["NARMAX_1"] = results_1["NARMAX_1"].astype(float)  # .clip(lower=10)
expected_ids = train["unique_id"].drop_duplicates().tolist()
pivot_df = results_1.pivot(
    index="unique_id", columns="ds", values="NARMAX_1"
).reindex(expected_ids)
results = pivot_df.to_numpy()
if len(expected_ids) != 414 or results.shape != (414, 48):
    raise RuntimeError("The M4 hourly evaluation requires 414 complete forecasts.")
if not np.isfinite(results).all():
    raise RuntimeError("The M4 forecast matrix contains non-finite values.")
M4Evaluation.evaluate("data", "Hourly", results)
```

|        |    SMAPE   |    MASE    |    OWA     |
|--------|------------|------------|------------|
| Hourly | 9.9497     | 0.8074     | 0.4392     |
> Table 4. SysIdentPy results using AOLS algorithm

For this configuration, the Overall Weighted Average (`OWA`) is slightly lower
than that of the `ESRNN` reference. This conclusion applies to the complete set
of 414 hourly series and the 48-step competition horizon; it is not a general
ranking of the algorithms.

Before we finish, let's verify how the performance of the `H147` model has improved with the `ylag=168` setting.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-05.png?raw=true)

> Based on the M4 benchmark paper, we could also clip the predictions lower than 10 to 10 and the results would be slightly better. But this is left to the user.

We could achieve even better performance with some fine-tuning of the model configuration. However, I’ll leave exploring these alternative adjustments as an exercise for the user. However, keep in mind that experimenting with different settings does not always guarantee improved results. A deeper theoretical knowledge can often lead you to better configurations and, hence, better results.

## Coupled Eletric Device

The CE8 coupled electric drives [dataset - Nonlinear Benchmark](https://www.nonlinearbenchmark.org/benchmarks) presents a compelling use case for demonstrating the performance of SysIdentPy. This system involves two electric motors driving a pulley with a flexible belt, creating a dynamic environment ideal for testing system identification tools.

> The [nonlinear benchmark website](https://www.nonlinearbenchmark.org/benchmarks) stands as a significant contribution to the system identification and machine learning community. The users are encouraged to explore all the papers referenced on the site.

### System Overview

The CE8 system, illustrated in Figure 1, features:
- **Two Electric Motors**: These motors independently control the tension and speed of the belt, providing symmetrical control around zero. This enables both clockwise and counterclockwise movements.
- **Pulley Mechanism**: The pulley is supported by a spring, introducing a lightly damped dynamic mode that adds complexity to the system.
- **Speed Control Focus**: The primary focus is on the speed control system. The pulley’s angular speed is measured using a pulse counter, which is insensitive to the direction of the velocity.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/ce8_design.png?raw=true)
> Figure 1. CE8 system design.

### Sensor and Filtering

The measurement process involves:
- **Pulse Counter**: This sensor measures the angular speed of the pulley without regard to the direction.
- **Analogue Low Pass Filtering**: This reduces high-frequency noise, followed by antialiasing filtering to prepare the signal for digital processing. The dynamic effects are mainly influenced by the electric drive time constants and the spring, with the low pass filtering having a minimal impact on the output.

### SOTA Results

SysIdentPy can be used to build robust models for identifying and modeling the complex dynamics of the CE8 system. The performance will be compared against a benchmark provided by [Max D. Champneys, Gerben I. Beintema, Roland Tóth, Maarten Schoukens, and Timothy J. Rogers - Baselines for Nonlinear Benchmarks, Workshop on Nonlinear System Identification Benchmarks, 2024.](https://arxiv.org/pdf/2405.10779)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/ce8_sota.png?raw=true)

The benchmark evaluate the average metric between the two experiments. That's why the SOTA method do not have the better metric for `test 1`, but it is still the best overall.  The goal of this case study is not only to showcase the robustness of SysIdentPy but also provides valuable insights into its practical applications in real-world dynamic systems.

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12 and
`nonlinear-benchmarks==1.0.1`. Install the repository checkout and the official
benchmark loader explicitly:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use a virtual environment to isolate the optional loader. Numerical results
should be recomputed if the environment or model configuration changes.

### SysIdentPy configuration

In this section, we will demonstrate the application of SysIdentPy to the CE8 coupled electric drives dataset. This example showcases the robust performance of SysIdentPy in modeling and identifying complex dynamic systems. The following code will guide you through the process of loading the dataset, configuring the SysIdentPy parameters, and building a model for CE8 system.

This practical example will help users understand how to effectively utilize SysIdentPy for their own system identification tasks, leveraging its advanced features to handle the complexities of real-world dynamic systems. Let's dive into the code and explore the capabilities of SysIdentPy.

```python
from warnings import catch_warnings, simplefilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.utils.display_results import results
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_mean_squared_error
from sysidentpy.utils.plotting import plot_results

import nonlinear_benchmarks

ced_url = (
    "https://web.archive.org/web/20210117142533id_/"
    "http://www.it.uu.se/research/publications/reports/2010-020/NonlinearData.zip"
)
train_val, test = nonlinear_benchmarks.CED(url=ced_url, atleast_2d=True)
data_train_1, data_train_2 = train_val
data_test_1, data_test_2 = test
```

We used the `nonlinear_benchmarks` package to load the data. The user is referred to the package documentation [GerbenBeintema - nonlinear_benchmarks: The official data load for nonlinear benchmark datasets](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) to check the details of how to use it.

The following plot detail the training and testing data of both experiments. Here we are trying to get two models, one for each experiment, that have a better performance than the mentioned baselines.

```python
plt.plot(data_train_1.u)
plt.plot(data_train_1.y)
plt.title("Experiment 1: training data")
plt.show()

plt.plot(data_test_1.u)
plt.plot(data_test_1.y)
plt.title("Experiment 1: testing data")
plt.show()

plt.plot(data_train_2.u)
plt.plot(data_train_2.y)
plt.title("Experiment 2: training data")
plt.show()

plt.plot(data_test_2.u)
plt.plot(data_test_2.y)
plt.title("Experiment 2: testing data")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-01.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-02.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-03.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-04.png?raw=true)

### Results

First, we will set the exactly same configuration to built models for both experiments. We can have better models by optimizing the configurations individually, but we will start simple.

A basic configuration of FROLS using a polynomial basis function with degree equal 2 is defined. The information criteria will be the default one, the `aic`. The `xlag` and `ylag` are set to $7$ in this first example.

Model for experiment 1:

```python
y_train = data_train_1.y
y_test = data_test_1.y
x_train = data_train_1.u
x_test = data_test_1.u

n = data_test_1.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=7,
    ylag=7,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
    n_info_values=120,
)

with catch_warnings():
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 1 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-05.png?raw=true)

Model for experiment 2:
```python
y_train = data_train_2.y
y_test = data_test_2.y
x_train = data_train_2.u
x_test = data_test_2.u

n = data_test_2.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=7,
    ylag=7,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
    n_info_values=120,
)

with catch_warnings():
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 2 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-06.png?raw=true)

With the current loader and the benchmark-defined initialization window, this
first configuration gives RMSE $0.102862$ for experiment 1 and $0.106816$ for
experiment 2. Their average is $0.104839$. The samples reserved for state
initialization are excluded once, and the recursion starts from the last
`model.max_lag` outputs inside that window.

The external table is still useful as context, but its values should only be
compared after matching the split, initialization, normalization and aggregation
rules. We will therefore use the present RMSE values to compare the SysIdentPy
configurations with one another. Before increasing the lags, the information
criterion is shown:

```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-07.png?raw=true)

It can be observed that after 22 regressors, adding new regressors do not improve the model performance (considering the configuration defined for that model). Because we want to try models with higher lags and higher nonlinearity degree, the stopping criteria will be changed to `err_tol` instead of information criteria. This will made the algorithm runs considerably faster.

```python
# experiment 1
y_train = data_train_1.y
y_test = data_test_1.y
x_train = data_train_1.u
x_test = data_test_1.u

n = data_test_1.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(),
    err_tol=0.9996,
    n_terms=22,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
print(model.final_model.shape, model.err.sum())
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]

rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 1 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-08.png?raw=true)

```python
# experiment 2
y_train = data_train_2.y
y_test = data_test_2.y
x_train = data_train_2.u
x_test = data_test_2.u

n = data_test_2.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aicc",
    err_tol=0.9996,
    n_terms=22,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]

rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 2 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-09.png?raw=true)

The 10-lag, 22-term models give RMSE $0.110933$ and $0.107076$ for experiments
1 and 2, respectively. The official state-initialization window contains 10
samples, so the former 14-lag configurations are not valid under this protocol.
Increasing the lag to the largest valid value does not improve this degree-2
configuration. Therefore, let's set the polynomial degree to $3$ and increase
the number of terms to `n_terms=40` if the `err_tol` is not reached. These values
are empirical; the estimator, error tolerance, structure-selection algorithm and
basis function are other possible tuning dimensions.

```python
# experiment 1
y_train = data_train_1.y
y_test = data_test_1.y
x_train = data_train_1.u
x_test = data_test_1.u

n = data_test_1.state_initialization_window_length

basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(),
    err_tol=0.9996,
    n_terms=40,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
print(model.final_model.shape, model.err.sum())
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]

rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 1 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-10.png?raw=true)

```python
# experiment 2
y_train = data_train_2.y
y_test = data_test_2.y
x_train = data_train_2.u
x_test = data_test_2.u

n = data_test_2.state_initialization_window_length

basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aicc",
    err_tol=0.9996,
    n_terms=40,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]

rmse = root_mean_squared_error(y_test[n:], yhat)
print(f"RMSE: {rmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=10000,
    title=f"Free Run simulation. Model 2 -> RMSE: {round(rmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-11.png?raw=true)

The degree-3 models give RMSE $0.112503$ and $0.096002$, with an average of
$0.104253$. The second experiment improves, whereas the first does not. This is
why the two experiments should be reported separately and why the external SOTA
table is not used to claim a ranking without an identical evaluation protocol.

## Wiener-Hammerstein

The description content primarily derives from the [benchmark website - Nonlinear Benchmark](https://www.nonlinearbenchmark.org/benchmarks) and [associated paper - Wiener-Hammerstein benchmark with process noise](https://data.4tu.nl/articles/_/12952124). For a detailed description, readers are referred to the linked references.

> The nonlinear benchmark website stands as a significant contribution to the system identification and machine learning community. The users are encouraged to explore all the papers referenced on the site.

This benchmark focuses on a Wiener-Hammerstein electronic circuit where process noise plays a significant role in distorting the output signal.

The Wiener-Hammerstein structure is a well-known block-oriented system which contains a static nonlinearity sandwiched between two Linear Time-Invariant (LTI) blocks (Figure 2). This arrangement presents a challenging identification problem due to the presence of these LTI blocks.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wh_system.png?raw=true)
> Figure 2: the Wiener-Hammerstein system

In Figure 2, the Wiener-Hammerstein system is illustrated with process noise $e_x(t)$ entering before the static nonlinearity $f(x)$, sandwiched between LTI blocks represented by $R(s)$ and $S(s)$ at the input and output, respectively. Additionally, small, negligible noise sources $e_u(t)$ and $e_y(t)$ affect the measurement channels. The measured input and output signals are denoted as $u_m(t)$ and $y_m(t)$.

The first LTI block $R(s)$ is effectively modeled as a third-order lowpass filter. The second LTI subsystem $S(s)$ is configured as an inverse Chebyshev filter with a stop-band attenuation of $40 dB$ and a cutoff frequency of $5 kHz$. Notably, $S(s)$ includes a transmission zero within the operational frequency range, complicating its inversion.

The static nonlinearity $f(x)$ is implemented using a diode-resistor network, resulting in saturation nonlinearity. Process noise $e_x(t)$ is introduced as filtered white Gaussian noise, generated from a discrete-time third-order lowpass Butterworth filter followed by zero-order hold and analog low-pass reconstruction filtering with a cutoff of $20 kHz$.

Measurement noise sources $e_u(t)$ and $e_y(t)$ are minimal compared to $e_x(t)$. The system's inputs and process noise are generated using an Arbitrary Waveform Generator (AWG), specifically the Agilent/HP E1445A, sampling at $78125 Hz$, synchronized with an acquisition system (Agilent/HP E1430A) to ensure phase coherence and prevent leakage errors. Buffering between the acquisition cards and the system's inputs and outputs minimizes measurement equipment distortion.

The benchmark provides two standard test signals through the benchmarking website: a random phase multi sine and a sine-sweep signal. Both signals have a $rms$ value of $0.71 Vrms$ and cover frequencies from DC to $15 kHz$ (excluding DC). The sine-sweep spans this frequency range at a rate of $4.29 MHz/min$. These test sets serve as targets for evaluating the model's performance, emphasizing accurate representation under varied conditions.

The Wiener-Hammerstein benchmark highlights three primary nonlinear system identification challenges:

1. **Process Noise:** Significant in the system, influencing output fidelity.
2. **Static Nonlinearity:** Indirectly accessible from measured data, posing identification challenges.
3. **Output Dynamics:** Complex inversion due to transmission zero presence in $S(s)$.

The goal of this benchmark is to develop and validate robust models using separate estimation data, ensuring accurate characterization of the Wiener-Hammerstein system's behavior.

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12 and
`nonlinear-benchmarks==1.0.1`. Install the repository checkout and the official
benchmark loader explicitly:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use a virtual environment to isolate the optional loader. Numerical results
should be recomputed if the environment or model configuration changes.

### SysIdentPy configuration

In this section, we will demonstrate the application of SysIdentPy to the Wiener-Hammerstein system dataset.  The following code will guide you through the process of loading the dataset, configuring the SysIdentPy parameters, and building a model for Wiener-Hammerstein system.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS, AOLS, MetaMSS
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.utils.display_results import results
from sysidentpy.parameter_estimation import (
    LeastSquares,
    BoundedVariableLeastSquares,
    NonNegativeLeastSquares,
    LeastSquaresMinimalResidual,
)

from sysidentpy.metrics import root_mean_squared_error
from sysidentpy.utils.plotting import plot_results

import nonlinear_benchmarks

train_val, test = nonlinear_benchmarks.WienerHammerBenchMark(atleast_2d=True)
x_train, y_train = train_val
x_test, y_test = test
```

We used the `nonlinear_benchmarks` package to load the data. The user is referred to the [package documentation](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) to check the details of how to use it.

The following plot detail the training and testing data of the experiment.

```python
plot_n = 800

plt.figure(figsize=(15, 4))
plt.plot(x_train[:plot_n])
plt.plot(y_train[:plot_n])
plt.title("Experiment: training data")
plt.legend(["x_train", "y_train"])
plt.show()

plt.figure(figsize=(15, 4))
plt.plot(x_test[:plot_n])
plt.plot(y_test[:plot_n])
plt.title("Experiment: testing data")
plt.legend(["x_test", "y_test"])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-01.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-02.png?raw=true)

The external benchmark provides useful context for the experiment. A direct
ranking, however, requires the same split, initialization window and
normalization; those conditions are made explicit below before any comparison is
drawn.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wh_sota_results.png?raw=true)
> State of the art results presented in the [benchmarking paper](https://arxiv.org/pdf/2405.10779). In this section we are only working with the Wiener-Hammerstein results, which are presented in the $W-H$  column.

### Results

We will start with a basic configuration of FROLS using a polynomial basis function with degree equal 2. The `xlag` and `ylag` are set to $7$ in this first example. Because the dataset is considerably large, we will start with `n_info_values=50`. This means the FROLS algorithm will not include all regressors when calculating the information criteria used to determine the model order. While this approach might result in a suboptimal model, it is a reasonable starting point for our first attempt.

```python
n = test.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=7,
    ylag=7,
    basis_function=basis_function,
    estimator=LeastSquares(unbiased=False),
    n_info_values=50,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=1000,
    title=f"SysIdentPy -> RMSE: {round(rmse, 4)}, NRMSE: {round(nrmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-03.png?raw=true)

The first configuration gives RMSE $0.020007$ and NRMSE $0.082029$. We started
with `xlag=ylag=7` to establish a compact baseline. The benchmarking paper uses
longer memories in some models, so the next configuration sets
`xlag=ylag=10`.

```python
x_train, y_train = train_val
x_test, y_test = test

n = test.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(unbiased=False),
    n_info_values=50,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=1000,
    title=f"SysIdentPy -> RMSE: {round(rmse, 4)}, NRMSE: {round(nrmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-04.png?raw=true)

The 10-lag configuration improves the result to RMSE $0.015202$ and NRMSE
$0.062328$. For now, we are not optimizing model complexity. The information-
criterion trace nevertheless shows that the 50-regressor model changes little
after many of the later additions.

```python
plt.plot(model.info_values)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-05.png?raw=true)

So, what happens if we set a model with half of the regressors?

```python
x_train, y_train = train_val
x_test, y_test = test

n = test.state_initialization_window_length

basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=10,
    ylag=10,
    basis_function=basis_function,
    estimator=LeastSquares(unbiased=False),
    n_info_values=50,
    n_terms=25,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=1000,
    title=f"SysIdentPy -> RMSE: {round(rmse, 4)}, NRMSE: {round(nrmse, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-06.png?raw=true)

The fixed 25-term model gives RMSE $0.018809$ and NRMSE $0.077117$. It is more
compact than the automatically selected 50-term model, at the cost of a larger
error. The three current results are summarized below.

| Configuration | RMSE | NRMSE |
| --- | ---: | ---: |
| lags 7, automatic order | 0.020007 | 0.082029 |
| lags 10, automatic order | 0.015202 | 0.062328 |
| lags 10, fixed at 25 terms | 0.018809 | 0.077117 |

The historical figures stored four decimal places, and the reproduced values
remain the same at that precision. Published tables may use a different
normalization or split, so they are not treated as a direct ranking. Users who
want to investigate the deep state-space alternatives can explore the
[deepsysid package](https://github.com/AlexandraBaier/deepsysid).

This basic configuration can serve as a starting point for users to develop even better models using SysIdentPy. Give it a try!

## Air Passenger Demand Forecasting - A Benchmarking

In this case study, we explore the capabilities of SysIdentPy by applying it to the Air Passenger dataset, a classic time series dataset widely used for evaluating time series forecasting methods. The primary goal of this analysis is to demonstrate that SysIdentPy can serve as a strong alternative for time series modeling, rather than to assert that one library is superior to another.

### Dataset Overview

The Air Passenger dataset consists of monthly totals of international airline passengers from 1949 to 1960. This dataset is characterized by its strong seasonal patterns, trend components, and variability, making it an ideal benchmark for evaluating various time series forecasting methods. Specifically, the dataset includes:

- **Total Monthly Passengers**: The number of passengers (in thousands) for each month.
- **Time Period**: From January 1949 to December 1960, providing 144 data points.

The dataset exhibits clear seasonal fluctuations and a trend, which poses a significant challenge for forecasting methods. It serves as a well-known benchmark for assessing the performance of different time series models due to its inherent complexity and well-documented behavior.

### Comparison with Other Libraries

We will compare the performance of SysIdentPy with other popular time series modeling libraries, focusing on the following tools:

- **sktime**: An extensive library for time series analysis in Python, offering various modeling techniques. For this case study, we will use:
  - `AutoARIMA`: Automatically selects the best ARIMA model based on the data.
  - `BATS` (Bayesian Structural Time Series): A model that captures complex seasonal patterns and trends.
  - `TBATS` (Trigonometric, Box-Cox, ARMA, Trend, and Seasonal): A model designed to handle multiple seasonal patterns.
  - `Exponential Smoothing`: A method that applies weighted averages to forecast future values.
  - `Prophet`: Developed by Facebook, it is particularly effective for capturing seasonality and holiday effects.
  - `AutoETS` (Automatic Exponential Smoothing): Selects the best exponential smoothing model for the data.

- **SysIdentPy**: A library designed for system identification and time series modeling. We will focus on:
  - `MetaMSS` (Meta-heuristic Model Structure Selection): Uses metaheuristic algorithms to select the best model structure.
  - `AOLS` (Accelerated Orthogonal Least Squares): A method for selecting relevant regressors in a model.
  - `FROLS` (Forward Regression with Orthogonal Least Squares, using polynomial base functions): A regression technique for model structure selection with polynomial terms.
  - `NARXNN` (Nonlinear Auto-Regressive model with Exogenous Inputs using Neural Networks): A flexible method for modeling nonlinear time series with external inputs.

### Objective

The objective of this case study is to evaluate and compare the performance of these methods on the Air Passenger dataset. We aim to assess how well each library handles the complex seasonal and trend components of the data and to showcase SysIdentPy as a viable option for time series forecasting.

### Required Packages and Versions

This comparison was verified with SysIdentPy 0.9.0 on Python 3.12.12. The
forecasting libraries are optional and were tested with the following versions:

```bash
python -m pip install -e .
python -m pip install sktime==1.0.1 neuralprophet==0.9.0 prophet==1.3.0
python -m pip install pmdarima==2.1.1 tbats==1.1.3 statsmodels==0.14.6
python -m pip install scipy==1.15.3 torch==2.5.1
```

The SciPy pin satisfies the `scipy<1.16` constraint enforced by the BATS/TBATS
adapters in this sktime version. Use a virtual environment because these packages
have a comparatively large and compatibility-sensitive dependency graph.
Randomized models use seed 42.

Let's begin by importing the necessary packages and setting up the environment for this analysis.

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

We use the `sktime` loader and reserve the final 24 monthly observations for
testing. Every method is evaluated on that same horizon.

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

The following image shows the data of the system to be modeled.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-01.png?raw=true)

## Results

Because we have several models to test, the current results are summarized in
the following table. The split and MSE definition are common to every row.

| No. | Package                   | Mean Squared Error |
| ---: | ------------------------- | -----------------: |
| 1    | SysIdentPy (AOLS)         | 440.9993           |
| 2    | SysIdentPy (MetaMSS)      | 510.3495           |
| 3    | NeuralProphet             | 514.0477           |
| 4    | Prophet                   | 910.7187           |
| 5    | Exponential Smoothing     | 1055.5128          |
| 6    | SysIdentPy (Neural NARX)  | 1621.5225          |
| 7    | SysIdentPy (FROLS)        | 1811.9000          |
| 8    | AutoARIMA                 | 2230.3321          |
| 9    | Manual ARIMA              | 2592.7244          |
| 10   | AutoETS                   | 3128.9366          |
| 11   | TBATS                     | 8825.0097          |
| 12   | BATS                      | 9043.4934          |

The final 13 training outputs initialize each SysIdentPy free-run forecast and
are excluded from the metric. MetaMSS uses an internal chronological split for
structure selection and does not automatically refit the selected parameters on
all 120 training observations. The table is a comparison of the stated compact
configurations, not a general ranking of the libraries.

## SysIdentPy: FROLS

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

## SysIdentPy: AOLS

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

## SysIdentPy: MetaMSS

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

## SysIdentPy: Neural NARX

The network architecture is just the same as the one used in to show how to build a Neural NARX model in SysIdentPy docs.

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

## sktime models

The following models are the ones available in the **sktime** package.

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-06.png?raw=true)


## sktime: Exponential Smoothing

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

## sktime: AutoETS

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

## sktime: AutoArima

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

## sktime: Arima

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

## sktime: BATS

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

## sktime: TBATS

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

## sktime: Prophet

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

The historical Neural NARX value of 316.5409 used a different protocol: the
network was trained on the first 108 observations, the next 13 measured outputs
initialized the recursion, and only the final 23 months were scored. Repeating
that protocol with the current stack gives 316.8340. Under the common 120/24
split, the former mini-batch configuration gives MSE 2967.6219. Selecting the
batch size, learning rate and epoch count on an internal 96/24 split, then
refitting on all 120 training observations, reduces the free-run MSE to
1621.5225. Its one-step-ahead MSE is 307.1843, indicating that recursive error
accumulation accounts for much of the remaining free-run error.

The final results can be summarized as follows, resulting in the table presented in the beginning of this case study:

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

## System With Hysteresis - Modeling a Magneto-rheological Damper Device

The memory effects between quasi-static input and output make the modeling of hysteretic systems very difficult. Physics-based models are often used to describe the hysteresis loops, but these models usually lack the simplicity and efficiency required in practical applications involving system characterization, identification, and control. As detailed in [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), NARX models have proven to be a feasible choice to describe the hysteresis loops. See Chapter 8 for a detailed background. However, even considering the sufficient conditions for rate independent hysteresis representation, classical structure selection algorithms fails to return a model with decent performance and the user needs to set a multi-valued function to ensure the occurrence of the bounding structure $\mathcal{H}$ ([Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)).

Even though some progress has been made, previous work has been limited to models with a single equilibrium point. The present case study aims to present new prospects in the model structure selection of hysteretic systems regarding the cases where the models have multiple inputs, and it is not restricted concerning the number of equilibrium points. For that, the MetaMSS algorithm will be used to build a model for a magneto-rheological damper (MRD) considering the mentioned sufficient conditions.

### A Brief description of the Bouc-Wen model of magneto-rheological damper device

The data used in this study-case is the Bouc-Wen model ([Bouc, R - Forced Vibrations of a Mechanical System with Hysteresis](https://www.scirp.org/reference/referencespapers?referenceid=726819)), ([Wen, Y. X. - Method for Random Vibration of Hysteretic Systems](https://ascelibrary.org/doi/10.1061/JMCEA3.0002106)) of an MRD whose schematic diagram is shown in the figure below.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/bouc_wen.png?raw=true)
> The model for a magneto-rheological damper proposed by [Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972).

The general form of the Bouc-Wen model can be described as ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)):

$$
\begin{equation}
\dfrac{dz}{dt} = g\left[x,z,sign\left(\dfrac{dx}{dt}\right)\right]\dfrac{dx}{dt},
\end{equation}
$$

where $z$ is the hysteretic model output, $x$ the input and $g[\cdot]$ a nonlinear function of $x$, $z$ and $sign (dx/dt)$. ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)) proposed the following phenomenological model for the aforementioned device:

$$
\begin{aligned}
f&= c_1\dot{\rho}+k_1(x-x_0),\nonumber\\
\dot{\rho}&=\dfrac{1}{c_0+c_1}[\alpha z+c_0\dot{x}+k_0(x-\rho)],\nonumber\\
\dot{z}&=-\gamma|\dot{x}-\dot{\rho}|z|z|^{n-1}-\beta(\dot{x}-\dot{\rho})|z|^n+A(\dot{x}-\dot{\rho}),\nonumber\\
\alpha&=\alpha_a+\alpha_bu_{bw},\nonumber\\
c_1&=c_{1a}+c_{1b}u_{bw},\nonumber\\
c_0&=c_{0a}+c_{0b}u_{bw},\nonumber\\
\dot{u}_{bw}&=-\eta(u_{bw}-E).
\end{aligned}
$$

where $f$ is the damping force, $c_1$ and $c_0$ represent the viscous coefficients, $E$ is the input voltage, $x$ is the displacement and $\dot{x}$ is the velocity of the model. The parameters of the system (see table below) were taken from [Leva, A. and Piroddi, L. - NARX-based technique for the modelling of magneto-rheological damping devices](https://iopscience.iop.org/article/10.1088/0964-1726/11/1/309).

| Parameter  | Value          | Parameter | Value        |
|------------|----------------|-----------|--------------|
| $c_{0_a}$  | $20.2 \, N \, s/cm$  | $\alpha_{a}$  | $44.9 \, N/cm$  |
| $c_{0_b}$  | $2.68 \, N \, s/cm \, V$ | $\alpha_{b}$  | $638 \, N/cm$   |
| $c_{1_a}$  | $350 \, N \, s/cm$   | $\gamma$      | $39.3 \, cm^{-2}$ |
| $c_{1_b}$  | $70.7 \, N \, s/cm \, V$  | $\beta$       | $39.3 \, cm^{-2}$ |
| $k_{0}$    | $15 \, N/cm$    | $n$           | $2$          |
| $k_{1}$    | $5.37 \, N/cm$   | $\eta$       | $251 \, s^{-1}$ |
| $x_{0}$    | $0 \, cm$      | $A$           | $47.2$       |

For this particular study, both displacement and voltage inputs, $x$ and $E$, respectively, were generated by filtering a white Gaussian noise sequence using a Blackman-Harris FIR filter with $6$Hz cutoff frequency. The integration step-size was set to $h = 0.002$, following the procedures described in [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968). These procedures are for identification purposes only since the inputs of a MRD could have several different characteristics.

The data used in this example is provided by the Professor Samir Angelo Milani Martins.

The challenges are:

- it possesses a nonlinearity featuring memory, i.e. a dynamic nonlinearity;
- the nonlinearity is governed by an internal variable z(t), which is not measurable;
- the nonlinear functional form in the Bouc Wen equation is nonlinear in the parameter;
- the nonlinear functional form in the Bouc Wen equation does not admit a finite Taylor series expansion because of the presence of absolute values

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12,
`pandas==2.3.3` and `scikit-learn==1.7.2`. Install the repository checkout and
the two data-preparation packages explicitly:

```bash
python -m pip install -e .
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

The dataset is loaded from an immutable `sysidentpy-data` URL. Randomized
examples use seed 42.

### SysIdentPy Configuration

```python
from warnings import catch_warnings, simplefilter
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.plotting import plot_results

df = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/bouc_wen/boucwen_histeretic_system.csv"
)
scaler_x = MaxAbsScaler()
scaler_y = MaxAbsScaler()

init = 400
x_train = df[["E", "v"]].iloc[init : df.shape[0] // 2, :]
x_train["sign_v"] = np.sign(df["v"])
x_train = scaler_x.fit_transform(x_train)

x_test = df[["E", "v"]].iloc[df.shape[0] // 2 + 1 : df.shape[0] - init, :]
x_test["sign_v"] = np.sign(df["v"])
x_test = scaler_x.transform(x_test)

y_train = df[["f"]].iloc[init : df.shape[0] // 2, :].values.reshape(-1, 1)
y_train = scaler_y.fit_transform(y_train)

y_test = (
    df[["f"]].iloc[df.shape[0] // 2 + 1 : df.shape[0] - init, :].values.reshape(-1, 1)
)
y_test = scaler_y.transform(y_test)

# Plotting the data
plt.figure(figsize=(10, 8))
plt.suptitle("Identification (training) data", fontsize=16)

plt.subplot(221)
plt.plot(y_train, "k")
plt.ylabel("Force - Output")
plt.xlabel("Samples")
plt.title("y")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.subplot(222)
plt.plot(x_train[:, 0], "k")
plt.ylabel("Control Voltage")
plt.xlabel("Samples")
plt.title("x_1")
plt.grid()
plt.axis([0, 1500, 0, 1])

plt.subplot(223)
plt.plot(x_train[:, 1], "k")
plt.ylabel("Velocity")
plt.xlabel("Samples")
plt.title("x_2")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.subplot(224)
plt.plot(x_train[:, 2], "k")
plt.ylabel("sign(Velocity)")
plt.xlabel("Samples")
plt.title("x_3")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-01.png?raw=true)

Let's check how is the hysteretic behavior considering each input:
```python
plt.figure()
plt.plot(x_train[:, 0], y_train)
plt.xlabel("x1 - Voltage")
plt.ylabel("y - Force")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-02.png?raw=true)

```python
plt.figure()
plt.plot(x_train[:, 1], y_train)
plt.xlabel("x2 - Velocity")
plt.ylabel("y - Force")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-03.png?raw=true)

```python
plt.figure()
plt.plot(x_train[:, 2], y_train)
plt.xlabel("u3 - sign(Velocity)")
plt.ylabel("y - Force")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-04.png?raw=true)

Now, we can build a NARX model. With all three inputs and `MaxAbsScaler`, the
free-run RRSE is $0.045104$:

```python
basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=[[1], [1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_test, y=y_test[: model.max_lag :, :])
rrse = root_relative_squared_error(y_test[model.max_lag :], yhat[model.max_lag :])
print(rrse)
plot_results(
    y=y_test[model.max_lag :],
    yhat=yhat[model.max_lag :],
    n=10000,
    title="FROLS: sign(v) and MaxAbsScaler",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-05.png?raw=true)


If we remove the `sign(v)` input and use the same configuration, free-run
simulation diverges at sample 203, as shown in the following figure:

```python
basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=[[1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
)

model.fit(X=x_train[:, :2], y=y_train)
with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    yhat = model.predict(
        X=x_test[:, :2], y=y_test[: model.max_lag]
    )
if np.isfinite(yhat).all():
    rrse = root_relative_squared_error(
        y_test[model.max_lag :], yhat[model.max_lag :]
    )
    print(rrse)
    plot_results(
        y=y_test[model.max_lag :],
        yhat=yhat[model.max_lag :],
        n=10000,
        title="FROLS without sign(v)",
    )
else:
    finite_mask = np.isfinite(yhat[:, 0])
    finite_stop = int(np.flatnonzero(~finite_mask)[0])
    print(f"Free-run simulation diverged at sample {finite_stop}.")
    plot_results(
        y=y_test[model.max_lag : finite_stop],
        yhat=yhat[model.max_lag : finite_stop],
        n=max(1, finite_stop - model.max_lag),
        title="FROLS without sign(v): trajectory before divergence",
    )
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-06.png?raw=true)

Using MetaMSS without `sign(v)` delays the loss of numerical stability, but does
not eliminate it: this free-run trajectory diverges at sample 1153.

```python
from sysidentpy.model_structure_selection import MetaMSS

basis_function = Polynomial(degree=3)
model = MetaMSS(
    xlag=[[1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    random_state=42,
)

with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train[:, :2], y=y_train)
    yhat = model.predict(
        X=x_test[:, :2], y=y_test[: model.max_lag]
    )
if np.isfinite(yhat).all():
    rrse = root_relative_squared_error(
        y_test[model.max_lag :], yhat[model.max_lag :]
    )
    print(rrse)
    finite_stop = len(yhat)
    plot_results(
        y=y_test[model.max_lag :],
        yhat=yhat[model.max_lag :],
        n=10000,
        title="MetaMSS without sign(v)",
    )
else:
    finite_mask = np.isfinite(yhat[:, 0])
    finite_stop = int(np.flatnonzero(~finite_mask)[0])
    print(f"Free-run simulation diverged at sample {finite_stop}.")
    plot_results(
        y=y_test[model.max_lag : finite_stop],
        yhat=yhat[model.max_lag : finite_stop],
        n=max(1, finite_stop - model.max_lag),
        title="MetaMSS without sign(v): trajectory before divergence",
    )
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-07.png?raw=true)

Before divergence, the oscillatory behavior becomes visible when the output
approaches its minimum value.

```python
window_stop = finite_stop
window_start = max(model.max_lag, window_stop - 100)
plot_results(
    y=y_test[window_start:window_stop],
    yhat=yhat[window_start:window_stop],
    n=100,
    title="MetaMSS without sign(v): last finite window",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-08.png?raw=true)

If we add the `sign(v)` input again and use MetaMSS, the free-run simulation is
finite and gives RRSE $0.055707$, close to the FROLS result with all inputs.

```python
basis_function = Polynomial(degree=3)
model = MetaMSS(
    xlag=[[1], [1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    random_state=42,
)

with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train, y=y_train)
    yhat = model.predict(X=x_test, y=y_test[: model.max_lag :, :])
rrse = root_relative_squared_error(y_test[model.max_lag :], yhat[model.max_lag :])
print(rrse)
plot_results(
    y=y_test[model.max_lag :],
    yhat=yhat[model.max_lag :],
    n=10000,
    title="MetaMSS: sign(v) and MaxAbsScaler",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-09.png?raw=true)

This case will also highlight the significance of data scaling. Previously, we used the `MaxAbsScaler` method, which resulted in great models when using the `sign(v)` inputs, but also resulted in unstable models when removing that input feature. When scaling is applied using `MinMaxScaler`, however, the overall stability of the results improves, and the model does not diverge, even when the `sign(v)` input is removed, using the `FROLS` algorithm.

The user can get the results bellow by just changing the data scaling method using

```python
minmax_scaler_x = MinMaxScaler()
minmax_scaler_y = MinMaxScaler()
midpoint = df.shape[0] // 2

x_train_minmax_frame = df[["E", "v"]].iloc[init:midpoint].copy()
x_train_minmax_frame["sign_v"] = np.sign(x_train_minmax_frame["v"])
x_test_minmax_frame = df[["E", "v"]].iloc[midpoint + 1 : df.shape[0] - init].copy()
x_test_minmax_frame["sign_v"] = np.sign(x_test_minmax_frame["v"])
x_train_minmax = minmax_scaler_x.fit_transform(x_train_minmax_frame)
x_test_minmax = minmax_scaler_x.transform(x_test_minmax_frame)

y_train_minmax = minmax_scaler_y.fit_transform(
    df[["f"]].iloc[init:midpoint].to_numpy()
)
y_test_minmax = minmax_scaler_y.transform(
    df[["f"]].iloc[midpoint + 1 : df.shape[0] - init].to_numpy()
)

def run_minmax_experiment(name, selector, use_sign):
    n_inputs = 3 if use_sign else 2
    x_train_variant = x_train_minmax[:, :n_inputs]
    x_test_variant = x_test_minmax[:, :n_inputs]
    if selector == "FROLS":
        candidate = FROLS(
            xlag=[[1]] * n_inputs,
            ylag=1,
            basis_function=Polynomial(degree=3),
            estimator=LeastSquares(),
            info_criteria="aic",
        )
    else:
        candidate = MetaMSS(
            xlag=[[1]] * n_inputs,
            ylag=1,
            basis_function=Polynomial(degree=3),
            estimator=LeastSquares(),
            random_state=42,
        )

    with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        simplefilter("ignore", RuntimeWarning)
        simplefilter("ignore", UserWarning)
        candidate.fit(X=x_train_variant, y=y_train_minmax)
        prediction = candidate.predict(
            X=x_test_variant,
            y=y_test_minmax[: candidate.max_lag],
        )

    finite_mask = np.isfinite(prediction[:, 0])
    if finite_mask.all():
        score = root_relative_squared_error(
            y_test_minmax[candidate.max_lag :],
            prediction[candidate.max_lag :],
        )
        print(f"{name}: RRSE={score:.6f}")
        stop = len(prediction)
    else:
        score = np.nan
        stop = int(np.flatnonzero(~finite_mask)[0])
        print(f"{name}: free-run simulation diverged at sample {stop}")

    plot_results(
        y=y_test_minmax[candidate.max_lag : stop],
        yhat=prediction[candidate.max_lag : stop],
        n=max(1, stop - candidate.max_lag),
        title=f"{name} with MinMaxScaler",
    )
    return prediction, score

minmax_results = {
    "FROLS with sign(v)": run_minmax_experiment(
        "FROLS with sign(v)", "FROLS", True
    ),
    "FROLS without sign(v)": run_minmax_experiment(
        "FROLS without sign(v)", "FROLS", False
    ),
    "MetaMSS without sign(v)": run_minmax_experiment(
        "MetaMSS without sign(v)", "MetaMSS", False
    ),
    "MetaMSS with sign(v)": run_minmax_experiment(
        "MetaMSS with sign(v)", "MetaMSS", True
    ),
}
yhat = minmax_results["MetaMSS with sign(v)"][0]
x_test = x_test_minmax
y_test = y_test_minmax
```

and running each model again. This change makes every free-run trajectory finite,
but it does not improve the complete-input configurations.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-10.png?raw=true)
> FROLS with `sign(v)` and `MinMaxScaler`: RRSE 0.115986.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-11.png?raw=true)
> FROLS without `sign(v)` and using `MinMaxScaler`: RRSE 0.163944.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-12.png?raw=true)
> MetaMSS without `sign(v)` and using `MinMaxScaler`: RRSE 0.185607.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-13.png?raw=true)
> MetaMSS with `sign(v)` and using `MinMaxScaler`: RRSE 0.104511.

The complete comparison is:

| Scaling | Selector | Inputs | Free-run result |
| --- | --- | --- | ---: |
| MaxAbs | FROLS | with `sign(v)` | RRSE 0.045104 |
| MaxAbs | FROLS | without `sign(v)` | diverged at sample 203 |
| MaxAbs | MetaMSS | without `sign(v)` | diverged at sample 1153 |
| MaxAbs | MetaMSS | with `sign(v)` | RRSE 0.055707 |
| MinMax | FROLS | with `sign(v)` | RRSE 0.115986 |
| MinMax | FROLS | without `sign(v)` | RRSE 0.163944 |
| MinMax | MetaMSS | without `sign(v)` | RRSE 0.185607 |
| MinMax | MetaMSS | with `sign(v)` | RRSE 0.104511 |

MetaMSS with `sign(v)` is the best of the MinMax-scaled configurations. The
best result overall remains FROLS with `sign(v)` and `MaxAbsScaler`. A non-finite
free-run output is reported as divergence rather than converted into a scalar
metric.

Here is the predicted hysteretic loop:
```python
plt.figure(figsize=(8, 6))
plt.plot(x_test[:, 1], yhat)
plt.xlabel("Scaled velocity")
plt.ylabel("Predicted scaled force")
plt.title("MetaMSS with sign(v) and MinMaxScaler")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-14.png?raw=true)

## Silver box

The description content mainly derives (copy and paste) from the [associated paper - Three free data sets for development and benchmarking in nonlinear system identification](https://ieeexplore.ieee.org/document/6669201). For a detailed description, readers are referred to the linked reference.

> The Silverbox system can be seen as an electronic implementation of the Duffing oscillator. It is build as a 2nd order linear time-invariant system with a 3rd degree polynomial static nonlinearity around it in feedback. This type of dynamics are, for instance, often encountered in mechanical systems [Nonlinear Benchmark - Silverbox](https://www.nonlinearbenchmark.org/benchmarks/silverbox).

In this case study, we will create a NARX model for the Silver box benchmark. The Silver box represents a simplified version of mechanical oscillating processes, which are a critical category of nonlinear dynamic systems. Examples include vehicle suspensions, where shock absorbers and progressive springs play vital roles. The data generated by the Silver box provides a simplified representation of such combined components. The electrical circuit generating this data closely approximates, but does not perfectly match, the idealized models described below.

As described in the original paper, the system was excited using a general waveform generator (HPE1445A). The input signal begins as a discrete-time signal $r(k)$, which is converted to an analog signal $r_c(t)$ using zero-order-hold reconstruction. The actual excitation signal $u_0(t)$ is then obtained by passing $r_c(t)$ through an analog low-pass filter $G(p)$ to eliminate high-frequency content around multiples of the sampling frequency. Here, $p$ denotes the differentiation operator. Thus, the input is given by:

$$
u_0(t) = G(p) r_c(t).
$$

The input and output signals were measured using HP1430A data acquisition cards, with synchronized clocks for the acquisition and generator cards. The sampling frequency was:

$$
f_s = \frac{10^7}{2^{14}} = 610.35 \, \text{Hz}.
$$

The silver box uses analog electrical circuitry to generate data representing a nonlinear mechanical resonating system with a moving mass $m$, viscous damping $d$, and a nonlinear spring $k(y)$. The electrical circuit is designed to relate the displacement $y(t)$ (the output) to the force $u(t)$ (the input) by the following differential equation:

$$
m \frac{d^2 y(t)}{dt^2} + d \frac{d y(t)}{dt} + k(y(t)) y(t) = u(t).
$$

The nonlinear progressive spring is described by a static, position-dependent stiffness:

$$
k(y(t)) = a + b y^2(t).
$$

The signal-to-noise ratio is sufficiently high to model the system without accounting for measurement noise. However, measurement noise can be included by replacing $y(t)$ with the artificial variable $x(t)$ in the equation above, and introducing disturbances $w(t)$ and $e(t)$ as follows:

$$
\begin{aligned}
& m \frac{d^2 x(t)}{dt^2} + d \frac{d x(t)}{dt} + k(x(t)) x(t) = u(t) + w(t), \\
& k(x(t)) = a + b x^2(t), \\
& y(t) = x(t) + e(t).
\end{aligned}
$$

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12 and
`nonlinear-benchmarks==1.0.1`. Install the repository checkout and the official
benchmark loader explicitly:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use a virtual environment to isolate the optional loader. Numerical results
should be recomputed if the environment or model configuration changes.

### SysIdentPy configuration

In this section, we will demonstrate the application of SysIdentPy to the Silver box dataset.  The following code will guide you through the process of loading the dataset, configuring the SysIdentPy parameters, and building a model for mentioned system.

```python
import numpy as np
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial, Fourier
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_mean_squared_error
from sysidentpy.utils.plotting import plot_results

import nonlinear_benchmarks

train_val, test = nonlinear_benchmarks.Silverbox(atleast_2d=True)

x_train, y_train = train_val.u, train_val.y
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
x_test, y_test = test_multisine.u, test_multisine.y

n = test_multisine.state_initialization_window_length
```

We used the `nonlinear_benchmarks` package to load the data. The user is referred to the [package documentation - GerbenBeintema/nonlinear_benchmarks: The official data load for http://www.nonlinearbenchmark.org/ (github.com)](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) to check the details of how to use it.

The following plot detail the training and testing data of the experiment.

```python
plt.plot(x_train)
plt.plot(y_train, alpha=0.3)
plt.title("Experiment 1: training data")
plt.show()

plt.plot(x_test)
plt.plot(y_test, alpha=0.3)
plt.title("Experiment 1: testing data")
plt.show()

plt.plot(test_arrow_full.u)
plt.plot(test_arrow_full.y, alpha=0.3)
plt.title("Experiment 2: training data")
plt.show()

plt.plot(test_arrow_no_extrapolation.u)
plt.plot(test_arrow_no_extrapolation.y, alpha=0.2)
plt.title("Experiment 2: testing data")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-01.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-02.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-03.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-04.png?raw=true)

> Important Note

The goal of this benchmark is to develop a model that outperforms the state-of-the-art (SOTA) model presented in the benchmarking paper. However, the results in the [paper](https://arxiv.org/pdf/2012.07697) differ from those provided in the  [GitHub repository](https://github.com/GerbenBeintema/SS-encoder-WH-Silver/blob/main/SS%20encoder%20Silverbox.ipynb).

| nx  | Set             | NRMS    | RMS (mV)   |
| --- | --------------- | ------- | ---------- |
| 2   | Train           | 0.10653 | 5.8103295  |
| 2   | Validation      | 0.11411 | 6.1938068  |
| 2   | Test            | 0.19151 | 10.2358533 |
| 2   | Test (no extra) | 0.12284 | 5.2789727  |
| 4   | Train           | 0.03571 | 1.9478290  |
| 4   | Validation      | 0.03922 | 2.1286373  |
| 4   | Test            | 0.12712 | 6.7943448  |
| 4   | Test (no extra) | 0.05204 | 2.2365904  |
| 8   | Train           | 0.03430 | 1.8707026  |
| 8   | Validation      | 0.03732 | 2.0254112  |
| 8   | Test            | 0.10826 | 5.7865255  |
| 8   | Test (no extra) | 0.04743 | 2.0382715  |
> Table: results presented in the github.

It appears that the values shown in the paper actually represent the training time, not the error metrics. I will contact the authors to confirm this information. According to the Nonlinear Benchmark website, the information is as follows:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver_sota.png?raw=true)

where the values in the "Training time" column matches the ones presented as error metrics in the paper.

> While we await confirmation of the correct values for this benchmark, we will demonstrate the performance of SysIdentPy. However, we will refrain from making any comparisons or attempting to improve the model at this stage.

### Results

We will start (as we did in every other case study) with a basic configuration
of FROLS using a polynomial basis function with degree equal to 2. The `xlag`
and `ylag` are set to $7$ in this first example. Because the dataset is large,
we use `err_tol` and set `n_terms=40`; the search stops when the error-reduction
tolerance is reached or 40 regressors have been tested. While this approach may
result in a suboptimal model, it is a reasonable starting point. There are three
tests: multisine, arrow (full), and arrow (without extrapolation).

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    xlag=7,
    ylag=7,
    basis_function=basis_function,
    estimator=LeastSquares(),
    err_tol=0.999,
    n_terms=40,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
rmse_mv = 1000 * rmse
print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")
plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=30000,
    figsize=(15, 4),
    title=f"Multisine. Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=300,
    figsize=(15, 4),
    title=f"Multisine. Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-05.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-06.png?raw=true)


```python
x_train, y_train = train_val.u, train_val.y
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
x_test, y_test = test_arrow_full.u, test_arrow_full.y

n = test_arrow_full.state_initialization_window_length

basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=14,
    ylag=14,
    basis_function=basis_function,
    estimator=LeastSquares(),
    err_tol=0.9999,
    n_terms=80,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
rmse_mv = 1000 * rmse

print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=30000,
    figsize=(15, 4),
    title=f"Arrow (full). Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=300,
    figsize=(15, 4),
    title=f"Arrow (full). Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-07.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-08.png?raw=true)

```python
x_train, y_train = train_val.u, train_val.y
test_multisine, test_arrow_full, test_arrow_no_extrapolation = test
x_test, y_test = test_arrow_no_extrapolation.u, test_arrow_no_extrapolation.y

n = test_arrow_no_extrapolation.state_initialization_window_length

basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=14,
    ylag=14,
    basis_function=basis_function,
    estimator=LeastSquares(),
    err_tol=0.9999,
    n_terms=40,
    order_selection=False,
)

model.fit(X=x_train, y=y_train)
if model.max_lag > n:
    raise ValueError("The model lag exceeds the benchmark initialization window.")
start = n - model.max_lag
yhat = model.predict(X=x_test[start:], y=y_test[start:n])
yhat = yhat[model.max_lag :]
rmse = root_mean_squared_error(y_test[n:], yhat)
nrmse = rmse / np.std(y_test[n:])
rmse_mv = 1000 * rmse
print(f"RMSE: {rmse:.6f}; NRMSE: {nrmse:.6f}")

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=30000,
    figsize=(15, 4),
    title=f"Arrow (no extrapolation). Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)

plot_results(
    y=y_test[n:],
    yhat=yhat,
    n=300,
    figsize=(15, 4),
    title=f"Free Run simulation. Model -> RMSE (x1000) mv: {round(rmse_mv, 4)}",
)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-09.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver-box-system-10.png?raw=true)

The current free-run results, evaluated after the 50-sample initialization
window supplied by the loader, are:

| Test set | RMSE | NRMSE |
| --- | ---: | ---: |
| Multisine | 0.007727 | 0.142302 |
| Arrow, complete | 0.004148 | 0.077565 |
| Arrow, without extrapolation | 0.002229 | 0.051822 |

The complete arrow test is harder than the version without extrapolation. The
metric excludes the initialization window once; it does not remove
`model.max_lag` a second time. The external deep state-space results remain
valuable context, but are not ranked directly against this table without first
matching the split, initialization and normalization.

## F-16 Ground Vibration Test Benchmark

The following examples are intended to demonstrate the application of SysIdentPy on a real-world dataset. Please note that these examples are not aimed at replicating the results presented in the cited manuscripts. The model parameters, such as `ylag` and `xlag`, as well as the size of the identification and validation data sets, differ from those used in the original studies. Additionally, adjustments related to sampling rates and other data preparation steps are not covered in this notebook.

**For a comprehensive reference regarding the F-16 Ground Vibration Test benchmark, please visit [the nonlinear benchmark website](http://www.nonlinearbenchmark.org/#F16).**

> **Note:** This notebook serves as a preliminary demonstration of SysIdentPy's performance on the F-16 dataset. A more detailed analysis will be provided in a future publication. The nonlinear benchmark website offers valuable resources and references related to system identification and machine learning, and readers are encouraged to explore the papers and information available there.

### Benchmark Overview

The F-16 Ground Vibration Test benchmark is a notable experiment in the field of system identification and nonlinear dynamics. It involves a high-order system with clearance and friction nonlinearities at the mounting interfaces of payloads on a full-scale F-16 aircraft.

**Experiment Details:**
- **Event:** Siemens LMS Ground Vibration Testing Master Class
- **Date:** September 2014
- **Location:** Saffraanberg military base, Sint-Truiden, Belgium

During the test, two dummy payloads were mounted on the wing tips of the F-16 to simulate the mass and inertia of real devices typically equipped on the aircraft during flight. Accelerometers were installed on the aircraft structure to capture vibration data. A shaker was placed under the right wing to apply input signals. The key source of nonlinearity in the system was identified as the mounting interfaces of the payloads, particularly the right-wing-to-payload interface, which exhibited significant nonlinear distortions.

**Data and Resources:**
- **Data Availability:** The dataset, including detailed system descriptions, estimation and test data sets, and setup images, is available for download in both .csv and .mat file formats.
- **Reference:** For in-depth information on the F-16 benchmark, refer to: J.P. Noël and M. Schoukens, "F-16 aircraft benchmark based on ground vibration test data," 2017 Workshop on Nonlinear System Identification Benchmarks, pp. 19-23, Brussels, Belgium, April 24-26, 2017.

The goal of this notebook is to illustrate how SysIdentPy can be applied to such complex datasets, showcasing its capabilities in modeling and analysis. For a thorough exploration of the benchmark and its methodologies, please consult the provided resources and references.

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12 and
`pandas==2.3.3`. Install the repository checkout and pandas explicitly:

```bash
python -m pip install -e .
python -m pip install pandas==2.3.3
```

The data are loaded from an immutable `sysidentpy-data` URL. Numerical results
should be recomputed if the environment or model configuration changes.

### SysIdentPy Configuration

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
```

## Procedure

```python
f_16 = pd.read_csv(
    r"https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/f_16_vibration_test/f-16.txt",
    header=None,
    names=["x1", "x2", "y"],
)
```

```python
f_16.shape
```

```python
f_16[["x1", "x2"]][0:500].plot(figsize=(12, 8))
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-01.png?raw=true)

```python
f_16["y"][0:2000].plot(figsize=(12, 8))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-02.png?raw=true)

The following code is to split the dataset into training and test sets

```python
x1_id, x1_val = f_16["x1"][0:16384].values.reshape(-1, 1), f_16["x1"][
    16384::
].values.reshape(-1, 1)
x2_id, x2_val = f_16["x2"][0:16384].values.reshape(-1, 1), f_16["x2"][
    16384::
].values.reshape(-1, 1)
x_id = np.concatenate([x1_id, x2_id], axis=1)
x_val = np.concatenate([x1_val, x2_val], axis=1)

y_id, y_val = f_16["y"][0:16384].values.reshape(-1, 1), f_16["y"][
    16384::
].values.reshape(-1, 1)
```

We will set the lags for both inputs as

```python
x1lag = list(range(1, 10))
x2lag = list(range(1, 10))
x2lag
```

and build a NARX model as follows

```python
basis_function = Polynomial(degree=1)
estimator = LeastSquares()

model = FROLS(
    order_selection=True,
    n_info_values=39,
    ylag=20,
    xlag=[x1lag, x2lag],
    info_criteria="bic",
    estimator=estimator,
    basis_function=basis_function,
)

model.fit(X=x_id, y=y_id)
y_hat = model.predict(X=x_val, y=y_val)
rrse = root_relative_squared_error(
    y_val[model.max_lag :], y_hat[model.max_lag :]
)
print(rrse)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=8,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)
```

After excluding the model's 20 initial lag samples, the free-run RRSE is
$0.291070$. The same aligned segment is used for the residual autocorrelation
and for the cross-correlation with the first input.

| Regressors | Parameters | ERR            |
|------------|------------|----------------|
| y(k-1)     | 1.8387E+00 | 9.43378253E-01 |
| y(k-2)     | -1.8938E+00| 1.95167599E-02 |
| y(k-3)     | 1.3337E+00 | 1.02432261E-02 |
| y(k-6)     | -1.6038E+00| 8.03485985E-03 |
| y(k-9)     | 2.6776E-01 | 9.27874557E-04 |
| x2(k-7)    | -2.2385E+01| 3.76837313E-04 |
| x1(k-1)    | 8.2709E+00 | 6.81508210E-04 |
| x2(k-3)    | 1.0587E+02 | 1.57459800E-03 |
| x1(k-8)    | -3.7975E+00| 7.35086279E-04 |
| x2(k-1)    | 8.5725E+01 | 4.85358786E-04 |
| y(k-7)     | 1.3955E+00 | 2.77245281E-04 |
| y(k-5)     | 1.3219E+00 | 8.64120037E-04 |
| y(k-10)    | -2.9306E-01| 8.51717688E-04 |
| y(k-4)     | -9.5479E-01| 7.23623116E-04 |
| y(k-8)     | -7.1309E-01| 4.44988077E-04 |
| y(k-12)    | -3.0437E-01| 1.49743148E-04 |
| y(k-11)    | 4.8602E-01 | 3.34613282E-04 |
| y(k-13)    | -8.2442E-02| 1.43738964E-04 |
| y(k-15)    | -1.6762E-01| 1.25546584E-04 |
| x1(k-2)    | -8.9698E+00| 9.76699739E-05 |
| y(k-17)    | 2.2036E-02 | 4.55983807E-05 |
| y(k-14)    | 2.4900E-01 | 1.10314107E-04 |
| y(k-19)    | -6.8239E-03| 1.99734771E-05 |
| x2(k-9)    | -9.6265E+01| 2.98523208E-05 |
| x2(k-8)    | 2.2620E+02 | 2.34402543E-04 |
| x2(k-2)    | -2.3609E+02| 1.04172323E-04 |
| y(k-20)    | -5.4663E-02| 5.37895336E-05 |
| x2(k-6)    | -2.3651E+02| 2.11392628E-05 |
| x2(k-4)    | 1.7378E+02 | 2.18396315E-05 |
| x1(k-7)    | 4.9862E+00 | 2.03811842E-05 |

```python
plot_results(
    y=y_val[model.max_lag :], yhat=y_hat[model.max_lag :], n=1000
)
ee = compute_residues_autocorrelation(
    y_val[model.max_lag :], y_hat[model.max_lag :]
)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(
    y_val[model.max_lag :],
    y_hat[model.max_lag :],
    x_val[model.max_lag :, 0],
)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-03.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-04.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-05.png?raw=true)

### Information criteria

The information-criterion trace used during order selection is shown below. It complements the selected-regressor table without changing the validation protocol.

```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")

# You can use the plot below to choose the "n_terms" and run the model again with the most adequate value of terms.
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-06.png?raw=true)

This is an illustrative SysIdentPy study, not an official F-16 benchmark
submission. The curated 32,768-sample record and its half-and-half split do not
reproduce every split distributed by the official loader, so the RRSE should not
be compared directly with official benchmark tables.

## PV Forecasting

In this case study, we evaluate SysIdentPy's capabilities for forecasting solar irradiance data, which can serve as a proxy for solar photovoltaic (PV) production. The objective is to demonstrate that SysIdentPy provides a competitive alternative for time series modeling, rather than claiming superiority over other libraries.

### Dataset Overview

The dataset used in this analysis consists of solar irradiance measurements, which are crucial for predicting solar PV production. Solar irradiance refers to the power of solar radiation received per unit area at the Earth's surface, typically measured in watts per square meter (W/m²). Accurate forecasting of solar irradiance is essential for optimizing energy production and managing grid stability in solar power systems.

**Dataset Details:**
- **Source:** The dataset can be accessed from the NeuralProphet GitHub repository.
- **Time Frame:** The dataset covers a continuous period with frequent measurements.
- **Variables:** Solar irradiance values over time, which will be used to model and forecast future irradiance levels.

### Comparison with Other Libraries

To assess the effectiveness of SysIdentPy, we will compare its performance with the NeuralProphet library. NeuralProphet is known for its flexibility and ability to capture complex seasonal patterns and trends, making it a suitable benchmark for this task.

For the comparison, we will use the following methods:

- **NeuralProphet:**
  - The configuration for NeuralProphet models will be based on examples provided in the [NeuralProphet documentation](https://neuralprophet.com/html/example_links/energy_data_example.html). This library employs advanced techniques for capturing temporal patterns and forecasting.

- **SysIdentPy:**
  - **MetaMSS (Meta-heuristic Model Structure Selection):** Utilizes metaheuristic algorithms to determine the optimal model structure.
  - **AOLS (Accelerated Orthogonal Least Squares):** A method designed for selecting relevant regressors in a model.
  - **FROLS (Forward Regression with Orthogonal Least Squares, using polynomial base functions):** A regression technique that incorporates polynomial terms to enhance model selection.

### Objective

The goal of this case study is to compare the performance of SysIdentPy's forecasting methods with NeuralProphet. We will specifically focus on:

- **1-Step Ahead Forecasting:** Evaluating the models' ability to predict the next time step in the series based on historical data.

We will train our models on 80% of the dataset and reserve the remaining 20% for validation purposes. This setup ensures that we test the models' performance on unseen data.

### Required Packages and Versions

This case study was verified with SysIdentPy 0.9.0 on Python 3.12.12,
`neuralprophet==0.9.0`, `torch==2.5.1`, `pandas==2.3.3` and
`scikit-learn==1.7.2`. Install the optional comparison stack explicitly:

```bash
python -m pip install -e .
python -m pip install neuralprophet==0.9.0 torch==2.5.1
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

The dataset is loaded from an immutable `sysidentpy-data` URL. Randomized
models use seed 42.

### Procedure

1. **Data Preparation:** Load and preprocess the solar irradiance dataset.
2. **Model Training:** Apply the chosen methods from SysIdentPy and NeuralProphet to the training data.
3. **Evaluation:** Assess the forecasting accuracy of each model on the validation set.

By comparing these approaches, we aim to showcase SysIdentPy as a viable option for time series forecasting, highlighting its strengths and versatility in practical applications.

Let’s start by importing the necessary libraries and setting up the environment for this analysis.

```python
from warnings import simplefilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sysidentpy.model_structure_selection import FROLS, AOLS, MetaMSS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.plotting import plot_results
from sysidentpy.metrics import mean_squared_error

from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

simplefilter("ignore", FutureWarning)
loss = mean_squared_error


def require_finite(name, values):
    if not np.isfinite(np.asarray(values)).all():
        raise FloatingPointError(f"{name} contains non-finite values.")
```

### Neural Prophet

```python
set_random_seed(42)

raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/"
    "4085901293ba5ed5674bb2911ef4d1fa20f3438d/"
    "datasets/san_francisco_pv_ghi/SanFrancisco_PV_GHI.csv"
)
df = pd.DataFrame(
    {
        "ds": pd.date_range("2015-01-01 01:00:00", freq="h", periods=8760),
        "y": raw.iloc[:, 0].to_numpy(),
    }
)

m = NeuralProphet(
    n_lags=24, ar_reg=0.5, epochs=100, learning_rate=0.01
)
split = 7008
df_train, df_val = df.iloc[:split], df.iloc[split:]
m.fit(df_train, freq="h", progress=None)
prediction_df = pd.concat([df_train.tail(m.config_ar.n_lags), df_val])
forecast = m.predict(prediction_df)
valid = (forecast["ds"] >= df_val["ds"].min()) & np.isfinite(
    forecast["yhat1"].to_numpy()
)
if valid.sum() != len(df_val):
    raise RuntimeError("NeuralProphet did not predict every validation sample.")
require_finite("NeuralProphet predictions", forecast.loc[valid, "yhat1"])
neuralprophet_loss = loss(
    forecast.loc[valid, "y"].to_numpy(),
    forecast.loc[valid, "yhat1"].to_numpy(),
)
print(neuralprophet_loss)
```

```python
plt.plot(forecast["y"][-104:], "ro-")
plt.plot(forecast["yhat1"][-104:], "k*-")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-04.png?raw=true)


The NeuralProphet baseline gives $MSE=2473.5397$ on the 1,752 validation
timestamps. Let's check how the SysIdentPy methods handle the same interval.

### FROLS

```python
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/san_francisco_pv_ghi/SanFrancisco_PV_GHI.csv"
)
df = pd.DataFrame()
df["ds"] = pd.date_range("1/1/2015 1:00:00", freq=str(60) + "Min", periods=8760)
df["y"] = raw.iloc[:, 0].values

df_train, df_val = df.iloc[:7008, :], df.iloc[7008:, :]

y = df["y"].values.reshape(-1, 1)
y_train = df_train["y"].values.reshape(-1, 1)
y_test = df_val["y"].values.reshape(-1, 1)

x_train = df_train["ds"].dt.hour.values.reshape(-1, 1)
x_test = df_val["ds"].dt.hour.values.reshape(-1, 1)

basis_function = Polynomial(degree=1)
sysidentpy = FROLS(
    order_selection=True,
    ylag=24,
    xlag=24,
    info_criteria="bic",
    basis_function=basis_function,
    model_type="NARMAX",
    estimator=LeastSquares(),
)

sysidentpy.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy.max_lag :], y_test])

yhat = sysidentpy.predict(X=x_test, y=y_test, steps_ahead=1)
require_finite("FROLS predictions", yhat[sysidentpy.max_lag :])
sysidentpy_loss = loss(
    y_test[sysidentpy.max_lag :],
    yhat[sysidentpy.max_lag :],
)
print(sysidentpy_loss)

plot_results(y=y_test[-104:], yhat=yhat[-104:])
```

The FROLS result is $MSE=2204.3336$.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-01.png?raw=true)

### MetaMSS

```python
set_random_seed(42)
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/san_francisco_pv_ghi/SanFrancisco_PV_GHI.csv"
)
df = pd.DataFrame()
df["ds"] = pd.date_range("1/1/2015 1:00:00", freq=str(60) + "Min", periods=8760)
df["y"] = raw.iloc[:, 0].values

df_train, df_val = df.iloc[:7008, :], df.iloc[7008:, :]

y = df["y"].values.reshape(-1, 1)
y_train = df_train["y"].values.reshape(-1, 1)
y_test = df_val["y"].values.reshape(-1, 1)

x_train = df_train["ds"].dt.hour.values.reshape(-1, 1)
x_test = df_val["ds"].dt.hour.values.reshape(-1, 1)

basis_function = Polynomial(degree=1)
estimator = LeastSquares()
sysidentpy_metamss = MetaMSS(
    basis_function=basis_function,
    xlag=24,
    ylag=24,
    estimator=estimator,
    maxiter=10,
    steps_ahead=1,
    n_agents=15,
    loss_func="metamss_loss",
    model_type="NARMAX",
    random_state=42,
)
sysidentpy_metamss.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy_metamss.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy_metamss.max_lag :], y_test])

yhat = sysidentpy_metamss.predict(X=x_test, y=y_test, steps_ahead=1)
require_finite("MetaMSS predictions", yhat[sysidentpy_metamss.max_lag :])
metamss_loss = loss(
    y_test[sysidentpy_metamss.max_lag :],
    yhat[sysidentpy_metamss.max_lag :],
)
print(metamss_loss)

plot_results(y=y_test[-104:], yhat=yhat[-104:])
```

MetaMSS selects the lowest-error model in this comparison, with
$MSE=2154.2684$.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-02.png?raw=true)

### AOLS

```python
set_random_seed(42)
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/san_francisco_pv_ghi/SanFrancisco_PV_GHI.csv"
)
df = pd.DataFrame()
df["ds"] = pd.date_range("1/1/2015 1:00:00", freq=str(60) + "Min", periods=8760)
df["y"] = raw.iloc[:, 0].values

df_train, df_val = df.iloc[:7008, :], df.iloc[7008:, :]

y = df["y"].values.reshape(-1, 1)
y_train = df_train["y"].values.reshape(-1, 1)
y_test = df_val["y"].values.reshape(-1, 1)

x_train = df_train["ds"].dt.hour.values.reshape(-1, 1)
x_test = df_val["ds"].dt.hour.values.reshape(-1, 1)
basis_function = Polynomial(degree=1)
sysidentpy_AOLS = AOLS(
    ylag=24, xlag=24, k=2, L=1, model_type="NARMAX", basis_function=basis_function
)
sysidentpy_AOLS.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy_AOLS.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy_AOLS.max_lag :], y_test])

yhat = sysidentpy_AOLS.predict(X=x_test, y=y_test, steps_ahead=1)
require_finite("AOLS predictions", yhat[sysidentpy_AOLS.max_lag :])
aols_loss = loss(
    y_test[sysidentpy_AOLS.max_lag :],
    yhat[sysidentpy_AOLS.max_lag :],
)
print(aols_loss)


plot_results(y=y_test[-104:], yhat=yhat[-104:])
```

The AOLS result is $MSE=2361.5617$.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-03.png?raw=true)

The reproduced comparison is therefore:

| Method | One-step-ahead MSE |
| --- | ---: |
| MetaMSS | 2154.2684 |
| FROLS | 2204.3336 |
| AOLS | 2361.5617 |
| NeuralProphet | 2473.5397 |

The SysIdentPy validation arrays are prefixed with the final 24 training
observations, and the metric excludes that prefix. NeuralProphet receives the
same 24 observations as context, while its MSE is restricted to validation
timestamps. Thus every value covers the same validation interval. This
one-step-ahead protocol measures local prediction with the latest measured
output available at every step; it does not establish free-run stability.
