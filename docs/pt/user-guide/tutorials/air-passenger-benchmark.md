# Benchmark de Passageiros Aéreos

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro fornece orientações detalhadas para apoiar seu trabalho com o SysIdentPy.

## Reprodutibilidade

Este tutorial foi verificado com SysIdentPy 0.9.0 no Python 3.12.12. Seus pacotes
opcionais são `sktime==1.0.1`, `neuralprophet==0.9.0`, `prophet==1.3.0`,
`pmdarima==2.1.1`, `tbats==1.1.3`, `statsmodels==0.14.6`,
`scipy==1.15.3` e `torch==2.5.1`. A versão fixada do SciPy satisfaz a
restrição `scipy<1.16` aplicada pelos adaptadores BATS/TBATS nesta versão do
sktime. Os modelos aleatórios usam semente explícita; os resultados devem ser
recalculados quando o ambiente ou a configuração mudar.

## Nota

O exemplo a seguir **não** tem a intenção de afirmar que uma biblioteca é melhor que outra. O foco principal destes exemplos é mostrar que o SysIdentPy pode ser uma boa alternativa para pessoas que desejam modelar séries temporais.

Comparamos o SysIdentPy com previsores do **sktime** e com as bibliotecas
independentes **Prophet** e **NeuralProphet**.

Do sktime, os seguintes modelos serão utilizados:

- AutoARIMA

- ARIMA

- BATS

- TBATS

- Exponential Smoothing

- AutoETS

Por questão de brevidade, do **SysIdentPy** apenas os métodos **MetaMSS**, **AOLS**, **FROLS** (com função base polinomial) e **NARXNN** serão utilizados. Consulte a documentação do SysIdentPy para conhecer outras formas de modelagem com a biblioteca.


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

## Dados de passageiros aéreos


```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-01.png?raw=true)

## Resultados reproduzidos

Todos os modelos usam as 120 primeiras observações para treinamento e preveem os
24 meses finais. A tabela apresenta o erro quadrático médio (MSE); quanto menor,
melhor. Os modelos aleatórios usam semente 42.

O MetaMSS também usa uma divisão cronológica interna: os 17% finais do bloco de
treino com 120 amostras orientam a seleção de estrutura. Os parâmetros escolhidos
são ajustados no segmento interno de identificação anterior; `fit` não os
reajusta no bloco completo.

| Posição | Método | MSE |
| ---: | --- | ---: |
| 1 | SysIdentPy (AOLS) | 440,9993 |
| 2 | SysIdentPy (MetaMSS) | 510,3495 |
| 3 | NeuralProphet | 514,0477 |
| 4 | Prophet | 910,7187 |
| 5 | Suavização exponencial | 1055,5128 |
| 6 | SysIdentPy (NARX neural) | 1621,5225 |
| 7 | SysIdentPy (FROLS) | 1811,9000 |
| 8 | AutoARIMA | 2230,3321 |
| 9 | ARIMA manual | 2592,7244 |
| 10 | AutoETS | 3128,9366 |
| 11 | TBATS | 8825,0097 |
| 12 | BATS | 9043,4934 |

Esta é uma comparação controlada das configurações apresentadas, não uma
classificação geral das bibliotecas.

O resultado histórico do NARX neural, MSE 316,5409, não pertence a esta divisão.
A rede era ajustada nas primeiras 108 observações, as 13 saídas medidas seguintes
serviam apenas para inicializar a recursão e os 23 meses finais entravam na
métrica. A reprodução desse protocolo na pilha atual resulta em 316,8340. Na
divisão comum 120/24, a configuração anterior de minilotes produz 2967,6219. Uma
divisão interna 96/24 restrita ao bloco de treino selecionou lote completo de 128
amostras, taxa de aprendizado 0,01 e 1.500 épocas; após o reajuste nas 120
observações, o MSE em simulação livre é 1621,5225. O MSE de um passo à frente é
307,1843, identificando a acumulação recursiva como a principal limitação
restante.


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
