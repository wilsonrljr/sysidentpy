# Benchmark de Previsão Fotovoltaica

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro fornece orientações detalhadas para apoiar seu trabalho com o SysIdentPy.

## Reprodutibilidade

Este tutorial foi verificado com SysIdentPy 0.9.0 no Python 3.12.12,
`neuralprophet==0.9.0`, `torch==2.5.1`, pandas 2.3.3 e scikit-learn 1.7.2. Os dados
hospedados em `sysidentpy-data` usam uma URL de commit imutável, e os modelos
aleatórios definem semente explícita.

## Nota

O exemplo a seguir **não** tem a intenção de afirmar que uma biblioteca é melhor que outra. O foco principal destes exemplos é mostrar que o SysIdentPy pode ser uma boa alternativa para pessoas que desejam modelar séries temporais.

Compararemos os resultados obtidos com a biblioteca **neural prophet**.

Por questão de brevidade, do **SysIdentPy** apenas os métodos **MetaMSS**, **AOLS** e **FROLS** (com função base polinomial) serão utilizados. Consulte a documentação do SysIdentPy para conhecer outras formas de modelagem com a biblioteca.


Compararemos um previsor de 1 passo à frente em dados de irradiância solar (que pode ser um proxy para produção fotovoltaica). A configuração do modelo neuralprophet foi retirada da documentação do neuralprophet (https://neuralprophet.com/html/example_links/energy_data_example.html)

O treinamento ocorrerá em 80% dos dados, reservando os últimos 20% para validação.

Nota: os dados usados neste exemplo podem ser encontrados no github do neuralprophet.


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

## FROLS


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

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-01.png?raw=true)


## MetaMSS


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

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-02.png?raw=true)


## AOLS


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

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-03.png?raw=true)


## Neural Prophet


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

## Resultados reproduzidos

Os 20% finais da série horária formam o conjunto de validação. Os quatro métodos
usam um contexto autorregressivo de 24 amostras e são comparados pelo MSE de um
passo à frente.

O MetaMSS ainda reserva os 25% finais do bloco de treino, com 7.008 amostras,
para sua perda cronológica interna de seleção. O modelo escolhido é ajustado no
segmento interno de identificação anterior; `fit` não o reajusta no bloco
completo.

| Método | MSE |
| --- | ---: |
| MetaMSS | 2154,2684 |
| FROLS | 2204,3336 |
| AOLS | 2361,5617 |
| NeuralProphet | 2473,5397 |

A comparação vale para os modelos, a semente e a divisão de dados informados; ela
não representa uma classificação geral das bibliotecas.
