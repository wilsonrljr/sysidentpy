# Benchmark de Previsão de Carga

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro fornece orientações detalhadas para apoiar seu trabalho com o SysIdentPy.

## Nota

O exemplo a seguir **não** pretende dizer que uma biblioteca é melhor que outra. O foco principal desses exemplos é mostrar que o SysIdentPy pode ser uma boa alternativa para pessoas que buscam modelar séries temporais.

Compararemos os resultados obtidos contra a biblioteca **neural prophet**.

Por questões de brevidade, do **SysIdentPy** apenas os métodos **MetaMSS**, **AOLS** e **FROLS** (com função de base polinomial) serão usados. Consulte a documentação do SysIdentPy para aprender outras formas de modelagem com a biblioteca.

Compararemos um previsor de 1 passo à frente no consumo de eletricidade de um edifício. A configuração do modelo neuralprophet foi retirada da documentação do neuralprophet (https://neuralprophet.com/html/example_links/energy_data_example.html)

O treinamento ocorrerá em 80% dos dados, reservando os últimos 20% para validação.

Nota: os dados usados neste exemplo podem ser encontrados no github do neuralprophet.

# Resultados do Benchmark:

| No. | Pacote | Erro Quadrático Médio |
| --- | ------ | -------------------- |
| 1 | SysIdentPy (FROLS) | 4183 |
| 2 | SysIdentPy (MetaMSS) | 5264 |
| 3 | SysIdentPy (AOLS) | 5264 |
| 4 | NeuralProphet | 11471 |


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

from sktime.datasets import load_airline
from neuralprophet import NeuralProphet
from neuralprophet import set_random_seed

simplefilter("ignore", FutureWarning)
np.seterr(all="ignore")

%matplotlib inline

loss = mean_squared_error
```

## FROLS


```python
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/san_francisco_hospital/SanFrancisco_Hospital.csv"
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
    info_criteria="bic",
    estimator=LeastSquares(),
    basis_function=basis_function,
)
sysidentpy.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy.max_lag :], y_test])

yhat = sysidentpy.predict(X=x_test, y=y_test, steps_ahead=1)
sysidentpy_loss = loss(
    pd.Series(y_test.flatten()[sysidentpy.max_lag :]),
    pd.Series(yhat.flatten()[sysidentpy.max_lag :]),
)
print(sysidentpy_loss)


plot_results(y=y_test[-504:], yhat=yhat[-504:], n=504, figsize=(18, 8))
```

    c:\Users\wilso\Desktop\projects\GitHub\sysidentpy\sysidentpy\model_structure_selection\ofr_base.py:537: UserWarning: n_info_values is greater than the maximum number of all regressors space considering the chosen y_lag, u_lag, and non_degree. We set as 5
      self.info_values = self.information_criterion(reg_matrix, y)


    4183.359498155755



    
![png](../../../en/user-guide/tutorials/load-forecasting-benchmark_files/load-forecasting-benchmark_6_2.png)
    


## MetaMSS


```python
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/san_francisco_hospital/SanFrancisco_Hospital.csv"
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
sysidentpy_metamss = MetaMSS(
    xlag=2,
    ylag=2,
    basis_function=basis_function,
    estimator=LeastSquares(),
    steps_ahead=1,
    n_agents=15,
    random_state=42,
)
sysidentpy_metamss.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy_metamss.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy_metamss.max_lag :], y_test])

yhat = sysidentpy_metamss.predict(X=x_test, y=y_test, steps_ahead=1)
metamss_loss = loss(
    pd.Series(y_test.flatten()[sysidentpy_metamss.max_lag :]),
    pd.Series(yhat.flatten()[sysidentpy_metamss.max_lag :]),
)
print(metamss_loss)


plot_results(y=y_test[:700], yhat=yhat[:700], n=504, figsize=(18, 8))
```

    5264.428783519863



    
![png](../../../en/user-guide/tutorials/load-forecasting-benchmark_files/load-forecasting-benchmark_8_1.png)
    


## AOLS


```python
set_random_seed(42)
raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/san_francisco_hospital/SanFrancisco_Hospital.csv"
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
sysidentpy_AOLS = AOLS(xlag=2, ylag=2, basis_function=basis_function)
sysidentpy_AOLS.fit(X=x_train, y=y_train)
x_test = np.concatenate([x_train[-sysidentpy_AOLS.max_lag :], x_test])
y_test = np.concatenate([y_train[-sysidentpy_AOLS.max_lag :], y_test])

yhat = sysidentpy_AOLS.predict(X=x_test, y=y_test, steps_ahead=1)
aols_loss = loss(
    pd.Series(y_test.flatten()[sysidentpy_AOLS.max_lag :]),
    pd.Series(yhat.flatten()[sysidentpy_AOLS.max_lag :]),
)
print(aols_loss)


plot_results(y=y_test[-504:], yhat=yhat[-504:], n=504, figsize=(18, 8))
```

    5264.42917196841



    
![png](../../../en/user-guide/tutorials/load-forecasting-benchmark_files/load-forecasting-benchmark_10_1.png)
    


## Neural Prophet


```python
set_random_seed(42)

raw = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/refs/heads/main/datasets/san_francisco_hospital/SanFrancisco_Hospital.csv"
)
df = pd.DataFrame()
df["ds"] = pd.date_range("1/1/2015 1:00:00", freq=str(60) + "Min", periods=8760)
df["y"] = raw.iloc[:, 0].values

m = NeuralProphet(
    n_lags=24, ar_sparsity=0.5, num_hidden_layers=2, d_hidden=20, learning_rate=0.001
)
metrics = m.fit(df, freq="H", valid_p=0.2)

df_train, df_val = m.split_df(df, valid_p=0.2)
m.test(df_val)

future = m.make_future_dataframe(df_val, n_historic_predictions=True)
forecast = m.predict(future)
print(loss(forecast["y"][24:-1], forecast["yhat1"][24:-1]))

neuralprophet_loss = loss(forecast["y"][24:-1], forecast["yhat1"][24:-1])
```

    WARNING: nprophet - fit: Parts of code may break if using other than daily data.


    INFO: nprophet.utils - set_auto_seasonalities: Disabling yearly seasonality. Run NeuralProphet with yearly_seasonality=True to override this.
    INFO: nprophet.config - set_auto_batch_epoch: Auto-set batch_size to 32
    INFO: nprophet.config - set_auto_batch_epoch: Auto-set epochs to 7
    Epoch[7/7]: 100%|██████████| 7/7 [00:04<00:00,  1.74it/s, SmoothL1Loss=0.0102, MAE=81.6, RegLoss=0.011] 
    INFO: nprophet - _evaluate: Validation metrics:    SmoothL1Loss    MAE
    1         0.011 84.733


    11397.103026422525



```python
plt.figure(figsize=(18, 8))
plt.plot(forecast["y"][-504:], "ro-")
plt.plot(forecast["yhat1"][-504:], "k*-")
```




    [<matplotlib.lines.Line2D at 0x237847417f0>]




    
![png](../../../en/user-guide/tutorials/load-forecasting-benchmark_files/load-forecasting-benchmark_13_1.png)
    



```python
results = {
    "SysIdentPy - FROLS": sysidentpy_loss,
    "SysIdentPy (AOLS)": aols_loss,
    "SysIdentPy (MetaMSS)": metamss_loss,
    "NeuralProphet": neuralprophet_loss,
}

sorted(results.items(), key=lambda result: result[1])
```




    [('SysIdentPy - FROLS', 4183.359498155755),
     ('SysIdentPy (MetaMSS)', 5264.429171346123),
     ('SysIdentPy (AOLS)', 5264.42917196841),
     ('NeuralProphet', 11397.103026422525)]


