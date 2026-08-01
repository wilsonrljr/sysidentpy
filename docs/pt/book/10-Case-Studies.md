## Dataset M4

O dataset M4 é um recurso bem conhecido para previsão de séries temporais, oferecendo uma ampla gama de séries de dados usadas para testar e melhorar métodos de previsão. Criado para a competição M4 organizada por Spyros Makridakis, este dataset impulsionou muitos avanços em técnicas de previsão.

O dataset M4 inclui 100.000 séries temporais de vários campos como demografia, finanças, indústria, macroeconomia e microeconomia, que foram selecionados aleatoriamente do banco de dados ForeDeCk. As séries vêm em diferentes frequências (anual, trimestral, mensal, semanal, diária e horária), tornando-o uma coleção abrangente para testar métodos de previsão.

Neste estudo de caso, focaremos no subconjunto horário do dataset M4. Este subconjunto consiste em dados de séries temporais registrados por hora, fornecendo uma visão detalhada e de alta frequência das mudanças ao longo do tempo. Dados horários apresentam desafios únicos devido à sua granularidade e ao potencial de capturar flutuações e padrões de curto prazo.

O dataset M4 fornece um benchmark padrão para comparar diferentes métodos de previsão, permitindo que pesquisadores e profissionais avaliem seus modelos de forma consistente. Com séries de vários domínios e frequências, o dataset M4 representa desafios de previsão do mundo real, tornando-o valioso para desenvolver técnicas de previsão robustas. A competição e o dataset em si levaram à criação de novos algoritmos e métodos, melhorando significativamente a precisão e confiabilidade das previsões.

Apresentaremos um passo a passo completo usando o dataset horário M4 para demonstrar as capacidades do SysIdentPy. O SysIdentPy oferece uma variedade de ferramentas e técnicas projetadas para lidar efetivamente com as complexidades de dados de séries temporais, mas focaremos em uma configuração rápida e fácil para este caso. Abordaremos a seleção de modelos e métricas de avaliação específicas para o dataset horário.

Ao final deste estudo de caso, você terá uma compreensão sólida de como usar o SysIdentPy para previsão com o dataset horário M4, preparando você para enfrentar desafios de previsão semelhantes em cenários do mundo real.

### Pacotes Necessários e Versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12 e
`datasetsforecast==1.0.1`. Instale explicitamente o checkout do repositório e o
carregador do M4:

```bash
python -m pip install -e .
python -m pip install datasetsforecast==1.0.1
```

Use um ambiente virtual para isolar essas dependências opcionais. Os exemplos
aleatórios usam a semente 42; os resultados numéricos devem ser recalculados se
o ambiente ou a configuração do modelo forem alterados.

### Configuração do SysIdentPy

Nesta seção, demonstraremos a aplicação do SysIdentPy ao subconjunto horário do
M4. O código a seguir mostra o carregamento dos dados, a configuração do
SysIdentPy e a construção dos modelos de previsão usados neste estudo.


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

Os gráficos a seguir fornecem uma visualização dos dados de treinamento para um pequeno subconjunto das séries temporais. O gráfico mostra os dados brutos, dando uma visão dos padrões e comportamentos inerentes a cada série.

Ao observar os dados, você pode ter uma noção da variedade e complexidade das séries temporais com as quais estamos trabalhando. Os gráficos podem revelar características importantes como tendências, padrões sazonais e anomalias potenciais dentro das séries temporais. Entender esses elementos é crucial para o desenvolvimento de modelos de previsão precisos.

No entanto, ao lidar com um grande número de séries temporais diferentes, é comum começar com suposições amplas em vez de análises individuais detalhadas. Neste contexto, adotaremos uma abordagem semelhante. Em vez de entrar nos detalhes de cada dataset, faremos algumas suposições gerais e veremos como o SysIdentPy as trata.

Esta abordagem fornece um ponto de partida prático, demonstrando como o SysIdentPy pode gerenciar diferentes tipos de dados de séries temporais sem muito trabalho. À medida que você se familiariza mais com a ferramenta, pode refinar seus modelos com insights mais detalhados. Por enquanto, vamos focar em usar o SysIdentPy para criar as previsões com base nessas suposições iniciais.

Nossa primeira suposição é que há um padrão sazonal de 24 horas nas séries. Examinando os gráficos abaixo, isso parece razoável. Portanto, começaremos a construir nossos modelos com `ylag=24`.


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


Vamos verificar e construir um modelo para o grupo `H20` antes de extrapolar as configurações para todos os grupos. Como não há features de entrada, usaremos um modelo tipo `NAR` no SysIdentPy. Para manter as coisas simples e rápidas, começaremos com função de base Polinomial com grau $1$.


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


Provavelmente, os resultados não são ótimos e não funcionarão para todos os grupos. No entanto, vamos verificar como esta configuração se compara ao modelo vencedor da [competição de séries temporais M4](https://www.researchgate.net/publication/325901666_The_M4_Competition_Results_findings_conclusion_and_way_forward): o Exponential Smoothing with Recurrent Neural Networks ([ESRNN](https://www.sciencedirect.com/science/article/abs/pii/S0169207019301153)).


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
> Tabela 1. Resultados de referência do ESRNN

O código a seguir levou apenas 49 segundos para rodar na minha máquina (processador AMD Ryzen 5 5600x, 32GB RAM a 3600MHz). Devido à sua eficiência, não criei uma versão paralela. Ao final deste caso de uso, você verá como o SysIdentPy pode ser rápido e eficaz, entregando bons resultados sem muita otimização.


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

|        | SMAPE      | MASE     | OWA      |
| ------ | ---------- | -------- | -------- |
| Hourly | 16.034196  | 0.958083 | 0.636132 |
> Tabela 2. Primeiro teste com o SysIdentPy

Os resultados iniciais são razoáveis, mas não correspondem exatamente ao desempenho do `ESRNN`. Esses resultados são baseados apenas em nossa primeira suposição. Para entender melhor o desempenho, vamos examinar os grupos com os piores resultados.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-03.png?raw=true)

O gráfico a seguir ilustra dois desses grupos, `H147` e `H136`. Ambos exibem um padrão sazonal de 24 horas.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_seasonal_h147_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/c10_m4_h136_seasonal_1.png?raw=true)

No entanto, uma observação mais atenta revela um insight adicional: além do padrão diário, essas séries também mostram um padrão semanal. Observe como os dados parecem quando dividimos a série em segmentos semanais.

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


Portanto, construiremos modelos definindo `ylag=168`.

> Note que este é um número muito alto para lags, então tenha cuidado se quiser tentar com graus polinomiais mais altos porque o tempo para rodar os modelos pode aumentar significativamente. Tentei algumas configurações com grau polinomial igual a 2 e levou apenas $6$ minutos para rodar (ainda menos, usando `AOLS`), sem fazer o código rodar em paralelo. Como você pode ver, o SysIdentPy pode ser muito rápido e você pode torná-lo mais rápido aplicando paralelização.


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

|        | SMAPE      | MASE     | OWA      |
| ------ | ---------- | -------- | -------- |
| Hourly | 10.475998  | 0.773749 | 0.446471 |
> Tabela 3. Resultados aprimorados com o SysIdentPy

Agora, os resultados estão muito mais próximos dos do modelo `ESRNN`! Enquanto o Erro Percentual Absoluto Médio Simétrico (`SMAPE`) é ligeiramente pior, o Erro Escalado Absoluto Médio (`MASE`) é melhor quando comparado ao `ESRNN`, levando a uma métrica de Média Ponderada Geral (`OWA`) muito semelhante. Notavelmente, esses resultados são alcançados usando apenas modelos `AR` simples. A seguir, vamos ver se o método `AOLS` pode fornecer resultados ainda melhores.


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

|        | SMAPE | MASE   | OWA    |
| ------ | ----- | ------ | ------ |
| Hourly | 9.9497 | 0.8074 | 0.4392 |
> Tabela 4. Resultados do SysIdentPy com o algoritmo AOLS

Para esta configuração, a Média Ponderada Geral (`OWA`) é ligeiramente menor
que a da referência `ESRNN`. Essa conclusão se aplica ao conjunto completo das
414 séries horárias e ao horizonte de 48 passos da competição; ela não
estabelece uma classificação geral entre os algoritmos.

Antes de terminar, vamos verificar como o desempenho do modelo `H147` melhorou com a configuração `ylag=168`.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/m4-benchmark-05.png?raw=true)

> Com base no artigo de benchmark M4, também poderíamos limitar as previsões menores que 10 para 10 e os resultados seriam ligeiramente melhores. Mas isso fica a critério do usuário.

Poderíamos alcançar um desempenho ainda melhor com algum ajuste fino da configuração do modelo. No entanto, deixarei a exploração desses ajustes alternativos como um exercício para o usuário. Porém, tenha em mente que experimentar com diferentes configurações nem sempre garante resultados melhores. Um conhecimento teórico mais profundo pode frequentemente levá-lo a melhores configurações e, portanto, melhores resultados.

## Dispositivo Elétrico Acoplado

O dataset CE8 de acionamentos elétricos acoplados [dataset - Nonlinear Benchmark](https://www.nonlinearbenchmark.org/benchmarks) apresenta um caso de uso interessante para demonstrar o desempenho do SysIdentPy. Este sistema envolve dois motores elétricos acionando uma polia com uma correia flexível, criando um ambiente dinâmico ideal para testar ferramentas de identificação de sistemas.

> O [site de benchmarks não lineares](https://www.nonlinearbenchmark.org/benchmarks) representa uma contribuição significativa para a comunidade de identificação de sistemas e aprendizado de máquina. Os usuários são encorajados a explorar todos os artigos referenciados no site.

### Visão Geral do Sistema

O sistema CE8, ilustrado na Figura 1, apresenta:
- **Dois Motores Elétricos**: Estes motores controlam independentemente a tensão e a velocidade da correia, fornecendo controle simétrico em torno do zero. Isso permite movimentos tanto horários quanto anti-horários.
- **Mecanismo de Polia**: A polia é suportada por uma mola, introduzindo um modo dinâmico levemente amortecido que adiciona complexidade ao sistema.
- **Foco no Controle de Velocidade**: O foco principal é o sistema de controle de velocidade. A velocidade angular da polia é medida usando um contador de pulsos, que é insensível à direção da velocidade.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/ce8_design.png?raw=true)
> Figura 1. Design do sistema CE8.

### Sensor e Filtragem

O processo de medição envolve:
- **Contador de Pulsos**: Este sensor mede a velocidade angular da polia sem considerar a direção.
- **Filtragem Analógica Passa-Baixa**: Reduz o ruído de alta frequência, seguido por filtragem anti-aliasing para preparar o sinal para processamento digital. Os efeitos dinâmicos são principalmente influenciados pelas constantes de tempo do acionamento elétrico e pela mola, com a filtragem passa-baixa tendo impacto mínimo na saída.

### Resultados SOTA

O SysIdentPy pode ser usado para construir modelos robustos para identificar e modelar as dinâmicas complexas do sistema CE8. O desempenho será comparado com um benchmark fornecido por [Max D. Champneys, Gerben I. Beintema, Roland Tóth, Maarten Schoukens, and Timothy J. Rogers - Baselines for Nonlinear Benchmarks, Workshop on Nonlinear System Identification Benchmarks, 2024.](https://arxiv.org/pdf/2405.10779)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/ce8_sota.png?raw=true)

O benchmark avalia a métrica média entre os dois experimentos. Por isso o método SOTA não tem a melhor métrica para o `teste 1`, mas ainda é o melhor no geral. O objetivo deste estudo de caso não é apenas demonstrar a robustez do SysIdentPy, mas também fornecer insights valiosos sobre suas aplicações práticas em sistemas dinâmicos do mundo real.

### Pacotes e Versões Necessários

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12 e
`nonlinear-benchmarks==1.0.1`. Instale explicitamente o checkout do repositório
e o carregador oficial do benchmark:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use um ambiente virtual para isolar o carregador opcional. Os resultados
numéricos devem ser recalculados se o ambiente ou a configuração do modelo forem
alterados.

### Configuração do SysIdentPy

Nesta seção, demonstraremos a aplicação do SysIdentPy ao dataset CE8 de acionamentos elétricos acoplados. Este exemplo mostra o desempenho robusto do SysIdentPy na modelagem e identificação de sistemas dinâmicos complexos. O código a seguir irá guiá-lo através do processo de carregamento do dataset, configuração dos parâmetros do SysIdentPy e construção de um modelo para o sistema CE8.

Este exemplo prático ajudará os usuários a entender como utilizar efetivamente o SysIdentPy para suas próprias tarefas de identificação de sistemas, aproveitando seus recursos avançados para lidar com as complexidades de sistemas dinâmicos do mundo real. Vamos mergulhar no código e explorar as capacidades do SysIdentPy.


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

Usamos o pacote `nonlinear_benchmarks` para carregar os dados. O usuário é direcionado à documentação do pacote [GerbenBeintema - nonlinear_benchmarks: The official dataload for nonlinear benchmark datasets](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) para verificar os detalhes de como usá-lo.

O gráfico a seguir detalha os dados de treinamento e teste de ambos os experimentos. Aqui estamos tentando obter dois modelos, um para cada experimento, que tenham um desempenho melhor que os baselines mencionados.


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


### Resultados

Primeiro, definiremos exatamente a mesma configuração para construir modelos para ambos os experimentos. Podemos ter modelos melhores otimizando as configurações individualmente, mas começaremos de forma simples.

Uma configuração básica do FROLS usando uma função base polinomial com grau igual a 2 é definida. O critério de informação será o padrão, o `aic`. Os `xlag` e `ylag` são definidos como $7$ neste primeiro exemplo.

Modelo para o experimento 1:


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


Modelo para o experimento 2:


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


Com o carregador atual e a janela de inicialização definida pelo benchmark, esta
primeira configuração produz RMSE $0.102862$ no experimento 1 e $0.106816$ no
experimento 2. A média é $0.104839$. As amostras reservadas à inicialização do
estado são excluídas uma única vez, e a recursão começa nas últimas
`model.max_lag` saídas contidas nessa janela.

A tabela externa continua sendo uma referência útil, mas seus valores só devem
ser comparados depois de igualar divisão dos dados, janela de inicialização,
normalização e regra de agregação. Portanto, usaremos os valores de RMSE desta
execução para comparar entre si as configurações do SysIdentPy. Antes de
aumentar os lags, mostramos o critério de informação:


```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-07.png?raw=true)


Pode-se observar que após 22 regressores, adicionar novos regressores não melhora o desempenho do modelo (considerando a configuração definida para aquele modelo). Como queremos experimentar modelos com lags maiores e grau de não linearidade maior, o critério de parada será alterado para `err_tol` em vez de critério de informação. Isso fará o algoritmo rodar consideravelmente mais rápido.


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


Os modelos com 10 lags e 22 termos produzem RMSE $0.110933$ e $0.107076$ nos
experimentos 1 e 2, respectivamente. A janela oficial de inicialização do estado
contém 10 amostras; por isso, as configurações anteriores com 14 lags não são
válidas sob este protocolo. Aumentar o lag até o maior valor válido não melhora
esta configuração de grau 2. Portanto, vamos definir o grau polinomial como $3$
e aumentar o número de termos para `n_terms=40` quando o `err_tol` não for
atingido. Esses valores são empíricos; o estimador, a tolerância de erro, o
algoritmo de seleção de estrutura e a função de base são outras dimensões que
podem ser ajustadas.


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


Os modelos de grau 3 produzem RMSE $0.112503$ e $0.096002$, com média
$0.104253$. O segundo experimento melhora, enquanto o primeiro não. Por esse
motivo, os dois experimentos devem ser apresentados separadamente, e a tabela
externa de estado da arte não é usada para estabelecer uma classificação sem um
protocolo de avaliação idêntico.

## Wiener-Hammerstein

O conteúdo da descrição deriva principalmente do [site do benchmark - Nonlinear Benchmark](https://www.nonlinearbenchmark.org/benchmarks) e do [artigo associado - Wiener-Hammerstein benchmark with process noise](https://data.4tu.nl/articles/_/12952124). Para uma descrição detalhada, os leitores são encaminhados às referências vinculadas.

> O site de benchmarks não lineares representa uma contribuição significativa para a comunidade de identificação de sistemas e aprendizado de máquina. Os usuários são encorajados a explorar todos os artigos referenciados no site.

Este benchmark foca em um circuito eletrônico Wiener-Hammerstein onde o ruído de processo desempenha um papel significativo na distorção do sinal de saída.

A estrutura Wiener-Hammerstein é um sistema orientado a blocos bem conhecido que contém uma não linearidade estática intercalada entre dois blocos Lineares Invariantes no Tempo (LTI) (Figura 2). Este arranjo apresenta um problema de identificação desafiador devido à presença desses blocos LTI.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wh_system.png?raw=true)
> Figura 2: o sistema Wiener-Hammerstein

Na Figura 2, o sistema Wiener-Hammerstein é ilustrado com ruído de processo $e_x(t)$ entrando antes da não linearidade estática $f(x)$, intercalado entre blocos LTI representados por $R(s)$ e $S(s)$ na entrada e saída, respectivamente. Além disso, pequenas fontes de ruído desprezíveis $e_u(t)$ e $e_y(t)$ afetam os canais de medição. Os sinais de entrada e saída medidos são denotados como $u_m(t)$ e $y_m(t)$.

O primeiro bloco LTI $R(s)$ é efetivamente modelado como um filtro passa-baixa de terceira ordem. O segundo subsistema LTI $S(s)$ é configurado como um filtro Chebyshev inverso com atenuação de banda de parada de $40 dB$ e frequência de corte de $5 kHz$. Notavelmente, $S(s)$ inclui um zero de transmissão dentro da faixa de frequência operacional, complicando sua inversão.

A não linearidade estática $f(x)$ é implementada usando uma rede de diodo-resistor, resultando em não linearidade de saturação. O ruído de processo $e_x(t)$ é introduzido como ruído gaussiano branco filtrado, gerado a partir de um filtro Butterworth passa-baixa de terceira ordem em tempo discreto seguido por zero-order hold e filtragem de reconstrução passa-baixa analógica com corte de $20 kHz$.

As fontes de ruído de medição $e_u(t)$ e $e_y(t)$ são mínimas comparadas a $e_x(t)$. As entradas do sistema e o ruído de processo são gerados usando um Gerador de Forma de Onda Arbitrária (AWG), especificamente o Agilent/HP E1445A, amostrando a $78125 Hz$, sincronizado com um sistema de aquisição (Agilent/HP E1430A) para garantir coerência de fase e prevenir erros de vazamento. O buffering entre as placas de aquisição e as entradas e saídas do sistema minimiza a distorção do equipamento de medição.

O benchmark fornece dois sinais de teste padrão através do site de benchmarking: um multisine de fase aleatória e um sinal de varredura senoidal. Ambos os sinais têm um valor $rms$ de $0.71 Vrms$ e cobrem frequências de DC a $15 kHz$ (excluindo DC). A varredura senoidal abrange esta faixa de frequência a uma taxa de $4.29 MHz/min$. Estes conjuntos de teste servem como alvos para avaliar o desempenho do modelo, enfatizando representação precisa sob condições variadas.

O benchmark Wiener-Hammerstein destaca três desafios principais de identificação de sistemas não lineares:

1. **Ruído de Processo:** Significativo no sistema, influenciando a fidelidade da saída.
2. **Não Linearidade Estática:** Indiretamente acessível a partir de dados medidos, apresentando desafios de identificação.
3. **Dinâmicas de Saída:** Inversão complexa devido à presença de zero de transmissão em $S(s)$.

O objetivo deste benchmark é desenvolver e validar modelos robustos usando dados de estimação separados, garantindo caracterização precisa do comportamento do sistema Wiener-Hammerstein.

### Pacotes Necessários e Versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12 e
`nonlinear-benchmarks==1.0.1`. Instale explicitamente o checkout do repositório
e o carregador oficial do benchmark:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use um ambiente virtual para isolar o carregador opcional. Os resultados
numéricos devem ser recalculados se o ambiente ou a configuração do modelo forem
alterados.

### Configuração do SysIdentPy

Nesta seção, demonstraremos a aplicação do SysIdentPy ao dataset do sistema Wiener-Hammerstein. O código a seguir guiará você através do processo de carregamento do dataset, configuração dos parâmetros do SysIdentPy e construção de um modelo para o sistema Wiener-Hammerstein.


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

Usamos o pacote `nonlinear_benchmarks` para carregar os dados. O usuário é encaminhado à [documentação do pacote](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) para verificar os detalhes de como usá-lo.

O gráfico a seguir detalha os dados de treinamento e teste do experimento.


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


O benchmark externo fornece um contexto útil para o experimento. Uma comparação
direta, porém, exige a mesma divisão dos dados, janela de inicialização e
normalização; essas condições são explicitadas abaixo antes de qualquer
comparação.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wh_sota_results.png?raw=true)
> Resultados estado-da-arte apresentados no [artigo de benchmarking](https://arxiv.org/pdf/2405.10779). Nesta seção estamos trabalhando apenas com os resultados Wiener-Hammerstein, que são apresentados na coluna $W-H$.

### Resultados

Começaremos com uma configuração básica do FROLS usando uma função de base polinomial com grau igual a 2. O `xlag` e `ylag` são definidos como $7$ neste primeiro exemplo. Como o dataset é consideravelmente grande, começaremos com `n_info_values=50`. Isso significa que o algoritmo FROLS não incluirá todos os regressores ao calcular os critérios de informação usados para determinar a ordem do modelo. Embora esta abordagem possa resultar em um modelo sub-ótimo, é um ponto de partida razoável para nossa primeira tentativa.


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


A primeira configuração produz RMSE $0.020007$ e NRMSE $0.082029$. Começamos
com `xlag=ylag=7` para estabelecer uma referência compacta. O artigo de
benchmarking usa memórias mais longas em alguns modelos; por isso, a próxima
configuração define `xlag=ylag=10`.


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


A configuração com 10 lags melhora o resultado para RMSE $0.015202$ e NRMSE
$0.062328$. Por enquanto, não estamos otimizando a complexidade do modelo. Ainda
assim, o traçado do critério de informação mostra que o modelo com 50
regressores muda pouco após várias das últimas inclusões.


```python
plt.plot(model.info_values)
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/wiener-hammerstein-system-05.png?raw=true)


Então, o que acontece se definirmos um modelo com metade dos regressores?


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


O modelo fixado em 25 termos produz RMSE $0.018809$ e NRMSE $0.077117$. Ele é
mais compacto que o modelo de 50 termos selecionado automaticamente, ao custo de
um erro maior. Os três resultados atuais são resumidos a seguir.

| Configuração | RMSE | NRMSE |
| --- | ---: | ---: |
| 7 lags, ordem automática | 0.020007 | 0.082029 |
| 10 lags, ordem automática | 0.015202 | 0.062328 |
| 10 lags, 25 termos fixos | 0.018809 | 0.077117 |

As figuras históricas armazenavam quatro casas decimais, e os valores
reproduzidos permanecem iguais nessa precisão. Tabelas publicadas podem usar
outra normalização ou divisão dos dados, portanto não são tratadas como uma
comparação direta. Quem quiser investigar alternativas baseadas em modelos de
estado profundos pode explorar o [pacote deepsysid](https://github.com/AlexandraBaier/deepsysid).

Esta configuração básica pode servir como ponto de partida para os usuários desenvolverem modelos ainda melhores usando o SysIdentPy. Experimente!

## Previsão de Demanda de Passageiros Aéreos — um benchmark

Neste estudo de caso, exploramos as capacidades do SysIdentPy aplicando-o ao
conjunto Air Passenger, uma série temporal clássica muito usada na avaliação de
métodos de previsão. O principal objetivo é demonstrar que o SysIdentPy pode ser
uma alternativa consistente para modelagem de séries temporais, e não afirmar
que uma biblioteca é superior às demais.

### Visão geral do conjunto de dados

O conjunto Air Passenger contém os totais mensais de passageiros de voos
internacionais entre 1949 e 1960. Ele apresenta sazonalidade acentuada,
tendência e variabilidade, o que o torna uma referência adequada para avaliar
diferentes métodos de previsão. Mais especificamente, o conjunto inclui:

- **Total mensal de passageiros:** o número de passageiros, em milhares, a cada
  mês.
- **Período:** de janeiro de 1949 a dezembro de 1960, totalizando 144
  observações.

As flutuações sazonais e a tendência são evidentes e impõem um desafio
significativo aos métodos de previsão. Por sua complexidade inerente e seu
comportamento bem documentado, esta série se tornou um benchmark conhecido para
comparar modelos de séries temporais.

### Comparação com outras bibliotecas

Compararemos o SysIdentPy com outras bibliotecas populares de modelagem de
séries temporais, considerando as seguintes ferramentas:

- **sktime:** uma biblioteca abrangente para análise de séries temporais em
  Python. Neste estudo, usaremos:
  - `AutoARIMA`: seleciona automaticamente o modelo ARIMA a partir dos dados.
  - `BATS` (Bayesian Structural Time Series): representa tendências e padrões
    sazonais complexos.
  - `TBATS` (Trigonometric, Box-Cox, ARMA, Trend, and Seasonal): foi projetado
    para lidar com múltiplos padrões sazonais.
  - `Exponential Smoothing`: aplica médias ponderadas para prever valores
    futuros.
  - `Prophet`: desenvolvido pelo Facebook, é especialmente adequado para
    representar sazonalidade e efeitos de feriados.
  - `AutoETS` (Automatic Exponential Smoothing): seleciona automaticamente o
    modelo de suavização exponencial.
- **SysIdentPy:** uma biblioteca voltada à identificação de sistemas e à
  modelagem de séries temporais. Usaremos:
  - `MetaMSS` (Meta-heuristic Model Structure Selection): usa algoritmos
    meta-heurísticos para selecionar a estrutura do modelo.
  - `AOLS` (Accelerated Orthogonal Least Squares): seleciona regressores
    relevantes para o modelo.
  - `FROLS` (Forward Regression with Orthogonal Least Squares, com função de
    base polinomial): realiza a seleção de estrutura por regressão ortogonal.
  - `NARXNN` (Nonlinear Auto-Regressive model with Exogenous Inputs using Neural
    Networks): oferece uma forma flexível de representar séries não lineares
    com entradas externas.

### Objetivo

O objetivo é avaliar e comparar o desempenho desses métodos no conjunto Air
Passenger. Queremos observar como cada biblioteca trata a sazonalidade e a
tendência da série e apresentar o SysIdentPy como uma opção viável para previsão
de séries temporais.

### Pacotes necessários e versões

Esta comparação foi verificada com o SysIdentPy 0.9.0 no Python 3.12.12. As
bibliotecas de previsão são opcionais e foram testadas com as seguintes versões:

```bash
python -m pip install -e .
python -m pip install sktime==1.0.1 neuralprophet==0.9.0 prophet==1.3.0
python -m pip install pmdarima==2.1.1 tbats==1.1.3 statsmodels==0.14.6
python -m pip install scipy==1.15.3 torch==2.5.1
```

A versão do SciPy satisfaz a restrição `scipy<1.16` imposta pelos adaptadores
BATS/TBATS nesta versão do sktime. Use um ambiente virtual, pois esses pacotes
possuem uma árvore de dependências relativamente grande e sensível a
compatibilidade. Os modelos aleatórios usam a semente 42.

Vamos começar importando os pacotes necessários e preparando o ambiente desta
análise.

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

Usamos o carregador do `sktime` e reservamos as 24 observações mensais finais
para teste. Todos os métodos são avaliados nesse mesmo horizonte.

```python
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=24)
plot_series(y_train, y_test, labels=["y_train", "y_test"])
fh = ForecastingHorizon(y_test.index, is_relative=False)
print(y_train.shape[0], y_test.shape[0])
```

A imagem a seguir mostra os dados do sistema que será modelado.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/air-passenger-benchmark-01.png?raw=true)

## Resultados

Como temos vários modelos para testar, os resultados atuais são resumidos na
tabela a seguir. A divisão dos dados e a definição do MSE são as mesmas em todas
as linhas.

| Nº | Pacote | Erro quadrático médio |
| ---: | --- | ---: |
| 1 | SysIdentPy (AOLS) | 440.9993 |
| 2 | SysIdentPy (MetaMSS) | 510.3495 |
| 3 | NeuralProphet | 514.0477 |
| 4 | Prophet | 910.7187 |
| 5 | Exponential Smoothing | 1055.5128 |
| 6 | SysIdentPy (Neural NARX) | 1621.5225 |
| 7 | SysIdentPy (FROLS) | 1811.9000 |
| 8 | AutoARIMA | 2230.3321 |
| 9 | ARIMA manual | 2592.7244 |
| 10 | AutoETS | 3128.9366 |
| 11 | TBATS | 8825.0097 |
| 12 | BATS | 9043.4934 |

As 13 saídas finais do treinamento inicializam a previsão livre de cada modelo
do SysIdentPy e não entram na métrica. O MetaMSS usa uma divisão cronológica
interna para selecionar a estrutura e não reajusta automaticamente os parâmetros
selecionados com todas as 120 observações de treinamento. A tabela compara as
configurações compactas declaradas aqui; ela não estabelece uma classificação
geral entre as bibliotecas.


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

A arquitetura da rede é a mesma usada na documentação do SysIdentPy para
demonstrar a construção de um modelo Neural NARX.

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

## Modelos do sktime

Os modelos a seguir estão disponíveis no pacote **sktime**.

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

O valor histórico de 316.5409 para o Neural NARX usava outro protocolo: a rede
era treinada nas primeiras 108 observações, as 13 saídas medidas seguintes
inicializavam a recursão e somente os 23 meses finais eram avaliados. Repetir
esse protocolo com o ambiente atual produz 316.8340. Na divisão comum de
120/24, a antiga configuração em minibatches produz MSE 2967.6219. Selecionar o
tamanho do lote, a taxa de aprendizado e o número de épocas em uma divisão
interna de 96/24 e, depois, reajustar a rede com todas as 120 observações de
treinamento reduz o MSE em simulação livre para 1621.5225. O MSE da predição de
um passo à frente é 307.1843, indicando que a acumulação recursiva de erros
explica boa parte do erro restante em simulação livre.

Os resultados finais podem ser resumidos como na tabela apresentada no início
deste estudo de caso:


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


## Sistema com Histerese - Modelagem de um Dispositivo Amortecedor Magneto-reológico

Os efeitos de memória entre entrada e saída quase-estáticas tornam a modelagem de sistemas histeréticos muito difícil. Modelos baseados em física são frequentemente usados para descrever os loops de histerese, mas esses modelos geralmente carecem da simplicidade e eficiência requeridas em aplicações práticas envolvendo caracterização, identificação e controle de sistemas. Como detalhado em [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), modelos NARX provaram ser uma escolha viável para descrever os loops de histerese. Veja o Capítulo 8 para um background detalhado. No entanto, mesmo considerando as condições suficientes para representação de histerese independente de taxa, algoritmos clássicos de seleção de estrutura falham em retornar um modelo com desempenho decente e o usuário precisa definir uma função multi-valorada para garantir a ocorrência da estrutura limitante $\mathcal{H}$ ([Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)).

Embora algum progresso tenha sido feito, trabalhos anteriores foram limitados a modelos com um único ponto de equilíbrio. O presente estudo de caso visa apresentar novas perspectivas na seleção de estrutura de modelos de sistemas histeréticos considerando os casos onde os modelos têm múltiplas entradas e não é restrito quanto ao número de pontos de equilíbrio. Para isso, o algoritmo MetaMSS será usado para construir um modelo para um amortecedor magneto-reológico (MRD) considerando as condições suficientes mencionadas.

### Uma Breve descrição do modelo Bouc-Wen de dispositivo amortecedor magneto-reológico

Os dados usados neste estudo de caso são do modelo Bouc-Wen ([Bouc, R - Forced Vibrations of a Mechanical System with Hysteresis](https://www.scirp.org/reference/referencespapers?referenceid=726819)), ([Wen, Y. X. - Method for Random Vibration of Hysteretic Systems](https://ascelibrary.org/doi/10.1061/JMCEA3.0002106)) de um MRD cujo diagrama esquemático é mostrado na figura abaixo.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/bouc_wen.png?raw=true)
> O modelo para um amortecedor magneto-reológico proposto por [Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972).

A forma geral do modelo Bouc-Wen pode ser descrita como ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)):

$$
\begin{equation}
\dfrac{dz}{dt} = g\left[x,z,sign\left(\dfrac{dx}{dt}\right)\right]\dfrac{dx}{dt},
\end{equation}
$$

onde $z$ é a saída do modelo histerético, $x$ a entrada e $g[\cdot]$ uma função não linear de $x$, $z$ e $sign (dx/dt)$. ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)) propuseram o seguinte modelo fenomenológico para o dispositivo mencionado:

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

onde $f$ é a força de amortecimento, $c_1$ e $c_0$ representam os coeficientes viscosos, $E$ é a tensão de entrada, $x$ é o deslocamento e $\dot{x}$ é a velocidade do modelo. Os parâmetros do sistema (veja a tabela abaixo) foram retirados de [Leva, A. and Piroddi, L. - NARX-based technique for the modelling of magneto-rheological damping devices](https://iopscience.iop.org/article/10.1088/0964-1726/11/1/309).

| Parâmetro  | Valor          | Parâmetro | Valor        |
|------------|----------------|-----------|--------------|
| $c_{0_a}$  | $20.2 \, N \, s/cm$  | $\alpha_{a}$  | $44.9 \, N/cm$  |
| $c_{0_b}$  | $2.68 \, N \, s/cm \, V$ | $\alpha_{b}$  | $638 \, N/cm$   |
| $c_{1_a}$  | $350 \, N \, s/cm$   | $\gamma$      | $39.3 \, cm^{-2}$ |
| $c_{1_b}$  | $70.7 \, N \, s/cm \, V$  | $\beta$       | $39.3 \, cm^{-2}$ |
| $k_{0}$    | $15 \, N/cm$    | $n$           | $2$          |
| $k_{1}$    | $5.37 \, N/cm$   | $\eta$       | $251 \, s^{-1}$ |
| $x_{0}$    | $0 \, cm$      | $A$           | $47.2$       |

Para este estudo particular, tanto as entradas de deslocamento quanto de tensão, $x$ e $E$, respectivamente, foram geradas filtrando uma sequência de ruído gaussiano branco usando um filtro FIR Blackman-Harris com frequência de corte de $6$Hz. O tamanho do passo de integração foi definido como $h = 0.002$, seguindo os procedimentos descritos em [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968). Estes procedimentos são apenas para fins de identificação, já que as entradas de um MRD podem ter várias características diferentes.

Os dados usados neste exemplo são fornecidos pelo Professor Samir Angelo Milani Martins.

Os desafios são:

- possui uma não linearidade com memória, ou seja, uma não linearidade dinâmica;
- a não linearidade é governada por uma variável interna z(t), que não é mensurável;
- a forma funcional não linear na equação de Bouc Wen é não linear no parâmetro;
- a forma funcional não linear na equação de Bouc Wen não admite uma expansão de série de Taylor finita devido à presença de valores absolutos

### Pacotes Necessários e Versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12,
`pandas==2.3.3` e `scikit-learn==1.7.2`. Instale explicitamente o checkout do
repositório e os dois pacotes de preparação dos dados:

```bash
python -m pip install -e .
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

O conjunto de dados é carregado por uma URL imutável do `sysidentpy-data`. Os
exemplos aleatórios usam a semente 42.

### Configuração do SysIdentPy


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


Vamos verificar como é o comportamento histerético considerando cada entrada:


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


Agora podemos construir um modelo NARX. Com as três entradas e o
`MaxAbsScaler`, o RRSE em simulação livre é $0.045104$:


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


Se removermos a entrada `sign(v)` e usarmos a mesma configuração, a simulação
livre diverge na amostra 203, como mostra a figura a seguir:


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


Usar o MetaMSS sem `sign(v)` adia a perda de estabilidade numérica, mas não a
elimina: esta trajetória em simulação livre diverge na amostra 1153.


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


Antes da divergência, o comportamento oscilatório se torna visível quando a
saída se aproxima de seu valor mínimo.


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


Ao adicionar novamente a entrada `sign(v)` e usar o MetaMSS, a simulação livre
permanece finita e produz RRSE $0.055707$, próximo ao resultado do FROLS com
todas as entradas.


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


Este caso também destacará a importância do escalonamento de dados. Anteriormente, usamos o método `MaxAbsScaler`, que resultou em ótimos modelos ao usar as entradas `sign(v)`, mas também resultou em modelos instáveis ao remover essa feature de entrada. Quando o escalonamento é aplicado usando `MinMaxScaler`, no entanto, a estabilidade geral dos resultados melhora, e o modelo não diverge, mesmo quando a entrada `sign(v)` é removida, usando o algoritmo `FROLS`.

O usuário pode obter os resultados abaixo apenas alterando o método de escalonamento de dados usando


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

e executando cada modelo novamente. Essa mudança torna finitas todas as
trajetórias em simulação livre, mas não melhora as configurações que usam todas
as entradas.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-10.png?raw=true)
> FROLS com `sign(v)` e `MinMaxScaler`: RRSE 0.115986.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-11.png?raw=true)
> FROLS sem `sign(v)` e com `MinMaxScaler`: RRSE 0.163944.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-12.png?raw=true)
> MetaMSS sem `sign(v)` e com `MinMaxScaler`: RRSE 0.185607.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-13.png?raw=true)
> MetaMSS com `sign(v)` e `MinMaxScaler`: RRSE 0.104511.

A comparação completa é:

| Escalonamento | Seletor | Entradas | Resultado em simulação livre |
| --- | --- | --- | ---: |
| MaxAbs | FROLS | com `sign(v)` | RRSE 0.045104 |
| MaxAbs | FROLS | sem `sign(v)` | divergiu na amostra 203 |
| MaxAbs | MetaMSS | sem `sign(v)` | divergiu na amostra 1153 |
| MaxAbs | MetaMSS | com `sign(v)` | RRSE 0.055707 |
| MinMax | FROLS | com `sign(v)` | RRSE 0.115986 |
| MinMax | FROLS | sem `sign(v)` | RRSE 0.163944 |
| MinMax | MetaMSS | sem `sign(v)` | RRSE 0.185607 |
| MinMax | MetaMSS | com `sign(v)` | RRSE 0.104511 |

O MetaMSS com `sign(v)` é o melhor entre as configurações com MinMax. O melhor
resultado geral continua sendo o FROLS com `sign(v)` e `MaxAbsScaler`. Uma saída
não finita em simulação livre é registrada como divergência, em vez de ser
convertida em uma métrica escalar.

Aqui está o loop histerético predito:


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

O conteúdo da descrição deriva principalmente (copiar e colar) do [artigo associado - Three free data sets for development and benchmarking in nonlinear system identification](https://ieeexplore.ieee.org/document/6669201). Para uma descrição detalhada, os leitores são encaminhados à referência vinculada.

> O sistema Silverbox pode ser visto como uma implementação eletrônica do oscilador de Duffing. É construído como um sistema linear invariante no tempo de 2ª ordem com uma não linearidade estática polinomial de 3º grau ao redor dele em feedback. Este tipo de dinâmica é, por exemplo, frequentemente encontrado em sistemas mecânicos [Nonlinear Benchmark - Silverbox](https://www.nonlinearbenchmark.org/benchmarks/silverbox).

Neste estudo de caso, criaremos um modelo NARX para o benchmark Silver box. O Silver box representa uma versão simplificada de processos oscilatórios mecânicos, que são uma categoria crítica de sistemas dinâmicos não lineares. Exemplos incluem suspensões de veículos, onde amortecedores e molas progressivas desempenham papéis vitais. Os dados gerados pelo Silver box fornecem uma representação simplificada de tais componentes combinados. O circuito elétrico que gera esses dados aproxima de perto, mas não corresponde perfeitamente, aos modelos idealizados descritos abaixo.

Conforme descrito no artigo original, o sistema foi excitado usando um gerador de forma de onda geral (HPE1445A). O sinal de entrada começa como um sinal de tempo discreto $r(k)$, que é convertido para um sinal analógico $r_c(t)$ usando reconstrução zero-order-hold. O sinal de excitação real $u_0(t)$ é então obtido passando $r_c(t)$ através de um filtro passa-baixa analógico $G(p)$ para eliminar o conteúdo de alta frequência em torno de múltiplos da frequência de amostragem. Aqui, $p$ denota o operador de diferenciação. Assim, a entrada é dada por:

$$
u_0(t) = G(p) r_c(t).
$$

Os sinais de entrada e saída foram medidos usando placas de aquisição de dados HP1430A, com relógios sincronizados para as placas de aquisição e gerador. A frequência de amostragem foi:

$$
f_s = \frac{10^7}{2^{14}} = 610.35 \, \text{Hz}.
$$

O silver box usa circuitos elétricos analógicos para gerar dados representando um sistema mecânico ressonante não linear com uma massa móvel $m$, amortecimento viscoso $d$, e uma mola não linear $k(y)$. O circuito elétrico é projetado para relacionar o deslocamento $y(t)$ (a saída) à força $u(t)$ (a entrada) pela seguinte equação diferencial:

$$
m \frac{d^2 y(t)}{dt^2} + d \frac{d y(t)}{dt} + k(y(t)) y(t) = u(t).
$$

A mola progressiva não linear é descrita por uma rigidez estática dependente da posição:

$$
k(y(t)) = a + b y^2(t).
$$

A relação sinal-ruído é suficientemente alta para modelar o sistema sem considerar o ruído de medição. No entanto, o ruído de medição pode ser incluído substituindo $y(t)$ pela variável artificial $x(t)$ na equação acima, e introduzindo perturbações $w(t)$ e $e(t)$ da seguinte forma:

$$
\begin{aligned}
& m \frac{d^2 x(t)}{dt^2} + d \frac{d x(t)}{dt} + k(x(t)) x(t) = u(t) + w(t), \\
& k(x(t)) = a + b x^2(t), \\
& y(t) = x(t) + e(t).
\end{aligned}
$$

### Pacotes Necessários e Versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12 e
`nonlinear-benchmarks==1.0.1`. Instale explicitamente o checkout do repositório
e o carregador oficial do benchmark:

```bash
python -m pip install -e .
python -m pip install nonlinear-benchmarks==1.0.1
```

Use um ambiente virtual para isolar o carregador opcional. Os resultados
numéricos devem ser recalculados se o ambiente ou a configuração do modelo forem
alterados.

### Configuração do SysIdentPy

Nesta seção, demonstraremos a aplicação do SysIdentPy ao dataset Silver box. O código a seguir guiará você através do processo de carregamento do dataset, configuração dos parâmetros do SysIdentPy e construção de um modelo para o sistema mencionado.


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

Usamos o pacote `nonlinear_benchmarks` para carregar os dados. O usuário é encaminhado à [documentação do pacote - GerbenBeintema/nonlinear_benchmarks: The official dataload for http://www.nonlinearbenchmark.org/ (github.com)](https://github.com/GerbenBeintema/nonlinear_benchmarks/tree/master) para verificar os detalhes de como usá-lo.

O gráfico a seguir detalha os dados de treinamento e teste do experimento.


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


> Nota Importante

O objetivo deste benchmark é desenvolver um modelo que supere o modelo estado-da-arte (SOTA) apresentado no artigo de benchmarking. No entanto, os resultados no [artigo](https://arxiv.org/pdf/2012.07697) diferem daqueles fornecidos no [repositório GitHub](https://github.com/GerbenBeintema/SS-encoder-WH-Silver/blob/main/SS%20encoder%20Silverbox.ipynb).

| nx  | Conjunto        | NRMS    | RMS (mV)   |
| --- | --------------- | ------- | ---------- |
| 2   | Treino          | 0.10653 | 5.8103295  |
| 2   | Validação       | 0.11411 | 6.1938068  |
| 2   | Teste           | 0.19151 | 10.2358533 |
| 2   | Teste (no extra)| 0.12284 | 5.2789727  |
| 4   | Treino          | 0.03571 | 1.9478290  |
| 4   | Validação       | 0.03922 | 2.1286373  |
| 4   | Teste           | 0.12712 | 6.7943448  |
| 4   | Teste (no extra)| 0.05204 | 2.2365904  |
| 8   | Treino          | 0.03430 | 1.8707026  |
| 8   | Validação       | 0.03732 | 2.0254112  |
| 8   | Teste           | 0.10826 | 5.7865255  |
| 8   | Teste (no extra)| 0.04743 | 2.0382715  |
> Tabela: resultados apresentados no github.

Parece que os valores mostrados no artigo realmente representam o tempo de treinamento, não as métricas de erro. Entrarei em contato com os autores para confirmar esta informação. De acordo com o site Nonlinear Benchmark, a informação é a seguinte:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/silver_sota.png?raw=true)

onde os valores na coluna "Training time" correspondem aos apresentados como métricas de erro no artigo.

> Enquanto aguardamos a confirmação dos valores corretos para este benchmark, demonstraremos o desempenho do SysIdentPy. No entanto, nos absteremos de fazer comparações ou tentar melhorar o modelo nesta fase.

### Resultados

Começaremos (como fizemos em todos os outros estudos de caso) com uma configuração básica do FROLS usando uma função de base polinomial com grau igual a 2. O `xlag` e `ylag` são definidos como $7$ neste primeiro exemplo. Como o dataset é consideravelmente grande, começaremos com `n_info_values=40`. Como estamos lidando com um grande dataset de treinamento, usaremos o `err_tol` em vez de critérios de informação para ter um desempenho mais rápido. Também definiremos `n_terms=40`, o que significa que a busca parará se o `err_tol` for atingido ou 40 regressores forem testados no algoritmo `ERR`. Embora esta abordagem possa resultar em um modelo sub-ótimo, é um ponto de partida razoável para nossa primeira tentativa. Existem três experimentos diferentes: multisine, arrow (full) e arrow (no extrapolation).


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

Os resultados atuais em simulação livre, avaliados após a janela de
inicialização de 50 amostras fornecida pelo carregador, são:

| Conjunto de teste | RMSE | NRMSE |
| --- | ---: | ---: |
| Multisine | 0.007727 | 0.142302 |
| Arrow, completo | 0.004148 | 0.077565 |
| Arrow, sem extrapolação | 0.002229 | 0.051822 |

O teste arrow completo é mais difícil que a versão sem extrapolação. A métrica
exclui a janela de inicialização uma única vez; ela não remove
`model.max_lag` pela segunda vez. Os resultados externos de modelos de estado
profundos continuam sendo uma referência útil, mas não são comparados
diretamente com esta tabela sem antes igualar divisão dos dados, inicialização e
normalização.

## F-16 Ground Vibration Test Benchmark

Os exemplos a seguir demonstram a aplicação do SysIdentPy a um conjunto de
dados real. Eles não procuram reproduzir os resultados dos manuscritos citados.
Parâmetros como `ylag` e `xlag`, assim como o tamanho dos conjuntos de
identificação e validação, diferem dos estudos originais. Ajustes relacionados à
taxa de amostragem e outras etapas de preparação dos dados também não são
tratados neste exemplo.

**Para uma referência abrangente sobre o benchmark F-16 Ground Vibration Test,
consulte o [site Nonlinear Benchmark](http://www.nonlinearbenchmark.org/#F16).**

> **Nota:** este exemplo é uma demonstração preliminar do desempenho do
> SysIdentPy no conjunto F-16. Uma análise mais detalhada será apresentada em
> uma publicação futura. O site do benchmark reúne recursos e referências úteis
> sobre identificação de sistemas e aprendizado de máquina, e o leitor é
> encorajado a consultá-los.

### Visão geral do benchmark

O F-16 Ground Vibration Test é um experimento relevante em identificação de
sistemas e dinâmica não linear. Trata-se de um sistema de alta ordem com
não linearidades de folga e atrito nas interfaces de montagem das cargas úteis
de uma aeronave F-16 em escala real.

**Detalhes do experimento:**

- **Evento:** Siemens LMS Ground Vibration Testing Master Class
- **Data:** setembro de 2014
- **Local:** base militar de Saffraanberg, Sint-Truiden, Bélgica

Durante o ensaio, duas cargas úteis simuladas foram montadas nas pontas das asas
para representar a massa e a inércia de dispositivos reais normalmente
instalados na aeronave durante o voo. A estrutura foi instrumentada com
acelerômetros, e um excitador sob a asa direita aplicou os sinais de entrada. A
principal fonte de não linearidade foi associada às interfaces de montagem das
cargas úteis, em particular à interface entre a asa direita e a carga útil, que
apresentou distorções não lineares significativas.

**Dados e referências:**

- **Disponibilidade dos dados:** o conjunto inclui a descrição detalhada do
  sistema, dados de estimação e teste e imagens da montagem, disponibilizados
  nos formatos `.csv` e `.mat`.
- **Referência:** J.P. Noël e M. Schoukens, “F-16 aircraft benchmark based on
  ground vibration test data”, 2017 Workshop on Nonlinear System Identification
  Benchmarks, pp. 19–23, Bruxelas, Bélgica, 24–26 de abril de 2017.

O objetivo é ilustrar como o SysIdentPy pode ser aplicado a um conjunto de dados
dessa complexidade. Para uma análise completa do benchmark e de sua metodologia,
consulte os recursos e as referências indicados.

### Pacotes necessários e versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12 e
`pandas==2.3.3`. Instale explicitamente o checkout do repositório e o pandas:

```bash
python -m pip install -e .
python -m pip install pandas==2.3.3
```

Os dados são carregados por uma URL imutável do `sysidentpy-data`. Os resultados
numéricos devem ser recalculados se o ambiente ou a configuração do modelo forem
alterados.

### Configuração do SysIdentPy


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

## Procedimento

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


O código a seguir divide o conjunto de dados em partes de treinamento e teste.

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

Definiremos os lags das duas entradas como:

```python
x1lag = list(range(1, 10))
x2lag = list(range(1, 10))
x2lag
```


e construiremos um modelo NARX da seguinte forma:

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

Depois de excluir as 20 amostras iniciais correspondentes aos lags do modelo, o
RRSE em simulação livre é $0.291070$. O mesmo trecho alinhado é usado na
autocorrelação dos resíduos e na correlação cruzada com a primeira entrada.

| Regressores | Parâmetros | ERR |
| --- | ---: | ---: |
| y(k-1) | 1.8387E+00 | 9.43378253E-01 |
| y(k-2) | -1.8938E+00 | 1.95167599E-02 |
| y(k-3) | 1.3337E+00 | 1.02432261E-02 |
| y(k-6) | -1.6038E+00 | 8.03485985E-03 |
| y(k-9) | 2.6776E-01 | 9.27874557E-04 |
| x2(k-7) | -2.2385E+01 | 3.76837313E-04 |
| x1(k-1) | 8.2709E+00 | 6.81508210E-04 |
| x2(k-3) | 1.0587E+02 | 1.57459800E-03 |
| x1(k-8) | -3.7975E+00 | 7.35086279E-04 |
| x2(k-1) | 8.5725E+01 | 4.85358786E-04 |
| y(k-7) | 1.3955E+00 | 2.77245281E-04 |
| y(k-5) | 1.3219E+00 | 8.64120037E-04 |
| y(k-10) | -2.9306E-01 | 8.51717688E-04 |
| y(k-4) | -9.5479E-01 | 7.23623116E-04 |
| y(k-8) | -7.1309E-01 | 4.44988077E-04 |
| y(k-12) | -3.0437E-01 | 1.49743148E-04 |
| y(k-11) | 4.8602E-01 | 3.34613282E-04 |
| y(k-13) | -8.2442E-02 | 1.43738964E-04 |
| y(k-15) | -1.6762E-01 | 1.25546584E-04 |
| x1(k-2) | -8.9698E+00 | 9.76699739E-05 |
| y(k-17) | 2.2036E-02 | 4.55983807E-05 |
| y(k-14) | 2.4900E-01 | 1.10314107E-04 |
| y(k-19) | -6.8239E-03 | 1.99734771E-05 |
| x2(k-9) | -9.6265E+01 | 2.98523208E-05 |
| x2(k-8) | 2.2620E+02 | 2.34402543E-04 |
| x2(k-2) | -2.3609E+02 | 1.04172323E-04 |
| y(k-20) | -5.4663E-02 | 5.37895336E-05 |
| x2(k-6) | -2.3651E+02 | 2.11392628E-05 |
| x2(k-4) | 1.7378E+02 | 2.18396315E-05 |
| x1(k-7) | 4.9862E+00 | 2.03811842E-05 |


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


### Critério de informação

O traçado do critério de informação usado durante a seleção de ordem é mostrado
abaixo. Ele complementa a tabela de regressores selecionados sem alterar o
protocolo de validação.


```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")

# You can use the plot below to choose the "n_terms" and run the model again with the most adequate value of terms.
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/f-16-aircraft-06.png?raw=true)

Este é um estudo ilustrativo com o SysIdentPy, não uma submissão oficial ao
benchmark F-16. O registro curado de 32.768 amostras e sua divisão em duas
metades não reproduzem todas as divisões distribuídas pelo carregador oficial;
por isso, o RRSE não deve ser comparado diretamente às tabelas oficiais do
benchmark.

## Previsão Fotovoltaica

Neste estudo de caso, avaliamos a capacidade do SysIdentPy de prever dados de
irradiância solar, que podem servir como aproximação para a produção solar
fotovoltaica (PV). O objetivo é demonstrar que o SysIdentPy oferece uma
alternativa competitiva para modelagem de séries temporais, sem afirmar que uma
biblioteca é superior às demais.

### Visão geral do conjunto de dados

O conjunto usado nesta análise contém medições de irradiância solar, uma
variável essencial para prever a produção fotovoltaica. A irradiância solar é a
potência da radiação recebida por unidade de área na superfície terrestre,
normalmente medida em watts por metro quadrado (W/m²). Previsões precisas dessa
grandeza ajudam a otimizar a produção de energia e a administrar a estabilidade
da rede elétrica.

**Detalhes do conjunto de dados:**

- **Fonte:** o conjunto pode ser obtido no repositório do NeuralProphet no
  GitHub.
- **Período:** os dados cobrem um intervalo contínuo com medições frequentes.
- **Variáveis:** valores de irradiância solar ao longo do tempo, usados para
  modelar e prever níveis futuros de irradiância.

### Comparação com outras bibliotecas

Para avaliar o SysIdentPy, compararemos seu desempenho com o da biblioteca
NeuralProphet. O NeuralProphet é capaz de representar padrões sazonais e
tendências complexas, servindo como uma referência adequada para esta tarefa.

Usaremos os seguintes métodos:

- **NeuralProphet:**
  - A configuração será baseada nos exemplos da [documentação do
    NeuralProphet](https://neuralprophet.com/html/example_links/energy_data_example.html),
    que apresenta recursos para capturar padrões temporais e fazer previsões.
- **SysIdentPy:**
  - **MetaMSS (Meta-heuristic Model Structure Selection):** usa algoritmos
    meta-heurísticos para determinar a estrutura do modelo.
  - **AOLS (Accelerated Orthogonal Least Squares):** seleciona os regressores
    relevantes do modelo.
  - **FROLS (Forward Regression with Orthogonal Least Squares, com função de
    base polinomial):** seleciona a estrutura por regressão ortogonal e pode
    incorporar termos polinomiais.

### Objetivo

O objetivo é comparar os métodos de previsão do SysIdentPy com o
NeuralProphet, com foco em:

- **Predição de um passo à frente:** avaliar a capacidade dos modelos de prever
  o próximo instante a partir dos dados históricos.

Os modelos são treinados com 80% do conjunto, e os 20% finais são reservados à
validação. Assim, o desempenho é medido em dados que não participaram do
treinamento.

### Pacotes necessários e versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12,
`neuralprophet==0.9.0`, `torch==2.5.1`, `pandas==2.3.3` e
`scikit-learn==1.7.2`. Instale explicitamente as dependências opcionais da
comparação:

```bash
python -m pip install -e .
python -m pip install neuralprophet==0.9.0 torch==2.5.1
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

O conjunto de dados é carregado por uma URL imutável do `sysidentpy-data`. Os
modelos aleatórios usam a semente 42.

### Procedimento

1. **Preparação dos dados:** carregar e preparar os dados de irradiância solar.
2. **Treinamento dos modelos:** aplicar os métodos escolhidos do SysIdentPy e o
   NeuralProphet aos dados de treinamento.
3. **Avaliação:** medir a precisão das previsões no conjunto de validação.

Ao comparar essas abordagens, procuramos mostrar o SysIdentPy como uma opção
viável para previsão de séries temporais e destacar sua utilidade em aplicações
práticas.

Vamos começar importando as bibliotecas necessárias e preparando o ambiente.

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

A referência NeuralProphet produz $MSE=2473.5397$ nos 1.752 instantes de
validação. Vamos verificar como os métodos do SysIdentPy se comportam no mesmo
intervalo.

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

O resultado do FROLS é $MSE=2204.3336$.

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

O MetaMSS seleciona o modelo com o menor erro desta comparação, com
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

O resultado do AOLS é $MSE=2361.5617$.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/PV-forecasting-benchmark-03.png?raw=true)

A comparação reproduzida é, portanto:

| Método | MSE de um passo à frente |
| --- | ---: |
| MetaMSS | 2154.2684 |
| FROLS | 2204.3336 |
| AOLS | 2361.5617 |
| NeuralProphet | 2473.5397 |

Os vetores de validação do SysIdentPy recebem como prefixo as 24 observações
finais do treinamento, e a métrica exclui esse prefixo. O NeuralProphet recebe
as mesmas 24 observações como contexto, enquanto seu MSE é restrito aos
instantes de validação. Portanto, todos os valores cobrem o mesmo intervalo. O
protocolo de um passo à frente mede a predição local com acesso à saída medida
mais recente a cada passo; ele não demonstra estabilidade em simulação livre.
