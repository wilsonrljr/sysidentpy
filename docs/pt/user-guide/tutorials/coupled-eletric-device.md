# Acionamentos Elétricos Acoplados

Nota: O exemplo mostrado neste notebook é retirado do livro [Nonlinear System Identification and Forecasting: Theory and Practice with SysIdentPy](https://sysidentpy.org/book/0-Preface/).

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

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-01.png?raw=true)

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


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-02.png?raw=true)


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-03.png?raw=true)


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-04.png?raw=true)


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-05.png?raw=true)


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


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/coupled-eletric-device-06.png?raw=true)


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


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/ce8_sota.png?raw=true)


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
