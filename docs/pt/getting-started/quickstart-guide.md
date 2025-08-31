---
template: overrides/main.html
title: Uso Básico
---

## 1. Pré-requisitos

Você precisa conhecer um pouco de Python.

Para executar os exemplos, além do NumPy você precisará do `pandas` instalado.

```bash
pip install sysidentpy pandas
# Opcional: Para redes neurais e recursos avançados
pip install sysidentpy["all"]
```

## 2. Principais Recursos

SysIdentPy oferece uma estrutura flexível para construir, validar e visualizar modelos não lineares de séries temporais e sistemas dinâmicos. O processo de modelagem envolve algumas etapas: definir a representação matemática, escolher o algoritmo de estimação de parâmetros, selecionar a estrutura do modelo e determinar o horizonte de previsão.

Os seguintes recursos estão disponíveis no SysIdentPy:

### Classes de Modelo
- NARMAX, NARX, NARMA, NAR, NFIR, ARMAX, ARX, AR e suas variantes.

### Representações Matemáticas
- Polynomial (Polinomial)
- Neural
- Fourier
- Laguerre
- Bernstein
- Bilinear
- Legendre
- Hermite
- HermiteNormalized

Você também pode definir modelos NARX como Bayesian e Gradient Boosting usando a classe GeneralNARX, que oferece integração direta com vários algoritmos de aprendizado de máquina.

### Algoritmos de Seleção de Estrutura
- Forward Regression Orthogonal Least Squares (FROLS)
- Meta-model Structure Selection (MeMoSS / MetaMSS)
- Accelerated Orthogonal Least Squares (AOLS)
- Entropic Regression (ER)
- Ultra Orthogonal Least Squares (UOLS)

### Métodos de Estimação de Parâmetros
- Mínimos Quadrados (MQ)
- Total Least Squares (TLS)
- Mínimos Quadrados Recursivos (MQR)
- Ridge Regression
- Non-Negative Least Squares (NNLS)
- Least Squares Minimal Residues (LSMR)
- Bounded Variable Least Squares (BVLS)
- Least Mean Squares (LMS) e suas variantes:
  - Affine LMS
  - LMS with Sign Error
  - Normalized LMS
  - LMS with Normalized Sign Error
  - LMS with Sign Regressor
  - Normalized LMS with Sign Sign
  - Leaky LMS
  - Fourth-Order LMS
  - Mixed Norm LMS

### Critérios de Seleção de Ordem
- Critério de Informação de Akaike (AIC)
- Critério de Informação de Akaike Corrigido (AICc)
- Critério de Informação Bayesiano (BIC)
- Final Prediction Error (FPE)
- Khundrin's Law of Iterated Logarithm Criterion (LILC)

### Métodos de Previsão
- Um passo à frente (one-step ahead)
- n passos à frente (n-steps ahead)
- Infinito passos à frente / simulação livre (infinity-steps / free run simulation)

### Ferramentas de Visualização
- Gráficos de previsão
- Análise de resíduos
- Visualização da estrutura do modelo
- Visualização de parâmetros

---

Como você pode ver, o SysIdentPy suporta diversas combinações de modelos. Não se preocupe em escolher todas as configurações logo no começo. Vamos começar com as configurações padrão.

<div class="custom-collapsible-card">
    <input type="checkbox" id="toggle-info">
    <label for="toggle-info">
        📚 <strong>Em busca de mais detalhes sobre modelos NARMAX?</strong>
        <span class="arrow">▼</span>
    </label>
    <div class="collapsible-content">
        <p>
            Para informações completas sobre modelos, métodos e um conjunto de exemplos e benchmarks implementados no <strong>SysIdentPy</strong>, confira nosso livro:
        </p>
        <a href="https://sysidentpy.org/book/0-Preface/" target="_blank">
            <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
        </a>
        <p>
            Esse livro oferece uma orientação detalhada para auxiliar no seu trabalho com o <strong>SysIdentPy</strong>.
        </p>
        <p>
            🛠️ Você também pode explorar os <a href="https://sysidentpy.org/user-guide/overview/" target="_blank"><strong>tutoriais na documentação</strong></a> para exemplos práticos.
        </p>
    </div>
</div>

## 3. Guia Rápido

Para manter as coisas simples, vamos carregar alguns dados simulados para os exemplos.

```python
from sysidentpy.utils.generate_data import get_siso_data

# Gera um conjunto de dados de um sistema dinâmico simulado.
x_train, x_valid, y_train, y_valid = get_siso_data(
        n=300,
        colored_noise=False,
        sigma=0.0001,
        train_percentage=80
)
```

### Construa seu primeiro modelo NARX

Com os dados carregados, vamos construir um modelo NARX Polinomial. Usando as configurações padrão, você precisa definir pelo menos o método de seleção de estrutura e a representação matemática (função base).

```python
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

O método de seleção de estrutura (MSS) habilita as operações de "treinamento" e previsão do modelo.

Embora diferentes algoritmos tenham diferentes hiperparâmetros, esse não é o foco aqui. Mostraremos como modificá-los, mas não discutiremos as melhores configurações nesse guia.

```python
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

Para avaliar o desempenho, você pode usar qualquer métrica disponível na biblioteca. Exemplo com Root Relative Squared Error (RRSE):

```python
from sysidentpy.metrics import root_relative_squared_error

rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
```

```console
0.00014
```

Para visualizar a equação final do modelo polinomial, use a função `results`. Ela requer a seguinte configuração:

- `final_model`: Regressoras selecionadas após o ajuste
- `theta`: Parâmetros estimados
- `err`: Error Reduction Ratio (ERR)

```python
from sysidentpy.utils.display_results import results

r = pd.DataFrame(
        results(
                model.final_model, model.theta, model.err,
                model.n_terms, err_precision=8, dtype='sci'
        ),
        columns=['Regressores', 'Parâmetros', 'ERR'])
print(r)
```

Resultado (exemplo):

```console
Regressores     Parâmetros        ERR
0        x1(k-2)     0.9000  0.95556574
1         y(k-1)     0.1999  0.04107943
2  x1(k-1)y(k-1)     0.1000  0.00335113
```

Para visualizar o desempenho do modelo:

```python
from sysidentpy.utils.plotting import plot_results

plot_results(y=y_valid, yhat=yhat, n=1000)
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-polynomial-narx.png?raw=true)

Analisar resíduos de um modelo é essencial. Podemos calcular a autocorrelação dos resíduos e correlação cruzada entre resíduos e entradas conforme exemplo abaixo:

```python
from sysidentpy.utils.plotting import plot_residues_correlation
from sysidentpy.residues.residues_correlation import (
        compute_residues_autocorrelation,
        compute_cross_correlation,
)

# Autocorrelação
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Resíduos", ylabel="$e^2$")

# Correlação cruzada com uma entrada
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Resíduos", ylabel="$x_1e$")
```

<div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-ee.png?raw=true" width="400" alt="Quickstart EE" />
    <img src="https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-xe.png?raw=true" width="400" alt="Quickstart XE" />
</div>

Código completo para referência:

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
        columns=['Regressores', 'Parâmetros', 'ERR'])
print(r)

plot_results(y=y_valid, yhat=yhat, n=1000, figsize=(15, 4))
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Resíduos", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Resíduos", ylabel="$x_1e$")
```

### Personalizando a configuração do modelo

#### Seleção de Estrutura

Para usar o algoritmo **AOLS** em vez de `FROLS`:

```python
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

Usando **MetaMSS**:

```python
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

Usando **Entropic Regression (ER)**:

```python
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

#### Estimação de Parâmetros

Listar algoritmos disponíveis:

```python
from sysidentpy import parameter_estimation
print("Algoritmos disponíveis:", parameter_estimation.__all__)
```

Definir estimador específico (ex: LSMR):

```python
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

#### Função Base (Representação Matemática)

Listar funções base:

```python
from sysidentpy import basis_function
print("Funções base disponíveis:", basis_function.__all__)
```

Exemplo com Fourier:

```python
from sysidentpy.model_structure_selection import AOLS
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

!!! note
        O método `results` suporta apenas a base **Polynomial** no momento. Suporte a todas as funções de base está planejado para a versão 1.0.

#### Customizando o Tipo de Modelo

Diferença entre **NARX** e **ARX**: presença de termos não lineares. `degree=2` (Polynomial) permite um modelo potencialmente NARX; `degree=1` resulta em ARX. Porém, a linearidade final depende da equação obtida pelo método de seleção de estrutura.

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=1)  # ARX
model = FROLS(
        ylag=2,
        xlag=2,
        basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

Para criar um **NAR**:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=1)
model = FROLS(
        ylag=2,
        basis_function=basis_function,
        model_type="NAR",
)
model.fit(y=y_train)
yhat = model.predict(y=y_valid, forecast_horizon=23)
```

Para **NFIR** (apenas entradas):

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=1)
model = FROLS(
        xlag=2,
        basis_function=basis_function,
        model_type="NFIR",
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

<div class="custom-collapsible-card">
    <input type="checkbox" id="initial-info">
    <label for="initial-info">
        📚 <strong>Quer saber mais detalhes sobre condições iniciais?</strong>
        <span class="arrow">▼</span>
    </label>
    <div class="collapsible-content">
        <p>
            Veja o capítulo 9 do nosso livro para entender por que modelos autorregressivos precisam de condições iniciais:
        </p>
        <a href="https://sysidentpy.org/book/9-Validation/" target="_blank">
            <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
        </a>
    </div>
</div>

#### Horizonte de Previsão

Por padrão, `predict` realiza previsão de infinitos passos a frente (ou simulação livre). Para um número específico de passos à frente, use `steps_ahead`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial

basis_function = Polynomial(degree=2)
model = FROLS(
        ylag=2,
        xlag=2,
        basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)

yhat_1 = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
yhat_4 = model.predict(X=x_valid, y=y_valid, steps_ahead=4)
```

<div class="custom-collapsible-card">
    <input type="checkbox" id="steps-info">
    <label for="steps-info">
        📚 <strong>Mais detalhes sobre previsão com diferentes passos a frente?</strong>
        <span class="arrow">▼</span>
    </label>
    <div class="collapsible-content">
        <p>
            Veja o capítulo 9 do nosso livro para saber como funcionam previsões um passo, n-passos e infinitos passos a frente:
        </p>
        <a href="https://sysidentpy.org/book/9-Validation/" target="_blank">
            <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
        </a>
    </div>
</div>

#### Seleção de Ordem

A seleção de ordem é uma abordagem clássica para determinar automaticamente a ordem ótima do modelo ao utilizar o algoritmo **FROLS**. Esse processo auxilia na identificação da melhor combinação dos atrasos e regressores por meio da avaliação de diferentes modelos com base em um critério de informação.

!!! Important
        Critérios de informação *só se aplicam* ao algoritmo **FROLS**.

Habilite com:
1. `order_selection=True`
2. `info_criteria="bic"` (ou `"aic"`, `"aicc"`, `"fpe"`, `"lilc"`).

```python
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

Controlar número de regressores testados: `n_info_values`.

```python
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
        Aumentar `n_info_values` pode melhorar a precisão, mas aumenta o tempo computacional.

#### Rede Neural NARX

Exemplo com PyTorch:

```python
from torch import nn
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_results

basis_function = Polynomial(degree=1)

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
        optim_params={'betas': (0.9, 0.999), 'eps': 1e-05}
)

narx_net.fit(X=x_train, y=y_train)
yhat = narx_net.predict(X=x_valid, y=y_valid)
plot_results(y=y_valid, yhat=yhat, n=1000, figsize=(15, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-neural-narx.png?raw=true)

#### Estimadores Gerais

Você pode integrar qualquer estimador (scikit-learn, xgboost, catboost etc.) desde que eles sigam o padrão `fit` e `predict`.

Exemplo CatBoost NARX:

```python
from sysidentpy.general_estimators import NARX
from catboost import CatBoostRegressor
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.plotting import plot_results

basis_function = Polynomial(degree=1)
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

Sem NARX (para comparação):

```python
from catboost import CatBoostRegressor
from sysidentpy.utils.plotting import plot_results

catboost = CatBoostRegressor(
        iterations=300,
        learning_rate=0.1,
        depth=6
)
catboost.fit(x_train, y_train, verbose=False)
plot_results(y=y_valid, yhat=catboost.predict(x_valid), figsize=(15, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/main/docs/images/quickstart-catboost-without-narx.png?raw=true)

Você ainda pode explorar combinações: usar função base Fourier, previsão multi-passos, diferentes estimadores etc.

Este é apenas um guia rápido. Para tutoriais completos, guias passo a passo, explicações detalhadas e casos avançados, veja a [documentação](https://sysidentpy.org/) e o [livro](https://sysidentpy.org/book/0-Preface/).
