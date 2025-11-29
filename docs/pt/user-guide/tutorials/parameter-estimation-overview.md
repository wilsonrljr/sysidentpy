# Estimação de Parâmetros - Visão Geral

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro oferece orientação aprofundada para apoiar seu trabalho com o SysIdentPy.

Aqui importamos o modelo NARMAX, a métrica para avaliação do modelo e os métodos para gerar dados de amostra para testes. Também importamos o pandas para uso específico.


```python
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import (
    TotalLeastSquares,
    RecursiveLeastSquares,
    NonNegativeLeastSquares,
    LeastMeanSquares,
    AffineLeastMeanSquares,
)
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results
```

## Gerando dados de amostra com 1 entrada e 1 saída

Os dados são gerados simulando o seguinte modelo:

$y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-1} + e_{k}$

Se *colored_noise* for definido como True:

$e_{k} = 0.8\nu_{k-1} + \nu_{k}$

onde $x$ é uma variável aleatória uniformemente distribuída e $\nu$ é uma variável com distribuição gaussiana com $\mu=0$ e $\sigma=0.1$

No próximo exemplo, geraremos dados com 1000 amostras com ruído branco e selecionando 90% dos dados para treinar o modelo.


```python
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)
```

### Existem vários métodos para estimação de parâmetros.

- Least Squares;
- Total Least Squares;
- Recursive Least Squares
- Ridge Regression
- NonNegative Least Squares
- Least Squares Minimal Residues
- Bounded Variable Least Squares
- Least Mean Squares
- Affine Least Mean Squares
- Least Mean Squares Sign Error
- Normalized Least Mean Squares
- Least Mean Squares Normalized Sign Error
- Least Mean Squares Sign Regressor
- Least Mean Squares Normalized Sign Regressor
- Least Mean Squares Sign Sign
- Least Mean Squares Normalized Sign Sign
- Least Mean Squares Normalized Leaky
- Least Mean Squares Leaky
- Least Mean Squares Fourth
- Least Mean Squares Mixed Norm


Modelos NARMAX polinomiais são lineares nos parâmetros, então métodos baseados em Least Squares funcionam bem para a maioria dos casos (usando com o algoritmo Extended Least Squares ao lidar com ruído colorido).

No entanto, o usuário pode escolher alguns métodos recursivos e de gradiente descendente estocástico (neste caso, o algoritmo Least Mean Squares e suas variantes) para essa tarefa também.

Escolher o método é simples: passe qualquer um dos métodos mencionados acima no parâmetro estimator.

- **Nota: Cada algoritmo tem parâmetros específicos que precisam ser ajustados. Nos exemplos a seguir, usaremos os valores padrão. Mais exemplos sobre ajuste de parâmetros estarão disponíveis em breve. Por enquanto, o usuário pode ler a documentação do método para mais informações.**

## Total Least Squares


```python
basis_function = Polynomial(degree=2)
estimator = TotalLeastSquares()

model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
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

    0.0021167167052431584
          Regressors  Parameters             ERR
    0        x1(k-2)  9.0000E-01  9.56200123E-01
    1         y(k-1)  1.9995E-01  4.05078042E-02
    2  x1(k-1)y(k-1)  1.0004E-01  3.28866604E-03


## Recursive Least Squares


```python
# recursive least squares
basis_function = Polynomial(degree=2)
estimator = RecursiveLeastSquares()

model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
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

    0.0020703083403116164
          Regressors  Parameters             ERR
    0        x1(k-2)  9.0012E-01  9.56200123E-01
    1         y(k-1)  2.0021E-01  4.05078042E-02
    2  x1(k-1)y(k-1)  9.9550E-02  3.28866604E-03


## Least Mean Squares


```python
basis_function = Polynomial(degree=2)
estimator = LeastMeanSquares()

model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
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

    0.015488793944313425
          Regressors  Parameters             ERR
    0        x1(k-2)  8.9775E-01  9.56200123E-01
    1         y(k-1)  2.0085E-01  4.05078042E-02
    2  x1(k-1)y(k-1)  7.5708E-02  3.28866604E-03


## Affine Least Mean Squares


```python
basis_function = Polynomial(degree=2)
estimator = AffineLeastMeanSquares()

model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
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

    0.0021441596280611167
          Regressors  Parameters             ERR
    0        x1(k-2)  8.9989E-01  9.56200123E-01
    1         y(k-1)  1.9992E-01  4.05078042E-02
    2  x1(k-1)y(k-1)  1.0003E-01  3.28866604E-03


# NonNegative Least Squares


```python
basis_function = Polynomial(degree=2)
estimator = NonNegativeLeastSquares()

model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
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

    0.0021170157359329173
          Regressors  Parameters             ERR
    0        x1(k-2)  9.0000E-01  9.56200123E-01
    1         y(k-1)  1.9995E-01  4.05078042E-02
    2  x1(k-1)y(k-1)  1.0004E-01  3.28866604E-03
