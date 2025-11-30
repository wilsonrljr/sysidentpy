# Definir Lags Específicos

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro oferece orientação aprofundada para apoiar seu trabalho com o SysIdentPy.

Diferentes formas de definir o lag máximo para entrada e saída


```python
pip install sysidentpy
```


```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
```

## Definindo lags usando um intervalo de valores

Se você passar valores inteiros para *ylag* e *xlag*, os lags são definidos como um intervalo de 1 até *ylag* e de 1 até *xlag*.

Por exemplo: se *ylag=4*, então os regressores candidatos são $y_{k-1}, y_{k-2}, y_{k-3}, y_{k-4}$


```python
basis_function = Polynomial(degree=1)

model = FROLS(
    order_selection=True,
    ylag=4,
    xlag=4,
    info_criteria="aic",
    basis_function=basis_function,
)
```

## Definindo lags específicos usando listas

Se você passar *ylag* e *xlag* como uma lista, apenas os lags relacionados aos valores na lista serão criados.
$y_{k-1}, y_{k-4}$,  $x_{k-1}, x_{k-4}$


```python
model = FROLS(
    order_selection=True,
    ylag=[1, 4],
    xlag=[1, 4],
    info_criteria="aic",
    basis_function=basis_function,
)
```

## Definindo lags para modelos Multiple Input Single Output (MISO)

O exemplo a seguir mostra como definir lags específicos para cada entrada. Note que precisamos usar uma lista aninhada neste caso.


```python
# O exemplo considera um modelo com 2 entradas, mas você pode usar o mesmo para qualquer quantidade de entradas.

model = FROLS(
    order_selection=True,
    ylag=[1, 4],
    xlag=[[1, 2, 3, 4], [1, 7]],
    info_criteria="aic",
    basis_function=basis_function,
)
# Os lags definidos são:
# x1(k-1), x1(k-2), x(k-3), x(k-4)
# x2(k-1), x1(k-7)
```


```python

```

