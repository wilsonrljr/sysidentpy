# Usar Extended Least Squares

Exemplo criado por Wilson Rocha Lacerda Junior

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla variedade de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro oferece orientação aprofundada para apoiar seu trabalho com o SysIdentPy.

Para usar o algoritmo **Extended Least Squares (ELS)**, defina o parâmetro `unbiased` como `True` ao definir o algoritmo de estimação de parâmetros.


```python
from sysidentpy.parameter_estimation import LeastSquares

estimator = LeastSquares(unbiased=True)
```


O hiperparâmetro `unbiased` está disponível em todos os algoritmos de estimação de parâmetros, com valor padrão `False`.

Além disso, o algoritmo **Extended Least Squares** é iterativo. No **SysIdentPy**, o número padrão de iterações é definido como 20 (`uiter=20`), já que estudos na literatura indicam que o algoritmo tipicamente converge entre 10 e 20 iterações. No entanto, você pode ajustar este valor para qualquer número de iterações que preferir.



```python
from sysidentpy.parameter_estimation import LeastSquares

estimator = LeastSquares(unbiased=True, uiter=40)
```

Um exemplo simples, porém completo, demonstrando a estimação de parâmetros usando o algoritmo **Extended Least Squares (ELS)** é mostrado abaixo.

*(Dados simulados são usados para fins ilustrativos.)*



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.generate_data import get_siso_data

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=True, sigma=0.2, train_percentage=90
)

basis_function = Polynomial(degree=2)
estimator = LeastSquares(unbiased=True)
parameters = np.zeros([3, 50])

for i in range(50):
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=3000, colored_noise=True, train_percentage=90
    )

    model = FROLS(
        order_selection=False,
        n_terms=3,
        ylag=2,
        xlag=2,
        elag=2,
        info_criteria="aic",
        estimator=estimator,
        basis_function=basis_function,
    )

    model.fit(X=x_train, y=y_train)
    parameters[:, i] = model.theta.flatten()

plt.figure(figsize=(14, 4))

# Compute and plot KDE for each parameter using scipy's gaussian_kde
x_grid = np.linspace(np.min(parameters), np.max(parameters), 1000)

for i, label in enumerate(["Parameter 1", "Parameter 2", "Parameter 3"]):
    kde = gaussian_kde(parameters[i, :])
    plt.plot(x_grid, kde(x_grid), label=label)

# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color="k", linestyle="--", label="Real Value 0.1")
plt.axvline(x=0.2, color="k", linestyle="--", label="Real Value 0.2")
plt.axvline(x=0.9, color="k", linestyle="--", label="Real Value 0.9")

plt.xlabel("Parameter Value")
plt.ylabel("Density")
plt.title("Kernel Density Estimate of Parameters (Matplotlib only)")
plt.legend()
plt.show()
```

