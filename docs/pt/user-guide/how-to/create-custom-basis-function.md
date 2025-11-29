# Criar uma Função Base Customizada

> Este guia espelha o exemplo disponível em `examples/custom-basis-function.ipynb` e mostra como é fácil integrar seu próprio gerador de features no SysIdentPy.

Neste how-to, estendemos a classe `BaseBasisFunction` para criar um mapeamento de features harmônicas usando apenas NumPy. A nova classe funciona exatamente como as funções base nativas, podendo ser reutilizada em qualquer estimador que espera a mesma interface.

## Requisitos

Você pode reutilizar o ambiente do projeto ou instalar um conjunto mínimo de pacotes:

```
sysidentpy
numpy
matplotlib
```

```bash
pip install -r requirements.txt
```

- O exemplo roda inteiramente em CPU.
- Nenhum dataset adicional é necessário.

## Gerando um dataset sintético

Construímos um sistema SISO simples com um forte componente senoidal controlado pela entrada. As primeiras 1600 amostras são usadas para treinamento e o restante para validação.

```python
import numpy as np
import matplotlib.pyplot as plt
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.plotting import plot_results
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function.basis_function_base import BaseBasisFunction

x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=False, sigma=0.0001, train_percentage=50
)
```

## Implementando a função base customizada

A nova classe `HarmonicBasis` precisa implementar apenas os métodos `fit` e `transform`. Internamente, criamos uma matriz que contém os sinais brutos mais as transformações seno/cosseno para as harmônicas solicitadas. Como a classe herda de `BaseBasisFunction`, o SysIdentPy pode utilizá-la da mesma forma que qualquer opção nativa.

```python
class HarmonicBasis(BaseBasisFunction):
    """Mapeia regressores defasados para features seno/cosseno."""

    def __init__(self, harmonics=(1,), include_linear=True, scale=np.pi):
        super().__init__(degree=1)
        self.harmonics = tuple(harmonics)
        self.include_linear = include_linear
        self.scale = scale

    def _build_matrix(self, data, predefined_regressors):
        features = []
        if self.include_linear:
            features.append(data)
        for harmonic in self.harmonics:
            scaled = self.scale * harmonic * data
            features.append(np.sin(scaled))
            features.append(np.cos(scaled))
        if not features:
            raise ValueError("The basis needs at least one active transformation.")
        psi = np.hstack(features)
        if predefined_regressors is not None:
            idx = np.asarray(predefined_regressors, dtype=int)
            psi = psi[:, idx]
        return psi

    def fit(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        psi = self._build_matrix(data, predefined_regressors)
        return psi[max_lag:, :]

    def transform(
        self,
        data,
        max_lag=1,
        ylag=1,
        xlag=1,
        model_type="NARMAX",
        predefined_regressors=None,
    ):
        return self.fit(data, max_lag, ylag, xlag, model_type, predefined_regressors)
```

## Treinando com a função base customizada

O fluxo de trabalho é idêntico a todos os outros exemplos. Simplesmente passamos uma instância de `HarmonicBasis` para o `FROLS` e procedemos com treinamento, avaliação e visualização.

```python
basis_function = HarmonicBasis(harmonics=(1, 2, 3), include_linear=True, scale=np.pi)

model = FROLS(
    ylag=2,
    xlag=2,
    order_selection=True,
    n_info_values=20,
    info_criteria="aic",
    estimator=LeastSquares(),
    basis_function=basis_function,
    model_type="NARX",
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

rrse = root_relative_squared_error(y_valid[model.max_lag:], yhat[model.max_lag:])
print(f"RRSE (validation): {rrse:.4f}")
```

```python
plot_results(
    y=y_valid[model.max_lag:],
    yhat=yhat[model.max_lag:],
    n=400,
    figsize=(12, 4),
    title="Validation results with HarmonicBasis",
)
```

## Conclusão

Com apenas algumas linhas de código, construímos um substituto direto para as funções base nativas. Qualquer transformação NumPy/SciPy/Scikit-Learn pode ser exportada para uma classe como `HarmonicBasis`, permitindo reutilizar mapeamentos de features personalizados em todos os estimadores do SysIdentPy sem alterar o restante do seu workflow.

