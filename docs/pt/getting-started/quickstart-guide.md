## 1. Pré-requisitos
Instale:
```bash
pip install sysidentpy pandas
```
Opcional (recursos avançados):
```bash
pip install sysidentpy["all"]
```

## 2. Exemplo Básico
Gerando dados simulados:
```python
from sysidentpy.utils.generate_data import get_siso_data
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=300, colored_noise=False, sigma=0.0001, train_percentage=80
)
```

Treinando um modelo NARX polinomial simples:
```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
basis_function = Polynomial(degree=2)
model = FROLS(ylag=2, xlag=2, basis_function=basis_function)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

Métrica:
```python
from sysidentpy.metrics import root_relative_squared_error
print(root_relative_squared_error(y_valid, yhat))
```

> Tradução resumida. Veja a versão em inglês para explicações completas e funcionalidades adicionais.
