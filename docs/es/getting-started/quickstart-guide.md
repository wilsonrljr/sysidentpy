---
template: overrides/main.html
title: Uso Básico
---

## 1. Requisitos previos

Necesitas tener conocimientos básicos de Python.

Para ejecutar los ejemplos, además de NumPy necesitarás `pandas` instalado.

```bash
pip install sysidentpy pandas
# Opcional: Para redes neuronales y funcionalidades avanzadas
pip install sysidentpy["all"]
```

## 2. Principales funcionalidades

SysIdentPy ofrece una estructura flexible para construir, validar y visualizar modelos no lineales de serie temporal y sistemas dinámicos. El proceso de modelado incluye algunos pasos: definir la representación matemática, elegir el algoritmo de estimación de parámetros, seleccionar la estructura del modelo y determinar el horizonte de predicción.

Las siguientes funcionalidades están disponibles en SysIdentPy:

### Clases de Modelo

- NARMAX, NARX, NARMA, NAR, NFIR, ARMAX, ARX, AR y sus variantes.

### Representaciones Matemáticas

- Polynomial (Polinomial)
- Neural
- Fourier
- Laguerre
- Bernstein
- Bilinear
- Legendre
- Hermite
- HermiteNormalized

También puedes definir modelos NARX como Bayesian y Gradient Boosting usando la clase GeneralNARX, que ofrece integración directa con varios algoritmos de aprendizaje automático.

### Algoritmos de Selección de Estructura

- Forward Regression Orthogonal Least Squares (FROLS)
- Meta-model Structure Selection (MeMoSS / MetaMSS)
- Accelerated Orthogonal Least Squares (AOLS)
- Entropic Regression (ER)
- Ultra Orthogonal Least Squares (UOLS)

### Métodos de Estimación de Parámetros

- Mínimos Cuadrados (MQ)
- Total Least Squares (TLS)
- Mínimos Cuadrados Recursivos (MQR)
- Ridge Regression
- Non-Negative Least Squares (NNLS)
- Least Squares Minimal Residues (LSMR)
- Bounded Variable Least Squares (BVLS)
- Least Mean Squares (LMS) y sus variantes:
  - Affine LMS
  - LMS with Sign Error
  - Normalized LMS
  - LMS with Normalized Sign Error
  - LMS with Sign Regressor
  - Normalized LMS with Sign Sign
  - Leaky LMS
  - Fourth-Order LMS
  - Mixed Norm LMS

### Criterios de Selección de Orden

- Criterio de Información de Akaike (AIC)
- Criterio de Información de Akaike Corregido (AICc)
- Criterio de Información Bayesiano (BIC)
- Final Prediction Error (FPE)
- Khundrin's Law of Iterated Logarithm Criterion (LILC)

### Métodos de Predicción

- Un paso adelante (one-step ahead)
- n pasos adelante (n-steps ahead)
- Paso infinito adelante / simulación libre (infinity-steps / free run simulation)

### Herramientas de Visualización

- Gráficos de predicción
- Análisis de residuos
- Visualización de la estructura del modelo
- Visualización de parámetros

---

Como puedes ver, SysIdentPy soporta diversas combinaciones de modelos. No te preocupes por elegir todas las configuraciones al principio. Comencemos con las configuraciones por defecto.

<div class="custom-collapsible-card">
    <input type="checkbox" id="toggle-info">
    <label for="toggle-info">
         <strong>Buscas más detalles sobre modelos NARMAX?</strong>
        <span class="arrow"></span>
    </label>
    <div class="collapsible-content">
        <p>
            Para información completa sobre modelos, métodos y un conjunto de ejemplos y benchmarks implementados en <strong>SysIdentPy</strong>, consulta nuestro libro:
        </p>
        <a href="https://sysidentpy.org/book/0-Preface/" target="_blank">
            <em><strong>Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy</strong></em>
        </a>
        <p>
            Este libro ofrece una guía detallada para ayudarte en tu trabajo con <strong>SysIdentPy</strong>.
        </p>
        <p>
             También puedes explorar los <a href="https://sysidentpy.org/user-guide/overview/" target="_blank"><strong>tutoriales en la documentación</strong></a> para ejemplos prácticos.
        </p>
    </div>
</div>

## 3. Guía Rápida

Para mantener las cosas simples, cargaremos algunos datos simulados para los ejemplos.

```python
from sysidentpy.utils.generate_data import get_siso_data

# Genera un conjunto de datos de un sistema dinámico simulado.
x_train, x_valid, y_train, y_valid = get_siso_data(
        n=300,
        colored_noise=False,
        sigma=0.0001,
        train_percentage=80
)
```

### Construye tu primer modelo NARX

Con los datos cargados, vamos a construir un modelo NARX Polinomial. Usando la configuración por defecto, necesitas definir al menos el método de selección de estructura y la representación matemática (función base).

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

El método de selección de estructura (MSS) habilita las operaciones de "entrenamiento" y predicción del modelo.

Aunque distintos algoritmos tienen diferentes hiperparámetros, ese no es el foco aquí. Mostraremos cómo modificarlos, pero no discutiremos las mejores configuraciones en esta guía.

```python
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
```

Para evaluar el rendimiento, puedes usar cualquier métrica disponible en la librería. Ejemplo con Root Relative Squared Error (RRSE):

```python
from sysidentpy.metrics import root_relative_squared_error

rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
```

```console
0.00014
```

Para mostrar la ecuación final del modelo polinomial, usa la función `results`. Requiere la siguiente configuración:

- `final_model`: Regresores seleccionados tras el ajuste
- `theta`: Parámetros estimados
- `err`: Error Reduction Ratio (ERR)

```python
from sysidentpy.utils.display_results import results

r = pd.DataFrame(
        results(
                model.final_model, model.theta, model.err,
                model.n_terms, err_precision=8, dtype='sci'
        ),
        columns=['Regressors', 'Parameters', 'ERR'])
print(r)
```

Resultado (ejemplo):

```console
Regressors     Parameters        ERR
0        x1(k-2)     0.9000  0.95556574
1         y(k-1)     0.1999  0.04107943
2  x1(k-1)y(k-1)     0.1000  0.00335113
```
