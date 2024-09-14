There are several NARMAX model representations, including polynomial, Fourier, generalized additive, neural networks, and wavelet ([Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594), [Aguirra, L. A](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas)). This book focuses on the model representations available in SysIdentPy and we’ll keep things updated as new methods are added to the package. If a particular representation is mentioned but is not available in SysIdentPy, it will be explicitly mentioned.

To reproduce the codes presented in this section, make sure you have these packages installed:

```
sysidentpy, scikit-learn, scipy, pytorch, matplotlib
```

## Basis Function

In System Identification, understanding the concept of basis functions is crucial for effectively modeling complex systems. Basis functions are predefined mathematical functions used to transform the input data into a new space, where the relationships within the data can be more easily modeled. By expressing the original data in terms of these basis functions, we can build nonlinear models in respect to it's structure while keeping it linear in the parameters, allowing the usage of straightforward parameter estimation methods.

Basis functions commonly used in system identification:

1. **Polynomial Basis Functions**: These functions are powers of the input variables. They are useful for capturing simple nonlinear relationships.

2. **Fourier Basis Functions**: These sinusoidal functions (sine and cosine) are ideal for representing periodic patterns within the data.

3. **Wavelet Basis Functions**: These functions are localized in both time and frequency, making them suitable for analyzing data with varying frequency components. Not available in SysIdentPy yet.

 In SysIdentPy you can define the basis function you want to use in your model by just import them:

```python
from sysidentpy.basis_function import Polynomial, Fourier, Bernstein
```

To keep things simple for now, we will show simple examples of how basis function can be used in a modeling task. We will show a simple polynomial basis functions, a triangular basis function, a radial basis function and a rectangular basis function.

> SysIdentPy does not currently include Vandermonde or any of the other basis functions defined below. These functions are provided solely as examples to illustrate the significance of the basis functions. The examples are based on Fredrik Bagge Carlson's [PhD thesis](https://arxiv.org/pdf/1906.02003), which I highly recommended for anyone interested in Nonlinear System Identification.

> Although Vandermonde and Radial Basis Functions (RBF) are planned for inclusion as native basis functions in SysIdentPy version 1.0, users can already create and use their own custom basis functions with SysIdentPy. An example of how to do this is available on the SysIdentPy [documentation page](https://sysidentpy.org/).

### Example: Vandermonde Matrix

The polynomial basis functions used in this example is defined as:

$$
\phi_i(x) = x^i
\tag{2.1}
$$

where $i$ is the degree of the polynomial and $x$ is the input variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate simulated quadratic polynomial data
np.random.seed(0)
x = np.linspace(-3, 3, 200)
y = 0.2 * x**2 - 0.3 * x + 0.1 + np.random.normal(0, 0.1, size=x.shape)

# Polynomial basis function
def poly_basis(x, degree):
    return np.vander(x, degree + 1, increasing=True)

# Create polynomial features
degree = 2
X_poly = poly_basis(x, degree)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_poly, y)
y_pred = model.predict(X_poly)
# Plot the original data (quadratic polynomial)
plt.scatter(x, y, color='#ffc865', s=25)
# Plot the polynomial approximation
plt.plot(x, y_pred, color='#00008c', linewidth=5)
# Plot the polynomial basis functions
basis_colors = ["#00b262", "#20007e", "#b20000"]
for i in range(degree + 1):
    plt.plot(x, poly_basis(x, degree)[:, i], linewidth=0.5, color=basis_colors[i % len(basis_colors)])

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks([])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/vandermode_example.png?raw=true)
> Figure 1. Approximation using Vandermode Matrix. The yellow dots show the system data, the bold blue line represents the predicted values, and the other lines depict the basis functions.

### Example: Rectangular Basis Functions

The rectangular basis functions are defined as:

$$
\phi_{i}(x) = \begin{cases}
1 & \text{if } c_i - \frac{w}{2} \leq x < c_i + \frac{w}{2} \\
0 & \text{otherwise}
\end{cases}
\tag{2.2}
$$

where $c_i$ represents the center of the basis function, $w$ is the width, and $x$ is the input variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate simulated quadratic polynomial data
np.random.seed(0)
x = np.linspace(-3, 3, 200)
y = 0.2 * x**2 - 0.3 * x + 0.1 + np.random.normal(0, 0.1, size=x.shape)
# Rectangular basis function
def rectangular_basis(x, centers, width):
    return np.column_stack([(np.abs(x - c) < width).astype(float) for c in centers])

# Create rectangular features
centers = np.linspace(-3, 3, 6)
width = 3
X_rect = rectangular_basis(x, centers, width)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_rect, y)
y_pred = model.predict(X_rect)
# Plot the original data (quadratic polynomial)
plt.scatter(x, y, color='#ffc865', s=25)
# Plot the rectangular approximation
plt.plot(x, y_pred, color='#00008c', linewidth=5)
# Plot the rectangular basis functions
basis_colors = ["#00b262", "#20007e", "#b20000"]
for i in range(len(centers)):
    plt.plot(x, rectangular_basis(x, centers, width)[:, i], linewidth=1, color=basis_colors[i % len(basis_colors)])

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks([])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/rectangular.png?raw=true)
> Figure 2. Approximation using Rectangular Basis Function. The yellow dots show the system data, the bold blue line represents the predicted values, and the other lines depict the basis functions.

### Example: Triangular Basis Functions

The triangular basis functions are defined as:

$$
\phi_{i}(x) = \max \left(0, 1 - \frac{|x - c_i|}{w} \right)
\tag{2.3}
$$

where $c_i$ is the center of the basis function, $w$ is the width, and $x$ is the input variable.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate simulated quadratic polynomial data
np.random.seed(0)
x = np.linspace(-3, 3, 200)
y = 0.2 * x**2 - 0.3 * x + 0.1 + np.random.normal(0, 0.1, size=x.shape)
# Triangular basis function
def triangular_basis(x, centers, width):
    return np.column_stack([np.maximum(0, 1 - np.abs((x - c) / width)) for c in centers])

# Create triangular features
centers = np.linspace(-3, 3, 6)
width = 1.5
X_tri = triangular_basis(x, centers, width)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_tri, y)
y_pred = model.predict(X_tri)
# Plot the original data (quadratic polynomial)
plt.scatter(x, y, color='#ffc865', s=25)
# Plot the triangular approximation
plt.plot(x, y_pred, color='#00008c', linewidth=5)
# Plot the triangular basis functions
basis_colors = ["#00b262", "#20007e", "#b20000"]
for i in range(len(centers)):
    plt.plot(x, triangular_basis(x, centers, width)[:, i], linewidth=1, color=basis_colors[i % len(basis_colors)])

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks([])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/triangular.png?raw=true)
 > Figure 3. Approximation using a Triangular Basis Function. The yellow dots show the system data, the bold blue line represents the predicted values, and the other lines depict the basis functions.

### Example: Radial Basis Function (RBF) - Gaussian

The Gaussian Radial Basis Function is defined as:

$$
\phi(x; c, \sigma) = \exp\left(- \frac{(x - c)^2}{2 \sigma^2}\right)
\tag{2.4}
$$

where:
- $x$ is the input variable.
- $c$ is the center of the RBF.
- $\sigma$ is the spread (or scale) of the RBF.

This function measures the distance between $x$ and the center $c$, and it decays exponentially based on the width $\sigma$. The smaller the $\sigma$, the more localized the basis function is around the center $c$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate simulated quadratic polynomial data
np.random.seed(0)
x = np.linspace(-3, 3, 200)  # More points for a smoother curve
y = 0.2 * x**2 - 0.3 * x + 0.1 + np.random.normal(0, 0.1, size=x.shape)  # Quadratic polynomial with noise
# RBF centers and sigma
centers = np.linspace(-3, 3, 6)  # More centers for better coverage
sigma = 0.5  # Spread of the RBF
# RBF basis function
def rbf_basis(x, c, sigma):
    return np.exp(- (x - c) ** 2 / (2 * sigma ** 2))

# Create RBF features
X_rbf = np.column_stack([rbf_basis(x, c, sigma) for c in centers])
# Fit a linear regression model
model = LinearRegression()
model.fit(X_rbf, y)
y_pred = model.predict(X_rbf)
# Plot the original data (quadratic polynomial)
plt.scatter(x, y, color='#ffc865', s=25)
# Basis function colors
basis_colors = ["#00b262", "#20007e", "#b20000"]
n_colors = len(basis_colors)
# Plot the basis functions
for i, c in enumerate(centers):
    color = basis_colors[i % n_colors]
    plt.plot(x, rbf_basis(x, c, sigma), linewidth=1, color=color, label=f'RBF Center {c:.2f}')

# Plot the approximation
plt.plot(x, y_pred, color='#00008c', linewidth=5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks([])
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/rbf_example.png?raw=true)
> Figure 4. Approximation using the Radial Basis Function. The yellow dots show the system data, the bold blue line represents the predicted values, and the other lines depict the basis functions.

## Linear Models

### ARMAX

You may have noticed the similarity between the acronym NARMAX with the well-known models ARX, ARMAX, etc., which are widely used for forecasting time series. And this resemblance is not by chance. The AutoRegressive models with Moving Average and Exogenous Input (ARMAX) and their variations AR, ARX, ARMA (to name just a few) are one of the most used mathematical representations for identifying linear systems. The ARMAX can be expressed as:

$$
y_k= \mathcal{\phi}[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
\tag{2.5}
$$

where $n_y\in \mathbb{N}$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ , are the maximum lags for the system output, input and noise regressors (representing the moving average part), respectively; $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$ is the system output at discrete time $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ stands for uncertainties and possible noise at discrete time $k$. In this case, $\mathcal{\phi}$ is some linear function of the input and output regressors and $d$ is a time delay typically set to $d=1$.

If $\mathcal{F}$ is a polynomial, we have a polynomial ARMAX model

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.6}
$$

where $\sum\nolimits_{0}$, $\Theta_{y}^{i}$, $\Theta_{e}^{j}$, and $\Theta_{x}^{m}$ are constant parameters.

The following example is a polynomial ARMAX model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139x_{k-1} -0.1691x_{k-1} + 0.2245e_{k-1}
\end{align}
\tag{2.7}
$$


You can easily build a polynomial ARMAX model using **SysIdentPy**:
```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=True)
)
```

In the example above, we define the linear polynomial basis function by importing the Polynomial basis and setting the degree equal to 1 (this ensure that we do not have a nonlinear combination of the regressors). Don't worry about the `FROLS` and `LeastSquares` yet. We'll talk about them in chapters 3 and 4, respectively.

For Figure 4, we conducted 10 separate simulations to analyse the effects of different noise process generation on the ARMAX system's behavior. Each simulation uses a unique sample of noise to observe how variations in this random component influence the overall system output. To illustrate this, we highlight one specific simulation while the others are displayed with less emphasis.

It's important to notice that all simulations, whether highlighted or not, are governed by the same underlying model. The deterministic part of the model equation explains the behavior of all the signals shown. The noticeable differences among the signals arise solely from the distinct noise samples used in each simulation. Despite these variations, the core dynamics of the signal remain consistent and are described by the model's deterministic component.

> Most of the code presented in this chapter is intended to illustrate fundamental concepts rather than demonstrating how to use SysIdentPy specifically. Many examples are implemented using pure Python to help you better understand the underlying concepts, replicate the examples, and adapt them as needed. SysIdentPy itself will be introduced and be used in the examples in the following chapter.


```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

random_samples = 50
n = np.arange(random_samples)
def system_equation(y, u, nu):
    yk = 0.9*y[0] - 0.24*y[1] + 0.92*u[0] + 0.92*nu[0] + nu[1]
    return yk

# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(12, 6))
u = np.random.normal(size=(random_samples,), scale=1)
for k in range(10):
    nu = np.random.normal(size=(random_samples,), scale=0.9)
    y = np.empty_like(nu)
    # Initial Conditions
    y0 = [0.5, -0.1]
    y[0:2] = y0
    for i in range(2, len(y)):
        y[i] = system_equation([y[i - 1], y[i - 2]], [u[i - 1]], [nu[i - 1], nu[i]])

    # Interpolate the data just to make the plot "nicer"
    interpolation_function = interp1d(n, y, kind='quadratic')
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
    y_interpolated = interpolation_function(n_fine)
    # Plotting the interpolated data
    if k == 0:
        ax.plot(n_fine, y_interpolated, color='k', alpha=1, linewidth=1.5)
    else:
        ax.plot(n_fine, y_interpolated, color='grey', linestyle=":", alpha=0.5, linewidth=1.5)

ax.set_xlabel("$n$", fontsize=18)
ax.set_ylabel("$y[n]$", fontsize=18)
ax.set_title("Simulation of an ARMAX model")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/armax_example.png?raw=true)
> Figure 4. Simulations to show the effects of different noise process generation on the ARMAX model's behavior.

### ARX

If we do not include noise terms $e_{k-n_e}$  in equation (1), we have ARX models.

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.8}
$$

The following example is a polynomial ARX model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139x_{k-1} -0.1691x_{k-1}
\end{align}
\tag{2.9}
$$

The only difference in SysIdentPy is setting the `unbiased=False`

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False)
)
```

The following example shows 10 separate simulations to analyse the effects of different noise process generation on the ARX system's behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

random_samples = 50
n = np.arange(random_samples)
def system_equation(y, u, nu):
    yk = 0.9*y[0] - 0.24*y[1] + 0.92*u[0] + nu[0]
    return yk

# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(12, 6))
u = np.random.normal(size=(random_samples,), scale=1)
for k in range(10):
    nu = np.random.normal(size=(random_samples,), scale=0.9)
    y = np.empty_like(nu)
    # Initial Conditions
    y0 = [0.5, -0.1]
    y[0:2] = y0
    for i in range(2, len(y)):
        y[i] = system_equation([y[i - 1], y[i - 2]], [u[i - 1]], [nu[i]])

    # Interpolate the data just to make the plot easier to understand
    interpolation_function = interp1d(n, y, kind='quadratic')
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
    y_interpolated = interpolation_function(n_fine)
    # Plotting the interpolated data
    if k == 0:
        ax.plot(n_fine, y_interpolated, color='k', alpha=1, linewidth=1.5)
    else:
        ax.plot(n_fine, y_interpolated, color='grey', linestyle=":", alpha=0.5, linewidth=1.5)

ax.set_xlabel("$n$", fontsize=18)
ax.set_ylabel("$y[n]$", fontsize=18)
ax.set_title("Simulation of an ARX model")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/arx_example.png?raw=true)
> Figure 5. Simulations to show the effects of different noise process generation on the ARX model's behavior.

### ARMA

if we do not include input terms in equation (1), it turns to ARMA model

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + e_k
\tag{2.10}
$$

The following example is a polynomial ARMA model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139y_{k-3} -0.1691y_{k-4} + 0.2245e_{k-1}
\end{align}
\tag{2.11}
$$

Since the model representation do not have inputs, we have to set the model type to `NAR` and set `unbiased=True` again in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=True),
	model_type="NAR"
)
```

The figure bellow shows 10 separate simulations to analyse the effects of different noise process generation on the ARX system's behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

random_samples = 50
n = np.arange(random_samples)
def system_equation(y, nu):
    yk = 0.5*y[0] - 0.4*y[1] + 0.8*nu[0] + nu[1]
    return yk

# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(12, 6))
for k in range(10):
    nu = np.random.normal(size=(random_samples,), scale=0.9)
    y = np.empty_like(nu)
    # Initial Conditions
    y0 = [0.5, -0.1]
    y[0:2] = y0
    for i in range(2, len(y)):
        y[i] = system_equation([y[i - 1], y[i - 2]], [nu[i - 1], nu[i]])

    # Interpolate the data just to make the plot easier to understand
    interpolation_function = interp1d(n, y, kind='quadratic')
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
    y_interpolated = interpolation_function(n_fine)
    # Plotting the interpolated data
    if k == 0:
        ax.plot(n_fine, y_interpolated, color='k', alpha=1, linewidth=1.5)
    else:
        ax.plot(n_fine, y_interpolated, color='grey', linestyle=":", alpha=0.5, linewidth=1.5)

ax.set_xlabel("$n$", fontsize=18)
ax.set_ylabel("$y[n]$", fontsize=18)
ax.set_title("Simulation of an ARMA model")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/arma_example.png?raw=true)
> Figure 6. Simulations to show the effects of different noise process generation on the ARMA model's behavior.

### AR

if we do not include input terms and noise terms in equation (1), it turns to AR model

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + e_k
\tag{2.12}
$$

The following example is a polynomial AR model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139y_{k-3} -0.1691y_{k-4}
\end{align}
\tag{2.13}
$$

In this case, we have to set the model type to `NAR` and set `unbiased=False` in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False),
	model_type="NAR"
)
```

The figure bellow shows 10 separate simulations to analyse the effects of different noise process generation on the AR system's behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

random_samples = 50
n = np.arange(random_samples)
def system_equation(y, nu):
    yk = 0.5*y[0] - 0.3*y[1] + nu[0]
    return yk

# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(12, 6))
for k in range(10):
    nu = np.random.normal(size=(random_samples,), scale=0.9)
    y = np.empty_like(nu)
    # Initial Conditions
    y0 = [0.5, -0.1]
    y[0:2] = y0
    for i in range(2, len(y)):
        y[i] = system_equation([y[i - 1], y[i - 2]], [nu[i]])

    # Interpolate the data just to make the plot easier to understand
    interpolation_function = interp1d(n, y, kind='quadratic')
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
    y_interpolated = interpolation_function(n_fine)
    # Plotting the interpolated data
    if k == 0:
        ax.plot(n_fine, y_interpolated, color='k', alpha=1, linewidth=1.5)
    else:
        ax.plot(n_fine, y_interpolated, color='grey', linestyle=":", alpha=0.5, linewidth=1.5)

ax.set_xlabel("$n$", fontsize=18)
ax.set_ylabel("$y[n]$", fontsize=18)
ax.set_title("Simulation of an AR model")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/ar_example.png?raw=true)
> Figure 7. Simulations to show the effects of different noise process generation on the AR model's behavior.

### FIR

if we only keep input terms in equation (1), it turns to NFIR model

$$
y_k = \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.14}
$$

The following example is a polynomial FIR model:

$$
\begin{align}
  y_k =& 0.7213x_{k-1}-0.5692x_{k-2}+0.1139x_{k-3} -0.1691x_{k-4}
\end{align}
\tag{2.15}
$$

In this case, we have to set the model type to `NFIR` and set `unbiased=False` in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False),
	model_type="NFIR"
)
```

The figure bellow shows 10 separate simulations to analyse the effects of different noise process generation on the FIR system's behavior.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

random_samples = 50
n = np.arange(random_samples)
def system_equation(u, nu):
    yk = 0.28*u[0] - 0.34*u[1] + nu[0]
    return yk

u = np.random.normal(size=(random_samples,), scale=1)
# Create a single figure and axis for all plots
fig, ax = plt.subplots(figsize=(12, 6))
for k in range(10):
    nu = np.random.normal(size=(random_samples,), scale=0.9)
    y = np.empty_like(nu)
    # Initial Conditions
    y0 = [0.5, -0.1]
    y[0:2] = y0
    for i in range(2, len(y)):
        y[i] = system_equation([0.1*u[i - 1], u[i - 2]], [nu[i]])

    # Interpolate the data just to make the plot easier to understand
    interpolation_function = interp1d(n, y, kind='quadratic')
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
    y_interpolated = interpolation_function(n_fine)
    # Plotting the interpolated data
    if k == 0:
        ax.plot(n_fine, y_interpolated, color='k', alpha=1, linewidth=1.5)
    else:
        ax.plot(n_fine, y_interpolated, color='grey', linestyle=":", alpha=0.5, linewidth=1.5)

ax.set_xlabel("$n$", fontsize=18)
ax.set_ylabel("$y[n]$", fontsize=18)
ax.set_title("Simulation of an FIR model")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/fir_example.png?raw=true)
> Figure 8. Simulations to show the effects of different noise process generation on the **FIR** model's behavior.

> We didn't set the model_type for ARMAX and ARX because the default is `NARMAX`. SysIdentPy allows three different model types: `NARMAX`, `NAR`, and `NFIR`. Because ARMAX, ARX and others linear variants are subsets of NARMAX models, there is no need for specific `ARMAX` model type. The idea is to have model types for model with input and output regressors; models with only output regressors; and models with only input regressors.

### Other Variants

For the sake of simplicity, we defined Equation 2.5 and only approach the polynomial representations. However, you can extend the representations to other basis functions, like the Fourier. If you set $\mathcal{F}$ as the Fourier extension

$$
\mathcal{F}(x) = [\cos(\pi x), \sin(\pi x), \cos(2\pi x), \sin(2\pi x), \ldots, \cos(N\pi x), \sin(N\pi x)]
\tag{2.16}
$$

In this case, the Fourier ARX representation will be:

$$
\begin{align}
y_k = &[ \cos(\pi y_{k-1}), \sin(\pi y_{k-1}), \cos(2\pi y_{k-1}), \sin(2\pi y_{k-1}), \ldots, \cos(N\pi y_{k-1}), \sin(N\pi y_{k-1}),  \\ &  \cos(\pi y_{k-n_y}), \sin(\pi y_{k-n_y}), \cos(2\pi y_{k-n_y}), \sin(2\pi y_{k-n_y}), \ldots, \cos(N\pi y_{k-n_y}), \sin(N\pi y_{k-n_y}), \\ & \cos(\pi x_{k-1}), \sin(\pi x_{k-1}), \cos(2\pi x_{k-1}), \sin(2\pi x_{k-1}), \ldots, \cos(N\pi x_{k-1}), \sin(N\pi x_{k-1}), \\ & \cos(\pi y_{k-n_y}), \sin(\pi y_{k-n_y}), \cos(2\pi y_{k-n_y}), \sin(2\pi y_{k-n_y}), \ldots, \cos(N\pi y_{k-n_y}), \sin(N\pi y_{k-n_y})] \\ & + e_k
\end{align}
\tag{2.17}
$$

To do that in SysIdentPy, just import the Fourier basis instead of the Polynomial

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Fourier
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Fourier(degree=1)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False),
	model_type="NARMAX"
)
```


## Nonlinear Models

### NARMAX

The NARMAX model was proposed by  Stephen A. Billings and I.J. Leontaritis in [1981](https://pdf.sciencedirectassets.com/314898/1-s2.0-S1474667082X74355/1-s2.0-S1474667017630398/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECcaCXVzLWVhc3QtMSJHMEUCIGSxH3DpnP8cywgmDASl%2FBUhnmjjJxeG%2BTay6JhHkUUKAiEAgdzFyYAYcYGDnHxy9T%2BzIVHT%2Fi8J0BVY4ZxI0ovAaUIqsgUIIBAFGgwwNTkwMDM1NDY4NjUiDNTsL0ESuGUnlT%2FyniqPBTUbIgulDeB6wJLJ6sql18H4y3NFfyReA0c45SRi1Rq%2F1TEI%2B0c2cC8v%2FfH8cJF7xGiU%2Fvq3apocGRSFYM6VCJ5IPeyqW3V9Z6lZCAHz3sMY2vX6xHf8czCB4jk85bQAr5Ct9C8m0YI33jvykw7ZC25zOVsyDyR%2Fnas4IOBAboSImvbTzYgDwcwL5S9eSlq%2Byw%2FZPyOMJslF6mOERjXYMrr1Xg51F%2B4xeDt%2BWSGe5qbu5s0RJwifIBp0%2F6WG1m6Mvf0wn%2BQyP8PMJyKMP31fbeA7g3Ndwtjs8ghLPjxax%2BSLby5hgi9ogzm1Gr%2Fv972XLW3TwNUod6yMdzVolsNcxgPBUtns%2Fdfr%2Fm2e%2BkvpETO%2FF1J7V%2FAievzJFE2Sl087x0ZACiy57fK3doa7UWpZj46DaU%2BeAEUbrSXel2PqHbLfSFM9NDsgZWBRXzqygQY4VMGANBEKCR%2BrcJdI3BG%2B%2Bkcg18OKr%2Bf42vgzGRiGB3szABhtRVq5QvGyXYYONv1SFcLBHAt67nuVukKMu%2BR%2BmKQT9CwNOL13oZ5ldS5Fowv1TC8B8o8qdlveFb6WywpTnsSc9%2Bm8Ufbn5W8zI5Fy2sECz4hjUY1zGRhH3O9027j87%2BwyVVfdT6ucHvOdHjQTr2tawgJmUvRunQlw7xN85t2IPEWdFZSXC9IgkG5QRxxzYzsyrCzG4Vsm3CGcShAu7Ms%2Bm7z7XO8QXgfN95JxcG4PX6siAkN5CGZwAdtUkp1VOXzpRt77mrTofG%2BHDGHeXPlyRnBNqR4Z7EJbevPWgUK9BRkVbKfnL0YCtuWbQ3sQ115UTR1a84VejuWmTg%2FJ8Fpf78ApwBE6igc1VfePXVJJwAzxzWcXPSwlhDQNqPIw8ffPtQY6sQHciEnI0Pd0iSUf53%2BhgScY6deC54cJfBcOtzm9RhrSeSoQyPdhrMHU0r2fWAhYRcr9ZeuObdxPtjpqardoQM%2BEizPzQNO8uBtOoYTK2inAXxp2z3U%2BDYHWJHNuapOhC2GsOPlPXco7e0Hm0GxiPUyk6%2B8FtqTAs6d4KxpT7CGLUNChK%2BCUbt%2BsnF84TjzK40Fkf0Gya9ESFUHmIEUmfXabOKfgra2QbSbcogWgoI5gChM%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240808T001847Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY343F44W4%2F20240808%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=95d8e3c94d76309850682e9e8f9025436a481f440bf152241e01eb47d70c4438&hash=fb1356fe353cd87c7fb931d951e83c97cce7568257b5a9c6c533aee21ef64f35&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1474667017630398&tid=spdf-a9063d37-b51a-42e8-97b3-bb5ab63b0384&sid=07d7477c73bfe241706ba2d-0fba8a3cbcd1gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=18165805090051570458&rr=8afb54c71f7e1ac0&cc=br) ([Billins, S. A.]((https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594))), and can be described as

$$
\begin{equation}
y_k= \mathcal{F}[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k,
\end{equation}
\tag{2.18}
$$

where $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ , are the maximum lags for the system output and input respectively; $x_k \in \mathbb{R}^{n_x}$ is the system input and $y_k \in \mathbb{R}^{n_y}$ is the system output at discrete time $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ represents uncertainties and possible noise at discrete time $k$. In this case, $\mathcal{F}$ is some nonlinear function of the input and output regressors and $d$ is a time delay typically set to $d=1$.

You can notice that the difference between Equation 2.5 and Equation 2.18 if the function representing the system. For NARMAX models, $\mathcal{F}$ can be any nonlinear function, while for Equation 2.5 only linear functions are allowed. Although there are many possible approximations of $\mathcal{F}(\cdot)$ (e.g., Neural Networks, Fuzzy, Wavelet, Radial Basis Function), the power-form Polynomial NARMAX model is the most commonly used ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594); [Khandelwal, D. and Schoukens, M. and Toth, R.](https://arxiv.org/abs/2001.05320)):

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}\prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.19}
$$

where $p$ is the number of regressors, $\Theta_i$ are the model parameters, and $a_i, m$, $b_i, j$ and $d_i, l \in \mathbb{N}$ are the exponents of the output, input and noise terms, respectively.

The Equation 2.20 describes a polynomial NARMAX model where the nonlinearity degree is equal to $2$, identified from experimental data of a DC motor/generator with no prior knowledge of the model form, taken from [Lacerda Junior, W. R, Almeida, V. M., Martins, S. A. M.]([LAM2017.pdf (ufsj.edu.br)](https://www.ufsj.edu.br/portal2-repositorio/File/gcom/LAM2017.pdf)):

$$
\begin{align}
  y_k =& 1.7813y_{k-1}-0.7962y_{k-2}+0.0339x_{k-1} -0.1597x_{k-1} y_{k-1} +0.0338x_{k-2} + \\
  & + 0.1297x_{k-1}y_{k-2} - 0.1396x_{k-2}y_{k-1}+ 0.1086x_{k-2}y_{k-2}+0.0085y_{k-2}^2 + 0.0247e_{k-1}e_{k-2}
\end{align}
\tag{2.20}
$$

The $\Theta$ values are the coefficients of each term of the polynomial equation.

Polynomial basis functions are one of the most used representations of NARMAX models due to several interesting attributes, such as ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)):

- All polynomial functions are smooth in $\mathbb{R}$.
- The Weierstrass [approximation theorem](https://www.researchgate.net/profile/P-Johnson/publication/353192512_Stone-Weierstrass_Theorem/links/60ec4cdc9541032c6d32c4a6/Stone-Weierstrass-Theorem.pdf) states that any continuous real-valued function defined on a closed and bounded space $[a,b]$ can be uniformly approximated using a polynomial on that interval.
- They can describe several nonlinear dynamical systems, including industrial processes, control systems, structural systems, economic and financial systems, biology, medicine, and social systems (some examples are detailed in [Lacerda Junior, W. R. and Martins, S. A. M. and Nepomuceno, E. G. and Lacerda, Marcio J.](https://ieeexplore.ieee.org/document/8751951) ; [Fung, E. H. K. and Wong, Y. K. and Ho, H. F. and Mignolet, M. P.](https://www.sciencedirect.com/science/article/pii/S0307904X03000714); [Kukreja, S. L. and Galiana, H. L. and Kearney, R. E.](https://ieeexplore.ieee.org/document/1179133); [Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594); [Aguirre, L. A.](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas?channel=doi&linkId=574cb16508ae82d2c6bc870f&showFulltext=true); and many others).
- Several algorithms have been developed for structure selection and parameter estimation of polynomial NARMAX models, and it remains an active area of research.
- Polynomial NARMAX models are versatile and can be used both for prediction and inference. The structure of polynomial NARMAX models are easy to interpret and can be related to the underlying system, which is much harder to achieve with neural networks or wavelet functions, for instance.

You can easily build a polynomial NARMAX model using SysIdentPy. Note that the difference for ARMAX, in this case, is the degree of the polynomial function.

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=True)
)
```

One could think that is a simple change, but in nonlinear scenarios the course of dimensionality becomes a real problem. The number of candidate regressors, $n_r$, of polynomial NARX can be defined as ([Korenberg, M. L. and Billings, S. A. and Liu, Y. P. and McIlroy, P. J.]([Orthogonal parameter estimation algorithm for non-linear stochastic systems](https://www.tandfonline.com/doi/abs/10.1080/00207178808906169))):

$$
\begin{equation}
    n_r = M+1,
\end{equation}
\tag{2.21}
$$

where

$$
\begin{align}
    M = & \sum_{i=1}^{\ell}n_i \\
    n_i = & \frac{n_{i-1}(n_y+n_x+i-1)}{i}, n_{0} = 1.
\end{align}
\tag{2.22}
$$

As we mentioned in the Introduction of the book, NARMAX methods aims to build the simplest models possible. The idea is to be reproduce a wide range of behaviors using a small subset of terms from the vast search space formed by candidate regressors.

Lets use SysIdentPy to see how the search space grows in the linear versus the nonlinear scenario. The `regressor_code` method available in `narmax_tools` can be used the check how many regressors exists in the search space given the number of inputs, the delays of `y` and `x` regressors and the basis function. We will use `xlag=ylag=10` and the polynomial basis function. The user can simulate different scenarios by setting different parameters.

```python
from sysidentpy.utils.narmax_tools import regressor_code
from sysidentpy.basis_function._basis_function import Polynomial]
import numpy as np
```

For the linear case with 1 input we have 21 regressors:
```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
basis_function = Polynomial(degree=1)
regressors = regressor_code(
    X=x_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 21
```

For the linear case with 2 inputs, the number of regressors jumps to 111:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 inputs
basis_function = Polynomial(degree=1)
xlag = [list(range(1, 11))] * x_train.shape[1]
regressors = regressor_code(
    X=x_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 111
```

If we consider a nonlinear case with 1 input by just changing the degree to 2, we have 231 regressors.

```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
basis_function = Polynomial(degree=2)
regressors = regressor_code(
    X=x_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 231
```

If we set the degree to 3, the number of terms increases significantly to 1771 regressors.

```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
basis_function = Polynomial(degree=2)
regressors = regressor_code(
    X=x_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 1771
```

If you have 2 inputs in the nonlinear scenario with `degree=2`, the number of regressors is 496:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 input
basis_function = Polynomial(degree=2)
xlag = [list(range(1, 11))] * x_train.shape[1]
regressors = regressor_code(
    X=x_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 496
```

If you have 2 inputs in the nonlinear scenario with `degree=3`, the number jumps to 5456 regressors:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 input
basis_function = Polynomial(degree=3)
xlag = [list(range(1, 11))] * x_train.shape[1]
regressors = regressor_code(
    X=x_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
n_regressors = regressors.shape[0]  # the number of features of the NARX net
n_regressors
>>> 5456
```

As you can notice, the number of regressors increases significantly as the degree of the polynomial and the number of inputs increases. That makes the model structure selection much more complex! In the linear case with 10 inputs we have `2^31=2.15e+09` possible model combinations. When `degree=2` with 2 inputs we have `2^496=2.05e+149` possible combinations! Try to get the number of possible model combinations when `degree=3` with 2 inputs. Moreover, try that with more inputs and higher nonlinear degree and see how the course of dimensionality is a big problem.

As you can see, getting a simple model in such a large search space is complex model structure selection task. To select the most significant terms from a huge dictionary of possible terms is not an easy task. And it is hard not only because the complex combinatoric problem and the uncertainty concerning the model order. Identifying the most significant terms in a nonlinear scenario is very difficult because depends on the type of the nonlinearity (sparse singularity or near-singular behavior, memory or dumping effects and many others), dynamical response (spatial-temporal systems, time-dependent), the steady-state response,  frequency of the data, the noise and many more.

Because of the model structure selection algorithms developed for NARMAX models, even linear models like ARMAX can have different performance when obtained using SysIdentPy when compared to other libraries, like Statsmodels. We have a case study showing exactly that in Chapter 10.

### NARX

If we do not include noise terms $e_{k-n_e}$  in Equation (2.19), we have NARX models.

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.23}
$$

The Equation 2.24 describes a simple polynomial NARX model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^2+0.1139y_{k-1}x_{k-1}
\end{align}
\tag{2.24}
$$

The only difference in SysIdentPy is setting the `unbiased=False`

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False)
)
```

> The user can use the codes provided for linear models to analyse the nonlinear models with different noise realizations.

### NARMA

if we do not include input terms in Equation 2.19, it turns to NARMA model

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times\prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.25}
$$

The following example is a polynomial NARMA model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^3+0.1139y_{k-3}y_{k-4} + 0.2245e_{k-1}
\end{align}
\tag{2.26}
$$

Since the model representation do not have inputs, we have to set the model type to `NAR` and set `unbiased=True` again in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=True),
	model_type="NAR"
)
```

### NAR

if we do not include input terms and noise terms in Equation 2.19, it turns to AR model

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.27}
$$

The following example is a polynomial NAR model:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^2+0.1139y_{k-3}^3 -0.1691y_{k-4}y_{k-5}
\end{align}
\tag{2.28}
$$

In this case, we have to set the model type to `NAR` and set `unbiased=False` in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False),
	model_type="NAR"
)
```

### NFIR

If we only keep input terms in Equation 2.19, it becomes a NFIR model

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}
\end{align}
\tag{2.29}
$$

The following example is a polynomial NFIR model:

$$
\begin{align}
  y_k =& 0.7213x_{k-1}-0.5692x_{k-2}^2+0.1139x_{k-3}x_{k-4} -0.1691x_{k-4}^3
\end{align}
\tag{2.30}
$$

In this case, we have to set the model type to `NFIR` and set `unbiased=False` in `LeastSquares`:

```python
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
	basis_function=basis_function,
	estimator=LeastSquares(unbiased=False),
	model_type="NFIR"
)
```

### Mixed NARMAX Models

In some applications, using a single basis functions cannot provide a satisfactory description for the relationship between the input (or independent) variables and the output (or response) variable. In order to improve the performance of the model, it has been proposed to use a linear combination of a set of nonlinear functions to replace the linear counterparts.

You can achieve that in SysIdentPy using ensembles in basis functions. You can build a Fourier model where terms have interactions. You can also build a model with mixed basis functions, using terms expanded by polynomial basis and Fourier basis or any other basis function available is the package.

> You can only mix a basis function with the polynomial basis for now in SysIdentPy. You can mix Fourier with Polynomial, but you can't mix Fourier with Bernstein.

To mix Fourier or Bernstein basis with Polynomial, the user just have to set `ensamble=True`
in the basis function definition

```python
from sysidentpy.basis_function import Fourier

basis_function = Fourier(degree=2, ensamble=True)
```

### Neural NARX Network

Neural networks are models composed of interconnected layers of nodes (neurons) designed for tasks like classification and regression. Each neuron is a basic unit within these networks. [Mathematically](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas?channel=doi&linkId=574cb16508ae82d2c6bc870f&showFulltext=true), a neuron is represented by a function $f$ that takes an input vector $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ and generates an output $y$. This function usually involves a weighted sum of the inputs, an optional bias term $b$, and an activation function $\phi$:

$$
y = \phi \left( \sum_{i=1}^{n} w_i x_i + b \right)
\tag{2.31}
$$

where $\mathbf{w} = [w_1, w_2, \ldots, w_n]$ are the weights associated with the inputs. The activation function $\phi$ introduces nonlinearity into the model, allowing the network to learn complex patterns. Common activation functions include:

- **Sigmoid**: $\phi(z) = \frac{1}{1 + e^{-z}}$
  Produces outputs between 0 and 1, making it useful for binary classification.

- **Hyperbolic Tangent (tanh)**: $\phi(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
  Outputs values between -1 and 1, often used to center data around zero.

- **Rectified Linear Unit (ReLU)**: $\phi(z) = \max(0, z)$
  Outputs zero for negative values and the input value itself for positive values, helping to mitigate the vanishing gradient problem.

- **Leaky ReLU**: $\phi(z) = \max(0.01z, z)$
  A variant of ReLU that allows a small, non-zero gradient when the input is negative, addressing the problem of dying neurons.

- **Softmax**: $\phi(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$
  Converts logits into probabilities for multi-class classification, ensuring that the outputs sum to 1.

Each activation function has its own advantages and is chosen based on the specific needs of the neural network and the task at hand.

As mentioned, neural network is composed of multiple layers, each consisting of several neurons. In this respect, the layers can be categorized into:

   - **Input Layer**: The layer that receives the input data.
   - **Hidden Layers**: Intermediate layers that process the inputs through weighted connections and activation functions.
   - **Output Layer**: The final layer that produces the output of the network.

> The network itself therefore has a very simple architecture. The terminology used in neural networks is also slightly different from the standard notation that is universal in system identification and statistics. So, instead of talking about model parameters, the term *network weights* is used, and instead of estimation, the term *learning* is used. This terminology was no doubt introduced to make it appear that something completely new was being discussed, whereas some of the problems addressed are quite traditional - [Stephen A. Billings](https://www.amazon.com.br/s/ref=dp_byline_sr_book_1?ie=UTF8&field-author=Stephen+A+Billings&text=Stephen+A+Billings&sort=relevancerank&search-alias=stripbooks)

Notice that the network itself is simply a collection of nonlinear activation units $\phi(\cdot)$  that are simple static functions. There are no dynamics within the network. This is fine for applications such as pattern recognition, but to use the network in system identification lagged inputs and outputs are necessary and these have to be supplied as inputs either explicitly or through a recurrent procedure. In this respect, if we set $\mathcal{F}$ as a neural function, we can adapt it to create a neural NARX model by transforming the neural architecture into a NARX architecture. The neural NARX, however, is not linear in the parameters like the NARMAX models based on basis functions. So, algorithms like Orthogonal Least Squares are not adequate to estimate the weights of the model.

**SysIdentPy** support a Series-Parallel (open-loop) Feedforward Network training process, which make the training process easier. We convert the NARX network from Series-Parallel to the Parallel (closed-loop) configuration for prediction.

Series-Parallel allows us to use `pytorch` directly for training, so **SysIdentPy** uses `pytorch` in the backend for neural NARX along with auxiliary methods available only in **SysIdentPy**.

A simple neural NARX model can be represented as a Multi-Layer Perceptron neural network with autoregressive component along with delayed inputs.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/Pasted%20image%2020240710121246.png?raw=true)
> Figure 9. Parallel and series-parallel neural network architectures for modeling the dynamic system $\mathbf{y}[k]=\mathbf{F}(\mathbf{y}[k-1], \mathbf{y}[k-2], \mathbf{u}[k-1], \mathbf{u}[k-2])$. The delay operator $q^{-1}$ is such that $\mathbf{y}[k-1]=q^{-1} \mathbf{y}[k]$. Reference: [Antonio H. Ribeiro and Luis A. Aguirre](https://arxiv.org/pdf/1706.07119)

> Neural NARX is not the same model as Recurrent Neural Networks (RNN). The user is referred to the following paper for more details [A Note on the Equivalence of NARX and RNN](https://link.springer.com/article/10.1007/s005210050005)

To build a Neural NARX network in SysIdentPy, the user must use `pytorch`. We use `pytorch` to make the definition of the network architecture flexible. However, this require that the user have a better understanding of how a neural networks. See the script bellow of how to build a simple Neural NARX model in **SysIdentPy**

```python
from torch import nn
import torch

from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.narmax_tools import regressor_code

# simulated data
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=False, sigma=0.01, train_percentage=80
)
```

The user can use `cuda` following the same approach when build a neural network in **pytorch**

```python
torch.cuda.is_available()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

The user can create a NARXNN object and choose the maximum lag of both input and output for building the regressor matrix to serve as input of the network. In addition, you can choose the loss function, the optimizer, the optional parameters of the optimizer, the number of epochs.

Because we built this feature on top of Pytorch, you can choose any of the loss function of the torch.nn.functional. [Click here](https://pytorch.org/docs/stable/nn.functional.html#loss-functions) for a list of the loss functions you can use. You just need to pass the name of the loss function you want.

Similarly, you can choose any of the optimizer of the torch.optim. [Click here](https://pytorch.org/docs/stable/optim.html) for a list of optimizer available.

```python
basis_function = Polynomial(degree=1)
narx_net = NARXNN(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
    model_type="NARMAX",
    loss_func="mse_loss",
    optimizer="Adam",
    epochs=2000,
    verbose=False,
    device=device,
    optim_params={
        "betas": (0.9, 0.999),
        "eps": 1e-05,
    },  # optional parameters of the optimizer
)
```

Because the NARXNN model were defined using $ylag=2$, $xlag=2$ and a polynomial basis function with $degree=1$, we have a regressor matrix with 4 features. We need the size of the regressor matrix to build the layers of our network. Our input data(`x_train`) have only one feature, but since we are creating an NARX network, a regressor matrix is built behind the scenes with new features based on the `xlag` and `ylag`.

If you need help finding how many regressors are created behind the scenes you can use the `narmax_tools` function `regressor_code` and take the size of the regressor code generated:

```python
basis_function = Polynomial(degree=1)
regressors = regressor_code(
    X=x_train, # t
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation="neural_network",
    basis_function=basis_function,
)

n_features = regressors.shape[0]  # the number of features of the NARX net
n_features
>>> 4
```

The configuration of your network follows exactly the same pattern of a network defined in Pytorch. The following representing our NARX neural network.

```python
class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_features, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()


    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z
```

The user have to pass the defined network to our NARXNN estimator and set `cuda` if available (or needed):

```python
narx_net.net = NARX()

if device == "cuda":
    narx_net.net.to(torch.device("cuda"))
```

Because we have a fit (for training) and predict function for Polynomial NARMAX, we create the same pattern for the NARX net. So, you only have to fit and predict using the following:

```python
narx_net.fit(X=x_train, y=y_train, X_test=x_valid, y_test=y_valid)
yhat = narx_net.predict(X=x_valid, y=y_valid)
```

If the net configuration is built before calling the NARXNN, just pass the model to the NARXNN as follows:

```python
class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_features, 30)
        self.lin2 = nn.Linear(30, 30)
        self.lin3 = nn.Linear(30, 1)
        self.tanh = nn.Tanh()


    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z


narx_net2 = NARXNN(
    net=NARX(),
    ylag=2,
    xlag=2,
    basis_function=basis_function,
    model_type="NARMAX",
    loss_func="mse_loss",
    optimizer="Adam",
    epochs=2000,
    verbose=False,
    optim_params={
        "betas": (0.9, 0.999),
        "eps": 1e-05,
    },  # optional parameters of the optimizer
)

narx_net2.fit(X=x_train, y=y_train)
yhat = narx_net2.predict(X=x_valid, y=y_valid)
```

### General Model Set Representation

Based on the ideia of transforming a static neural network in a neural NARX model, we can extend the method for basically any model class. SysIdentPy do not aim to implement every model class that exists in literature. However, we created a functionality that allows the usage of any other machine learning package that follows a `fit` and `predict` API inside SysIdentPy to convert such models to NARX versions of them.

Lets take XGboost (eXtreme Gradient Boosting) Algorithm as an example. XGBoost is a well kown model class used for regression tasks. XGboost, however, are not a common choice when you are dealing with a dynamical system identification task because they are originally made for modeling static systems. You can easily transform XGboost into a NARX model using SysIdentPy.

Scikit-learn, for example, is another great example. You can transform any Scikit-learn model into NARX models using SysIdentPy. We will see such applications in detail at Chapter 11, but you can see how easy it in the script bellow

``` python
from sysidentpy.general_estimators import NARX
from sysidentpy.basis_function._basis_function import Polynomial
from sklearn.linear_model import BayesianRidge
import xgboost as xgb

basis_function = Fourier(degree=1)
# define the scikit estimator
scikit_estimator = BayesianRidge()
# transform scikit_estimator into NARX model
gb_narx = NARX(
    base_estimator=scikit_estimator,
    xlag=2,
    ylag=2,
    basis_function=basis_function,
    model_type="NARMAX",
)

gb_narx.fit(X=x_train, y=y_train)
yhat = gb_narx.predict(X=x_valid, y=y_valid)

# XGboost examples
xgb_estimator = xgb.XGBRegressor()
xgb_narx = NARX(
    base_estimator=xgb_estimator,
    xlag=2,
    ylag=2,
    basis_function=basis_function,
    model_type="NARMAX",
)

xgb_narx.fit(X=x_train, y=y_train)
yhat = xgb_narx.predict(X=x_valid, y=y_valid)
```

You can use any other model by just changing the model class and passing it to the `base_estimator` in `NARX` functionality.

### MIMO Models

To keep things simple, only SISO models were represented in previous sections. However,  the NARMAX  models can effortlessly be extended to MIMO case ([Billings, S. A. and Chen, S. and Korenberg, M. J.](https://www.tandfonline.com/doi/abs/10.1080/00207178908559767)):

$$
\begin{align}
	 y_{{_i}k}=& F_{{_i}}^\ell \bigl[y_{{_1}k-1},  \dotsc, y_{{_1}k-n^i_{y{_1}}},\dotsc, y_{{_s}k-1},  \dotsc, y_{{_s}k-n^i_{y{_s}}}, x_{{_1}k-d}, \\
	 & x_{{_1}k-d-1}, \dotsc, x_{{_1}k-d-n^i_{x{_1}}}, \dotsc, x_{{_r}k-d}, x_{{_r}k-d-1}, \dotsc, x_{{_r}k-d-n^i_{x{_r}}}\bigr] + \xi_{{_i}k},
\end{align}
\tag{2.32}
$$

where for $i = 1, \dotsc, s$, each linear in the parameter sub-model can change regarding different maximum lags. More generally, considering

$$
\begin{align}
    Y_k = \begin{bmatrix}
    y_{{_1}k} \\
    y_{{_2}k} \\
    \vdots \\
    y_{{_s}k}
    \end{bmatrix},
    X_k = \begin{bmatrix}
    x_{{_1}k} \\
    x_{{_2}k} \\
    \vdots \\
    x_{{_r}k}
    \end{bmatrix},
    \Xi_k = \begin{bmatrix}
    \xi_{{_1}k} \\
    \xi_{{_2}k} \\
    \vdots \\
    \xi_{{_r}k}
    \end{bmatrix},
\end{align}
\tag{2.33}
$$

the MIMO model can be denoted as

$$
\begin{equation}
             Y_k= F^\ell[Y_{k-1},  \dotsc, Y_{k-n_y},X_{k-d}, X_{k-d-1}, \dotsc, X_{k-d-n_x}] + \Xi_k,
\end{equation}
\tag{2.34}
$$

where $Xk ~= \{x_{{_1}k}, x_{{_2}k}, \dotsc, x_{{_r}k}\}\in \mathbb{R}^{n^i_{x{_r}}}$ and $Yk~= \{y_{{_1}k}, y_{{_2}k}, \dotsc, y_{{_s}k}\}\in \mathbb{R}^{n^i_{y{_s}}}$. The number of possibles terms of MIMO NARX model given the $i$-th polynomial degree, $\ell_i$, is:

$$
\begin{equation}
    n_{{_{m}}r} = \sum_{j = 0}^{\ell_i}n_{ij},
\end{equation}
\tag{2.35}
$$

where

$$
\begin{align}
    n_{ij} = \frac{ n_{ij-1} \biggl[ \sum\limits_{k=1}^{s} n^i_{y_k} + \sum\limits_{k=1}^{r} n^i_{x_k} + j - 1 \biggr]}{j}, \qquad n_{i0}=1, j=1, \dotsc, \ell_i.
\end{align}
\tag{2.36}
$$

If $s=1$, we have a MISO model that can be represented by a single polynomial function. Additionally, a MIMO model can be decomposed into MISO models, as presented in the following figure:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/mimo_split.png?raw=true)
> Figure 10. A MIMO model split into individual MISO models.

> SysIdentPy do not support MIMO models yet, only MISO models. You can, however, decompose a MIMO system as presented in Figure 9 and use SysIdentPy to create models for each subsystem.
