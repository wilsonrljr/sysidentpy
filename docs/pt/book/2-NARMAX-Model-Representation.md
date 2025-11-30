Existem diversas representações de modelos NARMAX, incluindo polinomial, Fourier, aditivo generalizado, redes neurais e wavelet ([Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594), [Aguirra, L. A](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas)). Este livro foca nas representações de modelos disponíveis no SysIdentPy, e manteremos o conteúdo atualizado à medida que novos métodos forem adicionados ao pacote. Se alguma representação específica for mencionada, mas ainda não estiver disponível no SysIdentPy, isso será explicitamente indicado.

Para reproduzir os códigos apresentados nesta seção, certifique-se de ter os seguintes pacotes instalados:

```
sysidentpy, scikit-learn, scipy, pytorch, matplotlib
```

## Funções Base (Basis Function)

Em Identificação de Sistemas, entender o conceito de funções base é fundamental para modelar de forma eficaz sistemas complexos. Funções base são funções matemáticas pré-definidas usadas para transformar os dados de entrada em um novo espaço, no qual as relações nos dados podem ser mais facilmente modeladas. Ao expressar os dados originais em termos dessas funções base, podemos construir modelos não lineares em relação à estrutura, mantendo-os lineares nos parâmetros, o que permite o uso de métodos diretos de estimação de parâmetros.

Funções base comumente usadas em Identificação de Sistemas:

1. **Funções Base Polinomiais**: São potências das variáveis de entrada. São úteis para capturar relações não lineares simples.

2. **Funções Base de Fourier**: São funções senoidais (seno e cosseno), ideais para representar padrões periódicos nos dados.

3. **Funções Base Wavelet**: São funções localizadas no tempo e na frequência, adequadas para analisar dados com componentes de frequência variáveis. Ainda não estão disponíveis no SysIdentPy.

No SysIdentPy você pode definir a função base que deseja usar no seu modelo simplesmente importando-as:

```python
from sysidentpy.basis_function import Polynomial, Fourier, Bernstein
```

Para manter as coisas simples por enquanto, vamos mostrar exemplos simples de como funções base podem ser usadas em uma tarefa de modelagem. Apresentaremos uma função base polinomial simples, uma função base triangular, uma função base radial e uma função base retangular.

> O SysIdentPy atualmente não inclui Vandermonde nem nenhuma das outras funções base definidas abaixo. Essas funções são fornecidas apenas como exemplos para ilustrar a importância das funções base. Os exemplos são baseados na [tese de doutorado de Fredrik Bagge Carlson](https://arxiv.org/pdf/1906.02003), que é altamente recomendada para quem tem interesse em Identificação de Sistemas Não Lineares.

> Embora Vandermonde e Radial Basis Functions (RBF) estejam planejadas para serem incluídas como funções base nativas no SysIdentPy versão 1.0, os usuários já podem criar e usar suas próprias funções base customizadas com o SysIdentPy. Um exemplo de como fazer isso está disponível na [página de documentação do SysIdentPy](https://sysidentpy.org/).

### Exemplo: Matriz de Vandermonde

As funções base polinomiais usadas neste exemplo são definidas como:

$$
\phi_i(x) = x^i
\tag{2.1}
$$

em que $i$ é o grau do polinômio e $x$ é a variável de entrada.

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
> Figura 1. Aproximação usando Matriz de Vandermonde. Os pontos amarelos representam os dados do sistema, a linha azul em negrito representa os valores preditos e as demais linhas representam as funções base.

### Exemplo: Funções Base Retangulares

As funções base retangulares são definidas como:

$$
\phi_{i}(x) = \begin{cases}
1 & \text{se } c_i - \frac{w}{2} \leq x < c_i + \frac{w}{2} \\
0 & \text{caso contrário}
\end{cases}
\tag{2.2}
$$

em que $c_i$ representa o centro da função base, $w$ é a largura e $x$ é a variável de entrada.

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
> Figura 2. Aproximação usando Função Base Retangular. Os pontos amarelos representam os dados do sistema, a linha azul em negrito representa os valores preditos e as demais linhas representam as funções base.

### Exemplo: Funções Base Triangulares

As funções base triangulares são definidas como:

$$
\phi_{i}(x) = \max \left(0, 1 - \frac{|x - c_i|}{w} \right)
\tag{2.3}
$$

em que $c_i$ é o centro da função base, $w$ é a largura e $x$ é a variável de entrada.

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
> Figura 3. Aproximação usando Função Base Triangular. Os pontos amarelos representam os dados do sistema, a linha azul em negrito representa os valores preditos e as demais linhas representam as funções base.

### Exemplo: Radial Basis Function (RBF) - Gaussiana

A Radial Basis Function Gaussiana é definida como:

$$
\phi(x; c, \sigma) = \exp\left(- \frac{(x - c)^2}{2 \sigma^2}\right)
\tag{2.4}
$$

em que:
- $x$ é a variável de entrada;
- $c$ é o centro da RBF;
- $\sigma$ é a largura (ou escala) da RBF.

Essa função mede a distância entre $x$ e o centro $c$, decaindo exponencialmente com base na largura $\sigma$. Quanto menor o $\sigma$, mais localizada é a função base ao redor do centro $c$.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate simulated quadratic polynomial data
np.random.seed(0)
x = np.linspace(-3, 3, 200)  # More points for a smoother curve
y = 0.2 * x**2 - 0.3 * x + 0.1 + np.random.normal(0, 0.1, size=x.shape)  # Quadratic polynomial with noise
# RBF centers and sigma
centers = np.linspace(-3, 3, 6)  # More centers for better coverage
sigma = 0.5  # Spread of the RBF
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
> Figura 4. Aproximação usando Radial Basis Function. Os pontos amarelos representam os dados do sistema, a linha azul em negrito representa os valores preditos e as demais linhas representam as funções base.

## Modelos Lineares

### ARMAX

Você provavelmente já notou a semelhança entre o acrônimo NARMAX e os modelos bem conhecidos ARX, ARMAX etc., amplamente usados para previsão de séries temporais. E essa semelhança não é por acaso. Os modelos AutoRegressivos com Média Móvel e Entrada Exógena (ARMAX) e suas variações AR, ARX, ARMA (para citar apenas algumas) estão entre as representações matemáticas mais utilizadas para identificação de sistemas lineares. O modelo ARMAX pode ser expresso como:

$$
y_k= \mathcal{\phi}[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
\tag{2.5}
$$

em que $n_y\in \mathbb{N}$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ são os máximos atrasos para os regressors de saída, entrada e ruído do sistema (representando a parte de média móvel), respectivamente; $x_k \in \mathbb{R}^{n_x}$ é a entrada do sistema e $y_k \in \mathbb{R}^{n_y}$ é a saída do sistema no instante discreto $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ representa incertezas e possíveis ruídos no instante discreto $k$. Neste caso, $\mathcal{\phi}$ é alguma função linear dos regressors de entrada e saída e $d$ é um atraso de tempo, tipicamente definido como $d=1$.

Se $\mathcal{F}$ é um polinômio, obtemos um modelo ARMAX polinomial:

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.6}
$$

em que $\sum\nolimits_{0}$, $\Theta_{y}^{i}$, $\Theta_{e}^{j}$ e $\Theta_{x}^{m}$ são parâmetros constantes.

O exemplo a seguir é um modelo ARMAX polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139x_{k-1} -0.1691x_{k-1} + 0.2245e_{k-1}
\end{align}
\tag{2.7}
$$

Você pode construir facilmente um modelo ARMAX polinomial usando o **SysIdentPy**:
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

No exemplo acima, definimos a função base polinomial linear importando a função base `Polynomial` e definindo o grau igual a 1 (isso garante que não tenhamos combinações não lineares dos regressors). Não se preocupe ainda com `FROLS` e `LeastSquares`. Falaremos sobre eles nos capítulos 3 e 4, respectivamente.

Para a Figura 4, realizamos 10 simulações independentes para analisar os efeitos de diferentes realizações de processo de ruído no comportamento do sistema ARMAX. Cada simulação usa uma amostra distinta de ruído para observar como variações nesse componente aleatório influenciam a saída do sistema. Para ilustrar isso, destacamos uma simulação específica enquanto as demais são exibidas com menor destaque.

É importante notar que todas as simulações, destacadas ou não, são governadas pelo mesmo modelo subjacente. A parte determinística da equação do modelo explica o comportamento de todos os sinais exibidos. As diferenças observadas entre os sinais surgem apenas das diferentes amostras de ruído usadas em cada simulação. Apesar dessas variações, a dinâmica central do sinal permanece consistente e é descrita pelo componente determinístico do modelo.

> A maior parte do código apresentado neste capítulo tem o objetivo de ilustrar conceitos fundamentais, e não de mostrar especificamente como utilizar o SysIdentPy. Muitos exemplos são implementados usando Python "puro" para ajudar você a compreender melhor os conceitos subjacentes, reproduzir os exemplos e adaptá-los conforme necessário. O SysIdentPy em si será introduzido e utilizado nos exemplos a partir do próximo capítulo.

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
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
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
> Figura 5. Simulações para mostrar os efeitos de diferentes realizações de processo de ruído no comportamento do modelo ARMAX.

### ARX

Se não incluirmos os termos de ruído $e_{k-n_e}$ na Equação (2.5), obtemos modelos ARX.

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.8}
$$

O exemplo a seguir é um modelo ARX polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139x_{k-1} -0.1691x_{k-1}
\end{align}
\tag{2.9}
$$

A única diferença no SysIdentPy é definir `unbiased=False`:

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

O exemplo a seguir mostra 10 simulações independentes para analisar os efeitos de diferentes realizações de processo de ruído no comportamento do sistema ARX.

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
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
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
> Figura 6. Simulações para mostrar os efeitos de diferentes realizações de processo de ruído no comportamento do modelo ARX.

### ARMA

Se não incluirmos termos de entrada na Equação (2.5), obtemos o modelo ARMA:

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + e_k
\tag{2.10}
$$

O exemplo a seguir é um modelo ARMA polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139y_{k-3} -0.1691y_{k-4} + 0.2245e_{k-1}
\end{align}
\tag{2.11}
$$

Como a representação do modelo não possui entradas, precisamos definir o tipo de modelo como `NAR` e `unbiased=True` novamente em `LeastSquares`:

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

A figura abaixo mostra 10 simulações independentes para analisar os efeitos de diferentes realizações de processo de ruído no comportamento do sistema ARMA.

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
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
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
> Figura 7. Simulações para mostrar os efeitos de diferentes realizações de processo de ruído no comportamento do modelo ARMA.

### AR

Se não incluirmos termos de entrada e de ruído na Equação (2.5), obtemos o modelo AR:

$$
y_k = \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + e_k
\tag{2.12}
$$

O exemplo a seguir é um modelo AR polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}+0.1139y_{k-3} -0.1691y_{k-4}
\end{align}
\tag{2.13}
$$

Nesse caso, precisamos definir o tipo de modelo como `NAR` e `unbiased=False` em `LeastSquares`:

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

A figura abaixo mostra 10 simulações independentes para analisar os efeitos de diferentes realizações de processo de ruído no comportamento do sistema AR.

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
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
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
> Figura 8. Simulações para mostrar os efeitos de diferentes realizações de processo de ruído no comportamento do modelo AR.

### FIR

Se mantivermos apenas os termos de entrada na Equação (2.5), obtemos o modelo NFIR:

$$
y_k = \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m} + e_k
\tag{2.14}
$$

O exemplo a seguir é um modelo FIR polinomial:

$$
\begin{align}
  y_k =& 0.7213x_{k-1}-0.5692x_{k-2}+0.1139x_{k-3} -0.1691x_{k-4}
\end{align}
\tag{2.15}
$$

Nesse caso, precisamos definir o tipo de modelo como `NFIR` e `unbiased=False` em `LeastSquares`:

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

A figura abaixo mostra 10 simulações independentes para analisar os efeitos de diferentes realizações de processo de ruído no comportamento do sistema FIR.

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
    n_fine = np.linspace(n.min(), n.max(), 10*len(n))  # More points for a smoother curve
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
> Figura 9. Simulações para mostrar os efeitos de diferentes realizações de processo de ruído no comportamento do modelo **FIR**.

> Não definimos o `model_type` para ARMAX e ARX porque o padrão é `NARMAX`. O SysIdentPy permite três tipos de modelo: `NARMAX`, `NAR` e `NFIR`. Como ARMAX, ARX e outras variantes lineares são subconjuntos de modelos NARMAX, não há necessidade de um tipo de modelo específico `ARMAX`. A ideia é ter tipos de modelos para: modelos com regressors de entrada e saída; modelos apenas com regressors de saída; e modelos apenas com regressors de entrada.

### Outras Variantes

Por simplicidade, definimos a Equação (2.5) e consideramos apenas as representações polinomiais. No entanto, você pode estender essas representações para outras funções base, como Fourier. Se definirmos $\mathcal{F}$ como a extensão de Fourier:

$$
\mathcal{F}(x) = [\cos(\pi x), \sin(\pi x), \cos(2\pi x), \sin(2\pi x), \ldots, \cos(N\pi x), \sin(N\pi x)]
\tag{2.16}
$$

Nesse caso, a representação ARX de Fourier será:

$$
\begin{aligned}
y_k = &\Big[ \cos(\pi y_{k-1}), \sin(\pi y_{k-1}), \cos(2\pi y_{k-1}), \sin(2\pi y_{k-1}), \ldots, \cos(N\pi y_{k-1}), \sin(N\pi y_{k-1}), \\
&\ \ \cos(\pi y_{k-2}), \sin(\pi y_{k-2}), \ldots, \cos(N\pi y_{k-n_y}), \sin(N\pi y_{k-n_y}), \\
&\ \ \cos(\pi x_{k-1}), \sin(\pi x_{k-1}), \cos(2\pi x_{k-1}), \sin(2\pi x_{k-1}), \ldots, \cos(N\pi x_{k-n_x}), \sin(N\pi x_{k-n_x}) \Big] \\
&\ \ + e_k
\end{aligned}
\tag{2.17}
$$

Para fazer isso no SysIdentPy, basta importar a função base de Fourier em vez da polinomial:

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

## Modelos Não Lineares

### NARMAX

O modelo NARMAX foi proposto por Stephen A. Billings e I. J. Leontaritis em [1981](https://pdf.sciencedirectassets.com/314898/1-s2.0-S1474667082X74355/1-s2.0-S1474667017630398/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjECcaCXVzLWVhc3QtMSJHMEUCIGSxH3DpnP8cywgmDASl%2FBUhnmjjJxeG%2BTay6JhHkUUKAiEAgdzFyYAYcYGDnHxy9T%2BzIVHT%2Fi8J0BVY4ZxI0ovAaUIqsgUIIBAFGgwwNTkwMDM1NDY4NjUiDNTsL0ESuGUnlT%2FyniqPBTUbIgulDeB6wJLJ6sql18H4y3NFfyReA0c45SRi1Rq%2F1TEI%2B0c2cC8v%2FfH8cJF7xGiU%2Fvq3apocGRSFYM6VCJ5IPeyqW3V9Z6lZCAHz3sMY2vX6xHf8czCB4jk85bQAr5Ct9C8m0YI33jvykw7ZC25zOVsyDyR%2Fnas4IOBAboSImvbTzYgDwcwL5S9eSlq%2Byw%2FZPyOMJslF6mOERjXYMrr1Xg51F%2B4xeDt%2BWSGe5qbu5s0RJwifIBp0%2F6WG1m6Mvf0wn%2BQyP8PMJyKMP31fbeA7g3Ndwtjs8ghLPjxax%2BSLby5hgi9ogzm1Gr%2Fv972XLW3TwNUod6yMdzVolsNcxgPBUtns%2Fdfr%2Fm2e%2BkvpETO%2FF1J7V%2FAievzJFE2Sl087x0ZACiy57fK3doa7UWpZj46DaU%2BeAEUbrSXel2PqHbLfSFM9NDsgZWBRXzqygQY4VMGANBEKCR%2BrcJdI3BG%2B%2Bkcg18OKr%2Bf42vgzGRiGB3szABhtRVq5QvGyXYYONv1SFcLBHAt67nuVukKMu%2BR%2BmKQT9CwNOL13oZ5ldS5Fowv1TC8B8o8qdlveFb6WywpTnsSc9%2Bm8Ufbn5W8zI5Fy2sECz4hjUY1zGRhH3O9027j87%2BwyVVfdT6ucHvOdHjQTr2tawgJmUvRunQlw7xN85t2IPEWdFZSXC9IgkG5QRxxzYzsyrCzG4Vsm3CGcShAu7Ms%2Bm7z7XO8QXgfN95JxcG4PX6siAkN5CGZwAdtUkp1VOXzpRt77mrTofG%2BHDGHeXPlyRnBNqR4Z7EJbevPWgUK9BRkVbKfnL0YCtuWbQ3sQ115UTR1a84VejuWmTg%2FJ8Fpf78ApwBE6igc1VfePXVJJwAzxzWcXPSwlhDQNqPIw8ffPtQY6sQHciEnI0Pd0iSUf53%2BhgScY6deC54cJfBcOtzm9RhrSeSoQyPdhrMHU0r2fWAhYRcr9ZeuObdxPtjpqardoQM%2BEizPzQNO8uBtOoYTK2inAXxp2z3U%2BDYHWJHNuapOhC2GsOPlPXco7e0Hm0GxiPUyk6%2B8FtqTAs6d4KxpT7CGLUNChK%2BCUbt%2BsnF84TjzK40Fkf0Gya9ESFUHmIEUmfXabOKfgra2QbSbcogWgoI5gChM%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240808T001847Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY343F44W4%2F20240808%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=95d8e3c94d76309850682e9e8f9025436a481f440bf152241e01eb47d70c4438&hash=fb1356fe353cd87c7fb931d951e83c97cce7568257b5a9c6c533aee21ef64f35&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1474667017630398&tid=spdf-a9063d37-b51a-42e8-97b3-bb5ab63b0384&sid=07d7477c73bfe241706ba2d-0fba8a3cbcd1gxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=18165805090051570458&rr=8afb54c71f7e1ac0&cc=br), ([Billings, S. A. - Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)), e pode ser descrito como

$$
\begin{equation}
y_k= \mathcal{F}[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k,
\end{equation}
\tag{2.18}
$$

em que $n_y\in \mathbb{N}^*$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ são os máximos atrasos para saída, entrada e ruído, respectivamente; $x_k \in \mathbb{R}^{n_x}$ é a entrada do sistema e $y_k \in \mathbb{R}^{n_y}$ é a saída do sistema no instante discreto $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ representa incertezas e possíveis ruídos no instante discreto $k$. Nesse caso, $\mathcal{F}$ é alguma função não linear dos regressors de entrada e saída, e $d$ é um atraso de tempo tipicamente definido como $d=1$.

Você pode notar que a diferença entre as Equações (2.5) e (2.18) está na função que representa o sistema. Para modelos NARMAX, $\mathcal{F}$ pode ser qualquer função não linear, enquanto na Equação (2.5) apenas funções lineares são permitidas. Embora existam muitas possíveis aproximações para $\mathcal{F}(\cdot)$ (por exemplo, redes neurais, fuzzy, wavelet, Radial Basis Function), o modelo NARMAX polinomial em forma de potência é o mais amplamente utilizado ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594); [Khandelwal, D. and Schoukens, M. and Toth, R.](https://arxiv.org/abs/2001.05320)):

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}\prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.19}
$$

em que $p$ é o número de regressors, $\Theta_i$ são os parâmetros do modelo e $a_i, m$, $b_i, j$ e $d_i, l \in \mathbb{N}$ são os expoentes dos termos de saída, entrada e ruído, respectivamente.

A Equação (2.20) descreve um modelo NARMAX polinomial com grau de não linearidade igual a $2$, identificado a partir de dados experimentais de um sistema motor/gerador CC, sem conhecimento prévio da forma do modelo, extraído de [Lacerda Junior, W. R., Almeida, V. M., & Martins, S. A. M. (2017)](https://www.ufsj.edu.br/portal2-repositorio/File/gcom/LAM2017.pdf):

$$
\begin{align}
  y_k =& 1.7813y_{k-1}-0.7962y_{k-2}+0.0339x_{k-1} -0.1597x_{k-1} y_{k-1} +0.0338x_{k-2} + \\
  & + 0.1297x_{k-1}y_{k-2} - 0.1396x_{k-2}y_{k-1}+ 0.1086x_{k-2}y_{k-2}+0.0085y_{k-2}^2 + 0.0247e_{k-1}e_{k-2}
\end{align}
\tag{2.20}
$$

Os valores de $\Theta$ são os coeficientes de cada termo da equação polinomial.

Funções base polinomiais são uma das representações de NARMAX mais utilizadas devido a diversas características interessantes ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)):

- Todas as funções polinomiais são suaves em $\mathbb{R}$.
- O [Teorema de Aproximação de Weierstrass](https://www.researchgate.net/profile/P-Johnson/publication/353192512_Stone-Weierstrass_Theorem/links/60ec4cdc9541032c6d32c4a6/Stone-Weierstrass-Theorem.pdf) afirma que qualquer função contínua real definida em um espaço fechado e limitado $[a,b]$ pode ser aproximada uniformemente por um polinômio nesse intervalo.
- Elas podem descrever diversos sistemas dinâmicos não lineares, incluindo processos industriais, sistemas de controle, sistemas estruturais, sistemas econômicos e financeiros, biologia, medicina e sistemas sociais (alguns exemplos são detalhados em [Lacerda Junior, W. R. and Martins, S. A. M. and Nepomuceno, E. G. and Lacerda, Marcio J.](https://ieeexplore.ieee.org/document/8751951); [Fung, E. H. K. and Wong, Y. K. and Ho, H. F. and Mignolet, M. P.](https://www.sciencedirect.com/science/article/pii/S0307904X03000714); [Kukreja, S. L. and Galiana, H. L. and Kearney, R. E.](https://ieeexplore.ieee.org/document/1179133); [Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594); [Aguirre, L. A.](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas?channel=doi&linkId=574cb16508ae82d2c6bc870f&showFulltext=true); entre muitos outros).
- Diversos algoritmos foram desenvolvidos para seleção de estrutura e estimação de parâmetros de modelos NARMAX polinomiais, e essa continua sendo uma área ativa de pesquisa.
- Modelos NARMAX polinomiais são versáteis e podem ser usados tanto para predição quanto para inferência. A estrutura desses modelos é relativamente fácil de interpretar e pode ser relacionada ao sistema subjacente, o que é muito mais difícil de alcançar com redes neurais ou funções wavelet, por exemplo.

Você pode construir facilmente um modelo NARMAX polinomial usando o SysIdentPy. Note que, neste caso, a diferença em relação ao ARMAX está no grau da função polinomial.

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

Pode parecer uma mudança simples, mas em cenários não lineares a "curse of dimensionality" se torna um problema real. O número de regressors candidatos, $n_r$, de um modelo NARX polinomial pode ser definido como em [Korenberg, M. L., Billings, S. A., Liu, Y. P., and McIlroy, P. J. - Orthogonal parameter estimation algorithm for non-linear stochastic systems](https://www.tandfonline.com/doi/abs/10.1080/00207178808906169):

$$
\begin{equation}
    n_r = M+1,
\end{equation}
\tag{2.21}
$$

em que

$$
\begin{align}
    M = & \sum_{i=1}^{\ell}n_i \\
    n_i = & \frac{n_{i-1}(n_y+n_x+i-1)}{i}, n_{0} = 1.
\end{align}
\tag{2.22}
$$

Como mencionamos na Introdução do livro, os métodos NARMAX buscam construir modelos o mais simples possível. A ideia é reproduzir uma ampla gama de comportamentos usando um pequeno subconjunto de termos do vasto espaço de busca formado pelos regressors candidatos.

Vamos usar o SysIdentPy para ver como o espaço de busca cresce no cenário linear versus o não linear. O método `count_model_regressors` disponível em `narmax_tools` pode ser usado para verificar quantos regressors existem no espaço de busca, dado o número de entradas, os atrasos de `y` e `x` e a função base. Vamos usar `xlag=ylag=10` e a função base polinomial. O usuário pode simular diferentes cenários definindo parâmetros distintos.

```python
from sysidentpy.utils.information_matrix import count_model_regressors
from sysidentpy.basis_function import Polynomial
import numpy as np
```

Para o caso linear com 1 entrada, temos 21 regressors:
```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=1)
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 21
```

Para o caso linear com 2 entradas, o número de regressors salta para 31:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 inputs
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=1)
xlag = [list(range(1, 11))] * x_train.shape[1]
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 31
```

Se considerarmos um caso não linear com 1 entrada apenas mudando o grau para 2, temos 231 regressors:

```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=2)
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 231
```

Se definirmos o grau como 3, o número de termos aumenta significativamente para 1771 regressors:

```python
x_train = np.random.rand(10, 1)  # simulating a case with 1 input
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=3)
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=10,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 1771
```

Se tivermos 2 entradas no cenário não linear com `degree=2`, o número de regressors é 496:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 inputs
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=2)
xlag = [list(range(1, 11))] * x_train.shape[1]
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 496
```

Se tivermos 2 entradas no cenário não linear com `degree=3`, o número salta para 5456 regressors:

```python
x_train = np.random.rand(10, 2)  # simulating a case with 2 inputs
y_train = np.random.rand(10, 1)
basis_function = Polynomial(degree=3)
xlag = [list(range(1, 11))] * x_train.shape[1]
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=xlag,
    ylag=10,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=False,
)
n_regressors
>>> 5456
```

Como você pode notar, o número de regressors aumenta significativamente à medida que o grau do polinômio e o número de entradas crescem. Isso torna a seleção da estrutura do modelo muito mais complexa! No caso linear com 10 entradas, temos `2^31=2.15e+09` combinações possíveis de modelos. Quando `degree=2` com 2 entradas, temos `2^496=2.05e+149` combinações possíveis! Tente obter o número de combinações possíveis quando `degree=3` com 2 entradas. Além disso, tente repetir o exercício com mais entradas e graus de não linearidade mais altos e veja como a "curse of dimensionality" é um grande problema.

Como se pode ver, obter um modelo simples em um espaço de busca tão grande é uma tarefa complexa de seleção de estrutura. Selecionar os termos mais significativos a partir de um dicionário enorme de termos possíveis não é uma tarefa trivial. E isso é difícil não apenas devido ao problema combinatório complexo e à incerteza sobre a ordem do modelo. Identificar os termos mais relevantes em um cenário não linear é muito desafiador porque depende do tipo de não linearidade (singularidade esparsa ou quase singular, efeitos de memória ou amortecimento, entre outros), da resposta dinâmica (sistemas espaço-temporais, dependentes do tempo), da resposta em regime permanente, da frequência dos dados, do ruído e de muitos outros fatores.

Devido aos algoritmos de seleção de estrutura desenvolvidos para modelos NARMAX, mesmo modelos lineares como ARMAX podem apresentar desempenho diferente quando obtidos usando o SysIdentPy em comparação com outras bibliotecas, como Statsmodels. Apresentamos um estudo de caso mostrando exatamente isso no Capítulo 10.

### NARX

Se não incluirmos termos de ruído $e_{k-n_e}$ na Equação (2.19), obtemos modelos NARX:

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.23}
$$

A Equação (2.24) descreve um modelo NARX polinomial simples:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^2+0.1139y_{k-1}x_{k-1}
\end{align}
\tag{2.24}
$$

A única diferença no SysIdentPy é definir `unbiased=False`:

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

> O usuário pode reutilizar os códigos fornecidos para modelos lineares a fim de analisar modelos não lineares com diferentes realizações de ruído.

### NARMA

Se não incluirmos termos de entrada na Equação (2.19), obtemos o modelo NARMA:

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times\prod_{l=1}^{n_e}e_{k-l}^{d_i, l}\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.25}
$$

O exemplo a seguir é um modelo NARMA polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^3+0.1139y_{k-3}y_{k-4} + 0.2245e_{k-1}
\end{align}
\tag{2.26}
$$

Como a representação do modelo não possui entradas, precisamos definir o tipo de modelo como `NAR` e `unbiased=True` novamente em `LeastSquares`:

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

Se não incluirmos termos de entrada e ruído na Equação (2.19), obtemos o modelo NAR:

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times\prod_{m=1}^{n_y}y_{k-m}^{a_i, m}
\end{align}
\tag{2.27}
$$

O exemplo a seguir é um modelo NAR polinomial:

$$
\begin{align}
  y_k =& 0.7213y_{k-1}-0.5692y_{k-2}^2+0.1139y_{k-3}^3 -0.1691y_{k-4}y_{k-5}
\end{align}
\tag{2.28}
$$

Nesse caso, precisamos definir o tipo de modelo como `NAR` e `unbiased=False` em `LeastSquares`:

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

Se mantivermos apenas os termos de entrada na Equação (2.19), obtemos o modelo NFIR:

$$
\begin{align}
  y_k = \sum_{i=1}^{p}\Theta_i \times \prod_{j=0}^{n_x}x_{k-j}^{b_i, j}
\end{align}
\tag{2.29}
$$

O exemplo a seguir é um modelo NFIR polinomial:

$$
\begin{align}
  y_k =& 0.7213x_{k-1}-0.5692x_{k-2}^2+0.1139x_{k-3}x_{k-4} -0.1691x_{k-4}^3
\end{align}
\tag{2.30}
$$

Nesse caso, precisamos definir o tipo de modelo como `NFIR` e `unbiased=False` em `LeastSquares`:

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

### Modelos NARMAX Mistos

Em algumas aplicações, o uso de uma única função base pode não fornecer uma descrição satisfatória da relação entre as variáveis de entrada (ou independentes) e a variável de saída (ou resposta). Para melhorar o desempenho do modelo, foi proposto o uso de uma combinação linear de um conjunto de funções não lineares para substituir as contrapartes lineares.

Você pode obter isso no SysIdentPy utilizando ensembles de funções base. É possível construir um modelo de Fourier em que os termos possuem interações, bem como modelos com funções base mistas, usando termos expandidos por funções base polinomiais e de Fourier, ou qualquer outra função base disponível no pacote.

> No SysIdentPy, por enquanto, você só pode misturar uma função base com a função base polinomial. Você pode, por exemplo, misturar Fourier com Polynomial, mas não pode misturar Fourier com Bernstein.

Para misturar as funções base Fourier ou Bernstein com Polynomial, basta definir `ensemble=True` na definição da função base:

```python
from sysidentpy.basis_function import Fourier

basis_function = Fourier(degree=2, ensemble=True)
```

### Rede Neural NARX

Redes neurais são modelos compostos por camadas interconectadas de nós (neurônios) projetados para tarefas como classificação e regressão. Cada neurônio é uma unidade básica dentro dessas redes. [Matematicamente](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas?channel=doi&linkId=574cb16508ae82d2c6bc870f&showFulltext=true), um neurônio é representado por uma função $f$ que recebe um vetor de entrada $\mathbf{x} = [x_1, x_2, \ldots, x_n]$ e gera uma saída $y$. Essa função normalmente envolve uma soma ponderada das entradas, um termo de bias opcional $b$ e uma função de ativação $\phi$:

$$
y = \phi \left( \sum_{i=1}^{n} w_i x_i + b \right)
\tag{2.31}
$$

em que $\mathbf{w} = [w_1, w_2, \ldots, w_n]$ são os pesos associados às entradas. A função de ativação $\phi$ introduz não linearidade no modelo, permitindo que a rede aprenda padrões complexos. Funções de ativação comuns incluem:

- **Sigmoid**: $\phi(z) = \frac{1}{1 + e^{-z}}$  
  Produz saídas entre 0 e 1, sendo útil para classificação binária.

- **Tangente Hiperbólica (tanh)**: $\phi(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$  
  Produz valores entre -1 e 1, frequentemente usada para centralizar os dados em torno de zero.

- **Rectified Linear Unit (ReLU)**: $\phi(z) = \max(0, z)$  
  Retorna zero para valores negativos e o próprio valor de entrada para valores positivos, ajudando a mitigar o problema do gradiente desvanecente.

- **Leaky ReLU**: $\phi(z) = \max(0.01z, z)$  
  Uma variação da ReLU que permite um gradiente pequeno e diferente de zero quando a entrada é negativa, abordando o problema de "neurônios mortos".

- **Softmax**: $\phi(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$  
  Converte logits em probabilidades para classificação multiclasse, garantindo que as saídas somem 1.

Cada função de ativação possui vantagens específicas e é escolhida de acordo com as necessidades da rede neural e da tarefa em questão.

Como mencionado, uma rede neural é composta por múltiplas camadas, cada uma contendo diversos neurônios. Sob esse aspecto, as camadas podem ser categorizadas em:

- **Camada de Entrada (Input Layer)**: Camada que recebe os dados de entrada.
- **Camadas Escondidas (Hidden Layers)**: Camadas intermediárias que processam as entradas por meio de conexões ponderadas e funções de ativação.
- **Camada de Saída (Output Layer)**: Camada final que produz a saída da rede.

> A própria rede possui, portanto, uma arquitetura muito simples. A terminologia usada em redes neurais também é ligeiramente diferente da notação padrão, universal em Identificação de Sistemas e Estatística. Em vez de falar de parâmetros do modelo, fala-se em *pesos da rede* e, em vez de estimação, fala-se em *aprendizado*. Essa terminologia foi, sem dúvida, introduzida para fazer parecer que algo completamente novo estava sendo discutido, quando alguns dos problemas abordados são bastante tradicionais — [Stephen A. Billings](https://www.amazon.com.br/s/ref=dp_byline_sr_book_1?ie=UTF8&field-author=Stephen+A+Billings&text=Stephen+A+Billings&sort=relevancerank&search-alias=stripbooks)

Note que a rede em si é simplesmente uma coleção de unidades de ativação não lineares $\phi(\cdot)$ que são funções estáticas simples. Não há dinâmica dentro da rede. Isso é aceitável para aplicações como reconhecimento de padrões, mas, para usar a rede em Identificação de Sistemas, são necessários atrasos de entrada e saída, que devem ser fornecidos como entradas, seja explicitamente, seja por meio de um procedimento recorrente. Nesse sentido, se definirmos $\mathcal{F}$ como uma função neural, podemos adaptá-la para criar um modelo neural NARX, transformando a arquitetura neural em uma arquitetura NARX. A rede neural NARX, porém, não é linear nos parâmetros, como os modelos NARMAX baseados em funções base. Assim, algoritmos como Orthogonal Least Squares não são adequados para estimar os pesos do modelo.

O **SysIdentPy** oferece suporte a redes NARX em configuração Série-Paralelo (open-loop) para treinamento, o que torna o processo de treinamento mais simples. Em seguida, convertemos a rede NARX da configuração Série-Paralelo para a configuração Paralelo (closed-loop) para predição.

A configuração Série-Paralelo nos permite usar `pytorch` diretamente para o treinamento, portanto o **SysIdentPy** utiliza `pytorch` no backend para redes neurais NARX, juntamente com métodos auxiliares disponíveis apenas no **SysIdentPy**.

Um modelo neural NARX simples pode ser representado como uma rede neural Perceptron Multicamadas (MLP) com componente autorregressivo, junto com entradas atrasadas.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/Pasted%20image%2020240710121246.png?raw=true)
> Figura 10. Arquiteturas de rede neural em paralelo e série-paralelo para modelagem do sistema dinâmico $\mathbf{y}[k]=\mathbf{F}(\mathbf{y}[k-1], \mathbf{y}[k-2], \mathbf{u}[k-1], \mathbf{u}[k-2])$. O operador de atraso $q^{-1}$ é tal que $\mathbf{y}[k-1]=q^{-1} \mathbf{y}[k]$. Referência: [Antonio H. Ribeiro and Luis A. Aguirre](https://arxiv.org/pdf/1706.07119)

> Neural NARX não é o mesmo que Redes Neurais Recorrentes (RNN). Para mais detalhes, consulte o artigo [A Note on the Equivalence of NARX and RNN](https://link.springer.com/article/10.1007/s005210050005).

Para construir uma rede Neural NARX no SysIdentPy, o usuário deve usar `pytorch`. Utilizamos `pytorch` para tornar flexível a definição da arquitetura da rede. No entanto, isso exige que o usuário tenha uma compreensão razoável de como redes neurais funcionam. Veja abaixo um script que mostra como construir um modelo Neural NARX simples no **SysIdentPy**:

```python
from torch import nn
import torch

from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.narmax_tools import regressor_code

# simulated data
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=False, sigma=0.01, train_percentage=80
)
```

O usuário pode utilizar `cuda` seguindo a mesma abordagem usada ao construir uma rede neural em **pytorch**:

```python
torch.cuda.is_available()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
```

O usuário pode criar um objeto `NARXNN` e escolher o atraso máximo tanto da entrada quanto da saída para construir a matriz de regressors que servirá como entrada da rede. Além disso, é possível escolher a função de custo, o otimizador, os parâmetros opcionais do otimizador e o número de épocas.

Como construímos esse recurso sobre o Pytorch, você pode escolher qualquer função de custo disponível em `torch.nn.functional`. [Clique aqui](https://pytorch.org/docs/stable/nn.functional.html#loss-functions) para ver a lista de funções de custo disponíveis. Basta passar o nome da função de custo desejada.

De forma análoga, você pode escolher qualquer otimizador disponível em `torch.optim`. [Clique aqui](https://pytorch.org/docs/stable/optim.html) para ver a lista de otimizadores disponíveis.

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
    },  # optional parameters of the optimizer
)
```

Como o modelo NARXNN foi definido com $ylag=2$, $xlag=2$ e função base polinomial com $degree=1$, obtemos uma matriz de regressors com 4 features. Precisamos do tamanho dessa matriz de regressors para construir as camadas da nossa rede. Nossos dados de entrada (`x_train`) possuem apenas uma feature, mas como estamos criando uma rede NARX, uma matriz de regressors é construída internamente com novas features baseadas em `xlag` e `ylag`.

Se precisar de ajuda para descobrir quantos regressors são criados internamente, você pode usar a função `regressor_code` de `narmax_tools` e obter o tamanho do código de regressors gerado:

```python
basis_function = Polynomial(degree=1)
n_regressors = count_model_regressors(
    x=x_train,
    y=y_train,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    basis_function=basis_function,
    is_neural_narx=True,
)
n_regressors
>>> 4
```

A configuração da sua rede segue exatamente o mesmo padrão de uma rede definida em Pytorch. O código a seguir representa nossa rede neural NARX:

```python
class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_regressors, 30)
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

O usuário deve passar a rede definida para o estimador `NARXNN` e configurar `cuda` se estiver disponível (ou se for necessário):

```python
narx_net.net = NARX()

if device == "cuda":
    narx_net.net.to(torch.device("cuda"))
```

Como temos funções `fit` (para treinamento) e `predict` para o NARMAX polinomial, criamos o mesmo padrão para a rede NARX. Assim, basta chamar `fit` e `predict` da seguinte forma:

```python
narx_net.fit(X=x_train, y=y_train, X_test=x_valid, y_test=y_valid)
yhat = narx_net.predict(X=x_valid, y=y_valid)
```

Se a configuração da rede for construída antes de chamar o `NARXNN`, basta passar o modelo para o `NARXNN` da seguinte forma:

```python
class NARX(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(n_regressors, 30)
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
    },  # optional parameters of the optimizer
)

narx_net2.fit(X=x_train, y=y_train)
yhat = narx_net2.predict(X=x_valid, y=y_valid)
```

### Representação Geral de Conjunto de Modelos

Com base na ideia de transformar uma rede neural estática em um modelo neural NARX, podemos estender o método basicamente para qualquer classe de modelo. O SysIdentPy não tem como objetivo implementar todas as classes de modelos existentes na literatura. Entretanto, criamos uma funcionalidade que permite o uso de qualquer outro pacote de machine learning que siga a API `fit` e `predict` dentro do SysIdentPy para convertê-los em versões NARX desses modelos.

Vamos tomar o XGBoost (eXtreme Gradient Boosting) como exemplo. XGBoost é uma classe de modelos bem conhecida para tarefas de regressão. No entanto, XGBoost não é uma escolha comum quando tratamos de identificação de sistemas dinâmicos, pois foi originalmente projetado para modelar sistemas estáticos. Com o SysIdentPy, você pode facilmente transformar XGBoost em um modelo NARX.

O Scikit-learn é outro excelente exemplo. Você pode transformar qualquer modelo do Scikit-learn em um modelo NARX usando o SysIdentPy. Veremos essas aplicações em detalhes no Capítulo 11, mas o script abaixo ilustra como isso é simples:

``` python
from sysidentpy.general_estimators import NARX
from sysidentpy.basis_function import Polynomial
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

Você pode usar qualquer outro modelo simplesmente trocando a classe do modelo e passando-a para o parâmetro `base_estimator` na funcionalidade `NARX`.

### Modelos MIMO

Para manter as coisas simples, apenas modelos SISO foram apresentados nas seções anteriores. No entanto, modelos NARMAX podem ser estendidos de forma natural para o caso MIMO ([Billings, S. A. and Chen, S. and Korenberg, M. J.](https://www.tandfonline.com/doi/abs/10.1080/00207178908559767)):

$$
\begin{align}
\t y_{{_i}k}=& F_{{_i}}^\ell \bigl[y_{{_1}k-1},  \dotsc, y_{{_1}k-n^i_{y{_1}}},\dotsc, y_{{_s}k-1},  \dotsc, y_{{_s}k-n^i_{y{_s}}}, x_{{_1}k-d}, \\
\t & x_{{_1}k-d-1}, \dotsc, x_{{_1}k-d-n^i_{x{_1}}}, \dotsc, x_{{_r}k-d}, x_{{_r}k-d-1}, \dotsc, x_{{_r}k-d-n^i_{x{_r}}}\bigr] + \xi_{{_i}k},
\end{align}
\tag{2.32}
$$

em que, para $i = 1, \dotsc, s$, cada submodelo linear nos parâmetros pode ter diferentes atrasos máximos. Mais genericamente, considerando

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

o modelo MIMO pode ser denotado como

$$
\begin{equation}
             Y_k= F^\ell[Y_{k-1},  \dotsc, Y_{k-n_y},X_{k-d}, X_{k-d-1}, \dotsc, X_{k-d-n_x}] + \Xi_k,
\end{equation}
\tag{2.34}
$$

em que $Xk ~= \{x_{{_1}k}, x_{{_2}k}, \dotsc, x_{{_r}k}\}\in \mathbb{R}^{n^i_{x{_r}}}$ e $Yk~= \{y_{{_1}k}, y_{{_2}k}, \dotsc, y_{{_s}k}\}\in \mathbb{R}^{n^i_{y{_s}}}$. O número de termos possíveis de um modelo MIMO NARX dado o $i$-ésimo grau polinomial, $\ell_i$, é:

$$
\begin{equation}
    n_{{_{m}}r} = \sum_{j = 0}^{\ell_i}n_{ij},
\end{equation}
\tag{2.35}
$$

em que

$$
\begin{align}
    n_{ij} = \frac{ n_{ij-1} \biggl[ \sum\limits_{k=1}^{s} n^i_{y_k} + \sum\limits_{k=1}^{r} n^i_{x_k} + j - 1 \biggr]}{j}, \qquad n_{i0}=1, j=1, \dotsc, \ell_i.
\end{align}
\tag{2.36}
$$

Se $s=1$, temos um modelo MISO que pode ser representado por uma única função polinomial. Além disso, um modelo MIMO pode ser decomposto em modelos MISO, como apresentado na figura a seguir:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/mimo_split.png?raw=true)
> Figura 11. Um modelo MIMO decomposto em modelos MISO individuais.

> O SysIdentPy ainda não oferece suporte a modelos MIMO, apenas modelos MISO. Você pode, no entanto, decompor um sistema MIMO como apresentado na Figura 11 e usar o SysIdentPy para criar modelos para cada subsistema.
