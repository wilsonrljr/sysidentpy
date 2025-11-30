## Mínimos Quadrados

Considere o modelo NARX descrito em uma forma genérica como

$$
\begin{equation}
    y_k = \psi^\top_{k-1}\hat{\Theta} + \xi_k,
\end{equation}
\tag{3.1}
$$

onde $\psi^\top_{k-1} \in \mathbb{R}^{n_r \times n}$ é a matriz de informação, também conhecida como matriz de regressores. A matriz de informação é a transformação da entrada e saída baseada em uma função base e $\hat{\Theta}~\in \mathbb{R}^{n_{\Theta}}$ é o vetor de parâmetros estimados. O modelo acima também pode ser representado na forma matricial como:

$$
\begin{equation}
    y = \Psi\hat{\Theta} + \Xi,
\end{equation}
\tag{3.2}
$$

onde

$$
\begin{align}
    Y = \begin{bmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
    \end{bmatrix},
    \Psi = \begin{bmatrix}
    \psi_{{_1}} \\
    \psi_{{_2}} \\
    \vdots \\
    \psi_{{_{n_{\Theta}}}}
    \end{bmatrix}^\top=
    \begin{bmatrix}
    \psi_{{_1}1} & \psi_{{_2}1} & \dots & \psi_{{_{n_{\Theta}}}1} \\
    \psi_{{_1}2} & \psi_{{_2}2} & \dots & \psi_{{_{n_{\Theta}}}2} \\
    \vdots & \vdots &       & \vdots \\
    \psi_{{_1}n} & \psi_{{_2}n} & \dots & \psi_{{_{n_{\Theta}}}n} \\
    \end{bmatrix},
    \hat{\Theta} = \begin{bmatrix}
    \hat{\Theta}_1 \\
    \hat{\Theta}_2 \\
    \vdots \\
    \hat{\Theta}_{n_\Theta}
    \end{bmatrix},
    \Xi = \begin{bmatrix}
    \xi_1 \\
    \xi_2 \\
    \vdots \\
    \xi_n
    \end{bmatrix}.
\end{align}
\tag{3.3}
$$


> Consideraremos a função base polinomial para manter os exemplos diretos, mas os métodos aqui funcionarão para qualquer outra função base.

O modelo NARX paramétrico é linear nos parâmetros $\Theta$, então podemos usar algoritmos bem conhecidos, como o algoritmo de Mínimos Quadrados desenvolvido por Gauss em $1795$, para estimar os parâmetros do modelo. A ideia é encontrar o vetor de parâmetros que minimiza a norma $l2$, também conhecida como soma dos quadrados dos resíduos, descrita como

$$
\begin{equation}
    J_{\hat{\Theta}} = \Xi^\top \Xi = (y - \Psi\hat{\Theta})^\top(y - \Psi\hat{\Theta}) = \lVert y - \Psi\hat{\Theta} \rVert^2.
\end{equation}
\tag{3.4}
$$

Na Equação 3.4, $\Psi\hat{\Theta}$ é a predição um passo à frente de $y_k$, expressa como

$$
\begin{equation}
    \hat{y}_{1_k} = g(y_{k-1}, u_{k-1}\lvert ~\Theta),
\end{equation}
\tag{3.5}
$$

onde $g$ é alguma função polinomial desconhecida. Se o gradiente de $J_{\Theta}$ em relação a $\Theta$ é igual a zero, então temos a equação normal e a estimativa de Mínimos Quadrados é expressa como

$$
\begin{equation}
    \hat{\Theta}  = (\Psi^\top\Psi)^{-1}\Psi^\top y,
\end{equation}
\tag{3.6}
$$

onde $(\Psi^\top\Psi)^{-1}\Psi^\top$ é chamada de pseudo-inversa da matriz $\Psi$, denotada $\Psi^+ \in \mathbb{R}^{n \times n_r}$.

Para ter um estimador não-viesado, as seguintes são as suposições básicas necessárias para o método dos mínimos quadrados:
- A1 - Não há correlação entre o vetor de erros, $\Xi$, e a matriz de regressores, $\Psi$. Matematicamente:
- $\mathrm{E}\{[(\Psi^\top\Psi)^{-1}\Psi^\top] \Xi\} = \mathrm{E}[(\Psi^\top\Psi)^{-1}\Psi^\top] \mathrm{E}[\Xi]; \tag{3.7}$
- A2 - O vetor de erros $\Xi$ é uma sequência de ruído branco com média zero:
- $\mathrm{E}[\Xi] = 0; \tag{3.8}$
- A3 - A matriz de covariância do vetor de erros é
- $\mathrm{Cov}[\hat{\Theta}] = \mathrm{E}[(\Theta - \hat{\Theta})(\Theta - \hat{\Theta})^\top] = \sigma^2(\Psi^\top\Psi); \tag{3.9}$
- A4 - A matriz de regressores, $\Psi$, tem posto completo.

As suposições mencionadas acima são necessárias para garantir que o algoritmo de Mínimos Quadrados produza um modelo final não-viesado.

#### Exemplo

Vamos ver um exemplo prático. Considere o modelo

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-2} + e_{k}
\tag{3.10}
$$

Podemos gerar a entrada `X` e a saída `y` usando o SysIdentPy. Antes de entrar nos detalhes, vamos executar um modelo simples usando o SysIdentPy. Como sabemos a priori que o sistema que estamos tentando identificar não é linear (o sistema simulado tem um termo de interação $0.1y_{k-1}x_{k-1}$) e a ordem é 2 (o atraso máximo da entrada e saída), definiremos os hiperparâmetros de acordo. Note que este é um cenário simulado, e você não terá essa informação a priori em uma tarefa de identificação real. Mas não se preocupe, a ideia, por enquanto, é apenas mostrar como as coisas funcionam e desenvolveremos alguns modelos reais ao longo do livro.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)

basis_function = Polynomial(degree=2)
estimator = LeastSquares()
model = FROLS(
    n_info_values=3,
    ylag=1,
    xlag=2,
    estimator=estimator,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
# print the identified model
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms
    ),
    columns=["Regressors", "Parameters", "ERR"],
)
print(r)

Regressors   Parameters             ERR
0        x1(k-2)  9.0001E-01  9.56885108E-01
1         y(k-1)  2.0000E-01  3.96313039E-02
2  x1(k-1)y(k-1)  1.0001E-01  3.48355000E-03
```

Como você pode ver, o modelo final tem os mesmos 3 regressores do sistema simulado e os parâmetros estão muito próximos dos usados para simular o sistema. Isso nos mostra que o Mínimos Quadrados teve bom desempenho para estes dados.

Neste exemplo, no entanto, estamos aplicando um algoritmo de Seleção de Estrutura de Modelo (FROLS), que veremos no capítulo 6. Por isso o modelo final tem apenas 3 regressores. O algoritmo de estimação de parâmetros não escolhe quais termos incluir no modelo, então se tivermos uma função base expandida com 6 regressores, ele estimará o parâmetro para cada um dos regressores.

Para verificar como isso funciona, podemos usar o SysIdentPy sem Seleção de Estrutura de Modelo gerando a matriz de informação e aplicando o algoritmo de estimação de parâmetros diretamente.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)
xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
theta = LeastSquares().optimize(psi, y_train[max_lag:, :])
theta

[[-4.1511e-06]
 [ 2.0002e-01]
 [ 1.1237e-05]
 [ 1.0068e-05]
 [ 8.9997e-01]
 [-6.3216e-05]
 [ 1.3298e-04]
 [ 1.0008e-01]
 [ 6.3118e-05]
 [-5.6031e-05]
 [-1.9073e-05]
 [-1.8223e-04]
 [ 1.1307e-04]
 [-1.6601e-04]
 [-8.5068e-05]]
```

Neste caso, temos 15 parâmetros do modelo. Se observarmos a expansão da função base onde o grau do polinômio é igual a 2 e os atrasos para `y` e `x` são definidos como 2, temos

```python
from sysidentpy.utils.narmax_tools import regressor_code
basis_function = Polynomial(degree=2)
regressors = regressor_code(
    X=x_train,
    xlag=2,
    ylag=2,
    model_type="NARMAX",
    model_representation="Polynomial",
    basis_function=basis_function,
)
regressors

array([[ 0, 0],
   [1001, 0],
   [1002, 0],
   [2001, 0],
   [2002, 0],
   [1001, 1001],
   [1002, 1001],
   [2001, 1001],
   [2002, 1001],
   [1002, 1002],
   [2001, 1002],
   [2002, 1002],
   [2001, 2001],
   [2002, 2001],
   [2002, 2002]]
   )
```

O regressors é como o SysIdentPy codifica a função base polinomial seguindo este padrão de codificação:

- $0$ é o termo constante,\n",
- $[1001] = y_{k-1}$
- $[100n] = y_{k-n}$
- $[200n] = x1_{k-n}$
- $[300n] = x2_{k-n}$
- $[1011, 1001] = y_{k-11} \\times y_{k-1}$
- $[100n, 100m] = y_{k-n} \times y_{k-m}$
- $[12001, 1003, 1001] = x11_{k-1} \times y_{k-3} \times y_{k-1}$,
- e assim por diante

 Então, se você observar os parâmetros, podemos ver que a estimação do algoritmo de Mínimos Quadrados para os termos que pertencem ao sistema simulado estão muito próximos dos valores reais.

```python
[1001, 0] -> [ 2.00002486e-01]
[2002, 0] -> [ 8.99927332e-01]
[2001, 1001] -> [ 1.00062340e-01]
```

Além disso, os parâmetros estimados para os outros regressores são valores consideravelmente menores do que os estimados para os termos corretos, indicando que os outros podem não ser relevantes para o modelo.

Você pode começar a pensar que só precisamos definir uma função base e aplicar alguma técnica de estimação de parâmetros para construir modelos NARMAX. No entanto, como mencionado antes, o principal objetivo dos métodos NARMAX é construir o melhor modelo possível mantendo-o simples. E isso é verdade para o caso em que aplicamos o algoritmo FROLS. Além disso, ao lidar com identificação de sistemas, queremos recuperar a dinâmica do sistema em estudo, então adicionar mais termos do que o necessário pode levar a comportamentos inesperados, desempenho ruim e modelos instáveis. Lembre-se, este é apenas um exemplo didático, então em casos reais a seleção de estrutura de modelo é fundamental.

Você pode implementar o método de Mínimos Quadrados de forma simples como

```python
import numpy as np

def simple_least_squares(psi, y):
    return np.linalg.pinv(psi.T @ psi) @ psi.T @ y

# use the psi and y data created in previous examples or
# create them again here to run the example.
theta = simple_least_squares(psi, y_train[max_lag:, :])

theta

array(
	[
	   [-1.08377785e-05],
	   [ 2.00002486e-01],
	   [ 1.73422294e-05],
	   [-3.50957931e-06],
	   [ 8.99927332e-01],
	   [ 2.04427279e-05],
	   [-1.47542408e-04],
	   [ 1.00062340e-01],
	   [ 4.53379771e-05],
	   [ 8.90006341e-05],
	   [ 1.15234873e-04],
	   [ 1.57770755e-04],
	   [ 1.58414037e-04],
	   [-3.09236444e-05],
	   [-1.60377753e-04]
	]
)
```

Como você pode ver, os parâmetros estimados são muito próximos. No entanto, tenha cuidado ao usar tal abordagem em sistemas subdeterminados, bem determinados ou sobredeterminados. Recomendamos usar os métodos `lstsq` do numpy ou scipy.

## Mínimos Quadrados Totais

Esta seção é baseada em [Markovsky, I., & Van Huffel, S. (2007). Overview of total least squares methods. Signal Processing.](https://people.duke.edu/~hpgavin/SystemID/References/Markovsky+VanHuffel-SP-2007.pdf).

O algoritmo de Mínimos Quadrados Totais (Total Least Squares - TLS) é um método estatístico usado para encontrar a melhor relação linear entre variáveis quando tanto os sinais de entrada quanto de saída apresentam perturbação de ruído branco. Diferente dos mínimos quadrados ordinários (OLS), que assume que apenas a variável dependente está sujeita a erro, o TLS considera erros em todas as variáveis medidas, fornecendo uma solução mais robusta em muitas aplicações práticas. O algoritmo foi proposto por Golub e Van Loan.

No TLS, assumimos erros tanto em $\mathbf{X}$ quanto em $\mathbf{Y}$, denotados como $\Delta \mathbf{X}$ e $\Delta \mathbf{Y}$, respectivamente. O modelo verdadeiro se torna:

$$
\mathbf{Y} + \Delta \mathbf{Y} = (\mathbf{X} + \Delta \mathbf{X}) \mathbf{B}
\tag{3.11}
$$

Rearranjando, obtemos:

$$
\Delta \mathbf{Y} = \Delta \mathbf{X} \mathbf{B}
\tag{3.12}
$$

### Função Objetivo

A solução TLS minimiza a norma de Frobenius das perturbações totais em $\mathbf{X}$ e $\mathbf{Y}$:

$$
\min_{\Delta \mathbf{X}, \Delta \mathbf{Y}} \|[\Delta \mathbf{X}, \Delta \mathbf{Y}]\|_F
\tag{3.13}
$$

sujeito a:

$$
(\mathbf{X} + \Delta \mathbf{X}) \mathbf{B} = \mathbf{Y} + \Delta \mathbf{Y}
\tag{3.14}
$$

onde $\| \cdot \|_F$ denota a norma de Frobenius.

### Solução Clássica

A abordagem clássica para resolver o problema TLS é usando a Decomposição em Valores Singulares (SVD). A matriz aumentada $[\mathbf{X}, \mathbf{Y}]$ é decomposta como:

$$
[\mathbf{X}, \mathbf{Y}] = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
\tag{3.15}
$$

onde $\mathbf{U}$ é uma matriz ortogonal $n \times n$, $\Sigma=\operatorname{diag}\left(\sigma_1, \ldots, \sigma_{n+d}\right)$ é uma matriz diagonal de valores singulares; e $\mathbf{V}$ é uma matriz ortogonal definida como

$$
V:=\left[\begin{array}{cc}
V_{11} & V_{12} \\
V_{21} & V_{22}
\end{array}\right] \quad \begin{aligned}
\end{aligned} \quad \text { e } \quad \Sigma:=\left[\begin{array}{cc}
\Sigma_1 & 0 \\
0 & \Sigma_2
\end{array}\right] \begin{gathered}
\end{gathered} .
\tag{3.16}
$$

Uma solução de mínimos quadrados totais existe se e somente se $V_{22}$ for não-singular. Além disso, é única se e somente se $\sigma_n \neq \sigma_{n+1}$. No caso em que a solução de mínimos quadrados totais existe e é única, ela é dada por

$$
\widehat{X}_{\mathrm{tls}}=-V_{12} V_{22}^{-1}
\tag{3.17}
$$

e a matriz de correção de mínimos quadrados totais correspondente é

$$
\Delta C_{\mathrm{tls}}:=\left[\begin{array}{ll}
\Delta A_{\mathrm{tls}} & \Delta B_{\mathrm{tls}}
\end{array}\right]=-U \operatorname{diag}\left(0, \Sigma_2\right) V^{\top} .
\tag{3.18}
$$

Isso é implementado no SysIdentPy da seguinte forma:

```python
def optimize(self, psi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Estimate the model parameters using Total Least Squares method.

    Parameters
    ----------
    psi : ndarray of floats
        The information matrix of the model.
    y : array-like of shape = y_training
        The data used to training the model.

    Returns
    -------
    theta : array-like of shape = number_of_model_elements
        The estimated parameters of the model.

    """
    check_linear_dependence_rows(psi)
    full = np.hstack((psi, y))
    n = psi.shape[1]
    _, _, v = np.linalg.svd(full, full_matrices=True)
    theta = -v.T[:n, n:] / v.T[n:, n:]
    return theta.reshape(-1, 1)
```

Para usá-lo na tarefa de modelagem, basta importá-lo como fizemos no exemplo de Mínimos Quadrados.

A partir de agora, os exemplos não incluirão a etapa de Seleção de Estrutura de Modelo. O objetivo aqui é focar nos métodos de estimação de parâmetros. No entanto, já fornecemos um exemplo incluindo MSS na seção de Mínimos Quadrados, então você não terá nenhum problema para testar isso com outros algoritmos de estimação de parâmetros.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import TotalLeastSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n=1000, colored_noise=False, sigma=0.001, train_percentage=90
)
xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
theta = TotalLeastSquares().optimize(psi, y_train[max_lag:, :])
theta

[[ 1.3321e-04]
 [ 2.0014e-01]
 [-1.1771e-04]
 [ 5.8085e-05]
 [ 9.0011e-01]
 [-1.5490e-04]
 [-1.3517e-05]
 [ 9.9824e-02]
 [ 8.2326e-05]
 [-2.2814e-04]
 [-7.0837e-05]
 [-5.4319e-05]
 [-1.7472e-04]
 [-2.0396e-04]
 [ 1.7416e-05]]
```

## Mínimos Quadrados Recursivos

Considere o modelo de regressão

$$ y_k = \mathbf{\Psi}_k^T \theta_k + \epsilon_k \tag{3.19}$$

onde:
- $y_k$ é a saída observada no tempo $ k $.
- $\mathbf{\Psi}_k$ é a matriz de informação no tempo $k$.
- $\theta_k$ é o vetor de parâmetros a ser estimado no tempo $k$.
- $\epsilon_k$ é o ruído no tempo $k$.

O algoritmo de Mínimos Quadrados Recursivos (RLS) atualiza a estimativa do parâmetro $\theta_k$ recursivamente à medida que novos pontos de dados $(\mathbf{x}_k, y_k)$ se tornam disponíveis, minimizando uma função de custo de mínimos quadrados lineares ponderados relacionada à matriz de informação de maneira sequencial. O RLS é particularmente útil em aplicações em tempo real onde os dados chegam sequencialmente e o modelo precisa de atualização contínua ou para modelar sistemas variantes no tempo (se o fator de esquecimento for incluído).

Por ser uma estimação recursiva, é útil relacionar $\hat{\Theta}_k$ a $\hat{\Theta}_{k-1}$. Em outras palavras, o novo $\hat{\Theta}_k$ depende do último valor estimado (k). Além disso, para estimar $\hat{\Theta}_k$, precisamos incorporar a informação atual presente em $y_k$.

Aguirre define o estimador de Mínimos Quadrados Recursivos com fator de esquecimento $\lambda$ como

$$
\left\{\begin{array}{c}
K_k= Q_k\psi_k = \frac{P_{k-1} \psi_k}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda} \\
\hat{\theta}_k=\hat{\theta}_{k-1}+K_k\left[y(k)-\psi_k^{\mathrm{T}} \hat{\theta}_{k-1}\right] \\
P_k=\frac{1}{\lambda}\left(P_{k-1}-\frac{P_{k-1} \psi_k \psi_k^{\mathrm{T}} P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda}\right)
\end{array}\right.
\tag{3.20}
$$

onde $K_k$ é o cálculo do vetor de ganho (também conhecido como ganho de Kalman), $P_k$ é a atualização da matriz de covariância, e $y_k - \mathbf{\Psi}_k^T \theta_{k-1}$ é o erro de estimação a priori. O fator de esquecimento $\lambda$ ($0 < \lambda \leq 1$) é geralmente definido entre $0.94$ e $0.99$. Se você definir $\lambda = 1$ você estará usando o algoritmo recursivo tradicional. A equação acima considera que o vetor de regressores $\psi(k-1)$ foi reescrito como $\psi_k$, já que este vetor é atualizado na iteração $k$ e contém informação até o instante de tempo $k-1$. Podemos inicializar a estimativa do parâmetro $\theta_0$ como

$$ \theta_0 = \mathbf{0} \tag{3.21}$$

e inicializar a inversa da matriz de covariância $\mathbf{P}_0$ com um valor grande:

$$ \mathbf{P}_0 = \frac{\mathbf{I}}{\delta} \tag{3.22}$$

onde $\delta$ é uma constante positiva pequena, e $\mathbf{I}$ é a matriz identidade.

O fator de esquecimento $\lambda$ controla quão rapidamente o algoritmo esquece dados passados:
- $\lambda = 1$ significa sem esquecimento, e todos os dados passados são igualmente ponderados.
- $\lambda < 1$ significa que quando novos dados estão disponíveis, todos os pesos são multiplicados por $\lambda$, o que pode ser interpretado como a razão entre pesos consecutivos para os mesmos dados.

Você pode acessar o código fonte para verificar como o SysIdentPy implementa o algoritmo RLS. O exemplo a seguir apresenta como você pode usá-lo no SysIdentPy.

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import RecursiveLeastSquares
from sysidentpy.utils import build_lagged_matrix
import matplotlib.pyplot as plt

x_train, x_test, y_train, y_test = get_siso_data(
    n = 1000, colored_noise = False, sigma = 0.001, train_percentage = 90
)

xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
estimator = RecursiveLeastSquares(lam=0.99)
theta = estimator.optimize(psi, y_train[max_lag:, :])
theta

[[-1.1778e-04]
 [ 1.9988e-01]
 [-9.3114e-05]
 [ 2.5119e-04]
 [ 9.0006e-01]
 [ 1.8339e-04]
 [-1.1943e-04]
 [ 9.9957e-02]
 [-4.6181e-05]
 [ 1.3155e-04]
 [ 3.4535e-04]
 [ 1.3843e-04]
 [-3.5454e-05]
 [ 1.5669e-04]
 [ 2.4311e-04]]
```

Você pode plotar a evolução dos parâmetros estimados ao longo do tempo acessando os valores de `theta_evolution`
```python
# plotting only the first 50 values
plt.plot(estimator.theta_evolution.T[:50, :])
plt.xlabel("iterations")
plt.ylabel("theta")
```
![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/theta_evolution.png?raw=true)
> Figura 1. Evolução dos parâmetros estimados ao longo do tempo usando o algoritmo RLS.

## Least Mean Squares

O filtro adaptativo Least Mean Squares (LMS) é um algoritmo de gradiente estocástico popular desenvolvido por Widrow e Hoff em 1960. O filtro adaptativo LMS visa alterar adaptativamente seus coeficientes de filtro para alcançar a melhor filtragem possível de um sinal. Isso é feito minimizando o erro entre o sinal desejado $d(n)$ e a saída do filtro $y(n)$. Podemos derivar o algoritmo LMS a partir da formulação RLS.

No RLS, o $\lambda$ está relacionado à minimização da soma dos quadrados ponderados da inovação

$$
J_k = \sum^k_{j=1}\lambda^{k-j}e^2_j.
\tag{3.23}
$$

O $Q_k$ na Equação 3.20, definido como

$$
Q_k = \frac{P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+\lambda} \\
\tag{3.24}
$$

é derivado da forma geral do algoritmo do Filtro de Kalman (KF).

$$
Q_k = \frac{P_{k-1}}{\psi_k^{\mathrm{T}} P_{k-1} \psi_k+v_0} \\
\tag{3.25}
$$

onde $v_0$ é a variância do ruído na definição do KF, na qual a função de custo é definida como a soma dos quadrados da inovação (ruído). Você pode verificar os detalhes em [Billings, S. A. - Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594).

Se alterarmos $Q_k$ na Equação 3.25 para uma matriz identidade escalada

$$
Q_k = \frac{\mu}{\Vert \psi_k \Vert^2}I
\tag{3.26}
$$

onde $\mu \in \mathbb{R}^+$, o $Q_k$ e $\hat{\theta}_k$ na Equação 3.20 se tornam

$$
\hat{\theta}_k=\hat{\theta}_{k-1}+\frac{\mu\left[y(k)-\psi_k^{\mathrm{T}} \hat{\theta}_{k-1}\right]}{\Vert \psi_k \Vert^2}\psi_k
\tag{3.27}
$$

onde $\psi_k^{\mathrm{T}} \hat{\theta}_{k-1} = \hat{y}_k$, que é conhecido como algoritmo LMS.

#### Convergência e Tamanho do Passo

O parâmetro de tamanho do passo $\mu$ desempenha um papel crucial no desempenho do algoritmo LMS. Se $\mu$ for muito grande, o algoritmo pode se tornar instável e falhar em convergir. Se $\mu$ for muito pequeno, o algoritmo convergirá lentamente. A escolha de $\mu$ é tipicamente:

$$
0 < \mu < \frac{2}{\lambda_{\max}}
\tag{3.28}
$$

onde $\lambda_{\max}$ é o maior autovalor da matriz de autocorrelação do sinal de entrada.

No SysIdentPy, você pode usar várias variantes do algoritmo LMS:

1. **LeastMeanSquareMixedNorm**
2. **LeastMeanSquares**
3. **LeastMeanSquaresFourth**
4. **LeastMeanSquaresLeaky**
5. **LeastMeanSquaresNormalizedLeaky**
6. **LeastMeanSquaresNormalizedSignRegressor**
7. **LeastMeanSquaresNormalizedSignSign**
8. **LeastMeanSquaresSignError**
9. **LeastMeanSquaresSignSign**
10. **AffineLeastMeanSquares**
11. **NormalizedLeastMeanSquares**
12. **NormalizedLeastMeanSquaresSignError**
13. **LeastMeanSquaresSignRegressor**

Para usar qualquer um dos métodos acima, você só precisa importá-lo e definir o `estimator` usando a opção desejada:

```python
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastMeanSquares
from sysidentpy.utils import build_lagged_matrix

x_train, x_test, y_train, y_test = get_siso_data(
    n = 1000, colored_noise = False, sigma = 0.001, train_percentage = 90
)

xlag = 2
ylag = 2
max_lag = 2
regressor_matrix = build_lagged_matrix(
    x=x_train, y=y_train, xlag=xlag, ylag=ylag, model_type="NARMAX",
)
# apply the basis function
psi = Polynomial(degree=2).fit(regressor_matrix, max_lag=max_lag, xlag=xlag, ylag=ylag)
estimator = LeastMeanSquares(mu=0.1)
theta = estimator.optimize(psi, y_train[max_lag:, :])
theta

[[ 1.5924e-04]
 [ 1.9950e-01]
 [ 3.2137e-04]
 [ 1.7824e-04]
 [ 8.9951e-01]
 [ 2.7314e-04]
 [ 3.3538e-04]
 [ 1.0062e-01]
 [ 3.5219e-04]
 [ 1.3544e-04]
 [ 3.4149e-04]
 [ 5.6315e-04]
 [-4.6664e-04]
 [ 2.2849e-04]
 [ 1.0536e-04]]
```

## Algoritmo Extended Least Squares

Vamos mostrar um exemplo do efeito de uma estimação de parâmetros viesada. Para simplificar, os dados são gerados simulando o seguinte modelo:

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-2} + e_{k}
$$

Neste caso, conhecemos os valores dos parâmetros verdadeiros, então será mais fácil entender como eles são afetados por uma estimação viesada. Os dados são gerados usando um método do SysIdentPy. Se *colored_noise* for definido como True no método, um ruído colorido é adicionado aos dados:

$$e_{k} = 0.8\nu_{k-1} + \nu_{k}$$

onde $x$ é uma variável aleatória uniformemente distribuída e $\nu$ é uma variável com distribuição gaussiana com $\mu=0$ e $\sigma$ é definido pelo usuário.

Vamos gerar dados com 1000 amostras com ruído branco e selecionar 90% dos dados para treinar o modelo.

```python
x_train, x_valid, y_train, y_valid = get_siso_data(
    n=1000, colored_noise=True, sigma=0.2, train_percentage=90
)
```

Primeiro, vamos treinar um modelo sem o Algoritmo Extended Least Squares para fins de comparação.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results

basis_function = Polynomial(degree=2)
estimator = LeastSquares(unbiased=False)
model = FROLS(
    order_selection=False,
    n_terms=3,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    estimator=estimator,
    basis_function=basis_function,
    err_tol=None,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)

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

| Regressors    | Parameters | ERR            |
| ------------- | ---------- | -------------- |
| x1(k-2)       | 9.0442E-01 | 7.55518391E-01 |
| y(k-1)        | 2.7405E-01 | 7.57565084E-02 |
| x1(k-1)y(k-1) | 9.8757E-02 | 3.12896171E-03 |

Claramente temos algo errado com o modelo obtido. Os parâmetros estimados diferem dos verdadeiros definidos na equação que gerou os dados. Como podemos observar acima, a estrutura do modelo é exatamente a mesma que gerou os dados. Você pode ver que o ERR ordenou os termos da maneira correta. E esta é uma nota importante sobre o algoritmo ERR: __ele é muito robusto ao ruído colorido!!__

Essa é uma ótima característica! No entanto, embora a estrutura esteja correta, os *parâmetros* do modelo não estão corretos! Aqui temos uma estimação viesada! Por exemplo, o parâmetro real para $y_{k-1}$ é $0.2$, não $0.274$.

Neste caso, estamos na verdade modelando usando um modelo NARX, não um NARMAX. A parte MA existe para permitir uma estimação não-viesada dos parâmetros. Para alcançar uma estimação não-viesada dos parâmetros, temos o algoritmo Extended Least Squares.

Antes de aplicar o Algoritmo Extended Least Squares, vamos executar vários modelos NARX para verificar quão diferentes os parâmetros estimados são dos reais.

```python
parameters = np.zeros([3, 50])
for i in range(50):
    x_train, x_valid, y_train, y_valid = get_siso_data(
        n=3000, colored_noise=True, train_percentage=90
    )
    model.fit(X=x_train, y=y_train)
    parameters[:, i] = model.theta.flatten()

# Set the theme for seaborn (optional)
sns.set_theme()
plt.figure(figsize=(14, 4))
# Plot KDE for each parameter
sns.kdeplot(parameters.T[:, 0], label='Parameter 1')
sns.kdeplot(parameters.T[:, 1], label='Parameter 2')
sns.kdeplot(parameters.T[:, 2], label='Parameter 3')
# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color='k', linestyle='--', label='Real Value 0.1')
plt.axvline(x=0.2, color='k', linestyle='--', label='Real Value 0.2')
plt.axvline(x=0.9, color='k', linestyle='--', label='Real Value 0.9')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Parameters')
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/biased_parameter.png?raw=true)
> Figura 2.: Estimativas de Densidade de Kernel (KDEs) dos parâmetros estimados obtidos de 50 realizações de modelos NARX, cada um ajustado a dados com ruído colorido. As linhas tracejadas verticais indicam os valores verdadeiros dos parâmetros usados para gerar os dados. Embora a estrutura do modelo seja identificada corretamente, os parâmetros estimados são viesados devido à omissão do componente de Média Móvel (MA), destacando a necessidade do algoritmo Extended Least Squares para alcançar uma estimação de parâmetros não-viesada.


Como mostrado na figura acima, temos um problema para estimar o parâmetro para $y_{k-1}$. Agora usaremos o Algoritmo Extended Least Squares. No SysIdentPy, basta definir `unbiased=True` na definição da estimação de parâmetros e o algoritmo ELS será aplicado.

```python
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
# Plot KDE for each parameter
sns.kdeplot(parameters.T[:, 0], label='Parameter 1')
sns.kdeplot(parameters.T[:, 1], label='Parameter 2')
sns.kdeplot(parameters.T[:, 2], label='Parameter 3')
# Plot vertical lines where the real values must lie
plt.axvline(x=0.1, color='k', linestyle='--', label='Real Value 0.1')
plt.axvline(x=0.2, color='k', linestyle='--', label='Real Value 0.2')
plt.axvline(x=0.9, color='k', linestyle='--', label='Real Value 0.9')
plt.xlabel('Parameter Value')
plt.ylabel('Density')
plt.title('Kernel Density Estimate of Parameters')
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/unbiased_estimator.png?raw=true)
> Figura 3. Estimativas de Densidade de Kernel (KDEs) dos parâmetros estimados obtidos de 50 modelos NARX usando o algoritmo Extended Least Squares (ELS) com estimação não-viesada. As linhas tracejadas verticais indicam os valores verdadeiros dos parâmetros usados para gerar os dados.

Diferente da estimação viesada anterior, estas KDEs na Figura 3 mostram que os parâmetros estimados estão agora intimamente alinhados com os valores verdadeiros, demonstrando a eficácia do algoritmo ELS em alcançar estimação de parâmetros não-viesada, mesmo na presença de ruído colorido.

> O algoritmo Extended Least Squares é iterativo por natureza. No SysIdentPy, o número padrão de iterações é definido como 30 (`uiter=30`). No entanto, a [literatura](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas) sugere que o algoritmo tipicamente converge rapidamente, frequentemente dentro de 10 a 20 iterações. Portanto, você pode querer testar diferentes números de iterações para encontrar o equilíbrio ideal entre velocidade de convergência e eficiência computacional.
