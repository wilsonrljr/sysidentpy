## Introdução

> Esta seção é baseada, em grande parte, na minha dissertação de mestrado, que por sua vez se baseia em [Billings, S. A](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594).

Selecionar a estrutura do modelo é um passo crucial para desenvolver modelos capazes de reproduzir corretamente o comportamento de um sistema. Se algumas informações prévias sobre o sistema forem conhecidas, por exemplo, a ordem dinâmica e o grau de não linearidade, determinar os termos e então estimar os parâmetros torna-se uma tarefa trivial. Em cenários reais, entretanto, geralmente não há informação sobre quais termos devem ser incluídos no modelo e os regressores corretos precisam ser selecionados no contexto de identificação. Se o processo de Model Structure Selection (MSS) não for realizado com o devido cuidado, a lei científica que descreve o sistema pode não ser revelada, resultando em interpretações equivocadas sobre o comportamento do sistema. Para ilustrar esse cenário, considere o seguinte exemplo.

Seja $\mathcal{D}$ um conjunto de dados arbitrário

$$
\begin{equation}
    \mathcal{D} = \{(x_k, y_k), k = 1, 2, \dotsc, n\},
\end{equation}
$$

em que $x_k \in \mathbb{R}^{n_x}$ e $y_k\in \mathbb{R}^{n_y}$ são, respectivamente, a entrada e a saída de um sistema desconhecido, e $n$ é o número de amostras no conjunto de dados. A seguir, considere dois modelos polinomiais NARX construídos para descrever esse sistema:

$$
\begin{align}
    y_{ak} &= 0.7077y_{ak-1} + 0.1642u_{k-1} + 0.1280u_{k-2}
\end{align}
\tag{2}
$$

$$
\begin{align}
    y_{bk} &= 0.7103y_{bk-1} + 0.1458u_{k-1} + 0.1631u_{k-2} \\
           &\quad - 1467y_{bk-1}^3 + 0.0710y_{bk-2}^3 + 0.0554y_{bk-3}^2u_{k-3}.
\end{align}
\tag{3}
$$

A Figura 1 mostra os valores previstos por cada modelo e os dados reais. Como pode ser observado, o modelo não linear 2 parece ajustar melhor os dados do que o modelo linear 1. O sistema original considerado é um circuito RLC, composto por um resistor (R), um indutor (L) e um capacitor (C) conectados em série com uma fonte de tensão. Sabe‑se que o comportamento de um circuito RLC em série pode ser descrito com precisão por uma equação diferencial linear de segunda ordem que relaciona a corrente $I(t)$ e a tensão aplicada $V(t)$:

$$
L\frac{d^2I(t)}{dt^2} + R\frac{dI(t)}{dt} + \frac{1}{C}I(t) = \frac{dV(t)}{dt}
\tag{4}
$$

Dada essa relação linear, um modelo adequado para o circuito RLC deve refletir essa linearidade de segunda ordem. Embora o Modelo 2, que inclui termos não lineares, possa fornecer um ajuste mais próximo aos dados, ele é claramente superparametrizado. Tal superparametrização pode introduzir efeitos não lineares espúrios, frequentemente chamados de "ghost nonlinearities", que não correspondem à dinâmica real do sistema. Portanto, esses modelos precisam ser interpretados com cautela, pois o uso de um modelo excessivamente complexo pode mascarar a verdadeira natureza linear do sistema e levar a conclusões equivocadas sobre o seu comportamento.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/rlc.png?raw=true)
> Figura 1. Resultados para dois modelos polinomiais NARX ajustados a dados de um sistema desconhecido. O Modelo 1 (esquerda) é linear, enquanto o Modelo 2 (direita) inclui termos não lineares. A figura ilustra que o Modelo 2 fornece um ajuste mais próximo aos dados em comparação com o Modelo 1. No entanto, como o sistema original é um circuito RLC linear de segunda ordem, o melhor ajuste do Modelo 2 pode ser enganoso devido à superparametrização. Isso destaca a importância de considerar as características físicas do sistema ao interpretar os resultados, para evitar interpretar como reais não linearidades artificiais. Referência: [Meta Model Structure Selection: An Algorithm For Building Polynomial NARX Models For Regression And Classification](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf)

Identificar corretamente a estrutura de um modelo é fundamental para analisar com precisão a dinâmica do sistema. Uma estrutura de modelo bem escolhida garante que o modelo reflita o comportamento real do sistema, permitindo uma análise consistente e significativa. Nesse sentido, diversos algoritmos foram desenvolvidos para selecionar os termos apropriados na construção de um modelo polinomial NARX. O objetivo principal dos algoritmos de Model Structure Selection (MSS) é revelar as características do sistema produzindo o modelo mais simples possível que ainda descreva adequadamente os dados. Embora alguns sistemas exijam, de fato, modelos mais complexos, é essencial buscar um equilíbrio entre simplicidade e acurácia. Como Einstein disse de forma bastante apropriada:

> Um modelo deve ser o mais simples possível, mas não mais simples do que isso.

Esse princípio enfatiza a importância de evitar complexidade desnecessária, garantindo ao mesmo tempo que o modelo capture as dinâmicas essenciais do sistema.

Vimos no Capítulo 2 que a seleção de regressores não é uma tarefa simples. Se o grau de não linearidade, a ordem do modelo e o número de entradas aumentam, o número de modelos candidatos se torna grande demais para uma abordagem de força bruta. Considerando o caso MIMO, esse problema é ainda mais severo do que no caso SISO quando muitas entradas e saídas são necessárias. O número de todos os modelos distintos pode ser calculado como

$$
\begin{align}
    n_m =
    \begin{cases}
    2^{n_r} & \text{para modelos SISO}, \\
    2^{n_{{_{m}}r}} & \text{para modelos MIMO},
    \end{cases}
\end{align}
\tag{5}
$$

em que $n_r$ e $n_{{_{m}}r}$ são os valores calculados usando as equações apresentadas no Capítulo 2.

Uma solução clássica para o problema de seleção de regressores é o algoritmo Forward Regression Orthogonal Least Squares (FROLS) associado ao algoritmo Error Reduction Ratio (ERR). Essa técnica é baseada no framework de Prediction Error Minimization e seleciona, termo a termo, o regressor mais relevante por meio de uma regressão step‑wise. O método FROLS adapta o conjunto de regressores no espaço de busca para um conjunto de vetores ortogonais, sobre os quais o ERR avalia a contribuição individual para a variância da saída desejada.

## O algoritmo Forward Regression Orthogonal Least Squares

Considere o modelo NARMAX geral definido na Equação 2.23, descrito de forma genérica como

$$
\begin{equation}
    y_k = \psi^\top_{k-1}\hat{\Theta} + \xi_k,
\end{equation}
\tag{6}
$$

em que $\psi^\top_{k-1} \in \mathbb{R}^{n_r \times n}$ é um vetor contendo combinações dos regressores e $\hat{\Theta} \in \mathbb{R}^{n_{\Theta}}$ é o vetor de parâmetros estimados. De forma mais compacta, o modelo NARMAX pode ser representado em forma matricial como:

$$
\begin{equation}
    y = \Psi\hat{\Theta} + \Xi,
\end{equation}
\tag{7}
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
\tag{8}
$$

Os parâmetros na equação acima podem ser estimados por um algoritmo baseado em Mínimos Quadrados, mas isso exigiria otimizar todos os parâmetros ao mesmo tempo devido à interação entre regressores causada pela falta de ortogonalidade. Consequentemente, a demanda computacional torna‑se impraticável para um número elevado de regressores. Nesse contexto, o FROLS transforma o modelo não ortogonal apresentado na equação acima em um modelo ortogonal.

A matriz de regressores $\Psi$ pode ser decomposta ortogonalmente como

$$
\begin{equation}
    \Psi = QA,
\end{equation}
\tag{9}
$$

em que $A \in \mathbb{R}^{n_{\Theta}\times n_{\Theta}}$ é uma matriz triangular superior unitária, dada por

$$
\begin{align}
A =
    \begin{bmatrix}
    1       & a_{12} & a_{13} & \dotsc & a_{1n_{\Theta}} \\
    0       &   1    & a_{23} & \dotsc & a_{2n_{\Theta}} \\
    0       &   0    &   1    & \dotsc &     \vdots       \\
    \vdots  & \vdots & \vdots & \ddots & a_{n_{\Theta}-1n_{\Theta}} \\
    0       &  0     &  0     &  0     & 1
    \end{bmatrix},
\end{align}
\tag{10}
$$

e $Q \in \mathbb{R}^{n\times n_{\Theta}}$ é uma matriz com colunas ortogonais $q_i$, descrita como

$$
\begin{equation}
    Q =
        \begin{bmatrix}
        q_{{_1}} & q_{{_2}} & q_{{_3}} & \dotsc & q_{{_{n_{\Theta}}}}
        \end{bmatrix},
\end{equation}
\tag{11}
$$

tal que $Q^\top Q = \Lambda$ e $\Lambda$ é diagonal com entradas $d_i$, que podem ser expressas como

$$
\begin{align}
    d_i = q_i^\top q_i = \sum^{k=1}_{n}q_{{_i}k}q_{{_i}k}, \qquad 1\leq i \leq n_{\Theta}.
\end{align}
$$

Como o subespaço gerado pela base ortogonal $Q$ (Equação 11) é o mesmo subespaço gerado pelo conjunto de base $\Psi$ (Equação 8) (isto é, contém todas as combinações lineares possíveis nesse subespaço), podemos reescrever a Equação 7 como

$$
\begin{equation}
    Y = \underbrace{(\Psi A^{-1})}_{Q}\underbrace{(A\Theta)}_{g}+ \Xi = Qg+\Xi,
\end{equation}
\tag{12}
$$

onde $g\in \mathbb{R}^{n_\Theta}$ é um vetor auxiliar de parâmetros. A solução do modelo descrito na Equação 12 é dada por

$$
\begin{equation}
    g = \left(Q^\top Q\right)^{-1}Q^\top Y = \Lambda^{-1}Q^\top Y
\end{equation}
\tag{13}
$$

ou, de forma equivalente,

$$
\begin{equation}
    g_{{_i}} = \frac{q_{{_i}}^\top Y}{q_{{_i}}^\top q_{{_i}}}.
\end{equation}
\tag{14}
$$

Como os vetores de parâmetros $\Theta$ e $g$ satisfazem o sistema triangular $A\Theta = g$, qualquer método de ortogonalização, como Householder, Gram‑Schmidt, Gram‑Schmidt modificado ou transformações de Givens, pode ser utilizado para resolver a equação e estimar os parâmetros originais. Assumindo que $E[\Psi^\top \Xi] = 0$, a variância da saída pode ser obtida multiplicando a Equação 12 por ela mesma e dividindo por $n$, o que resulta em

$$
\begin{equation}
    \frac{1}{n}Y^\top Y = \underbrace{\frac{1}{n}\sum^{i = 1}_{n_{\Theta}}g_{{_i}}^2q^\top_{{_i}}q_{{_i}}}_{\text{{variância explicada pelos regressores}}} + \underbrace{\frac{1}{n}\Xi^\top \Xi}_{\text{{variância não explicada}}}.
\end{equation}
\tag{15}
$$

Assim, o ERR associado à inclusão do regressor $q_{{_i}}$ pode ser expresso como

$$
[\text{ERR}]_i = \frac{g_{i}^2 \cdot q_{i}^\top q_{i}}{Y^\top Y}, \qquad \text{para } i=1,2,\dotsc, n_\Theta.
$$

Existem diversas formas de encerrar o algoritmo. Uma abordagem frequentemente utilizada é parar quando a variância da saída não explicada pelo modelo cai abaixo de um limite pré‑definido $\varepsilon$:

$$
\begin{equation}
    1 - \sum_{i = 1}^{n_{\Theta}}\text{ERR}_i \leq \varepsilon,
\end{equation}
\tag{17}
$$

### Mantendo as coisas simples

Para fins didáticos, vamos apresentar o FROLS usando exemplos simples, de forma a tornar a intuição por trás do método mais clara. Primeiro, definimos o cálculo do ERR e, em seguida, explicamos a ideia do FROLS em termos simples.

#### Caso ortogonal

Considere o caso em que temos um conjunto de entradas $x_1, x_2, \ldots, x_n$ e uma saída $y$. Suponha que esses vetores de entrada sejam ortogonais entre si.

Suponha que queremos construir um modelo para aproximar $y$ usando $x_1, x_2, \ldots, x_n$ da seguinte forma:

$$
y=\hat{\theta}_1 x_1+\hat{\theta}_2 x_2+\ldots+\hat{\theta}_n x_n+e
\tag{18}
$$

onde $\hat{\theta}_1, \hat{\theta}_2, \ldots, \hat{\theta}_n$ são parâmetros e $e$ é um ruído branco, independente de $x$ e $y$ (note a hipótese $E[\Psi^\top \Xi] = 0$ mencionada na seção anterior). Nesse caso, podemos reescrever a equação acima como

$$
y = \hat{\theta} x
\tag{19}
$$

de modo que

$$
\left\langle x, y\right\rangle = \left\langle \hat{\theta} x, x\right\rangle = \hat{\theta} \left\langle x, x\right\rangle
\tag{20}
$$

o que implica

$$
\hat{\theta} = \frac{\left\langle x, y\right\rangle}{\left\langle x, x\right\rangle}
\tag{21}
$$

Portanto, podemos mostrar que

$$
\begin{align}
& \left\langle x_1, y\right\rangle=\hat{\theta}_1\left\langle x_1, x_1\right\rangle \Rightarrow \hat{\theta}_1=\frac{\left\langle x_1, y\right\rangle}{\left\langle x_1, x_1\right\rangle}=\frac{x_1^T y}{x_1^T x_1} \\
& \left\langle x_2, y\right\rangle=\hat{\theta}_2\left\langle x_2, x_2\right\rangle \Rightarrow \hat{\theta}_2=\frac{\left\langle x_2, y\right\rangle}{\left\langle x_2, x_2\right\rangle}=\frac{x_2^T y}{x_2^T x_2}, \ldots \\
& \left\langle x_n, y\right\rangle=\hat{\theta}_n\left\langle x_n, x_n\right\rangle \Rightarrow \hat{\theta}_n=\frac{\left\langle x_n, y\right\rangle}{\left\langle x_n, x_n\right\rangle}=\frac{x_n^T y}{x_n^T x_n},
\end{align}
\tag{22}
$$


Seguindo a mesma ideia, também podemos mostrar que

$$
\langle y, y\rangle=\hat{\theta}_1^2\left\langle x_1, x_1\right\rangle+\hat{\theta}_2^2\left\langle x_2, x_2\right\rangle+\ldots+\hat{\theta}_n^2\left\langle x_n, x_n\right\rangle+\langle e, e\rangle
\tag{23}
$$

que pode ser descrito como

$$
y^T y=\hat{\theta}_1^2 x_1^T x_1+\hat{\theta}_2^2 x_2^T x_2+\ldots+\hat{\theta}_n^2 x_n^T x_n+e^T e
\tag{24}
$$

ou ainda

$$
\|y\|^2=\hat{\theta}_1^2\left\|x_1\right\|^2+\hat{\theta}_2^2\left\|x_2\right\|^2+\ldots+\hat{\theta}_n^2\left\|x_n\right\|^2+\|e\|^2
\tag{25}
$$

Dividindo ambos os lados da equação por $\|y\|^2$ e rearranjando, obtemos

$$
\frac{\|e\|^2}{\|y\|^2}=1-\hat{\theta}_1^2 \frac{\left\|x_1\right\|^2}{\|y\|^2}-\hat{\theta}_2^2 \frac{\left\|x_2\right\|^2}{\|y\|^2}-\ldots-\hat{\theta}_n^2 \frac{\left\|x_n\right\|^2}{\|y\|^2}
\tag{26}
$$

Como $\hat{\theta}_k=\frac{x_k^T y}{x_k^T x_k}=\frac{x_k^T y}{\left\|x_k\right\|^2}, k=1,2, . ., n$, temos

$$
\begin{align}
\frac{\|e\|^2}{\|y\|^2} & =1-\left(\frac{x_1^T y}{\left\|x_1\right\|^2}\right)^2 \frac{\left\|x_1\right\|^2}{\|y\|^2}-\left(\frac{x_2^T y}{\left\|x_2\right\|^2}\right)^2 \frac{\left\|x_2\right\|^2}{\|y\|^2}-\ldots-\left(\frac{x_n^T y}{\left\|x_n\right\|^2}\right)^2 \frac{\left\|x_n\right\|^2}{\|y\|^2} \\
& =1-\frac{\left(x_1^T y\right)^2}{\left\|x_1\right\|^2\| y \|^2}-\frac{\left(x_2^T y\right)^2}{\left\|x_2\right\|^2\|y\|^2}-\cdots-\frac{\left(x_n^T y\right)^2}{\left\|x_n\right\|^2\|y\|^2} \\
& =1-ERR_1 \quad-ERR_2-\cdots-ERR_n
\end{align}
\tag{27}
$$

onde $\operatorname{ERR}_k(k=1,2 \ldots, n)$ é o Error Reduction Ratio definido na seção anterior.

Veja o exemplo abaixo usando a base canônica:

```python
import numpy as np

y = np.array([3, 7, 8])
# Orthogonal Basis
x1 = np.array([1, 0, 0])
x2 = np.array([0, 1, 0])
x3 = np.array([0, 0, 1])

theta1 = (x1.T@y)/(x1.T@x1)
theta2 = (x2.T@y)/(x2.T@x2)
theta3 = (x3.T@y)/(x3.T@x3)

squared_y = y.T @ y
err1 = (x1.T@y)**2/((x1.T@x1) * squared_y)
err2 = (x2.T@y)**2/((x2.T@x2) * squared_y)
err3 = (x3.T@y)**2/((x3.T@x3) * squared_y)

print(f"x1 represents {round(err1*100, 2)}% of the variation in y, \n x2 represents {round(err2*100, 2)}% of the variation in y, \n x3 represents {round(err3*100, 2)}% of the variation in y")

x1 represents 7.38% of the variation in y,
x2 represents 40.16% of the variation in y,
x3 represents 52.46% of the variation in y
```

Vamos agora analisar o que acontece em um cenário não ortogonal.

```python
y = np.array([3, 7, 8])
x1 = np.array([1, 2, 2])
x2 = np.array([-1, 0, 2])
x3 = np.array([0, 0, 1])

theta1 = (x1.T@y)/(x1.T@x1)
theta2 = (x2.T@y)/(x2.T@x2)
theta3 = (x3.T@y)/(x3.T@x3)

squared_y = y.T @ y
err1 = (x1.T@y)**2/((x1.T@x1) * squared_y)
err2 = (x2.T@y)/((x2.T@x2) * squared_y)
err3 = (x3.T@y)**2/((x3.T@x3) * squared_y)

print(f"x1 represents {round(err1*100, 2)}% of the variation in y, \n x2 represents {round(err2*100, 2)}% of the variation in y, \n x3 represents {round(err3*100, 2)}% of the variation in y")

>>> x1 represents 99.18% of the variation in y,
>>> x2 represents 2.13% of the variation in y,
>>> x3 represents 52.46% of the variation in y
```

Nesse caso, $x1$ apresenta o maior valor de $err$, então o escolhemos como o primeiro vetor ortogonal.

```python
q1 = x1.copy()

v1 = x2 - (q1.T@x2)/(q1.T@q1)*q1
errv1 = (v1.T@y)**2/((v1.T@v1) * squared_y)

v2 = x3 - (q1.T@x3)/(q1.T@q1)*q1
errv2 = (v2.T@y)**2/((v2.T@v2) * squared_y)

print(f"v1 represents {round(errv1*100, 2)}% of the variation in y, \n v2 represents {round(errv2*100, 2)}% of the variation in y")

>>> v1 represents 0.82% of the variation in y,
>>> v2 represents 0.66% of the variation in y
```

Assim, neste caso, ao somarmos os valores de ERR dos dois primeiros vetores ortogonais, $x1$ e $v1$, obtemos $err_3 + errv1 = 100\%$. Não há necessidade de continuar a busca por mais termos: o modelo com esses dois termos já explica toda a variância dos dados.

> Essa é a ideia do algoritmo FROLS. Calculamos o ERR, escolhemos o vetor com maior ERR como o primeiro vetor ortogonal, ortogonalizamos todos os demais vetores em relação a ele, calculamos o ERR de cada um deles, escolhemos novamente aquele com maior ERR e repetimos o processo até que algum critério de parada seja satisfeito.

No SysIdentPy, temos dois hiperparâmetros chamados `n_terms` e `err_tol`. Ambos podem ser usados para interromper as iterações. O primeiro faz com que o algoritmo selecione até `n_terms` regressores. O segundo interrompe o processo quando $\sum ERR_i > err_{tol}$. Se ambos forem definidos, o algoritmo pára quando qualquer uma das condições for satisfeita.

```python
model = FROLS(
        n_terms=50,
        ylag=7,
        xlag=7,
        basis_function=basis_function,
        err_tol=0.98
    )
```

O SysIdentPy aplica o método de Golub‑Householder para a decomposição ortogonal. Uma discussão mais detalhada sobre Householder e procedimentos de ortogonalização em geral pode ser encontrada em [Chen, S. and Billings, S. A. and Luo, W.](https://www.tandfonline.com/doi/abs/10.1080/00207178908953472)

## Estudo de caso

Um exemplo usando dados reais será apresentado utilizando o SysIdentPy. Neste exemplo, construiremos modelos lineares e não lineares para descrever o comportamento de um motor de corrente contínua operando como gerador. Detalhes do experimento usado para gerar esses dados podem ser encontrados no artigo (em português) [IDENTIFICAÇÃO DE UM MOTOR/GERADOR CC POR MEIO DE MODELOS POLINOMIAIS AUTORREGRESSIVOS E REDES NEURAIS ARTIFICIAIS](https://www.researchgate.net/publication/320418710_Identificacao_de_um_motorgerador_CC_por_meio_de_modelos_polinomiais_autorregressivos_e_redes_neurais_artificiais)

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results

df1 = pd.read_csv("examples/datasets/x_cc.csv")
df2 = pd.read_csv("examples/datasets/y_cc.csv")

# checking the ouput
df2[5000:80000].plot(figsize=(10, 4))
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/generator_example.png?raw=true)
> Figura 2. Saída do sistema eletromecânico.


Neste exemplo, iremos dizimar os dados usando $d = 500$. A motivação para a dizimação é que os dados estão superamostrados devido à configuração experimental. Em uma seção futura discutiremos em mais detalhes como lidar com dados superamostrados no contexto de identificação de sistemas. Por ora, considere essa abordagem como a mais apropriada para este caso.

```python
x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)
```

Neste caso, construiremos um modelo NARX. No SysIdentPy, isso significa definir `unbiased=False` na classe `LeastSquares`. Usaremos uma função base `Polynomial` e definiremos o atraso máximo tanto para a entrada quanto para a saída como 2. Essa configuração resulta em 15 termos na matriz de informação; portanto, definiremos `n_terms=15`. Essa especificação é necessária porque, neste exemplo, `order_selection=False`. Discutiremos `order_selection` com mais detalhes na seção de Critérios de Informação.

> No SysIdentPy, `order_selection` é `True` por padrão. Quando `order_selection=False`, o usuário deve informar um valor para `n_terms`, pois esse hiperparâmetro é opcional e o valor padrão é `None`. Se definirmos `n_terms=5`, por exemplo, o FROLS irá parar após selecionar os primeiros 5 regressores. Não queremos isso neste caso, pois desejamos que o FROLS pare apenas quando `e_tol` for atingido.

```python
basis_function = Polynomial(degree=2)

model = FROLS(
	order_selection=False,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
    e_tol=0.9999
    n_terms=15
)
```

O SysIdentPy foi projetado para simplificar o uso de algoritmos como o `FROLS`. Construir, treinar ou ajustar um modelo é feito por meio de uma interface simples chamada `fit`. Ao utilizar esse método, todo o processo é tratado internamente, sem a necessidade de interação adicional do usuário.

```python
model.fit(X=x_train, y=y_train)
```

O SysIdentPy também oferece um método para recuperar informações detalhadas sobre o modelo ajustado. É possível inspecionar os termos incluídos no modelo, os parâmetros estimados, os valores de Error Reduction Ratio (ERR) e muito mais.

> Estamos usando `pandas` aqui apenas para tornar a saída mais legível, mas isso é opcional.

```python
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

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| y(k-1)         | 1.0998E+00  | 9.86000384E-01 |
| x1(k-1)^2      | 1.0165E+02  | 7.94805130E-03 |
| y(k-2)^2       | -1.9786E-05 | 2.50905908E-03 |
| x1(k-1)y(k-1)  | -1.2138E-01 | 1.43301039E-03 |
| y(k-2)         | -3.2621E-01 | 1.02781443E-03 |
| x1(k-1)y(k-2)  | 5.3596E-02  | 5.35200312E-04 |
| x1(k-2)        | 3.4655E+02  | 2.79648078E-04 |
| x1(k-2)y(k-1)  | -5.1647E-02 | 1.12211942E-04 |
| x1(k-2)x1(k-1) | -8.2162E+00 | 4.54743448E-05 |
| y(k-2)y(k-1)   | 4.0961E-05  | 3.25346101E-05 |
>Table 1

A tabela acima mostra que 10 regressores (de um total de 15 disponíveis) foram suficientes para atingir o `e_tol` definido, com a soma dos ERR para os regressores selecionados igual a $0.99992$.

Em seguida, vamos avaliar o desempenho do modelo usando os dados de validação. De forma análoga ao método `fit`, o SysIdentPy oferece o método `predict`. Para obter os valores previstos e plotar os resultados, basta fazer o seguinte:

```python
yhat = model.predict(X=x_valid, y=y_valid)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/generator_predict_c4.png?raw=true)
> Figura 3. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado.

## Critérios de Informação

Mencionamos que existem diversas maneiras de encerrar o algoritmo e selecionar os termos do modelo, mas até agora só definimos o critério baseado em ERR. Uma forma alternativa de parar o algoritmo é utilizando critérios de informação, por exemplo, o Akaike Information Criterion (AIC). Para regressão baseada em Mínimos Quadrados, o AIC indica o número de regressores ao minimizar a função objetivo ([Akaike, H. - A new look at the statistical model identification](https://ieeexplore.ieee.org/document/1100705)):

$$
\begin{equation}
    J_{\text{AIC}} = \underbrace{n\log\left(Var[\xi_k]\right)}_{\text{primeiro termo}}+\underbrace{2n_{\Theta}}_{\text{{segundo termo}}}.
\end{equation}
\tag{28}
$$

É importante notar que a equação acima ilustra o trade‑off entre ajuste do modelo e complexidade do modelo. Em particular, esse trade‑off envolve equilibrar a capacidade do modelo de ajustar bem os dados (primeiro termo) com a sua complexidade, relacionada ao número de parâmetros incluídos (segundo termo). À medida que mais termos são incluídos no modelo, o valor do AIC tende inicialmente a diminuir, atingindo um mínimo que representa um equilíbrio ótimo entre complexidade e desempenho preditivo. No entanto, se o número de parâmetros se tornar excessivo, a penalização por complexidade passa a superar o benefício de um melhor ajuste, fazendo com que o valor de AIC volte a crescer. O AIC e várias de suas variantes têm sido amplamente utilizados na identificação de sistemas lineares e não lineares. Ver, por exemplo, [Wei, H. and Zhu, D. and Billings, S. A. and Balikhin, M. A. - Forecasting the geomagnetic activity of the Dst index using multiscale radial basis function networks](https://www.sciencedirect.com/science/article/abs/pii/S0273117707002086), [Martins, S. A. M. and Nepomuceno, E. G. and Barroso, M. F. S. - Improved Structure Detection For Polynomial NARX Models Using a Multiobjective Error Reduction Ratio](https://link.springer.com/article/10.1007/s40313-013-0071-9), [Hafiz, F. and Swain, A. and Mendes, E. M. A. M. and Patel, N. - Structure Selection of Polynomial NARX Models Using Two Dimensional (2D) Particle Swarms](https://ieeexplore.ieee.org/document/8477782), [Gu, Y. and Wei, H. and Balikhin, M. M. - Nonlinear predictive model selection and model averaging using information criteria](https://www.tandfonline.com/doi/full/10.1080/21642583.2018.1496042) e referências aí citadas.


Apesar de bastante eficazes em muitos problemas de seleção de modelos lineares, critérios de informação como AIC podem ter dificuldade em selecionar um número adequado de parâmetros quando lidamos com sistemas fortemente não lineares. Além disso, esses critérios podem levar a modelos subótimos se o espaço de busca não contiver todos os termos necessários para representar adequadamente o modelo "verdadeiro". Consequentemente, em sistemas altamente não lineares ou quando componentes importantes do modelo estão ausentes, os critérios de informação podem não fornecer orientações confiáveis, resultando em modelos com desempenho ruim.

Além do AIC, o SysIdentPy disponibiliza outros quatro critérios de informação: [Bayesian Information Criteria](https://wires.onlinelibrary.wiley.com/doi/abs/10.1002/wics.199) (BIC), [Final Prediction Error](https://www.researchgate.net/publication/303679484_Introducao_a_Identificacao_de_Sistemas) (FPE), [Low of Iterated Logarithm Criteria](https://www.sciencedirect.com/science/article/abs/pii/S0169743902000515) (LILC) e [Corrected Akaike Information Criteria](https://www.sciencedirect.com/science/article/abs/pii/S0167715296001289) (AICc), definidos, respectivamente, como

$$
\begin{align}
\operatorname{FPE}\left(n_\theta\right) & =N \ln \left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right]+N \ln \left[\frac{N+n_\theta}{N-n_\theta}\right] \\
BI C\left(n_\theta\right) & =N \ln \left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right]+n_\theta \ln N \\
AICc &=AIC+2 n_p * \frac{n_p+1}{N-n_p-1} \\
LILC &= 2n_{\theta}\ln(\ln(N)) + N \ln(\left[\sigma_{\text {erro }}^2\left(n_\theta\right)\right])
\end{align}
\tag{29}
$$

Para usar qualquer critério de informação no SysIdentPy, defina `order_selection=True` (como mencionado, esse já é o valor padrão). Além de `order_selection`, você pode definir quantos regressores deseja avaliar antes de interromper o algoritmo por meio do hiperparâmetro `n_info_values`. O valor padrão é $15$, mas o usuário deve aumentá‑lo de acordo com o número de regressores disponíveis, dado por `ylag`, `xlag` e o grau da função base.

> O uso de critérios de informação pode ser computacionalmente custoso, dependendo do número de regressores avaliados e da quantidade de amostras. Para calcular o critério, o algoritmo ERR é executado `n` vezes, onde `n` é o valor definido em `n_info_values`. Certifique‑se de entender bem o funcionamento do método antes de decidir se realmente precisa utilizá‑lo.

Executando o mesmo exemplo, mas agora usando o critério de informação BIC para selecionar a ordem do modelo, temos:

```python
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function
)
model.fit(X=x_train, y=y_train)
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

| Regressors       | Parameters  | ERR           |
|------------------|-------------|---------------|
| y(k-1)           | 1.3666E+00  | 9.86000384E-01|
| x1(k-1)^2        | 1.0500E+02  | 7.94805130E-03|
| y(k-2)^2         | -5.8577E-05 | 2.50905908E-03|
| x1(k-1)y(k-1)    | -1.2427E-01 | 1.43301039E-03|
| y(k-2)           | -5.1414E-01 | 1.02781443E-03|
| x1(k-1)y(k-2)    | 5.3001E-02  | 5.35200312E-04|
| x1(k-2)          | 3.1144E+02  | 2.79648078E-04|
| x1(k-2)y(k-1)    | -4.8013E-02 | 1.12211942E-04|
| x1(k-2)x1(k-1)   | -8.0561E+00 | 4.54743448E-05|
| x1(k-2)y(k-2)    | 4.1381E-03  | 3.25346101E-05|
| 1                | -5.6653E+01 | 7.54107553E-06|
| y(k-2)y(k-1)     | 1.5679E-04  | 3.52002717E-06|
| y(k-1)^2         | -9.0164E-05 | 6.17373260E-06|
>Table 2

Nesse caso, em vez de 8 regressores, o modelo final possui 13 termos.

Atualmente, o número de regressores é determinado identificando o índice da última posição em que a diferença entre o valor atual do critério e o valor anterior é menor que 0. Para inspecionar esses valores, você pode utilizar a abordagem a seguir:

```python
xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dc_generator_aic_c4.png?raw=true)
> Figura 4. O gráfico mostra os valores do critério de informação (BIC) em função do número de termos incluídos no modelo. O processo de seleção de modelo, com base no BIC, adiciona regresssores iterativamente até que o BIC atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de BIC deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 13 termos.

A predição do modelo neste caso é mostrada na Figura 5.

```python
yhat = model.predict(X=x_valid, y=y_valid)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dc_generator_bic_c4.png?raw=true)
> Figura 5. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando o BIC.

### Visão geral dos métodos de Critério de Informação

Nesta seção, utilizamos dados simulados para fornecer ao leitor uma visão mais clara dos critérios de informação disponíveis no SysIdentPy.

> Aqui estamos trabalhando com uma estrutura de modelo conhecida, o que nos permite focar apenas em como os diferentes critérios de informação se comportam. Em dados reais, o número correto de termos do modelo é desconhecido, o que torna esses métodos ferramentas importantes para guiar a seleção de modelos.

> Se você observar as métricas reportadas abaixo, notará que o desempenho é excelente para todos os modelos. No entanto, é fundamental lembrar que Identificação de Sistemas não trata apenas de obter boas métricas de predição — o objetivo é encontrar a estrutura de modelo mais adequada. Model Structure Selection está no coração dos métodos NARMAX!

Os dados são gerados pela simulação do seguinte modelo:

$$
y_k = 0.2y_{k-1} + 0.1y_{k-1}x_{k-1} + 0.9x_{k-1} + e_k
\tag{30}
$$

Se `colored_noise` for definido como `True`, o termo de ruído é dado por:

$$
e_k = 0.8\nu_{k-1} + \nu_k
\tag{31}
$$

em que $x$ é uma variável aleatória uniformemente distribuída e $\nu$ é uma variável Gaussiana com $\mu = 0$ e $\sigma = 0.1$.

No próximo exemplo, geraremos dados com 100 amostras, usando ruído branco, e selecionaremos 70% dos dados para treinar o modelo.

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.generate_data import get_siso_data
from sysidentpy.utils.display_results import results


x_train, x_valid, y_train, y_valid = get_siso_data(
    n=100, colored_noise=False, sigma=0.1, train_percentage=70
)
```

A ideia aqui é mostrar o impacto dos critérios de informação na seleção do número de termos que compõem o modelo final. Você verá por que esses critérios são ferramentas auxiliares e por que deixar o algoritmo selecionar o número de termos apenas com base no valor mínimo do critério nem sempre é uma boa ideia quando lidamos com dados muito contaminados por ruído (mesmo ruído branco).

#### AIC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    ylag=2,
    xlag=2,
    info_criteria="aic",
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

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

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

Os regressores, a simulação em regime livre e os valores de AIC são detalhados a seguir.

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| x1(k-2)        | 9.4236E-01  | 9.26094341E-01 |
| y(k-1)         | 2.4933E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1)  | 1.3001E-01  | 2.35736200E-03 |
| x1(k-1)        | 8.4024E-02  | 4.11741791E-03 |
| x1(k-1)^2      | 7.0807E-02  | 2.54231877E-03 |
| x1(k-2)^2      | -9.1138E-02 | 1.39658893E-03 |
| y(k-1)^2       | 1.1698E-01  | 1.70257419E-03 |
| x1(k-2)y(k-2)  | 8.3745E-02  | 1.11056684E-03 |
| y(k-2)^2       | -4.1946E-02 | 1.01686239E-03 |
| x1(k-2)x1(k-1) | 5.9034E-02  | 7.47435512E-04 |
>Table 3

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predict_aic_c4.png?raw=true)
> Figura 5. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando AIC.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aic_c4.png?raw=true)
> Figura 6. O gráfico mostra os valores do critério de informação (AIC) em função do número de termos incluídos no modelo. O processo de seleção de modelo, baseado no AIC, adiciona regressores iterativamente até que o critério atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de AIC deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 10 termos.

Neste caso, obtemos um modelo com 10 termos. Sabemos, porém, que o número correto é 3, pois estamos trabalhando com um sistema simulado conhecido.

#### AICc

A única modificação necessária para usar AICc em vez de AIC é alterar o hiperparâmetro de critério de informação: `information_criteria="aicc"`.

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="aicc",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
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
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors       | Parameters  | ERR           |
|------------------|-------------|---------------|
| x1(k-2)          | 9.2282E-01  | 9.26094341E-01|
| y(k-1)           | 2.4294E-01  | 3.35898283E-02|
| x1(k-1)y(k-1)    | 1.2753E-01  | 2.35736200E-03|
| x1(k-1)          | 6.9597E-02  | 4.11741791E-03|
| x1(k-1)^2        | 7.0578E-02  | 2.54231877E-03|
| x1(k-2)^2        | -1.0523E-01 | 1.39658893E-03|
| y(k-1)^2         | 1.0949E-01  | 1.70257419E-03|
| x1(k-2)y(k-2)    | 7.1821E-02  | 1.11056684E-03|
| y(k-2)^2         | -3.9756E-02 | 1.01686239E-03|
>Table 4


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_aicc_c4.png?raw=true)
> Figura 7. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando AICc.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aicc_c4.png?raw=true)
> Figura 8. O gráfico mostra os valores do critério de informação (AICc) em função do número de termos incluídos no modelo. O processo de seleção de modelo, baseado no AICc, adiciona regressores iterativamente até que o critério atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de AICc deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 9 termos.

Desta vez, obtemos um modelo com 9 regressores.

#### BIC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
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
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors | Parameters | ERR            |
| ---------- | ---------- | -------------- |
| x1(k-2)    | 9.1726E-01 | 9.26094341E-01 |
| y(k-1)     | 1.8670E-01 | 3.35898283E-02 |
>Table 5


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_bic_c4.png?raw=true)
> Figura 9. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando BIC.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bic_c4.png?raw=true)
> Figura 10. O gráfico mostra os valores do critério de informação (BIC) em função do número de termos incluídos no modelo. O processo de seleção de modelo, baseado no BIC, adiciona regressores iterativamente até que o critério atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de BIC deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 2 termos.

O BIC retornou um modelo com apenas 2 regressores.

#### LILC

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="lilc",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
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
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors    | Parameters  | ERR            |
| ------------- | ----------- | -------------- |
| x1(k-2)       | 9.1160E-01  | 9.26094341E-01 |
| y(k-1)        | 2.3178E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1) | 1.2080E-01  | 2.35736200E-03 |
| x1(k-1)       | 6.3113E-02  | 4.11741791E-03 |
| x1(k-1)^2     | 5.4088E-02  | 2.54231877E-03 |
| x1(k-2)^2     | -9.0683E-02 | 1.39658893E-03 |
| y(k-1)^2      | 8.2157E-02  | 1.70257419E-03 |
>Table 6

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_lilc_c4.png?raw=true)
> Figura 11. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando LILC.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/lilc_c4.png?raw=true)
>Figura 12. O gráfico mostra os valores do critério de informação (LILC) em função do número de termos incluídos no modelo. O processo de seleção de modelo, baseado no LILC, adiciona regressores iterativamente até que o critério atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de LILC deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 7 termos.

O LILC retornou um modelo com 7 regressores.

#### FPE

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="fpe",
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
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
plot_results(y=y_valid, yhat=yhat, n=1000)

xaxis = np.arange(1, model.n_info_values + 1)
plt.plot(xaxis, model.info_values)
plt.xlabel("n_terms")
plt.ylabel("Information Criteria")
```

| Regressors     | Parameters  | ERR            |
| -------------- | ----------- | -------------- |
| x1(k-2)        | 9.4236E-01  | 9.26094341E-01 |
| y(k-1)         | 2.4933E-01  | 3.35898283E-02 |
| x1(k-1)y(k-1)  | 1.3001E-01  | 2.35736200E-03 |
| x1(k-1)        | 8.4024E-02  | 4.11741791E-03 |
| x1(k-1)^2      | 7.0807E-02  | 2.54231877E-03 |
| x1(k-2)^2      | -9.1138E-02 | 1.39658893E-03 |
| y(k-1)^2       | 1.1698E-01  | 1.70257419E-03 |
| x1(k-2)y(k-2)  | 8.3745E-02  | 1.11056684E-03 |
| y(k-2)^2       | -4.1946E-02 | 1.01686239E-03 |
| x1(k-2)x1(k-1) | 5.9034E-02  | 7.47435512E-04 |
>Table 7

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/predicted_fpe_c4.png?raw=true)
> Figura 13. Simulação em regime livre (ou previsão em passos infinitos à frente) do modelo ajustado utilizando FPE.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/fpe_c4.png?raw=true)
> Figura 14. O gráfico mostra os valores do critério de informação (FPE) em função do número de termos incluídos no modelo. O processo de seleção de modelo, baseado no FPE, adiciona regressores iterativamente até que o critério atinja um mínimo, indicando o balanço ótimo entre complexidade e ajuste. O ponto em que o valor de FPE deixa de diminuir define o número ótimo de termos, resultando em um modelo final com 10 termos.

O FPE retornou um modelo com 10 regressores.

## Meta Model Structure Selection (MetaMSS)

> Esta seção reflete em grande parte o conteúdo de um artigo que publiquei no [ArXiv](https://arxiv.org/abs/2109.09917), intitulado *"Meta-Model Structure Selection: Building Polynomial NARX Models for Regression and Classification."* Esse artigo foi inicialmente escrito para submissão em periódico, com base nos resultados da minha [dissertação de mestrado](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). No entanto, como passei a atuar como Data Scientist e considerando o longo processo de submissão em periódicos, decidi não seguir com a publicação formal neste momento. Assim, o artigo permanece disponível apenas no ArXiv.

> Esse trabalho estende um artigo anterior apresentado em um [congresso brasileiro](https://proceedings.science/sbai-2019/trabalhos/identificacao-de-sistemas-nao-lineares-utilizando-o-algoritmo-hibrido-e-binario?lang=pt-br), onde parte dos resultados foi inicialmente divulgada.

Esta seção apresenta uma abordagem meta‑heurística para seleção da estrutura de modelos polinomiais NARX em problemas de regressão. O método proposto considera simultaneamente a complexidade do modelo e a contribuição de cada termo para construir modelos parcimoniosos por meio de uma nova formulação de função de custo. A robustez do algoritmo é avaliada em diversos sistemas simulados e experimentais com diferentes características de não linearidade. Os resultados mostram que o algoritmo é capaz de identificar corretamente o modelo quando a estrutura verdadeira é conhecida e de produzir modelos parcimoniosos com dados experimentais, mesmo em casos em que métodos clássicos e contemporâneos frequentemente falham. A nova abordagem é comparada com métodos clássicos, como FROLS, e com técnicas recentes baseadas em amostragem aleatória.

Comentamos anteriormente que selecionar termos adequados de modelo é crucial para capturar corretamente a dinâmica do sistema original. Problemas como superparametrização e má condição numérica frequentemente surgem devido às limitações dos algoritmos de identificação existentes em selecionar os termos corretos para o modelo final. Ver, por exemplo, [Aguirre, L. A. and Billings, S. A. - Dynamical effects of overparametrization in nonlinear models](https://www.sciencedirect.com/science/article/abs/pii/0167278995900535), [Piroddi, L. and Spinelli, W. - An identification algorithm for polynomial NARX models based on simulation error minimization](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419). Também mencionamos que um dos algoritmos mais tradicionais para seleção de estrutura em modelos NARMAX polinomiais é o ERR. Diversas variantes do algoritmo FROLS foram desenvolvidas para melhorar o desempenho da seleção de modelo, como em [Billings, S. A., Chen, S., and Korenberg, M. J. - Identification of MIMO non-linear systems using a forward-regression orthogonal estimator](https://www.tandfonline.com/doi/abs/10.1080/00207178908559767), [Farina, M. and Piroddi, L. - Simulation Error Minimization–Based Identification of Polynomial Input–Output Recursive Models](https://www.sciencedirect.com/science/article/pii/S1474667016388462), [Guo, Y., Guo, L. Z., Billings, S. A., and Wei, H. - A New Iterative Orthogonal Forward Regression Algorithm](https://eprints.whiterose.ac.uk/107315/3/A%20New%20Iterative%20Orthogonal%20Forward%20Regression%20Algorithm%20-%20R2.pdf), [Mao, K. Z. and Billings, S. A. - VARIABLE SELECTION IN NON-LINEAR SYSTEMS MODELLING](https://www.sciencedirect.com/science/article/abs/pii/S0888327098901807). As limitações do FROLS foram amplamente discutidas na literatura, por exemplo, em [Billings, S. A. and Aguirre, L. A.](https://core.ac.uk/download/pdf/29031334.pdf), [Palumbo, P. and Piroddi, L.](https://ui.adsabs.harvard.edu/abs/2001JSV...239..405P/abstract), [Falsone, A., Piroddi, L., and Prandini, M.](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088). A maior parte desses pontos fracos está relacionada a (i) o framework de Prediction Error Minimization (PEM); (ii) a inadequação do índice ERR para medir a importância absoluta dos regressores; e (iii) o uso de critérios de informação como AIC, FPE e BIC para selecionar a ordem do modelo. Em relação aos critérios de informação, embora funcionem bem para modelos lineares, em um contexto não linear não é possível estabelecer uma relação simples entre tamanho do modelo e acurácia [Falsone, A., Piroddi, L., and Prandini, M. - A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088), [Chen, S., Hong, X., and Harris, C. J. - Sparse kernel regression modeling using combined locally regularized orthogonal least squares and D-optimality experimental design](https://ieeexplore.ieee.org/document/1205199).

Devido às limitações dos algoritmos baseados em Ordinary Least Squares (OLS), pesquisas recentes propuseram soluções que se afastam da abordagem clássica FROLS. Novos métodos reformularam o processo de Model Structure Selection (MSS) em um arcabouço probabilístico e passaram a empregar técnicas de amostragem aleatória [Falsone, A., Piroddi, L., and Prandini, M. - A randomized algorithm for nonlinear model structure selection](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088), [Tempo, R., Calafiore, G., and Dabbene, F. - Randomized Algorithms for Analysis and Control of Uncertain Systems: With Applications](https://link.springer.com/book/10.1007/978-1-4471-4610-0), [Baldacchino, T., Anderson, S. R., and Kadirkamanathan, V. - Computational system identification for Bayesian NARMAX modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063), [Rodriguez-Vazquez, K., Fonseca, C. M., and Fleming, P. J. - Identifying the structure of nonlinear dynamic systems using multiobjective genetic programming](https://ieeexplore.ieee.org/document/1306531), [Severino, A. G. V. and Araujo, F. M. U. de](https://repositorio.ufrn.br/bitstream/123456789/24900/1/AlcemyGabrielVitorSeverino_DISSERT.pdf). Apesar desses avanços, abordagens meta‑heurísticas e probabilísticas ainda apresentam algumas limitações. Em particular, muitas dependem de critérios de informação como AIC, FPE e BIC para definir a função de custo da otimização, o que frequentemente leva a modelos superparametrizados.

Considere $\mathcal{F}$ como uma classe de funções limitadas $\phi: \mathbf{R} \mapsto \mathbf{R}$. Se as propriedades de $\phi(x)$ satisfazem

$$
\begin{align}
    &\lim\limits_{x \to \infty} \phi(x) = \alpha \nonumber \\
    &\lim\limits_{x \to -\infty} \phi(x) = \beta \quad \text{com } \alpha > \beta,  \nonumber
\end{align}
\tag{32}
$$

então $\phi(x)$ é chamada de função sigmoide.

No caso particular considerado aqui, seguindo a definição da Equação 32 com $\alpha = 1$ e $\beta = 0$, podemos escrever uma curva em "S" como

$$
\begin{equation}
    \varsigma(x) = \frac{1}{1+e^{-a(x-c)}}.
\end{equation}
\tag{33}
$$

Nesse caso, podemos especificar o parâmetro $a$, que controla a taxa de variação. Se $a$ é próximo de zero, a função sigmoide é suave. Se $a$ é grande, a transição da sigmoide é abrupta. Se $a$ é negativo, a sigmoide varia de 1 para 0. O parâmetro $c$ corresponde ao valor de $x$ para o qual $y = 0.5$.

A Sigmoid Linear Unit Function (SiLU) é definida como o produto da função sigmoide pela sua entrada

$$
\begin{equation}
    \text{silu}(x) = x \varsigma(x),
\end{equation}
\tag{34}
$$

que pode ser vista como uma função sigmoide "mais íngreme" com overshoot.

### Metaheurísticas

Nas últimas duas décadas, algoritmos de otimização inspirados na natureza têm ganhado destaque devido à sua flexibilidade, simplicidade, versatilidade e capacidade de evitar mínimos locais em aplicações reais.

Algoritmos metaheurísticos são caracterizados por duas propriedades fundamentais: exploração e explotação [Blum, C. and Roli, A. - Metaheuristics in combinatorial optimization: Overview and conceptual comparison](https://dl.acm.org/doi/10.1145/937503.937505). **Exploitation** (exploração local) foca em utilizar informações locais para refinar a busca em torno da melhor solução atual, melhorando a qualidade das soluções vizinhas. Por outro lado, **exploration** (exploração global) procura investigar regiões mais amplas do espaço de busca, de modo a descobrir soluções potencialmente superiores e evitar que o algoritmo fique preso em mínimos locais.

Embora não haja consenso absoluto sobre as definições exatas de exploração e explotação em computação evolutiva, como discutido em [Eiben, Agoston E and Schippers, Cornelis A](https://www.researchgate.net/publication/220443407_On_Evolutionary_Exploration_and_Exploitation), existe um entendimento geral de que esses conceitos atuam como forças opostas e difíceis de equilibrar. Para lidar com esse desafio, metaheurísticas híbridas combinam múltiplos algoritmos, buscando tirar proveito tanto da exploração global quanto da explotação local, resultando em métodos de otimização mais robustos.

#### O algoritmo Binary hybrid Particle Swarm Optimization and Gravitational Search Algorithm (BPSOGSA)

Alcançar um bom equilíbrio entre exploração e explotação é um dos grandes desafios da maioria dos algoritmos metaheurísticos. Na abordagem considerada aqui, aumentamos o desempenho e a flexibilidade do processo de busca utilizando uma estratégia híbrida que combina Binary Particle Swarm Optimization (BPSO) com Gravitational Search Algorithm (GSA), conforme proposto em [Mirjalili, S. and Hashim, S. Z. M.](https://ieeexplore.ieee.org/abstract/document/6141614). Esse método híbrido incorpora uma técnica de coevolução heterogênea de baixo nível, originalmente introduzida por [Talbi, E. G.](https://link.springer.com/article/10.1023/A:1016540724870).

O BPSOGSA tira proveito das forças de ambos os algoritmos: a componente Particle Swarm Optimization (PSO) é especialmente eficiente em explorar o espaço de busca em escala global, buscando o ótimo global, enquanto a componente Gravitational Search Algorithm (GSA) é eficaz em refinar a busca em torno de soluções locais dentro de um espaço binário. Essa combinação visa fornecer uma estratégia de otimização mais abrangente e eficiente, garantindo um melhor equilíbrio entre exploração e explotação.

#### Particle Swarm Optimization (PSO) padrão

No Particle Swarm Optimization (PSO) [Kennedy, J. and Eberhart, R. C.](https://ieeexplore.ieee.org/document/488968), [Kennedy, J.](https://ieeexplore.ieee.org/document/488968), cada partícula representa uma solução candidata e é caracterizada por dois componentes: sua posição no espaço de busca, denotada por $\vec{x}_{\,np,d} \in \mathbb{R}^{np \times d}$, e sua velocidade, $\vec{v}_{\,np,d} \in \mathbb{R}^{np \times d}$. Aqui, $np = 1, 2, \ldots, n_a$, onde $n_a$ é o tamanho do enxame, e $d$ é a dimensionalidade do problema. A população inicial é representada por

$$
\vec{x}_{\,np,d} =
\begin{bmatrix}
x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\
x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\
\vdots  & \vdots  & \ddots & \vdots \\
x_{n_a,1} & x_{n_a,2} & \cdots & x_{n_a,d}
\end{bmatrix}
\tag{35}
$$

A cada iteração $t$, a posição e a velocidade de uma partícula são atualizadas pelas seguintes equações:

$$
v_{np,d}^{t+1} = \zeta v_{np,d}^{t} + c_1 \kappa_1 (pbest_{np}^{t} - x_{np,d}^{t})
+ c_2 \kappa_2 (gbest_{np}^{t} - x_{np,d}^{t}),
\tag{36}
$$

em que $\kappa_j \in \mathbb{R}$ para $j = [1,2]$ são variáveis aleatórias contínuas no intervalo $[0,1]$, $\zeta \in \mathbb{R}$ é o fator de inércia que controla a influência da velocidade anterior na velocidade atual (representando o trade‑off entre exploração e explotação), $c_1$ é o fator cognitivo associado à melhor posição pessoal $pbest$, e $c_2$ é o fator social associado à melhor posição global $gbest$. A velocidade $\vec{v}_{\,np,d}$ é tipicamente limitada ao intervalo $[v_{min}, v_{max}]$ para evitar que as partículas saiam do espaço de busca. A posição é então atualizada por

$$
x_{np,d}^{t+1} = x_{np,d}^{t} + v_{np,d}^{t+1}.
\tag{37}
$$

#### Gravitational Search Algorithm (GSA) padrão

No Gravitational Search Algorithm (GSA) [Rashedi, Esmat, Nezamabadi-Pour, Hossein, and Saryazdi, Saeid - GSA: A Gravitational Search Algorithm](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200), os agentes são representados por massas, cujo valor é proporcional ao fitness (valor da função objetivo) associado a cada agente. Essas massas interagem mediante forças gravitacionais, atraindo‑se mutuamente em direção a regiões mais promissoras do espaço de busca. Massas mais pesadas (agentes com melhor fitness) movem‑se mais lentamente, enquanto massas mais leves (agentes com pior fitness) tendem a se mover mais rapidamente. Cada massa no GSA possui quatro propriedades: posição, massa inercial, massa gravitacional ativa e massa gravitacional passiva. A posição de uma massa representa uma solução candidata, e suas massas gravitacional e inercial são derivadas da função de fitness.

Considere uma população de agentes descrita pelas seguintes equações. Em um instante de tempo $t$, a velocidade e a posição de cada agente são atualizadas como

$$
\begin{align}
    v_{i,d}^{t+1} &= \kappa_i \times v_{i,d}^t + a_{i,d}^t, \\
    x_{i,d}^{t+1} &= x_{i,d}^t + v_{i,d}^{t+1}.
\end{align}
\tag{38}
$$

Aqui, $\kappa_i$ introduz características estocásticas no processo de busca. A aceleração $a_{i,d}^t$ é calculada de acordo com a lei do movimento [Rashedi, Esmat and Nezamabadi-Pour, Hossein and Saryazdi, Saeid](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200):

$$
\begin{equation}
    a_{i,d}^t = \frac{F_{i,d}^t}{M_{ii}^{t}},
\end{equation}
\tag{39}
$$

em que $M_{ii}^{t}$ é a massa inercial do agente $i$ e $F_{i,d}^t$ representa a força gravitacional atuando sobre o agente $i$ na dimensão $d$. Os detalhes do cálculo de $F_{i,d}$ e $M_{ii}$ podem ser encontrados em [Rashedi, Esmat and Nezamabadi-Pour, Hossein and Saryazdi, Saeid](https://www.sciencedirect.com/science/article/abs/pii/S0020025509001200).

#### O algoritmo híbrido binário de otimização

A combinação dos algoritmos segue a formulação descrita em [Mirjalili, S. and Hashim, S. Z. M. - A new hybrid PSOGSA algorithm for function optimization](https://ieeexplore.ieee.org/abstract/document/6141614):

$$
\begin{align}
    v_{i}^{t+1} = \zeta \times v_i^t + \mathrm{c}'_{1} \times \kappa \times a_i^t + \mathrm{c}'_2 \times \kappa \times (gbest - x_i^t),
\end{align}
\tag{40}
$$

onde $\mathrm{c}'_j \in \mathbb{R}$ são coeficientes de aceleração. Essa formulação intensifica a fase de explotação ao incorporar a melhor posição encontrada até o momento. Por outro lado, essa mesma característica pode prejudicar a fase de exploração. Para contornar esse problema, [Mirjalili, S., Mirjalili, S. M., and Lewis, A. - Grey Wolf Optimizer](https://www.sciencedirect.com/science/article/abs/pii/S0965997813001853) propuseram valores adaptativos para $\mathrm{c}'_j$, conforme descrito em [Mirjalili, S., Wang, Gai-Ge, and Coelho, L. dos S. - Binary optimization using hybrid particle swarm optimization and gravitational search algorithm](https://dl.acm.org/doi/10.1007/s00521-014-1629-6):

$$
\begin{align}
    \mathrm{c}_1' &= -2 \times \frac{t^3}{\max(t)^3} + 2, \\
    \mathrm{c}_2' &= 2 \times \frac{t^3}{\max(t)^3} + 2.
\end{align}
\tag{41}
$$

Em cada iteração, as posições das partículas são atualizadas segundo as regras acima, sendo o espaço contínuo mapeado para soluções discretas por meio de uma função de transferência [Mirjalili, S. and Lewis, A. - S-shaped versus V-shaped transfer functions for binary Particle Swarm Optimization](https://www.sciencedirect.com/science/article/abs/pii/S2210650212000648):

$$
\begin{equation}
    S(v_{ik}) = \left|\frac{2}{\pi}\arctan\left(\frac{\pi}{2}v_{ik}\right)\right|.
\end{equation}
\tag{42}
$$

Com um número aleatório $\kappa \in (0,1)$ uniformemente distribuído, as posições dos agentes no espaço binário são atualizadas como

$$
\begin{equation}
    x_{np,d}^{t+1} =
    \begin{cases}
        (x_{np,d}^{t})^{-1}, & \text{se } \kappa < S(v_{ik}^{t+1}), \\
        x_{np,d}^{t}, & \text{se } \kappa \geq S(v_{ik}^{t+1}).
    \end{cases}
\end{equation}
\tag{43}
$$

### Meta-Model Structure Selection (MetaMSS): Construindo NARX para regressão

Nesta subseção, exploramos a abordagem meta‑heurística para seleção da estrutura de modelos NARX com BPSOGSA proposta na minha [dissertação de mestrado](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). A ideia é buscar, no espaço de decisão definido por um dicionário de regressores pré‑especificado, a estrutura de modelo que minimiza uma função de custo. A função objetivo considerada é baseada no root mean squared error (RMSE) da saída em regime livre (free‑run simulation), acrescida de um termo de penalização que leva em conta a complexidade do modelo e a contribuição de cada regressor.

#### Esquema de codificação

O processo de uso do BPSOGSA para seleção de estrutura envolve definir as dimensões da função de teste. Em particular, $n_y$, $n_x$ e $\ell$ são escolhidos de forma a cobrir todos os regressores possíveis, e uma matriz geral de regressores $\Psi$ é construída. O número de colunas de $\Psi$ é denotado por $noV$, e o número de agentes, por $N$. Uma matriz binária $noV \times N$, denotada $\mathcal{X}$, é então gerada aleatoriamente para representar a posição de cada agente no espaço de busca. Cada coluna de $\mathcal{X}$ representa uma solução candidata, isto é, uma estrutura de modelo a ser avaliada em cada iteração. Nessa matriz, um valor 1 indica que a coluna correspondente de $\Psi$ é incluída na matriz reduzida de regressores, enquanto um valor 0 indica a exclusão.

Como exemplo, considere o caso em que todos os regressores possíveis são definidos com $\ell = 1$ e $n_y = n_u = 2$. A matriz $\Psi$ é dada por

$$
\begin{align}
[ \text{constant} \quad y(k-1) \quad y(k-2) \quad u(k-1) \quad u(k-2) ]
\end{align}
\tag{44}
$$

Com 5 regressores possíveis, temos $noV = 5$. Supondo $N = 5$, a matriz $\mathcal{X}$ pode ser representada como

$$
\begin{equation}
    \mathcal{X} =
    \begin{bmatrix}
        0 & 1 & 0 & 0 & 0 \\
        1 & 1 & 1 & 0 & 1 \\
        0 & 0 & 1 & 1 & 0 \\
        0 & 1 & 0 & 0 & 1 \\
        1 & 0 & 1 & 1 & 0
    \end{bmatrix}
\end{equation}
\tag{45}
$$

Cada coluna de $\mathcal{X}$ é transposta para gerar uma solução candidata. Por exemplo, a primeira coluna resulta em

$$
\begin{equation*}
    \mathcal{X} =
    \begin{bmatrix}
        \text{constant} & y(k-1) & y(k-2) & u(k-1) & u(k-2) \\
        1 & 1 & 1 & 0 & 1
    \end{bmatrix}
\end{equation*}
\tag{46}
$$

Neste cenário, o primeiro modelo a ser avaliado é $\alpha y(k-1) + \beta u(k-2)$, cujos parâmetros $\alpha$ e $\beta$ podem ser estimados com qualquer método de estimação de parâmetros disponível. O mesmo procedimento é repetido para cada coluna de $\mathcal{X}$.

#### Formulação da função objetivo

Para cada estrutura de modelo candidata, o sistema linear nos parâmetros pode ser resolvido diretamente utilizando o algoritmo de Mínimos Quadrados (Least Squares) ou qualquer outro método disponível no SysIdentPy. A variância dos parâmetros estimados pode ser calculada como

$$
\hat{\sigma}^2 = \hat{\sigma}_e^2 V_{jj},
\tag{47}
$$

em que $\hat{\sigma}_e^2$ é a variância do erro estimada, dada por

$$
\hat{\sigma}_e^2 = \frac{1}{N-m} \sum_{k=1}^{N} (y_k - \psi_{k-1}^\top \hat{\Theta}),
\tag{48}
$$

e $V_{jj}$ é o $j$‑ésimo elemento da diagonal de $(\Psi^\top \Psi)^{-1}$.

O erro padrão estimado para o $j$‑ésimo coeficiente de regressão $\hat{\Theta}_j$ é dado pela raiz quadrada positiva dos elementos diagonais de $\hat{\sigma}^2$:

$$
\mathrm{se}(\hat{\Theta}_j) = \sqrt{\hat{\sigma}^2_{jj}}.
\tag{49}
$$

Para avaliar a relevância estatística de cada regressor, propõe‑se um teste de penalização baseado no erro padrão dos coeficientes de regressão. Neste caso, utilizamos o teste t (t‑student) para realizar um teste de hipótese sobre os coeficientes, avaliando a significância de cada regressor no modelo de regressão linear múltipla. As hipóteses consideradas são

$$
\begin{align*}
   H_0 &: \Theta_j = 0, \\
   H_a &: \Theta_j \neq 0.
\end{align*}
\tag{50}
$$

O valor da estatística de teste t é calculado como

$$
T_0 = \frac{\hat{\Theta}_j}{\mathrm{se}(\hat{\Theta}_j)},
\tag{51}
$$

que mede quantos desvios padrão $\hat{\Theta}_j$ está distante de zero. Mais precisamente, se

$$
-t_{\alpha/2, N-m} < T_0 < t_{\alpha/2, N-m},
\tag{52}
$$

onde $t_{\alpha/2, N-m}$ é o valor crítico da distribuição t para nível de significância $\alpha$ e $N-m$ graus de liberdade, então, se $T_0$ estiver fora dessa região de aceitação, rejeitamos a hipótese nula $H_0: \Theta_j = 0$. Isso implica que $\Theta_j$ é estatisticamente significativo ao nível $\alpha$. Caso contrário, se $T_0$ estiver dentro da região de aceitação, não rejeitamos $H_0$ e consideramos que $\Theta_j$ não é significativamente diferente de zero.

#### Valor de penalização baseado na derivada da Sigmoid Linear Unit

Propomos um valor de penalização baseado na derivada da função sigmoide, definida como

$$
\dot{\varsigma}(x(\varrho)) = \varsigma(x) [1 + (a(x - c))(1 - \varsigma(x))].
\tag{53}
$$

Nessa formulação, os parâmetros são definidos da seguinte forma: $x$ tem dimensão $noV$; $c = noV / 2$; e $a$ é definido como a razão entre o número de regressores do modelo em teste e $c$. Esse procedimento leva a uma curva específica para cada modelo, sendo que o declive da sigmoide se torna mais acentuado à medida que o número de regressores aumenta. O valor de penalização $\varrho$ corresponde ao valor de $y$ da curva sigmoide para o número de regressores considerado em $x$. Como a derivada da função sigmoide pode assumir valores negativos, normalizamos $\varsigma$ como

$$
\varrho = \varsigma - \mathrm{min}(\varsigma),
\tag{54}
$$

garantindo que $\varrho \in \mathbb{R}^{+}$.

Note que dois modelos distintos com o mesmo número de regressores podem apresentar desempenhos bastante diferentes, devido à importância relativa de cada termo. Para lidar com isso, incorporamos o teste t‑student na penalização para quantificar a relevância estatística dos regressores. Mais especificamente, calculamos $n_{\Theta, H_{0}}$, o número de regressores considerados não significativos para o modelo. O valor de penalização é então ajustado com base no tamanho efetivo do modelo:

$$
\mathrm{model\_size} = n_{\Theta} + n_{\Theta, H_{0}}.
\tag{55}
$$

A função objetivo que combina o root mean squared error relativo com o termo de penalização $\varrho$ é definida como

$$
\mathcal{F} = \frac{\sqrt{\sum_{k=1}^{n} (y_k - \hat{y}_k)^2}}{\sqrt{\sum_{k=1}^{n} (y_k - \bar{y})^2}} \times \varrho.
\tag{56}
$$

Essa abordagem garante que, mesmo para modelos com o mesmo número de regressores, aqueles contendo termos redundantes sejam mais penalizados.

#### Estudos de caso: resultados de simulação

Nesta subseção, apresentamos seis exemplos de simulação para ilustrar a eficácia do algoritmo MetaMSS. Uma análise detalhada do desempenho do algoritmo é realizada considerando diferentes configurações de parâmetros. Os sistemas selecionados são amplamente utilizados como benchmarks em problemas de seleção de estrutura de modelos e foram extraídos de [Wei, H. and Billings, S. A., "Model structure selection using an integrated forward orthogonal search algorithm assisted by squared correlation and mutual information"](https://www.inderscienceonline.com/doi/abs/10.1504/IJMIC.2008.020543), [Falsone, A. and Piroddi, L. and Prandini, M., "A randomized algorithm for nonlinear model structure selection"](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088), [Baldacchino, T. and Anderson, S. R. and Kadirkamanathan, V., "Computational system identification for Bayesian NARMAX modelling"](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063), [Piroddi, L. and Spinelli, W., "An identification algorithm for polynomial NARX models based on simulation error minimization"](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419), [Guo, Y. and Guo, L. Z. and Billings, S. A. and Wei, H., "A New Iterative Orthogonal Forward Regression Algorithm"](https://eprints.whiterose.ac.uk/107315/3/A%20New%20Iterative%20Orthogonal%20Forward%20Regression%20Algorithm%20-%20R2.pdf), [Bonin, M. and Seghezza, V. and Piroddi, L., "NARX model selection based on simulation error minimization and LASSO"](https://www.researchgate.net/publication/224153379_NARX_model_selection_based_on_simulation_error_minimisation_and_LASSO) e [Aguirre, L. A. and Barbosa, B. H. G. and Braga, A. P., "Prediction and simulation errors in parameter estimation for nonlinear systems"](https://www.sciencedirect.com/science/article/abs/pii/S0888327010001469). Finalmente, realizamos uma análise comparativa com o [Randomized Model Structure Selection (RaMSS), "A randomized algorithm for nonlinear model structure selection"](https://www.sciencedirect.com/science/article/abs/pii/S0005109815003088), o FROLS e o algoritmo [Reversible-jump Markov chain Monte Carlo (RJMCMC), "Computational system identification for Bayesian NARMAX modelling"](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063) para avaliar a eficácia do método proposto.

Os modelos de simulação são definidos como

$$
\begin{align}
    & S_1: \quad y_k = -1.7y_{k-1} - 0.8y_{k-2} + x_{k-1} + 0.81x_{k-2} + e_k, \\
    & \qquad \quad \text{com } x_k \sim \mathcal{U}(-2, 2) \text{ e } e_k \sim \mathcal{N}(0, 0.01^2); \\ 
    & S_2: \quad y_k = 0.8y_{k-1} + 0.4x_{k-1} + 0.4x_{k-1}^2 + 0.4x_{k-1}^3 + e_k, \\
    & \qquad \quad \text{com } x_k \sim \mathcal{N}(0, 0.3^2) \text{ e } e_k \sim \mathcal{N}(0, 0.01^2). \\ 
    & S_3: \quad y_k = 0.2y_{k-1}^3 + 0.7y_{k-1}x_{k-1} + 0.6x_{k-2}^2 \\
    &- 0.7y_{k-2}x_{k-2}^2 -0.5y_{k-2}+ e_k, \\
    & \qquad \quad \text{com } x_k \sim \mathcal{U}(-1, 1) \text{ e } e_k \sim \mathcal{N}(0, 0.01^2). \\ 
    & S_4: \quad y_k = 0.7y_{k-1}x_{k-1} - 0.5y_{k-2} + 0.6x_{k-2}^2 \\
    &- 0.7y_{k-2}x_{k-2}^2 + e_k, \\
    & \qquad \quad \text{com } x_k \sim \mathcal{U}(-1, 1) \text{ e } e_k \sim \mathcal{N}(0, 0.04^2). \\ 
    & S_5: \quad y_k = 0.7y_{k-1}x_{k-1} - 0.5y_{k-2} + 0.6x_{k-2}^2 \\
    &- 0.7y_{k-2}x_{k-2}^2 + 0.2e_{k-1} \\
    & \qquad \quad - 0.3x_{k-1}e_{k-2} + e_k,\\
    & \qquad \quad \text{com } x_k \sim \mathcal{U}(-1, 1) \text{ e } e_k \sim \mathcal{N}(0, 0.02^2); \\ 
    & S_6: \quad y_k = 0.75y_{k-2} + 0.25x_{k-2} - 0.2y_{k-2}x_{k-2} + e_k \\
    & \qquad \quad \text{com } x_k \sim \mathcal{N}(0, 0.25^2) \text{ e } e_k \sim \mathcal{N}(0, 0.02^2); 
\end{align}
\tag{57}
$$

em que $\mathcal{U}(a, b)$ denota amostras uniformemente distribuídas em $[a, b]$ e $\mathcal{N}(\eta, \sigma^2)$ denota amostras com distribuição Gaussiana de média $\eta$ e desvio padrão $\sigma$. Todas as realizações dos sistemas são compostas por 500 amostras de entrada e saída. Além disso, a mesma semente aleatória é utilizada para garantir reprodutibilidade.

Todos os resultados apresentados nesta subseção são baseados na implementação original e foram extraídos da minha dissertação de mestrado. Na época, o algoritmo foi implementado em Matlab $2018$a, executado em um Dell Inspiron $5448$ Core i$5-5200$U CPU $2.20$GHz com $12$GB de RAM. No entanto, não é difícil adaptar o código para o SysIdentPy.

Seguindo os estudos mencionados, escolhemos lags máximos $n_u=n_y=4$ para entrada e saída e grau de não linearidade $\ell = 3$. Os parâmetros relacionados ao BPSOGSA são detalhados na Tabela 8.

| Parameters | $n_u$ | $n_y$ | $\ell$ | p-value | max\_iter | n\_agents | $\alpha$ | $G_0$ |
|------------|-------|-------|--------|---------|-----------|-----------|----------|-------|
| Values     | $4$   | $4$   | $3$    | $0.05$  | $30$      | $10$      | $23$     | $100$ |
>Table 8. Parâmetros usados no MetaMSS

Foram realizadas 300 execuções do algoritmo MetaMSS para cada modelo, com o objetivo de comparar estatísticas sobre o desempenho do método. O tempo de execução (elapsed time) e a taxa de acerto (correctness), isto é, a porcentagem de vezes em que a estrutura correta foi selecionada, foram analisados.

Os resultados na Tabela 9 foram obtidos com os parâmetros configurados de acordo com a Tabela 8.

|                     | $S_1$ | $S_2$ | $S_3$ | $S_4$ | $S_5$ | $S_6$ |
| ------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| Correct model       | 100\% | 100\% | 100\% | 100\% | 100\% | 100\% |
| Elapsed time (mean) | 5.16s | 3.90s | 3.40s | 2.37s | 1.40s | 3.80s |
>Table 9. Desempenho geral do MetaMSS

A Tabela 9 mostra que todos os termos dos modelos foram corretamente selecionados usando o MetaMSS. Vale destacar que até mesmo o modelo $S_5$, que possui ruído autoregressivo, foi corretamente identificado pelo algoritmo. Esse resultado se deve ao fato de que todos os regressores são avaliados individualmente, e aqueles considerados redundantes são removidos do modelo.

A Figura 15 apresenta a convergência de cada execução do MetaMSS. Observa‑se que a maioria das execuções converge para a estrutura correta com 10 ou menos iterações. Isso está relacionado ao número máximo de iterações e ao número de agentes de busca. O primeiro parâmetro influencia diretamente os coeficientes de aceleração do algoritmo, que reforçam a fase de exploração, enquanto o segundo aumenta o número de modelos candidatos avaliados. Intuitivamente, ambos influenciam o tempo de execução e, ainda mais importante, a estrutura de modelo selecionada como solução final. Assim, escolhas inadequadas desses parâmetros podem levar à seleção de modelos sub ou superparametrizados, uma vez que o algoritmo pode convergir para um ótimo local. A subseção a seguir apresenta uma análise da influência de `max_iter` e `n_agents` no desempenho do algoritmo.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_convergence.png?raw=true)
> Figura 15. Convergência do MetaMSS para diferentes estruturas de modelo. Cada curva representa a trajetória de convergência do algoritmo para uma estrutura específica, de $S_1$ a $S_6$, ao longo de, no máximo, 30 iterações.

#### Influência dos parâmetros $max\_iter$ e $n\_agents$

Os modelos simulados são usados para avaliar o desempenho do MetaMSS considerando diferentes configurações para `max_iter` e `n_agents`. Primeiro, fixamos `max_iter=30` e variamos `n_agents`. Em seguida, fixamos `n_agents` e variamos `max_iter`. Os resultados apresentados nesta subseção foram obtidos com os demais parâmetros configurados conforme a Tabela 8.

|                                    |                         | $S_1$     | $S_2$     | $S_3$   | $S_4$     | $S_5$     | $S_6$     |
| ---------------------------------- | ----------------------- | --------- | --------- | ------- | --------- | --------- | --------- |
| **max\_iter = 30, n\_agents = 1**  | **Correct model**       | $65\%$    | $55.66\%$ | $14\%$  | $14\%$    | $7.3\%$   | $20.66\%$ |
|                                    | **Elapsed time (mean)** | $0.26$s   | $0.19$s   | $0.15$s | $0.11$s   | $0.13$s   | $0.13$s   |
| **max\_iter = 30, n\_agents = 5**  | **Correct model**       | $100\%$   | $100\%$   | $99\%$  | $98\%$    | $91.66\%$ | $98.33\%$ |
|                                    | **Elapsed time (mean)** | $2.08$s   | $1.51$s   | $1.41$s | $0.99$s   | $0.59$s   | $1.13$s   |
| **max\_iter = 30, n\_agents = 20** | **Correct model**       | $100\%$   | $100\%$   | $100\%$ | $100\%$   | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $12.88$s  | $9.10$s   | $8.77$s | $5.70$s   | $3.37$s   | $9.50$s   |
| **max\_iter = 5, n\_agents = 10**  | **Correct model**       | $96.33\%$ | $99\%$    | $86\%$  | $93.66\%$ | $93\%$    | $97.33\%$ |
|                                    | **Elapsed time (mean)** | $0.92$s   | $0.73$s   | $0.72$s | $0.52$s   | $0.29$s   | $0.64$s   |
| **max\_iter = 15, n\_agents = 10** | **Correct model**       | $100\%$   | $100\%$   | $99\%$  | $99\%$    | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $2.80$s   | $2.33$s   | $2.25$s | $1.60$s   | $0.90$s   | $2.30$s   |
| **max\_iter = 50, n\_agents = 10** | **Correct model**       | $100\%$   | $100\%$   | $100\%$ | $100\%$   | $100\%$   | $100\%$   |
|                                    | **Elapsed time (mean)** | $7.38$s   | $5.44$s   | $4.56$s | $3.01$s   | $2.10$s   | $4.52$s   |
>Table 10.

Os resultados agregados da Tabela 10 confirmam o comportamento esperado em relação ao tempo de execução e à taxa de acerto. Ambos aumentam significativamente à medida que o número de agentes e o número máximo de iterações crescem. O número de agentes é particularmente relevante, pois amplia a capacidade de exploração do espaço de busca. Todos os sistemas são impactados pelo aumento no número de agentes e no máximo de iterações.

Observando todos os sistemas testados, fica claro que uma exploração mais ampla tem impacto significativo na exatidão da seleção de modelos. Quando poucos agentes são utilizados, o desempenho do MetaMSS se deteriora de forma importante, especialmente para os sistemas $S_3$, $S_4$ e $S_5$. O número máximo de iterações, por sua vez, permite que os agentes explorem, de forma global e local, regiões em torno dos modelos candidatos já testados. Assim, quanto maior o número de iterações, mais o algoritmo pode explorar o espaço e examinar diferentes conjuntos de regressores.

Se esses parâmetros forem escolhidos de forma inadequada, o algoritmo pode não ser capaz de encontrar a estrutura ideal. Nesse sentido, os resultados apresentados aqui se referem apenas aos sistemas analisados. Quanto maior o espaço de busca, maior precisará ser o número de agentes e o número de iterações. Embora o esforço computacional aumente com valores grandes de `n_agents` e `max_iteration`, o algoritmo permanece bastante eficiente em termos de tempo de execução para todas as configurações que garantem a seleção das estruturas verdadeiras.

#### Seleção de modelos super e sub‑parametrizados

Mesmo diante do sucesso na seleção das estruturas de todos os modelos pelo MetaMSS, é natural perguntar como os modelos selecionados diferem do modelo verdadeiro nos casos apresentados na Tabela 10 em que o algoritmo não garantiu 100\% de acerto. A Figura 16 ilustra a distribuição do número de termos selecionados em cada caso. É evidente que o número de modelos superparametrizados é, em geral, maior do que o de modelos subparametrizados. Nos casos em que o número de agentes é baixo, devido à baixa capacidade de exploração e explotação, o algoritmo tende a convergir prematuramente, resultando em modelos com muitos regressores espúrios. Em particular, para $S_2$ e $S_5$ com `n_agents=1`, o algoritmo selecionou modelos com mais de 20 termos. Pode‑se argumentar que esse é um cenário extremo usado apenas para fins de comparação, mas a escolha adequada dos parâmetros está intrinsecamente ligada à dimensão do espaço de busca. Para casos com `n_agents`$\geq 5$, por exemplo, o número de termos espúrios diminui significativamente quando o algoritmo não consegue selecionar o modelo verdadeiro.

Além disso, é importante destacar a relevância de uma boa calibração dos parâmetros, já que as fases de exploração e explotação dependem fortemente deles. Uma convergência prematura pode levar à seleção de modelos com o número correto de termos, mas com regressores errados. Isso aconteceu em todos os casos com `n_agents=1`. Por exemplo, para $S_3$, o algoritmo produziu modelos com o número correto de termos em 33.33\% das execuções, mas a Tabela 10 mostra que apenas 14\% desses modelos são, de fato, equivalentes à estrutura verdadeira.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_terms_distribution.png?raw=true)
> Figura 16. Distribuição do número de termos selecionados em cada modelo simulado, considerando diferentes configurações de `max_iter` e `n_agents`.

Os sistemas $S_1$, $S_2$, $S_3$, $S_4$ e $S_6$ foram utilizados como benchmark por [Bianchi, F., Falsone, A., Prandini, M. and Piroddi, L.](https://www.tandfonline.com/doi/abs/10.1080/00207721.2016.1244309), o que permite comparar diretamente nossos resultados com aqueles reportados na literatura. Todas as técnicas utilizaram $n_y=n_u=4$ e $\ell = 3$. O RaMSS e o RaMSS com Conditional Linear Family (C-RaMSS) foram configurados com: $K=1$, $\alpha = 0.997$, $NP = 200$ e $v=0.1$. O MetaMSS foi ajustado conforme os parâmetros indicados na Tabela 8.

|                     |                         | $S_1$     | $S_2$    | $S_3$    | $S_4$    | $S_6$    |
| ------------------- | ----------------------- | --------- | -------- | -------- | -------- | -------- |
| **Meta-MSS**        | **Correct model**       | $100\%$   | $100\%$  | $100\%$  | $100\%$  | $100\%$  |
|                     | **Elapsed time (mean)** | $5.16$s   | $3.90$s  | $3.40$s  | $2.37$s  | $3.80$s  |
| **RaMSS- $NP=100$** | **Correct model**       | $90.33\%$ | $100\%$  | $100\%$  | $100\%$  | $66\%$   |
|                     | **Elapsed time (mean)** | $3.27$s   | $1.24$s  | $2.59$s  | $1.67$s  | $6.66$s  |
| **RaMSS- $NP=200$** | **Correct model**       | $78.33\%$ | $100\%$  | $100\%$  | $100\%$  | $82\%$   |
|                     | **Elapsed time (mean)** | $6.25$s   | $2.07$s  | $4.42$s  | $2.77$s  | $9.16$s  |
| **C-RaMSS**         | **Correct model**       | $93.33\%$ | $100\%$  | $100\%$  | $100\%$  | $100\%$  |
|                     | **Elapsed time (mean)** | $18$s     | $10.50$s | $16.96$s | $10.56$s | $48.52$s |
> Table 11. Análise comparativa entre MetaMSS, RaMSS e C-RaMSS

Em termos de taxa de acerto, o MetaMSS supera (ou pelo menos iguala) o RaMSS e o C-RaMSS para todos os sistemas analisados, como mostrado na Tabela 11. Para o sistema $S_6$, por exemplo, a taxa de acerto aumenta em 18\% em relação ao RaMSS, enquanto o tempo de execução necessário para que o C-RaMSS atinja 100\% de acerto é 1276.84\% maior do que o do MetaMSS. Além disso, o MetaMSS é visivelmente mais eficiente do que o C-RaMSS e possui desempenho computacional semelhante ao RaMSS.

#### MetaMSS vs FROLS

O algoritmo FROLS foi aplicado a todos os sistemas testados, com os resultados resumidos na Tabela 12. O método foi capaz de selecionar corretamente a estrutura para $S_2$ e $S_6$. No entanto, falhou em identificar dois dos quatro regressores de $S_1$. Para $S_3$, o FROLS incluiu $y_{k-1}$ no lugar do termo correto $y_{k-1}^3$. De forma semelhante, em $S_4$, o termo $y_{k-4}$ foi selecionado em vez de $y_{k-2}$. Para $S_5$, o algoritmo resultou em uma estrutura incorreta ao incluir o termo espúrio $y_{k-4}$.

|                       | Meta-MSS Regressor  | Correct | FROLS Regressor  | Correct |
|-----------------------|--------------------|---------|------------------|---------|
| **$S_1$**             | $y_{k-1}$          | yes     | $y_{k-1}$        | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $x_{k-2}$          | yes     | $x_{k-4}$        | no      |
| **$S_2$**             | $y_{k-1}$          | yes     | $y_{k-1}$        | yes     |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $x_{k-1}^2$        | yes     | $x_{k-1}^2$      | yes     |
|                       | $x_{k-1}^3$        | yes     | $x_{k-1}^3$      | yes     |
| **$S_3$**             | $y_{k-1}^3$        | yes     | $y_{k-1}$        | no      |
|                       | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
|                       | $y_{k-2}$          | yes     | $y_{k-2}$        | yes     |
| **$S_4$**             | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
| **$S_5$**             | $y_{k-1}x_{k-1}$   | yes     | $y_{k-1}x_{k-1}$ | yes     |
|                       | $y_{k-2}$          | yes     | $y_{k-4}$        | no      |
|                       | $x_{k-2}^2$        | yes     | $x_{k-2}^2$      | yes     |
|                       | $y_{k-2}x_{k-2}^2$ | yes     | $y_{k-2}x_{k-2}^2$ | yes   |
| **$S_6$**             | $y_{k-2}$          | yes     | $y_{k-2}$        | yes     |
|                       | $x_{k-1}$          | yes     | $x_{k-1}$        | yes     |
|                       | $y_{k-2}x_{k-2}$   | yes     | $y_{k-2}x_{k-1}$ | yes     |
> Table 12. Análise comparativa entre MetaMSS e FROLS

#### Meta-MSS vs RJMCMC

O modelo $S_4$ foi tomado do trabalho de Baldacchino, Anderson e Kadirkamanathan ([Computational System Identification for Bayesian NARMAX Modelling](https://www.sciencedirect.com/science/article/abs/pii/S0005109813003063)). No estudo, os lags máximos são $n_y = n_u = 4$ e o grau de não linearidade é $\ell = 3$. Os autores executaram o algoritmo RJMCMC 10 vezes sobre os mesmos dados de entrada e saída. O método RJMCMC identificou a estrutura verdadeira 7 de 10 vezes. Em contraste, o algoritmo MetaMSS identificou a estrutura correta em todas as execuções. Esses resultados são resumidos na Tabela 13.

Além disso, o RJMCMC apresenta algumas desvantagens que são mitigadas pelo MetaMSS. Em particular, o RJMCMC é computacionalmente intensivo, exigindo 30.000 iterações para obter os resultados. Ademais, ele depende de diversas distribuições de probabilidade para simplificar o processo de estimação de parâmetros, o que torna as contas mais complexas. Já o MetaMSS oferece uma abordagem mais simples e eficiente, evitando esses problemas.

|                  | Meta-MSS Model       | Correct | RJMCMC Model 1 ($7\times$) | RJMCMC Model 2      | RJMCMC Model 3      | RJMCMC Model 4      | Correct |
|------------------|----------------------|---------|---------------------------|---------------------|---------------------|---------------------|---------|
| **$S_4$**        | $y_{k-1}x_{k-1}$      | yes     | $y_{k-1}x_{k-1}$           | $y_{k-1}x_{k-1}$    | $y_{k-1}x_{k-1}$    | $y_{k-1}x_{k-1}$    | yes     |
|                  | $y_{k-2}$             | yes     | $y_{k-2}$                  | $y_{k-2}$           | $y_{k-2}$           | $y_{k-2}$           | yes     |
|                  | $x_{k-2}^2$           | yes     | $x_{k-2}^2$                | $x_{k-2}^2$         | $x_{k-2}^2$         | $x_{k-2}^2$         | yes     |
|                  | $y_{k-2}x_{k-2}^2$    | yes     | $y_{k-2}x_{k-2}^2$         | $y_{k-2}x_{k-2}^2$  | $y_{k-2}x_{k-2}^2$  | $y_{k-2}x_{k-2}^2$  | yes     |
|                  | -                      | -       | -                           | $y_{k-3}x_{k-3}$    | $x_{k-4}^2$         | $x_{k-1}x_{k-3}^2$  | no      |
> Table 13. Análise comparativa entre MetaMSS e RJMCMC.

### MetaMSS usando SysIdentPy

Considere agora os mesmos dados utilizados na subseção Overview of the Information Criteria Methods.

```python
from sysidentpy.model_structure_selection import MetaMSS


basis_function = Polynomial(degree=2)
model = MetaMSS(
    ylag=2,
    xlag=2,
    random_state=42,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

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
plot_results(y=y_valid, yhat=yhat, n=1000)
```

O algoritmo MetaMSS não depende de critérios de informação como ERR para seleção de estrutura de modelo e, por isso, não utiliza esses hiperparâmetros. O mesmo vale para os algoritmos AOLS e ER. Para mais detalhes sobre o uso desses métodos e seus hiperparâmetros, consulte a documentação.

No que diz respeito à estimação de parâmetros, o SysIdentPy permite empregar qualquer método disponível, independentemente do algoritmo de seleção de estrutura utilizado. Ou seja, o usuário pode combinar métodos de estrutura (FROLS, AOLS, ER, MetaMSS, etc.) com diferentes estimadores de parâmetros. Essa flexibilidade permite explorar diversas abordagens de modelagem e personalizar o processo de identificação. Embora os exemplos fornecidos utilizem o método de estimação padrão, o usuário é encorajado a testar outras opções para encontrar a melhor solução para o seu problema.

Os resultados do MetaMSS são

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| y(k-1)     | 1.8004E-01  | 0.00000000E+00|
| x1(k-2)    | 8.9747E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/metamss_result_c4_example.png?raw=true)
> Figura 17. Simulação em regime livre para o modelo ajustado utilizando MetaMSS.

O método `results` retorna ERR igual a 0 para todos os regressores porque, como mencionado, o algoritmo ERR não é executado neste caso.

## Accelerated Orthogonal Least Squares (AOLS) e Entropic Regression (ER)

Além de FROLS e MetaMSS, o SysIdentPy inclui outros dois métodos para seleção de estrutura de modelo: Accelerated Orthogonal Least Squares (AOLS) e Entropic Regression (ER). Nesta seção não entraremos em detalhes sobre os métodos, como fizemos com FROLS e MetaMSS, mas apresentaremos uma visão geral e referências para leitura adicional:

- **Accelerated Orthogonal Least Squares (AOLS):** para uma discussão detalhada sobre AOLS, ver o artigo original [aqui](https://www.sciencedirect.com/science/article/abs/pii/S1051200418305311).
- **Entropic Regression (ER):** detalhes sobre ER podem ser encontrados no artigo original [aqui](https://arxiv.org/pdf/1905.08061).

A seguir, mostramos como utilizar esses métodos no SysIdentPy.

### Accelerated Orthogonal Least Squares

```python
from sysidentpy.model_structure_selection import AOLS

basis_function = Polynomial(degree=2)
model = AOLS(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

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
plot_results(y=y_valid, yhat=yhat, n=1000)
```

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| x1(k-2)    | 9.1542E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/aols_example_c4.png?raw=true)
> Figura 18. Simulação em regime livre para o modelo ajustado com AOLS.

### Entropic Regression

```python

from sysidentpy.model_structure_selection import ER

basis_function = Polynomial(degree=2)
model = ER(
    ylag=2,
    xlag=2,
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)

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
plot_results(y=y_valid, yhat=yhat, n=1000)
```

| Regressors | Parameters  | ERR           |
|------------|-------------|---------------|
| 1          | -2.4554E-02 | 0.00000000E+00|
| x1(k-2)    | 9.0273E-01  | 0.00000000E+00|

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/er_example_c4.png?raw=true)
> Figura 19. Simulação em regime livre para o modelo ajustado com Entropic Regression.
