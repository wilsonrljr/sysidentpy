Multiobjective parameter estimation representa uma mudança de paradigma fundamental na forma como abordamos o problema de ajuste de parâmetros para modelos NARMAX. Em vez de buscar um único conjunto de valores de parâmetros que ajuste o modelo de forma ótima aos dados, abordagens multiobjetivo visam identificar um conjunto de soluções de parâmetros, conhecido como *Pareto front*, que fornece um trade-off entre objetivos conflitantes. Esses objetivos frequentemente abrangem um espectro de critérios de desempenho do modelo, como qualidade de ajuste (*goodness-of-fit*), complexidade do modelo e robustez.

O que isso significa? Significa que, quando estamos modelando um sistema dinâmico, na maior parte do tempo estamos construindo modelos que são bons apenas para representar o comportamento dinâmico do sistema em estudo. Isso é válido na maioria dos casos porque estamos construindo modelos dinâmicos; portanto, se o modelo não tiver um bom desempenho em cenários estáticos, isso pode não ser um problema. Entretanto, nem sempre é assim, e podemos desejar um modelo que apresente bom desempenho tanto do ponto de vista dinâmico quanto do estático. Nesses casos, métodos desenvolvidos apenas para sistemas puramente dinâmicos não são adequados e algoritmos multiobjetivo podem nos auxiliar nessa tarefa.

A ideia principal na estimação de parâmetros multiobjetivo é a inclusão da *affine information* (informação afim). A informação afim é uma informação auxiliar que pode ser definida *a priori*, como o ganho estático e a função estática do sistema. Formalmente, a informação afim pode ser definida [como segue](https://www.researchgate.net/publication/216771768_Multiobjective_parameter_estimation_for_nonlinear_systems_Affine_information_and_least-squares_formulation):

Seja o vetor de parâmetros $\Theta \in \mathbb{R}^{n_{\Theta}}$, um vetor $\mathrm{v}\in \mathbb{R}^p$ e uma matriz $\mathrm{G}\in \mathbb{R}^{n_{\Theta}\times p}$, em que $\mathrm{v}$ e $\mathrm{G}$ são assumidos acessíveis. Suponha que $\mathrm{G}\Theta$ seja uma estimativa de $\mathrm{v}$. Então, $\mathrm{v} = \mathrm{G}\Theta + \xi$. Logo, $[\mathrm{v}, \mathrm{G}]$ é um par de informação afim do sistema.

## Multi-objective optimization problem

Vamos definir o que é um problema multiobjetivo. Dadas $m$ funções objetivo

$$
\begin{equation}
	\mathrm{J}(\hat{\Theta}) = [J_1(\hat{\Theta}), J_2(\hat{\Theta}), \cdots, J_m(\hat{\Theta})]^\top,
\end{equation}
	ag{5.1}
$$

em que $\mathrm{J}(\cdot):\mathbb{R}^n \mapsto \mathbb{R}^m$, um problema geral de otimização multiobjetivo pode ser escrito como ([A. Baykasoglu, S. Owen, e N. Gindy](https://www.tandfonline.com/doi/abs/10.1080/03052159908941394))

$$
\begin{equation}
	\begin{aligned}
		 & \underset{\Theta}{\text{minimize}} & & \mathrm{J}(\Theta) \\
		 & \text{subject to} & & \Theta \in \mathrm{S} = \left\{\Theta \mid \Theta \in \mathrm{A}^n, g_i(\Theta) \leq a_i, h_j(\Theta) = b_j \right\}, \\
		 & & & i = 1, \ldots, m, \quad j = 1, \ldots, n
	\end{aligned}
\end{equation}
	ag{5.2}
$$

em que $\Theta$ é um vetor $n$-dimensional de variáveis de decisão, $\mathrm{S}$ é o conjunto de soluções factíveis limitado por $m$ restrições de desigualdade ($g_i$) e $n$ restrições de igualdade ($h_j$), e $a_i$ e $b_j$ são constantes. Para variáveis contínuas, $A = \mathbb{R}$, enquanto $A$ contém o conjunto de valores permitidos para variáveis discretas.

> Normalmente, problemas com $1 < m < 4$ são *chamados de problemas de otimização multiobjetivo*. Quando há mais objetivos ($m\geq 4$), eles são chamados de *many-objective optimization problems*, uma classe emergente de problemas multiobjetivo voltada para a solução de tarefas reais complexas e modernas. Mais detalhes podem ser encontrados em ([Fleming, P. J., Purshouse, R. C., and Lygoe, R. J., "Many-Objective Optimization: An Engineering Design Perspective"](https://www.researchgate.net/publication/216300612_Many-Objective_Optimization_An_Engineering_Design_Perspective)), ([Li, B., Li, J., Tang, K., and Yao, X., "A survey on multi-objective evolutionary algorithms for many-objective problems"](https://link.springer.com/article/10.1007/s10589-014-9644-1)).

## Pareto Optimal Definition and Pareto Dominance

> Considere $[y^{(1)}, y^{(2)}] \in \mathbb{R}^m$ dois vetores no espaço objetivo. Se, e somente se, $\forall i \in \{1, \ldots, m \}: y_i^{(1)}\leq y_i^{(2)}$ e $\exists j \in \{1, \ldots, m \}: y_j^{(1)} < y_j^{(2)}$, pode-se dizer que $y^{(1)} \prec y^{(2)}$ ([P. L. Yu, "Cone convexity, cone extreme points, and non dominated solutions in decision problems with multiobjectives"](https://link.springer.com/article/10.1007/BF00932614)).

O conceito de otimalidade de Pareto é geralmente usado para descrever o trade-off entre a minimização de diferentes objetivos. Seguindo a definição de Pareto: o ótimo de Pareto é qualquer vetor de parâmetros que represente uma solução eficiente tal que nenhuma função objetivo possa ser melhorada sem piorar pelo menos uma outra função objetivo; tal vetor será referido como um Pareto-model.

No contexto de identificação de sistemas, isso significa encontrar um modelo em que não seja possível obter um melhor desempenho dinâmico sem piorar o desempenho estático.

Um conjunto de Pareto hipotético é mostrado na Figura 1.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/pareto_example.png?raw=true)
> Figura 1. A figura ilustra o conceito de otimalidade de Pareto, em que cada ponto no espaço objetivo representa uma solução. A frente de Pareto é representada por uma curva, mostrando o trade-off entre dois objetivos conflitantes. Pontos na frente não podem ser melhorados em um dos objetivos sem piorar o outro, destacando o equilíbrio nas soluções ótimas.

Neste caso, assume-se que a estrutura do modelo é conhecida e, portanto, existe uma correspondência biunívoca entre cada vetor de parâmetros na solução ótima de Pareto e um modelo ([Nepomuceno, E. G., Takahashi, R. H. C., and Aguirre, L. A., "Multiobjective parameter estimation for non-linear systems: affine information and least-squares formulation"](https://www.tandfonline.com/doi/abs/10.1080/00207170601185053)). Pode-se construir um conjunto de Pareto aplicando o *Weighted Sum Method*, no qual um conjunto de objetivos é escalarizado em um único objetivo pela soma de cada objetivo multiplicado por um peso fornecido pelo usuário. Considere

$$
\begin{equation}
	\mathrm{W} = \Bigg\{ w\mid w \in \mathbb{R}^m, w_j\geq 0 \quad \textrm{and} \quad \sum^{m}_{j=1}w_j=1 \Bigg\}
\end{equation}
	ag{5.3}
$$

como pesos não negativos. Então, o problema de otimização convexo pode ser escrito como

$$
\begin{equation} \begin{aligned} \Theta^* &= \underset{\Theta}{\text{argmin}} \, \langle w, \mathrm{J}(\Theta) \rangle \end{aligned}
\end{equation}
	ag{5.4}
$$

em que $w$ é uma combinação de pesos para as diferentes funções objetivo. Portanto, o conjunto de Pareto está associado ao conjunto de realizações de $w \in \mathrm{W}$. Uma estratégia computacional eficiente em passo único foi apresentada em ([Nepomuceno, E. G., Takahashi, R. H. C., and Aguirre, L. A., "Multiobjective parameter estimation for non-linear systems: affine information and least-squares formulation"](https://www.tandfonline.com/doi/abs/10.1080/00207170601185053)) para resolver a Equação 5.4 por meio de uma formulação em *Least Squares*, apresentada na próxima seção.

## Affine Information Least Squares Algorithm

Considere os $m$ pares de informação afim $[\mathrm{v}_i \in \mathbb{R}^{p_i}, \mathrm{G}_i \in \mathbb{R}^{p_i\times n}]$ com $i = 1, \ldots, m$. Assuma que existe $\mathrm{G}_i$ de posto coluna completo (*full column rank*) e seja $M$ um modelo da forma

$$
y = \Psi\Theta + \epsilon.
	ag{5.5}
$$

Então, os $m$ pares de informação afim podem ser considerados na estimação de parâmetros resolvendo

$$
\begin{equation}
\begin{aligned}
\Theta^* &= \underset{\Theta}{\text{argmin}} \sum_{i=1}^{m} w_i (\mathrm{v}_i - \mathrm{G}_i \Theta)^\top (\mathrm{v}_i - \mathrm{G}_i \Theta)
\end{aligned}
\end{equation}
	ag{5.6}
$$

com $w = [w_i, \ldots, w_m]^\top \in \mathrm{W}$. A solução da equação acima é dada por

$$
\begin{equation}
	\Theta^* = \left[\sum^{m}_{i=1}w_i\mathrm{G}_i^\top\mathrm{G}_i\right]^{-1}  \left[\sum^{m}_{i=1}w_i\mathrm{G}_i^\top\mathrm{v}_i\right].
\end{equation}
	ag{5.7}
$$

Se existir apenas uma informação, o problema se reduz à solução monoobjetivo de *Least Squares*.

Para tornar as coisas mais claras, vamos analisar um estudo de caso detalhado.

## Estudo de Caso - Conversor Buck

Um conversor Buck é um tipo de conversor CC/CC (DC/DC) que reduz a tensão (enquanto aumenta a corrente) de sua entrada (fonte de alimentação) para sua saída (carga). Ele é similar a um conversor Boost (elevador) e é um tipo de fonte de alimentação chaveada (*switched-mode power supply*, SMPS) que tipicamente contém pelo menos dois semicondutores (um diodo e um transistor, embora conversores Buck modernos substituam o diodo por um segundo transistor usado para retificação síncrona) e pelo menos um elemento de armazenamento de energia, um capacitor, um indutor ou ambos combinados.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.multiobjective_parameter_estimation import AILS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_results
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.narmax_tools import set_weights
```

### Comportamento Dinâmico

```python
df_train = pd.read_csv(r"datasets/buck_id.csv")
df_valid = pd.read_csv(r"datasets/buck_valid.csv")

# Plotting the measured output (identification and validation data)
plt.figure(1)
plt.title("Output")
plt.plot(df_train.sampling_time, df_train.y, label="Identification", linewidth=1.5)
plt.plot(df_valid.sampling_time, df_valid.y, label="Validation", linewidth=1.5)
plt.xlabel("Samples")
plt.ylabel("Voltage")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_data.png?raw=true)

```python
# Plotting the measured input (identification and validation data)
plt.figure(2)
plt.title("Input")
plt.plot(df_train.sampling_time, df_train.input, label="Identification", linewidth=1.5)
plt.plot(df_valid.sampling_time, df_valid.input, label="Validation", linewidth=1.5)
plt.ylim(2.1, 2.6)
plt.ylabel("u")
plt.xlabel("Samples")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_input_data.png?raw=true)

### Função Estática do Conversor Buck

O *duty cycle*, representado pelo símbolo $D$, é definido como a razão entre o tempo em que o sistema permanece ligado ($T_{on}$) e o tempo total de operação do ciclo ($T$). Matematicamente, pode ser expresso como $D=\frac{T_{on}}{T}$. O complementar do ciclo de trabalho, representado por $D'$, é definido como a razão entre o tempo em que o sistema permanece desligado ($T_{off}$) e o tempo total de operação ($T$) e pode ser expresso como $D'=\frac{T_{off}}{T}$.

A tensão na carga ($V_o$) está relacionada à tensão da fonte ($V_d$) pela equação $V_o = D\cdot V_d = (1-D')\cdot V_d$. Para este conversor em particular, sabe-se que $D′=\frac{\bar{u}-1}{3}$, o que significa que a função estática deste sistema pode ser derivada da teoria como:

$$
V_o = \frac{4V_d}{3} - \frac{V_d}{3}\cdot \bar{u}
$$

Se assumirmos que a tensão da fonte $V_d$ é igual a 24 V, podemos reescrever a expressão acima como:

$$
V_o = (4 - \bar{u})\cdot 8
$$

```python
# Static data
Vd = 24
Uo = np.linspace(0, 4, 50)
Yo = (4 - Uo) * Vd / 3
Uo = Uo.reshape(-1, 1)
Yo = Yo.reshape(-1, 1)
plt.figure(3)
plt.title("Buck Converter Static Curve")
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{y}$")
plt.plot(Uo, Yo, linewidth=1.5, linestyle="-", marker="o")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_data.png?raw=true)

### Ganho Estático do Conversor Buck

O ganho de um conversor Buck é uma medida de como sua tensão de saída varia em resposta a alterações na tensão de entrada. Matematicamente, o ganho pode ser calculado como a derivada da função estática do conversor, que descreve a relação entre as tensões de entrada e saída.

Neste caso, a função estática do conversor Buck é dada por

$$
V_o = (4 - \bar{u})\cdot 8
$$

Derivando essa equação em relação a $\hat{u}$, obtemos que o ganho do conversor Buck é igual a −8. Em outras palavras, para cada unidade de aumento na tensão de entrada $\hat{u}$, a tensão de saída $V_o$ diminui em 8 unidades. Assim,

$$
gain=V_o'=-8
$$

```python
# Defining the gain
gain = -8 * np.ones(len(Uo)).reshape(-1, 1)
plt.figure(3)
plt.title("Buck Converter Static Gain")
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{gain}$")
plt.plot(Uo, gain, linewidth=1.5, label="gain", linestyle="-", marker="o")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_gain.png?raw=true)

### Construindo um modelo dinâmico usando a abordagem mono-objetivo

```python
x_train = df_train.input.values.reshape(-1, 1)
y_train = df_train.y.values.reshape(-1, 1)
x_valid = df_valid.input.values.reshape(-1, 1)
y_valid = df_valid.y.values.reshape(-1, 1)

basis_function = Polynomial(degree=2)
model = FROLS(
	order_selection=True,
	n_info_values=8,
	extended_least_squares=False,
	ylag=2,
	xlag=2,
	info_criteria="aic",
	estimator="least_squares",
	basis_function=basis_function,
)

model.fit(X=x_train, y=y_train)
```

### Affine Information Least Squares Algorithm (AILS)

AILS é um algoritmo de estimação de parâmetros multiobjetivo, baseado em um conjunto de pares de informação afim. A abordagem multiobjetivo proposta no artigo citado e implementada em SysIdentPy leva a um problema de otimização multiobjetivo convexo, que pode ser resolvido por AILS. AILS é um esquema não iterativo do tipo *Least Squares* para encontrar as soluções do conjunto de Pareto (Pareto-set solutions) para o problema multiobjetivo.

Assim, com a estrutura do modelo definida (neste exemplo, usaremos a obtida a partir dos dados dinâmicos anteriores), podemos estimar os parâmetros usando a abordagem multiobjetivo.

As informações sobre a função estática e o ganho estático, além dos dados dinâmicos usuais de entrada/saída, podem ser usadas para construir o par de informação afim utilizado na estimação dos parâmetros do modelo. Podemos modelar a função de custo como:

$$
\gamma(\hat\theta) = w_1\cdot J_{LS}(\hat{\theta})+w_2\cdot J_{SF}(\hat{\theta})+w_3\cdot J_{SG}(\hat{\theta})
$$

### Estimação de parâmetros multiobjetivo considerando 3 objetivos diferentes: o erro de predição, a função estática e o ganho estático

```python
# you can use any set of model structure you want in your use case, but in this notebook we will use the one obtained above the compare with other work
mo_estimator = AILS(final_model=model.final_model)
# setting the log-spaced weights of each objective function
w = set_weights(static_function=True, static_gain=True)
# you can also use something like
# w = np.array(
# 	    [
# 	        [0.98, 0.7, 0.5, 0.35, 0.25, 0.01, 0.15, 0.01],
# 	        [0.01, 0.1, 0.3, 0.15, 0.25, 0.98, 0.35, 0.01],
# 	        [0.01, 0.2, 0.2, 0.50, 0.50, 0.01, 0.50, 0.98],
# 	    ]
# )

# to set the weights. Each row correspond to each objective
```

AILS possui um método `estimate` que retorna as funções de custo (J), a norma euclidiana das funções de custo (E), os parâmetros estimados associados a cada vetor de pesos (theta), a matriz de regressores associada ao ganho estático (HR) e à função estática (QR), respectivamente.

```python
J, E, theta, HR, QR, position = mo_estimator.estimate(
	X=x_train, y=y_train, gain=gain, y_static=Yo, X_static=Uo, weighing_matrix=w
)
result = {
	"w1": w[0, :],
	"w2": w[2, :],
	"w3": w[1, :],
	"J_ls": J[0, :],
	"J_sg": J[1, :],
	"J_sf": J[2, :],
	"||J||:": E,
}
pd.DataFrame(result)
```

| w1       | w2       | w3       | J_ls     | J_sg         | J_sf     | $\lVert J \rVert$ |
| -------- | -------- | -------- | -------- | ------------ | -------- | ----------------- |
| 0.006842 | 0.003078 | 0.990080 | 0.999970 | 1.095020e-05 | 0.000013 | 0.245244          |
| 0.007573 | 0.002347 | 0.990080 | 0.999938 | 2.294665e-05 | 0.000016 | 0.245236          |
| 0.008382 | 0.001538 | 0.990080 | 0.999885 | 6.504913e-05 | 0.000018 | 0.245223          |
| 0.009277 | 0.000642 | 0.990080 | 0.999717 | 4.505541e-04 | 0.000021 | 0.245182          |
| 0.006842 | 0.098663 | 0.894495 | 1.000000 | 7.393246e-08 | 0.000015 | 0.245251          |
| ...      | ...      | ...      | ...      | ...          | ...      | ...               |
| 0.659632 | 0.333527 | 0.006842 | 0.995896 | 3.965699e-04 | 1.000000 | 0.244489          |
| 0.730119 | 0.263039 | 0.006842 | 0.995632 | 5.602981e-04 | 0.972842 | 0.244412          |
| 0.808139 | 0.185020 | 0.006842 | 0.995364 | 8.321071e-04 | 0.868299 | 0.244300          |
| 0.894495 | 0.098663 | 0.006842 | 0.995100 | 1.364999e-03 | 0.660486 | 0.244160          |
| 0.990080 | 0.003078 | 0.006842 | 0.992584 | 9.825987e-02 | 0.305492 | 0.261455          |

Agora podemos definir $\theta$ associado a qualquer combinação de pesos desejada.

```python
model.theta = theta[-1, :].reshape(
	-1, 1
)  # setting the theta estimated for the last combination of the weights

# the model structure is exactly the same, but the order of the regressors is changed in estimate method. Thats why you have to change the model.final_model

model.final_model = mo_estimator.final_model
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
	results(
		model.final_model,
		model.theta,
		model.err,
		model.n_terms,
		err_precision=3,
		dtype="sci",
	),
	columns=["Regressors", "Parameters", "ERR"],
)
r
```

| Regressors    | Parameters  | ERR       |
| ------------- | ----------- | --------- |
| 1             | 2.2930E+00  | 9.999E-01 |
| y(k-1)        | 2.3307E-01  | 2.042E-05 |
| y(k-2)        | 6.3209E-01  | 1.108E-06 |
| x1(k-1)       | -5.9333E-01 | 4.688E-06 |
| y(k-1)^2      | 2.7673E-01  | 3.922E-07 |
| y(k-2)y(k-1)  | -5.3228E-01 | 8.389E-07 |
| x1(k-1)y(k-1) | 1.6667E-02  | 5.690E-07 |
| y(k-2)^2      | 2.5766E-01  | 3.827E-06 |

#### Os resultados dinâmicos para o theta escolhido são

```python
plot_results(y=y_valid, yhat=yhat, n=1000)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_dynamic_results.png?raw=true)

#### O resultado do ganho estático é

```python
plt.figure(4)
plt.title("Gain")
plt.plot(
	Uo,
	gain,
	linewidth=1.5,
	linestyle="-",
	marker="o",
	label="Buck converter static gain",
)
plt.plot(
	Uo,
	HR.dot(model.theta),
	linestyle="-",
	marker="^",
	linewidth=1.5,
	label="NARX model gain",
)
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{g}$")
plt.ylim(-16, 0)
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_gain_results.png?raw=true)

#### O resultado da função estática é

```python
plt.figure(5)
plt.title("Static Curve")
plt.plot(Uo, Yo, linewidth=1.5, label="Static curve", linestyle="-", marker="o")
plt.plot(
	Uo,
	QR.dot(model.theta),
	linewidth=1.5,
	label="NARX \u200b\u200bstatic representation",
	linestyle="-",
	marker="^",
)
plt.xlabel("$\\bar{u}$")
plt.xlabel("$\\bar{y}$")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_function_results.png?raw=true)

### Obtendo a melhor combinação de pesos com base na norma da função de custo

A variável `position` retornada pelo método `estimate` fornece a posição da melhor combinação de pesos. A estrutura do modelo é exatamente a mesma, mas a ordem dos regressores é alterada no método `estimate`. Por isso é necessário atualizar `model.final_model`. Os resultados dinâmico, de ganho estático e da função estática para o $\theta$ escolhido são mostrados a seguir.

```python
model.theta = theta[position, :].reshape(
	-1, 1
)  # setting the theta estimated for the best combination of the weights

# changing the model.final_model

model.final_model = mo_estimator.final_model
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
	results(
		model.final_model,
		model.theta,
		model.err,
		model.n_terms,
		err_precision=3,
		dtype="sci",
	),
	columns=["Regressors", "Parameters", "ERR"],
)
print(r)

# dynamic results
plot_results(y=y_valid, yhat=yhat, n=1000)

# static gain
plt.figure(4)
plt.title("Gain")
plt.plot(
	Uo,
	gain,
	linewidth=1.5,
	linestyle="-",
	marker="o",
	label="Buck converter static gain",
)
plt.plot(
	Uo,
	HR.dot(model.theta),
	linestyle="-",
	marker="^",
	linewidth=1.5,
	label="NARX model gain",
)
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{g}$")
plt.ylim(-16, 0)
plt.legend()
plt.show()

# static function
plt.figure(5)
plt.title("Static Curve")
plt.plot(Uo, Yo, linewidth=1.5, label="Static curve", linestyle="-", marker="o")
plt.plot(
	Uo,
	QR.dot(model.theta),
	linewidth=1.5,
	label="NARX \u200b\u200bstatic representation",
	linestyle="-",
	marker="^",
)

plt.xlabel("$\\bar{u}$")
plt.xlabel("$\\bar{y}$")
plt.legend()
plt.show()
```

| Regressors    | Parameters  | ERR       |
| ------------- | ----------- | --------- |
| 1             | 1.5405E+00  | 9.999E-01 |
| y(k-1)        | 2.9687E-01  | 2.042E-05 |
| y(k-2)        | 6.4693E-01  | 1.108E-06 |
| x1(k-1)       | -4.1302E-01 | 4.688E-06 |
| y(k-1)^2      | 2.7671E-01  | 3.922E-07 |
| y(k-2)y(k-1)  | -5.3474E-01 | 8.389E-07 |
| x1(k-1)y(k-1) | 4.0624E-03  | 5.690E-07 |
| y(k-2)^2      | 2.5832E-01  | 3.827E-06 |


![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_result_mo_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_mo_static_gain_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_curve_mo_1.png?raw=true)

Você também pode plotar as soluções do conjunto de Pareto

```python
plt.figure(6)
ax = plt.axes(projection="3d")
ax.plot3D(J[0, :], J[1, :], J[2, :], "o", linewidth=0.1)
ax.set_title("Pareto-set solutions", fontsize=15)
ax.set_xlabel("$J_{ls}$", fontsize=10)
ax.set_ylabel("$J_{sg}$", fontsize=10)
ax.set_zlabel("$J_{sf}$", fontsize=10)
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/pareto_set_solutions_1.png?raw=true)

### Detalhando o AILS

O modelo polinomial NARX construído usando a abordagem mono-objetivo possui a seguinte estrutura:

$$
y(k) = \theta_1 y(k-1) + \theta_2 y(k-2) + \theta_3 u(k-1) y(k-1) + \theta_4 + \theta_5 y(k-1)^2 + \theta_6 u(k-1) + \theta_7 y(k-2)y(k-1) + \theta_8 y(k-2)^2
$$

Assim, o objetivo ao usar as informações da função estática e do ganho estático no cenário multiobjetivo é estimar o vetor $\hat{\theta}$ com base em:

$$
		heta = [w_1\Psi^T\Psi + w_2(HR)^T(HR) + w_3(QR)(QR)^T]^{-1} [w_1\Psi^T y + w_2(HR)^T\overline{g}+w_3(QR)^T\overline{y}]
$$

A matriz $\Psi$ é construída usando a abordagem usual de modelagem dinâmica mono-objetivo no SysIdentPy. No entanto, ainda é necessário encontrar as matrizes Q, H e R. O AILS possui métodos para calcular todas essas matrizes. Basicamente, para isso, $q_i^T$ é primeiro estimado:$$
q_i^T =
\begin{bmatrix}
1 & \overline{y_i} & \overline{u_1} & \overline{y_i}^2 & \cdots & \overline{y_i}^l & F_{yu} & \overline{u_i}^2 & \cdots & \overline{u_i}^l
\end{bmatrix}
$$

onde $F_{yu}$ representa todos os monômios não lineares no modelo que estão relacionados a $y(k)$ e $u(k)$, $l$ é a maior não linearidade no modelo para termos de entrada e saída. Para um modelo com grau de não linearidade igual a 2, podemos obter:

$$
q_i^T =
\begin{bmatrix}
1 & \overline{y_i} & \overline{u_i} & \overline{y_i}^2 & \overline{u_i}\:\overline{y_i} & \overline{u_i}^2
\end{bmatrix}
$$

É possível codificar a matriz $q_i^T$ de forma que ela siga a codificação do modelo definida no SysIdentPy. Para isso, 0 é considerado como uma constante, $y_i$ igual a 1 e $u_i$ igual a 2. O número de colunas indica o grau de não linearidade do sistema e o número de linhas reflete o número de termos:

$$
q_i =
\begin{bmatrix}
0 & 0\\
1 & 0\\
2 & 0\\
1 & 1\\
2 & 1\\
2 & 2\\
\end{bmatrix}
=
\begin{bmatrix}
1 \\
\overline{y_i}\\
\overline{u_i}\\
\overline{y_i}^2\\
\overline{u_i}\:\overline{y_i}\\
\overline{u_i}^2\\
\end{bmatrix}
$$

Finally, the result can be easily obtained using the ‘regressor_space’ method of SysIdentPy

```python
from sysidentpy.narmax_base import RegressorDictionary

object_qit = RegressorDictionary(xlag=1, ylag=1)
R_example = object_qit.regressor_space(n_inputs=1) // 1000
print(f"R = {R_example}")
```

$$
R = \begin{bmatrix} 0 & 0 \\ 1 & 0 \\ 2 & 0 \\ 1 & 1 \\ 2 & 1 \\ 2 & 2 \end{bmatrix}
$$

de modo que:

$$
\overline{y_i} = q_i^T R\theta
$$

e:

$$
\overline{g_i} = H R\theta
$$

onde $R$ é o mapeamento linear dos regressores estáticos representados por $q_i^T$. Além disso, a matriz $H$ contém informação afim referente a $\overline{g_i}$, que é igual a $\overline{g_i} = \frac{d\overline{y}}{d\overline{u}}{\big |}_{(\overline{u_i}\:\overline{y_i})}$.

A partir de agora, começaremos a aplicar a estimação de parâmetros de forma multiobjetivo. Isso será feito tendo em mente o modelo polinomial NARX do conversor BUCK. Neste contexto, $q_i^T$ será genérico e assumirá um formato específico para o problema em questão. Para esta tarefa, será utilizado o método `build_linear_mapping`, cujo objetivo é retornar o $q_i^T$ relacionado ao modelo e a matriz do mapeamento linear $R$:

```python
R, qit = mo_estimator.build_linear_mapping()
print("R matrix:")
print(R)
print("qit matrix:")
print(qit)
```

$$
R = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 1 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
\end{bmatrix}
$$

and

$$
qit = \begin{bmatrix}
0 & 0 \\
1 & 0 \\
0 & 1 \\
2 & 0 \\
1 & 1 \\
\end{bmatrix}
$$

Portanto

$$
q_i =
\begin{bmatrix}
0 & 0 \\
1 & 0 \\
2 & 0 \\
1 & 1 \\
2 & 1 \\
\end{bmatrix}
=
\begin{bmatrix}
1 \\
\overline{y} \\
\overline{u} \\
\overline{y^2} \\
\overline{u} \cdot \overline{y} \\
\end{bmatrix}
$$

Você pode notar que o método produz saídas consistentes com o esperado:

$$
y(k) = \theta_1 y(k-1) + \theta_2 y(k-2) + \theta_3 u(k-1) y(k-1) + \theta_4 + \theta_5 y(k-1)^2 + \theta_6 u(k-1) + \theta_7 y(k-2)y(k-1) + \theta_8 y(k-2)^2
$$

e:

$$
R =
\begin{bmatrix}
term/\theta & \theta_1 & \theta_2 & \theta_3 & \theta_4 & \theta_5 & \theta_6 & \theta_7 & \theta_8\\
1 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
\overline{y} & 1 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
\overline{u} & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
\overline{y^2} & 0 & 0 & 0 & 0 & 1 & 0 & 1 & 1\\
\overline{y}\:\overline{u} & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
\end{bmatrix}
$$
 
### Validação

A seguinte estrutura de modelo será usada para validar a abordagem:

$$
y(k) = \theta_1 y(k-1) + \theta_2 y(k-2) + \theta_3 + \theta_4 u(k-1) + \theta_5 u(k-1)^2 + \theta_6 u(k-2)u(k-1)+\theta_7 u(k-2) + \theta_8 u(k-2)^2
$$

$$
	herefore
$$

$$
final\_model =
\begin{bmatrix}
1001 & 0\\
1002 & 0\\
0 & 0\\
2001 & 0\\
2001 & 2001\\
2002 & 2001\\
2002 & 0\\
2002 & 2002
\end{bmatrix}
$$

definindo em código:

```python
final_model = np.array(
	[
		[1001, 0],
		[1002, 0],
		[0, 0],
		[2001, 0],
		[2001, 2001],
		[2002, 2001],
		[2002, 0],
		[2002, 2002],
	]
)
final_model
```

| 1001 | 0    |
| ---- | ---- |
| 1002 | 0    |
| 0    | 0    |
| 2001 | 0    |
| 2001 | 2001 |
| 2002 | 2001 |
| 2002 | 0    |
| 2002 | 2002 |

```python
mult2 = AILS(final_model=final_model)

def psi(X, Y):
	PSI = np.zeros((len(X), 8))
	for k in range(2, len(Y)):
		PSI[k, 0] = Y[k - 1]
		PSI[k, 1] = Y[k - 2]
		PSI[k, 2] = 1
		PSI[k, 3] = X[k - 1]
		PSI[k, 4] = X[k - 1] ** 2
		PSI[k, 5] = X[k - 2] * X[k - 1]
		PSI[k, 6] = X[k - 2]
		PSI[k, 7] = X[k - 2] ** 2
	return np.delete(PSI, [0, 1], axis=0)
```

O valor de theta com o menor erro quadrático médio obtido com o mesmo código implementado em Scilab foi:

$$
W_{LS} = 0.3612343
$$

e:

$$
W_{SG} = 0.3548699
$$

e:

$$
W_{SF} = 0.3548699
$$

```python
PSI = psi(x_train, y_train)
w = np.array([[0.3612343], [0.2838959], [0.3548699]])
J, E, theta, HR, QR, position = mult2.estimate(
	y=y_train, X=x_train, gain=gain, y_static=Yo, X_static=Uo, weighing_matrix=w
)
result = {
	"w1": w[0, :],
	"w2": w[2, :],
	"w3": w[1, :],
	"J_ls": J[0, :],
	"J_sg": J[1, :],
	"J_sf": J[2, :],
	"||J||:": E,
}

pd.DataFrame(result)
```

| w1       | w2      | w3       | J_ls | J_sg | J_sf | $\lVert J \rVert$ |
| -------- | ------- | -------- | ---- | ---- | ---- | ----------------- |
| 0.361234 | 0.35487 | 0.283896 | 1.0  | 1.0  | 1.0  | 1.0               |
A ordem dos pesos é diferente devido à forma como implementamos em Python, mas os resultados são muito próximos, como esperado.

#### Resultados dinâmicos

```python
model.theta = theta[position, :].reshape(-1, 1)
model.final_model = mult2.final_model
yhat = model.predict(X=x_valid, y=y_valid)

rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
	results(
		model.final_model,
		model.theta,
		model.err,
		model.n_terms,
		err_precision=3,
		dtype="sci",
	),
	columns=["Regressors", "Parameters", "ERR"],
)
r
```

| Regressors     | Parameters  | ERR       |
| -------------- | ----------- | --------- |
| 1              | 1.4287E+00  | 9.999E-01 |
| y(k-1)         | 5.5147E-01  | 2.042E-05 |
| y(k-2)         | 4.0449E-01  | 1.108E-06 |
| x1(k-1)        | -1.2605E+01 | 4.688E-06 |
| x1(k-2)        | 1.2257E+01  | 3.922E-07 |
| x1(k-1)^2      | 8.3274E+00  | 8.389E-07 |
| x1(k-2)x1(k-1) | -1.1416E+01 | 5.690E-07 |
| x1(k-2)^2      | 3.0846E+00  | 3.827E-06 |

```python
plot_results(y=y_valid, yhat=yhat, n=1000)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_mo_results_2.png?raw=true)

#### Ganho estático

```python
plt.figure(7)
plt.title("Gain")
plt.plot(
	Uo,
	gain,
	linewidth=1.5,
	linestyle="-",
	marker="o",
	label="Buck converter static gain",
)

plt.plot(
	Uo,
	HR.dot(model.theta),
	linestyle="-",
	marker="^",
	linewidth=1.5,
	label="NARX model gain",
)
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{g}$")
plt.ylim(-16, 0)
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_gain_mo_2.png?raw=true)

#### Função estática

```python
plt.figure(8)
plt.title("Static Curve")
plt.plot(Uo, Yo, linewidth=1.5, label="Static curve", linestyle="-", marker="o")
plt.plot(
	Uo,
	QR.dot(model.theta),
	linewidth=1.5,
	label="NARX \u200b\u200bstatic representation",
	linestyle="-",
	marker="^",
)

plt.xlabel("$\\bar{u}$")
plt.xlabel("$\\bar{y}$")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_curve_mo_2.png?raw=true)

#### Soluções do conjunto de Pareto

```python
plt.figure(9)
ax = plt.axes(projection="3d")
ax.plot3D(J[0, :], J[1, :], J[2, :], "o", linewidth=0.1)
ax.set_title("Optimum pareto-curve", fontsize=15)
ax.set_xlabel("$J_{ls}$", fontsize=10)
ax.set_ylabel("$J_{sg}$", fontsize=10)
ax.set_zlabel("$J_{sf}$", fontsize=10)
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_pareto_2.png?raw=true)

A tabela a seguir mostra os resultados reportados em `IniciacaoCientifica2007` e os obtidos com a implementação do SysIdentPy

| Theta      | SysIdentPy   | IniciacaoCientifica2007   |
|------------|--------------|---------------------------|
| $\theta_1$ | 0.5514725    | 0.549144                  |
| $\theta_2$ | 0.40449005   | 0.408028                  |
| $\theta_3$ | 1.42867821   | 1.45097                   |
| $\theta_4$ | -12.60548863 | -12.55788                 |
| $\theta_5$ | 8.32740057   | 8.1516315                 |
| $\theta_6$ | -11.41574116 | -11.09728                 |
| $\theta_7$ | 12.25729955  | 12.215782                 |
| $\theta_8$ | 3.08461195   | 2.9319577                 |

onde:

$$
E_{Scilab} =    17.426613
$$

e:

$$
E_{Python} = 17.474865
$$

Nota: como mencionado anteriormente, a ordem dos regressores no modelo muda, mas é a mesma estrutura. As tabelas mostram o respectivo parâmetro do regressor referente ao `SysIdentPy` e `IniciacaoCientifica2007`, mas a ordem $\Theta_1$, $\Theta_2$ e assim por diante não é a mesma dos valores em `model.final_model`

```python
R, qit = mult2.build_linear_mapping()
print("R matrix:")
print(R)
print("qit matrix:")
print(qit)
```

$$
R = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 1 & 1 \\
\end{bmatrix}
$$

and

$$
qit = \begin{bmatrix}
0 & 0 \\
1 & 0 \\
0 & 1 \\
0 & 2 \\
\end{bmatrix}
$$

estrutura do modelo que será utilizada (`IniciacaoCientifica2007`):

$$
y(k) = \theta_1 y(k-1) + \theta_2 y(k-2) + \theta_3 + \theta_4 u(k-1) + \theta_5 u(k-1)^2 + \theta_6 u(k-2)u(k-1)+\theta_7 u(k-2) + \theta_8 u(k-2)^2
$$

$$
q_i =
\begin{bmatrix}
0 & 0 \\
1 & 0 \\
2 & 0 \\
2 & 2 \\
\end{bmatrix}
=
\begin{bmatrix}
1 \\
\overline{y} \\
\overline{u} \\
\overline{u^2}
\end{bmatrix}
$$

### Otimização biobjetivo

#### Um caso de uso aplicado ao conversor Buck CC-CC usando como objetivos a informação da curva estática e o erro de predição (dinâmico)

```python
bi_objective = AILS(
	static_function=True, static_gain=False, final_model=final_model, normalize=True
)
```

o valor de theta com o menor erro quadrático médio obtido através da rotina em Scilab foi:

$$
W_{LS} = 0.9931126
$$

e:

$$
W_{SF} = 0.0068874
$$

```python
w = np.zeros((2, 2000))
w[0, :] = np.logspace(-0.01, -6, num=2000, base=2.71)
w[1, :] = np.ones(2000) - w[0, :]
J, E, theta, HR, QR, position = bi_objective.estimate(
	y=y_train, X=x_train, y_static=Yo, X_static=Uo, weighing_matrix=w
)

result = {"w1": w[0, :], "w2": w[1, :], "J_ls": J[0, :], "J_sg": J[1, :], "||J||:": E}

pd.DataFrame(result)
```

| w1       | w2       | J_ls     | J_sg     | $\lVert J \rVert$ |
| -------- | -------- | -------- | -------- | ----------------- |
| 0.990080 | 0.009920 | 0.990863 | 1.000000 | 0.990939          |
| 0.987127 | 0.012873 | 0.990865 | 0.987032 | 0.990939          |
| 0.984182 | 0.015818 | 0.990867 | 0.974307 | 0.990939          |
| 0.981247 | 0.018753 | 0.990870 | 0.961803 | 0.990940          |
| 0.978320 | 0.021680 | 0.990873 | 0.949509 | 0.990941          |
| ...      | ...      | ...      | ...      | ...               |
| 0.002555 | 0.997445 | 0.999993 | 0.000072 | 0.999993          |
| 0.002547 | 0.997453 | 0.999994 | 0.000072 | 0.999994          |
| 0.002540 | 0.997460 | 0.999996 | 0.000071 | 0.999996          |
| 0.002532 | 0.997468 | 0.999998 | 0.000071 | 0.999998          |
| 0.002525 | 0.997475 | 1.000000 | 0.000070 | 1.000000          |


```python
model.theta = theta[position, :].reshape(-1, 1)
model.final_model = bi_objective.final_model
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=3,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

r
```

|     | Regressors     | Parameters  | ERR       |
| --- | -------------- | ----------- | --------- |
| 0   | 1              | 1.3873E+00  | 9.999E-01 |
| 1   | y(k-1)         | 5.4941E-01  | 2.042E-05 |
| 2   | y(k-2)         | 4.0804E-01  | 1.108E-06 |
| 3   | x1(k-1)        | -1.2515E+01 | 4.688E-06 |
| 4   | x1(k-2)        | 1.2227E+01  | 3.922E-07 |
| 5   | x1(k-1)^2      | 8.1171E+00  | 8.389E-07 |
| 6   | x1(k-2)x1(k-1) | -1.1047E+01 | 5.690E-07 |
| 7   | x1(k-2)^2      | 2.9043E+00  | 3.827E-06 |

```python
plot_results(y=y_valid, yhat=yhat, n=1000)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_mo_3.png?raw=true)

```python
plt.figure(10)
plt.title("Static Curve")
plt.plot(Uo, Yo, linewidth=1.5, label="Static curve", linestyle="-", marker="o")
plt.plot(
    Uo,
    QR.dot(model.theta),
    linewidth=1.5,
    label="NARX ​​static representation",
    linestyle="-",
    marker="^",
)

plt.xlabel("$\\bar{u}$")
plt.xlabel("$\\bar{y}$")
plt.legend()
plt.show()
```

```python
plt.figure(11)
plt.title("Costs Functions")
plt.plot(J[1, :], J[0, :], "o")
plt.xlabel("Static Curve Information")
plt.ylabel("Prediction Error")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_pareto_mo_3.png?raw=true)

onde o melhor $\Theta$ estimado é

| Theta       | SysIdentPy   | IniciacaoCientifica2007  |
|-------------|--------------|--------------------------|
| $\theta_1$  | 0.54940883   | 0.5494135                |
| $\theta_2$  | 0.40803995   | 0.4080312                |
| $\theta_3$  | 1.38725684   | 3.3857601                |
| $\theta_4$  | -12.51466378 | -12.513688               |
| $\theta_5$  | 8.11712897   | 8.116575                 |
| $\theta_6$  | -11.04664789 | -11.04592                |
| $\theta_7$  | 12.22693907  | 12.227184                |
| $\theta_8$  | 2.90425844   | 2.9038468                |

onde:

$$
E_{Scilab} = 17.408934
$$

e:

$$
E_{Python} = 17.408947
$$

### Estimação de parâmetros multiobjetivo

#### Caso de uso considerando 2 objetivos diferentes: o erro de predição e o ganho estático

```python
bi_objective_gain = AILS(
    static_function=False, static_gain=True, final_model=final_model, normalize=False
)
```

o valor de theta com o menor erro quadrático médio obtido através da rotina em Scilab foi:

$$
W_{LS} = 0.9931126
$$

e:

$$
W_{SF} = 0.0068874
$$

```python
w = np.zeros((2, 2000))
w[0, :] = np.logspace(0, -6, num=2000, base=2.71)
w[1, :] = np.ones(2000) - w[0, :]
J, E, theta, HR, QR, position = bi_objective_gain.estimate(
    X=x_train, y=y_train, gain=gain, y_static=Yo, X_static=Uo, weighing_matrix=w
)

result = {"w1": w[0, :], "w2": w[1, :], "J_ls": J[0, :], "J_sg": J[1, :], "||J||:": E}

pd.DataFrame(result)
```

| w1       | w2       | J_ls      | J_sg         | $\lVert J \rVert$ |
| -------- | -------- | --------- | ------------ | ----------------- |
| 1.000000 | 0.000000 | 17.407256 | 3.579461e+01 | 39.802849         |
| 0.997012 | 0.002988 | 17.407528 | 2.109260e-01 | 17.408806         |
| 0.994033 | 0.005967 | 17.407540 | 2.082067e-01 | 17.408785         |
| 0.991063 | 0.008937 | 17.407559 | 2.056636e-01 | 17.408774         |
| 0.988102 | 0.011898 | 17.407585 | 2.031788e-01 | 17.408771         |
| ...      | ...      | ...       | ...          | ...               |
| 0.002555 | 0.997445 | 17.511596 | 3.340081e-07 | 17.511596         |
| 0.002547 | 0.997453 | 17.511596 | 3.320125e-07 | 17.511596         |
| 0.002540 | 0.997460 | 17.511597 | 3.300289e-07 | 17.511597         |
| 0.002532 | 0.997468 | 17.511598 | 3.280571e-07 | 17.511598         |
| 0.002525 | 0.997475 | 17.511599 | 3.260972e-07 | 17.511599         |

```python
# Writing the results
model.theta = theta[position, :].reshape(-1, 1)
model.final_model = bi_objective_gain.final_model
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
r = pd.DataFrame(
    results(
        model.final_model,
        model.theta,
        model.err,
        model.n_terms,
        err_precision=3,
        dtype="sci",
    ),
    columns=["Regressors", "Parameters", "ERR"],
)

r
```

|   | Regressors        | Parameters   | ERR         |
|---|------------------|--------------|-------------|
| 0 | 1                | 1.4853E+00   | 9.999E-01   |
| 1 | y(k-1)           | 5.4940E-01   | 2.042E-05   |
| 2 | y(k-2)           | 4.0806E-01   | 1.108E-06   |
| 3 | x1(k-1)          | -1.2581E+01  | 4.688E-06   |
| 4 | x1(k-2)          | 1.2210E+01   | 3.922E-07   |
| 5 | x1(k-1)^2        | 8.1686E+00   | 8.389E-07   |
| 6 | x1(k-2)x1(k-1)   | -1.1122E+01  | 5.690E-07   |
| 7 | x1(k-2)^2        | 2.9455E+00   | 3.827E-06   |


```python
plot_results(y=y_valid, yhat=yhat, n=1000)
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_mo_4.png?raw=true)

```python
plt.figure(12)
plt.title("Gain")
plt.plot(
    Uo,
    gain,
    linewidth=1.5,
    linestyle="-",
    marker="o",
    label="Buck converter static gain",
)

plt.plot(
    Uo,
    HR.dot(model.theta),
    linestyle="-",
    marker="^",
    linewidth=1.5,
    label="NARX model gain",
)
plt.xlabel("$\\bar{u}$")
plt.ylabel("$\\bar{g}$")
plt.legend()
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_static_gain_mo_4.png?raw=true)

```python
plt.figure(11)
plt.title("Costs Functions")
plt.plot(J[1, :], J[0, :], "o")
plt.xlabel("Gain Information")
plt.ylabel("Prediction Error")
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/buck_pareto_mo_4.png?raw=true)

sendo o $\theta$ selecionado:

| Theta      | SysIdentPy   | IniciacaoCientifica2007 |
| ---------- | ------------ | ----------------------- |
| $\theta_1$ | 0.54939785   | 0.54937289              |
| $\theta_2$ | 0.40805603   | 0.40810168              |
| $\theta_3$ | 1.48525190   | 1.48663719              |
| $\theta_4$ | -12.58066084 | -12.58127183            |
| $\theta_5$ | 8.16862622   | 8.16780294              |
| $\theta_6$ | -11.12171897 | -11.11998621            |
| $\theta_7$ | 12.20954849  | 12.20927355             |
| $\theta_8$ | 2.94548501   | 2.9446532               |

onde:

$$
E_{Scilab} =  17.408997
$$

e:

$$
E_{Python} = 17.408781
$$

## Informações Adicionais

Você também pode acessar as matrizes Q e H usando os seguintes métodos

Matriz Q:

```python
bi_objective_gain.build_static_function_information(Uo, Yo)[1]
```

Matriz H+R:

```python
bi_objective_gain.build_static_gain_information(Uo, Yo, gain)[1]
```


