## O Método `predict` no SysIdentPy

Antes de entrar no processo de validação em Identificação de Sistemas, é essencial entender como o método `predict` funciona no SysIdentPy.

### Usando o Método `predict`

Um uso típico do método `predict` no SysIdentPy é assim:

```python
yhat = model.predict(X=x_test, y=y_test)
```

Usuários do SysIdentPy frequentemente têm duas dúvidas comuns sobre este método:

1. Por que precisamos passar os dados de teste, `y_test`, como argumento no método `predict`?
2. Por que os valores iniciais preditos são idênticos aos valores nos dados de teste?

Para responder a essas perguntas, vamos primeiro explicar os conceitos de predição infinitos passos à frente, predição n passos à frente e predição um passo à frente em sistemas dinâmicos.

### Predição Infinitos Passos à Frente

A predição infinitos passos à frente, também conhecida como *free run simulation*, refere-se a fazer predições usando valores previamente **preditos**, $\hat{y}_{k-n_y}$, no loop de predição.

Por exemplo, considere os seguintes dados de entrada e saída de teste:

$$
x_{test} = [1, 2, 3, 4, 5, 6, 7]
$$

$$
y_{test} = [8, 9, 10, 11, 12, 13, 14]
$$

Suponha que queremos validar um modelo $m$ definido por:

$$
m \rightarrow y_k = 1*y_{k-1} + 2*x_{k-1}
$$

Para predizer o primeiro valor, precisamos de acesso tanto a $y_{k-1}$ quanto a $x_{k-1}$. Esse requisito explica por que você precisa passar `y_test` como argumento no método `predict`. Isso também responde à segunda pergunta: o SysIdentPy requer que o usuário forneça as condições iniciais explicitamente. Os dados `y_test` passados no método `predict` não são usados inteiramente; apenas os valores iniciais necessários para a estrutura de lags do modelo são usados.

Neste exemplo, o lag máximo do modelo é 1, então precisamos apenas de 1 condição inicial. Os valores preditos, `yhat`, são então calculados da seguinte forma:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*14 + 2*3 = 20
yhat(4) = 1*20 + 2*4 = 28
```

Como mostrado, o primeiro valor de `yhat` corresponde ao primeiro valor de `y_test` porque ele serve como condição inicial. Outro ponto importante é que o loop de predição usa os valores previamente **preditos**, não os valores reais de `y_test`, e é por isso que é chamado de infinitos passos à frente ou free run simulation.

Em identificação de sistemas, frequentemente buscamos modelos que tenham bom desempenho em predições infinitos passos à frente. Como o erro de predição se propaga ao longo do tempo, um modelo que mostra bom desempenho em free run simulation é considerado um modelo robusto.

No SysIdentPy, os usuários só precisam passar as condições iniciais ao realizar uma predição infinitos passos à frente. Se você passar apenas as condições iniciais, os resultados serão os mesmos! Portanto

```python
yhat = model.predict(X=x_test, y=y_test)
```

é na verdade o mesmo que

```python
yhat = model.predict(X=x_test, y=y_test[:model.max_lag].reshape(-1, 1))
```

> `model.max_lag` pode ser acessado após ajustarmos o modelo usando o código abaixo.

```python
model = FROLS(
	order_selection=False,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
    e_tol=0.9999
    n_terms=15
)
model.fit(X=x, y=y)
model.max_lag
```

> É importante mencionar que, na versão atual do SysIdentPy, o lag máximo considerado é na verdade o lag máximo entre as definições de `xlag` e `ylag`. Isso é importante porque você pode passar `ylag = xlag = 10` e o modelo final, após a seleção de estrutura de modelo, selecionar termos onde o lag máximo é 3. Você tem que passar 10 condições iniciais, mas internamente os cálculos são feitos usando os regressores corretos. Isso é necessário devido à forma como os regressores são criados após o modelo ser ajustado. Portanto, é recomendado usar `model.max_lag` para ter certeza.

### Predição 1 Passo à Frente

A diferença entre predição 1 passo à frente e predição infinitos passos à frente é que o modelo usa os valores reais anteriores de `y_test` no loop ao invés dos valores preditos `yhat`. E essa é uma diferença enorme e importante. Vamos fazer a predição usando o método 1 passo à frente:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*9 + 2*2 = 13
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*11 + 2*4 = 19
e assim por diante
```

O modelo usa valores reais no loop e apenas prediz o próximo valor. O erro de predição, neste caso, é sempre corrigido porque não estamos propagando o erro usando os valores preditos no loop.

O método `predict` do SysIdentPy permite que o usuário realize uma predição 1 passo à frente configurando `steps_ahead=1`

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=1)
```

Neste caso, como você pode imaginar, precisamos passar todos os dados de `y_test` porque o método precisa acessar os valores reais em cada iteração. Se você passar apenas as condições iniciais, `yhat` terá apenas as condições iniciais mais 1 amostra adicional, que é a predição 1 passo à frente. Para predizer outro ponto, você precisaria passar as novas condições iniciais novamente e assim por diante. O SysIdentPy já faz tudo isso para você, então apenas passe todos os dados que você quer validar usando o método 1 passo à frente.

### Predição n Passos à Frente

A predição n passos à frente é quase a mesma que a de 1 passo à frente, mas aqui você pode definir o número de passos à frente que quer testar seu modelo. Se você configurar `steps_ahead=5`, por exemplo, significa que os primeiros 5 valores serão preditos usando `yhat` no loop, mas então o processo é *reiniciado* alimentando os valores reais em `y_test` na próxima iteração, então realizando outras 5 predições usando o `yhat` e assim por diante. Vamos verificar o exemplo considerando `steps_ahead=2`:

```python
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*16 + 2*4 = 24
e assim por diante
```

## Desempenho do Modelo

A validação de modelos é uma das partes mais cruciais em identificação de sistemas. Como mencionamos antes, em identificação de sistemas estamos tentando modelar a dinâmica do processo para tarefas como projeto de controle. Em tais casos, não podemos apenas confiar em métricas de regressão, mas também garantir que os resíduos sejam imprevisíveis em várias combinações de entradas e saídas passadas ([Billings, S. A. and Voon, W. S. F., "Structure detection and model validity tests in the identification of nonlinear systems"](https://digital-library.theiet.org/content/journals/10.1049/ip-d.1983.0034)). Um teste estatístico frequentemente usado é o RMSE normalizado, chamado RRSE, que pode ser expresso por

$$
\begin{equation}
        \textrm{RRSE}= \frac{\sqrt{\sum\limits_{k=1}^{n}(y_k-\hat{y}_k)^2}}{\sqrt{\sum\limits_{k=1}^{n}(y_k-\bar{y})^2}},
\end{equation}
\tag{1}
$$

onde $\hat{y}_k \in \mathbb{R}$ é a saída predita pelo modelo e $\bar{y} \in \mathbb{R}$ é a média da saída medida $y_k$. O RRSE fornece alguma indicação sobre a qualidade do modelo, mas concluir sobre o melhor modelo avaliando apenas essa quantidade pode levar a uma interpretação incorreta, como mostrado no exemplo a seguir.

Considere os modelos

$$
y_{{_a}k} = 0.7077y_{{_a}k-1} + 0.1642u_{k-1} + 0.1280u_{k-2}
$$

e

$$y_{{_b}k}=0.7103y_{{_b}k-1} + 0.1458u_{k-1} + 0.1631u_{k-2} -1467y^3_{{_b}k-1} + 0.0710y^3_{{_b}k-2} +0.0554y^2_{{_b}k-3}u_{k-3}$$

definidos em [Meta Model Structure Selection: An Algorithm For Building Polynomial NARX Models For Regression And Classification](https://ufsj.edu.br/portal2-repositorio/File/ppgel/225-2020-02-17-DissertacaoWilsonLacerda.pdf). O primeiro resulta em $RRSE = 0.1202$ enquanto o último resulta em $RRSE~=0.0857$. Embora o modelo $y_{{_b}k}$ ajuste melhor os dados, ele é apenas uma representação enviesada para um conjunto de dados e não uma boa descrição de todo o sistema.

O RRSE (ou qualquer outra métrica) mostra que testes de validação podem precisar ser realizados cuidadosamente. Outra prática tradicional é dividir o conjunto de dados em duas partes. Nesse sentido, pode-se testar os modelos obtidos da parte de estimação dos dados usando dados específicos para validação. No entanto, o desempenho de um passo à frente de modelos NARX geralmente resulta em interpretações equivocadas porque mesmo modelos fortemente enviesados podem ajustar bem os dados. Portanto, uma abordagem de free run simulation geralmente permite uma melhor interpretação se o modelo é adequado ou não ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)).

Testes estatísticos para modelos SISO baseados nas funções de correlação foram propostos em ([Billings, S. A. and Voon, W. S. F., "A prediction-error and stepwise-regression estimation algorithm for non-linear systems"](https://www.tandfonline.com/doi/abs/10.1080/00207178608933633)), ([Model validity tests for non-linear signal processing applications](https://www.tandfonline.com/doi/abs/10.1080/00207179108934155)). Os testes são:

$$
\begin{align}
    \phi_{_{\xi \xi}\tau} &= E\{\xi_k \xi_{k-\tau}\} = \delta_{\tau}, \\
    \phi_{_{\xi x}\tau} &= E\{\xi_k x_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{\xi \xi x}\tau} &= E\{\xi_k \xi_{k-\tau} x_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{x^2 \xi}\tau} &= E\{(u^2_k - E\{x^2_k\})\xi_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{x^2 \xi^2}\tau} &= E\{(u^2_k - E\{x^2_k\})\xi^2_{k-\tau}\} = 0 \forall \tau, \\
    \phi_{_{(y\xi) x^2}\tau} &= E\{(y_k\xi_k - E\{y_k\xi_k\})(x^2_{k-\tau} - E\{x^2_k\})\} = 0 \forall \tau,
\end{align}
\tag{2}
$$


onde $\delta$ é a função delta de Dirac e a função de correlação cruzada $\phi$ é denotada por ([Billings, S. A. and Voon, W. S. F.](https://digital-library.theiet.org/content/journals/10.1049/ip-d.1983.0034)):

$$
\begin{equation}
\phi_{{_{ab}}\tau} = \frac{\frac{1}{n}\sum\limits_{k=1}^{n-\tau}(a_k - \hat{a})(b_{k+\tau}-\hat{b})}{\sqrt{\frac{1}{n}\sum\limits_{k=1}^{n}(a_k-\hat{a})^2} \sqrt{\frac{1}{n}\sum\limits_{k=1}^{n}(b_k-\hat{b})^2}} = \frac{\sum\limits_{k=1}^{n-\tau}(a_k - \hat{a})(b_{k+\tau}-\hat{b})}{\sqrt{\sum\limits_{k=1}^{n}(a_k-\hat{a})^2} \sqrt{\sum\limits_{k=1}^{n}(b_k-\hat{b})^2}},
\end{equation}
\tag{3}
$$

onde $a$ e $b$ são duas sequências de sinais. Se os testes são verdadeiros, então os resíduos do modelo podem ser considerados como ruído branco.

### Métricas Disponíveis no SysIdentPy

O SysIdentPy fornece as seguintes métricas de regressão prontas para uso:

- forecast_error
- mean_forecast_error
- mean_squared_error
- root_mean_squared_error
- normalized_root_mean_squared_error
- root_relative_squared_error
- mean_absolute_error
- mean_squared_log_error
- median_absolute_error
- explained_variance_score
- r2_score
- symmetric_mean_absolute_percentage_error

Para usá-las, o usuário só precisa importar a métrica desejada usando, por exemplo

```python
from sysidentpy.metrics import root_relative_squared_error
```

O SysIdentPy também fornece métodos para calcular e analisar a correlação dos resíduos

```python

from sysidentpy.utils.plotting import plot_residues_correlation
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
```

Vamos verificar as métricas do sistema eletromecânico modelado no Capítulo 4.

```python
import numpy as np
import pandas as pd
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.utils.display_results import results
from sysidentpy.utils.plotting import plot_residues_correlation, plot_results
from sysidentpy.residues.residues_correlation import (
    compute_residues_autocorrelation,
    compute_cross_correlation,
)
from sysidentpy.metrics import root_relative_squared_error

df1 = pd.read_csv("examples/datasets/x_cc.csv")
df2 = pd.read_csv("examples/datasets/y_cc.csv")

x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)

basis_function = Polynomial(degree=2)
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
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)

ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c9_dc_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_1.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_1.png?raw=true)

O RRSE é 0.0800, que é uma métrica muito boa. No entanto, podemos ver que os resíduos têm algumas autocorrelações altas e com a entrada. Isso significa que nosso modelo talvez não seja bom o suficiente como poderia ser.

Vamos verificar o que acontece se aumentarmos `xlag`, `ylag` e mudarmos o algoritmo de estimação de parâmetros de Least Squares para Recursive Least Squares

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=50,
    ylag=5,
    xlag=5,
    info_criteria="bic",
    estimator=RecursiveLeastSquares(unbiased=False),
    basis_function=basis_function
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_valid, y=y_valid)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

Agora o RRSE é 0.0568 e temos uma melhor correlação residual!

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_dc_2.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_2.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_2.png?raw=true)

No final das contas, o melhor modelo será aquele que satisfaz as necessidades do usuário. No entanto, é importante entender como analisar os modelos para que você possa ter uma ideia se pode obter algumas melhorias sem muito trabalho.

Por curiosidade, vamos verificar como o modelo se comporta se executarmos uma predição 1 passo à frente. Não precisamos ajustar o modelo novamente, apenas fazer outra predição usando a opção 1 passo.

```python
yhat = model.predict(X=x_valid, y=y_valid, steps_ahead=1)
rrse = root_relative_squared_error(y_valid, yhat)
print(rrse)
# plot only the first 100 samples (n=100)
plot_results(y=y_valid, yhat=yhat, n=100)
ee = compute_residues_autocorrelation(y_valid, yhat)
plot_residues_correlation(data=ee, title="Residues", ylabel="$e^2$")
x1e = compute_cross_correlation(y_valid, yhat, x_valid)
plot_residues_correlation(data=x1e, title="Residues", ylabel="$x_1e$")
```

O mesmo modelo, mas avaliando a predição 1 passo à frente, agora retorna um RRSE$= 0.02044$ e os resíduos estão ainda melhores. Mas lembre-se, isso é esperado, como explicado na seção anterior.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_dc_3.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ee_3.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/c09_ex_03.png?raw=true)
