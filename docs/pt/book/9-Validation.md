## O Método `predict` no SysIdentPy

Antes de entrar no processo de validação, é importante entender o que o método `predict` calcula. Em sistemas dinâmicos, a saída predita pode depender de valores passados da própria saída. Portanto, duas predições feitas pelo mesmo modelo podem ser muito diferentes dependendo de quais valores são realimentados no loop de predição.

Um uso típico do método `predict` no SysIdentPy é

```python
yhat = model.predict(X=x_test, y=y_test)
```

As explicações a seguir usam o FROLS com base polinomial, a mesma configuração adotada no restante do capítulo. Esse caminho oferece simulação livre e predições de um e de $n$ passos para modelos NAR e NARMAX. Em modelos NFIR não existe realimentação da saída; por isso, os três modos são equivalentes à mesma predição feed-forward.

Duas dúvidas aparecem com frequência:

1. Por que precisamos passar `y_test` para predizer a saída?
2. Por que os primeiros valores de `yhat` são iguais aos primeiros valores de `y_test`?

A resposta está nas condições iniciais. Considere o modelo

$$
y_k = y_{k-1} + 2x_{k-1}.
\tag{9.1}
$$

Para calcular $\hat{y}_1$, o modelo precisa conhecer $y_0$. Se o maior lag do modelo for $n_{\text{lag}}$, serão necessárias pelo menos $n_{\text{lag}}$ amostras da saída para iniciar a recursão. No SysIdentPy, esse número fica disponível em `model.max_lag` assim que o modelo é configurado.

```python
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares

basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=False,
    n_terms=15,
    ylag=2,
    xlag=2,
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.max_lag
```

Essas `max_lag` amostras são as condições iniciais necessárias para começar a predição. O SysIdentPy as copia para o início de `yhat`; elas não são predições produzidas pelo modelo e, por isso, devem ser excluídas do cálculo das métricas de validação.

Os dados que definem o intervalo de predição dependem do modo de validação. Na simulação livre, modelos com entradas seguem a sequência de entrada disponível, enquanto um modelo NAR usa `forecast_horizon`. Nas predições de um e de $n$ passos à frente, a sequência de saída medida define o intervalo, pois ela é usada para corrigir ou reiniciar a recursão. As próximas seções ilustram essas diferenças.

Para o modelo NARMAX desse exemplo, `max_lag` é obtido a partir dos lags configurados em `xlag` e `ylag`, e não apenas dos termos que permaneceram no modelo final. Se `xlag=ylag=10`, por exemplo, ainda será necessário fornecer dez condições iniciais mesmo que o maior lag entre os regressores selecionados seja menor.

### Predição de Infinitos Passos à Frente

A predição de infinitos passos à frente, também chamada de *free run simulation* ou simulação livre, usa as saídas anteriormente **preditas** para continuar a recursão. No método `predict`, esse é o comportamento obtido quando `steps_ahead=None`, que é o valor padrão.

Considere

$$
x_{test} = [1, 2, 3, 4, 5, 6, 7]
$$

e

$$
y_{test} = [8, 9, 10, 11, 12, 13, 14].
$$

Para o modelo da Equação 9.1, com lag máximo igual a 1, a simulação começa com $y_0=8$:

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*14 + 2*3 = 20
yhat(4) = 1*20 + 2*4 = 28
```

Depois da condição inicial, nenhum outro valor real de `y_test` é usado. O erro cometido em uma amostra passa a influenciar as amostras seguintes. É justamente essa propagação que torna a simulação livre um teste importante para modelos dinâmicos.

Quando `X` é fornecido, seu número de amostras define o horizonte da simulação. Por isso, passar toda a saída ou apenas as condições iniciais produz o mesmo resultado:

```python
yhat = model.predict(X=x_test, y=y_test)
yhat_from_initial_conditions = model.predict(
    X=x_test,
    y=y_test[: model.max_lag],
)
```

Nesse caso, um valor fornecido em `forecast_horizon` não substitui o comprimento de `X`: é `X.shape[0]` que determina quantas amostras serão retornadas, incluindo o prefixo das condições iniciais.

Para modelos NAR, que não possuem entrada `X`, o horizonte normalmente deve ser informado por meio de `forecast_horizon`:

```python
yhat = model.predict(
    X=None,
    y=y_initial,
    forecast_horizon=100,
)
```

Nesse exemplo, o resultado contém as condições iniciais seguidas de 100 valores preditos. Portanto, `forecast_horizon` controla o comprimento da previsão quando não existe `X`; ele não altera quais saídas são realimentadas. Essa última escolha é feita por `steps_ahead`. O valor de `forecast_horizon` deve ser um inteiro não negativo.

### Predição de Um Passo à Frente

Na predição um passo à frente, o modelo usa a saída **medida** anterior em cada nova predição. Para o mesmo exemplo,

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*9 + 2*2 = 13
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*11 + 2*4 = 19
```

O erro não é propagado indefinidamente porque a recursão é corrigida pela saída medida a cada amostra. No SysIdentPy, essa opção é definida por `steps_ahead=1`:

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=1)
```

Neste caso, `y_test` precisa conter todo o intervalo que será avaliado. Modelos NARMAX e NFIR também exigem `X` com o mesmo número de amostras. Resultados de um passo à frente costumam ser melhores do que resultados de simulação livre, mas respondem a uma pergunta menos exigente: quão bem o modelo prediz a próxima amostra quando conhece o histórico real da saída?

### Predição de n Passos à Frente

A predição de $n$ passos fica entre os dois casos anteriores. Dentro de cada bloco, o modelo usa as próprias predições. Ao completar `steps_ahead` amostras, a recursão é reiniciada com valores medidos e um novo bloco é calculado.

Com `steps_ahead=2`, temos

```text
y_initial = yhat(0) = 8
yhat(1) = 1*8 + 2*1 = 10
yhat(2) = 1*10 + 2*2 = 14
yhat(3) = 1*10 + 2*3 = 16
yhat(4) = 1*16 + 2*4 = 24
```

No SysIdentPy:

```python
yhat = model.predict(X=x_test, y=y_test, steps_ahead=2)
```

`steps_ahead` deve ser um inteiro positivo. Cada bloco usa as `max_lag` saídas medidas anteriores como condições iniciais. Se restarem menos de `steps_ahead` amostras, o último bloco é reduzido ao intervalo restante. Quanto maior o valor, mais o teste se aproxima da simulação livre e maior é a oportunidade para que erros sejam propagados.

Em modelos NFIR não existe recursão da saída a reiniciar. Consequentemente, `steps_ahead=None`, `steps_ahead=1` e qualquer valor válido de $n$ passos usam o mesmo cálculo feed-forward. O argumento `y` continua obrigatório para o prefixo inicial e o alinhamento da saída, mas seus valores após `max_lag` não afetam as predições NFIR.

### Alinhando a Saída Antes da Avaliação

O SysIdentPy copia as primeiras `model.max_lag` amostras de `y` para o início de `yhat`, pois elas são condições iniciais, não predições. Incluí-las no cálculo de uma métrica reduz o erro artificialmente. A comparação deve começar depois desse prefixo:

```python
start = model.max_lag
y_eval = y_test[start:]
yhat_eval = yhat[start:]
```

O mesmo alinhamento deve ser usado nas métricas e nas correlações dos resíduos. Essa regra é especialmente importante ao comparar modelos com valores diferentes de `max_lag`.

## Desempenho e Validação do Modelo

Uma métrica resume a distância entre a saída medida e a saída predita. Isso é necessário, mas não suficiente para validar um modelo dinâmico. Dois modelos podem apresentar erros numéricos semelhantes e, ainda assim, representar dinâmicas muito diferentes.

Sempre que possível, o desempenho deve ser medido em dados que não participaram da estimação dos parâmetros nem da seleção da estrutura. O erro no conjunto de estimação mostra quão bem o modelo se ajustou aos dados usados para construí-lo; o erro em um conjunto de validação separado fornece evidência sobre sua capacidade de generalização. Essa separação, porém, não substitui a análise dos resíduos nem a identificação do modo de predição usado.

Defina o resíduo como

$$
e_k = y_k - \hat{y}_k.
\tag{9.2}
$$

Se o modelo explicou toda a informação determinística disponível nos dados, não deve ser possível predizer $e_k$ usando resíduos anteriores ou entradas passadas. Em termos práticos, procuramos resíduos pequenos e sem padrões sistemáticos. Um RRSE baixo com resíduos fortemente correlacionados indica que ainda existe estrutura não explicada. Essa combinação entre desempenho numérico e análise residual é central nos testes de validade de modelos NARMAX ([Billings e Voon, 1983](https://doi.org/10.1049/ip-d.1983.0034)).

Essa distinção também ajuda a entender por que o modo de predição precisa ser informado junto com qualquer métrica. Um erro calculado em predição de um passo não é diretamente comparável ao mesmo erro calculado em simulação livre.

### Correlação dos Resíduos no SysIdentPy

O SysIdentPy disponibiliza duas funções públicas para essa análise:

```python
from sysidentpy.residues.residues_correlation import (
    compute_cross_correlation,
    compute_residues_autocorrelation,
)
from sysidentpy.utils.plotting import plot_residues_correlation
```

Os argumentos já devem estar alinhados e representar um sinal por chamada. Se
`X` tiver várias colunas de entrada, chame `compute_cross_correlation`
separadamente para cada entrada, em vez de passar a matriz completa como `arr`.

`compute_residues_autocorrelation(y, yhat)` calcula a autocorrelação normalizada dos resíduos e retorna os $N$ lags não negativos. A implementação correlaciona diretamente $e$ com ele mesmo, sem subtrair a média dos resíduos, e normaliza o resultado pelo valor no lag zero. Quando a energia dos resíduos é diferente de zero, esse primeiro valor é 1; são os demais lags que devem ser inspecionados. Valores relevantes fora de zero sugerem que resíduos passados ainda carregam informação sobre o resíduo atual.

`compute_cross_correlation(y, yhat, arr)` calcula a correlação cruzada entre os resíduos e outro sinal alinhado, normalmente a entrada usada para excitar o sistema, e retorna os primeiros $\lfloor N/2\rfloor$ lags não negativos. Correlações relevantes sugerem que parte do efeito dessa entrada não foi capturada pelo modelo. Esses dois testes procuram dependências lineares específicas; não esgotam todos os padrões que podem existir nos resíduos.

As duas funções retornam uma tupla na ordem `correlation, upper_bound, lower_bound`. Para uma série residual com $N$ amostras, a implementação usa os limites aproximados

$$
\pm\frac{1.96}{\sqrt{2N-1}}.
\tag{9.3}
$$

Esses limites são a referência visual nominal e aproximada de 95% retornada pela implementação. Alguns pontos fora da faixa podem aparecer por acaso, principalmente quando muitos lags são avaliados. Portanto, os gráficos são ferramentas de diagnóstico: eles não provam sozinhos que um modelo é válido ou inválido.

`plot_residues_correlation(data=...)` recebe diretamente a tupla retornada por uma dessas funções, desenha a correlação e sombreia os limites. Por padrão, o gráfico mostra os primeiros 100 lags; o argumento `n` altera essa quantidade. Se todos os resíduos forem zero, a normalização da autocorrelação envolve uma divisão $0/0$ e a implementação retorna um vetor com `NaN`. A correlação cruzada também fica indefinida quando algum dos sinais normalizados tem energia zero, mas, nesse caso, a implementação atual lança `ZeroDivisionError`. Esses resultados representam uma correlação normalizada que não pode ser calculada, e não a detecção de correlação.

## Métricas Disponíveis no SysIdentPy

O módulo `sysidentpy.metrics` fornece 13 funções públicas para erros de regressão e previsão. Todas comparam valores observados, `y`, com valores preditos, `yhat`. `forecast_error` retorna o erro de cada amostra; as demais retornam um escalar. O MASE é a única métrica desse módulo que também exige os dados de treinamento e, opcionalmente, um período sazonal.

| Grupo | Funções | Unidade ou escala | Melhor valor | Principal característica |
|---|---|---|---:|---|
| Erro com sinal | `forecast_error`, `mean_forecast_error` | Mesma unidade de $y$ | 0 | Mostram a direção do erro e o viés médio |
| Erro quadrático | `mean_squared_error`, `root_mean_squared_error` | $y^2$ no MSE; unidade de $y$ no RMSE | 0 | Penalizam mais os erros grandes |
| Erro quadrático normalizado | `normalized_root_mean_squared_error`, `root_relative_squared_error` | Adimensional | 0 | Facilitam comparações de escala, mas usam referências diferentes |
| Erro absoluto | `mean_absolute_error`, `median_absolute_error` | Mesma unidade de $y$ | 0 | Têm interpretação direta e menor influência de erros extremos |
| Erro absoluto escalado | `mean_absolute_scaled_error` | Adimensional | 0 | Compara o MAE com uma previsão ingênua calculada no treino |
| Erro logarítmico e percentual | `mean_squared_log_error`, `symmetric_mean_absolute_percentage_error` | Diferença logarítmica ao quadrado; porcentagem | 0 | Avaliam diferenças relativas, sob restrições próprias |
| Qualidade do ajuste | `explained_variance_score`, `r2_score` | Adimensional | 1 | Comparam o erro com a variabilidade da saída |

### Erro de Previsão e Erro Médio de Previsão

`forecast_error` implementa diretamente a Equação 9.2. Para uma única saída, retorna o vetor $[e_1,\ldots,e_N]$; de forma mais geral, a operação `y - yhat` preserva a forma resultante dos arrays. Como o SysIdentPy usa $e_k=y_k-\hat{y}_k$, um erro positivo indica subestimação e um erro negativo indica superestimação.

`mean_forecast_error` calcula

$$
\mathrm{MFE}=\frac{1}{N}\sum_{k=1}^{N}e_k.
\tag{9.4}
$$

O MFE, expresso na mesma unidade de $y$, é útil para detectar viés médio, mas erros positivos e negativos podem se cancelar. Um MFE próximo de zero não implica, por si só, predições precisas.

### MSE e RMSE

O erro quadrático médio é

$$
\mathrm{MSE}=\frac{1}{N}\sum_{k=1}^{N}(y_k-\hat{y}_k)^2,
\tag{9.5}
$$

e sua raiz é

$$
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\tag{9.6}
$$

O MSE está na unidade de $y$ ao quadrado. O RMSE volta à unidade original da saída e costuma ser mais fácil de interpretar. Como ambos elevam o erro ao quadrado, erros grandes recebem peso elevado.

### NRMSE e RRSE

No SysIdentPy, `normalized_root_mean_squared_error` normaliza o RMSE pelo intervalo observado da saída:

$$
\mathrm{NRMSE}=\frac{\mathrm{RMSE}}{\max(y)-\min(y)}.
\tag{9.7}
$$

Essa definição torna o resultado adimensional, mas também o deixa sensível a valores extremos que ampliem o intervalo. Se `y` for constante, o denominador será zero. A implementação retorna 0 para uma predição perfeita e `inf` para uma predição imperfeita; um erro contendo `NaN` continua sendo `NaN`.

O `root_relative_squared_error` usa a média da saída como referência:

$$
\mathrm{RRSE}=
\sqrt{
\frac{\sum_{k=1}^{N}(y_k-\hat{y}_k)^2}
{\sum_{k=1}^{N}(y_k-\bar{y})^2}
}.
\tag{9.8}
$$

Para uma saída não constante, RRSE menor que 1 significa que o modelo supera a predição constante $\hat{y}_k=\bar{y}$ no mesmo conjunto de avaliação. RRSE igual a 1 indica desempenho equivalente e RRSE maior que 1 indica desempenho pior. Para uma saída constante, o comportamento é o mesmo do NRMSE: 0 para uma predição perfeita, `inf` para uma imperfeita e `NaN` quando o erro não é finito.

Embora ambas sejam chamadas de métricas normalizadas, NRMSE e RRSE não são intercambiáveis. A primeira divide pelo intervalo da saída; a segunda compara a soma dos erros quadráticos com a variabilidade em torno da média.

### MAE, Erro Absoluto Mediano e MASE

O erro absoluto médio é

$$
\mathrm{MAE}=\frac{1}{N}\sum_{k=1}^{N}|y_k-\hat{y}_k|.
\tag{9.9}
$$

O `median_absolute_error` substitui a média pela mediana dos erros absolutos. MAE e erro absoluto mediano mantêm a unidade de `y`, mas a mediana é menos influenciada por uma pequena quantidade de erros extremos. Essa robustez se refere à agregação: um valor extremo ainda produz um erro absoluto grande, mas tem menor influência sobre a mediana do conjunto.

O `mean_absolute_scaled_error` normaliza o MAE usando o erro de uma previsão ingênua calculada nos dados de treinamento. A implementação generaliza para um período sazonal configurável a escala proposta por [Hyndman e Koehler (2006)](https://doi.org/10.1016/j.ijforecast.2006.03.001):

$$
\mathrm{MASE}=
\frac{\frac{1}{N}\sum_{k=1}^{N}|y_k-\hat{y}_k|}
{\frac{1}{T-m}\sum_{t=m+1}^{T}|y^{\mathrm{train}}_t-y^{\mathrm{train}}_{t-m}|},
\tag{9.10}
$$

onde $m$ é `seasonal_period`. O valor padrão é $m=1$, correspondente à previsão ingênua de um passo. Um MASE menor que 1 indica que o MAE do modelo é menor que o erro ingênuo médio dentro do conjunto de treinamento; MASE igual a 1 indica desempenho equivalente.

Sua assinatura é diferente das demais métricas:

```python
mean_absolute_scaled_error(
    y,
    yhat,
    y_train,
    seasonal_period=1,
)
```

O MASE aceita apenas uma saída, em vetores unidimensionais ou matrizes com uma coluna. `y` e `yhat` devem ter a mesma forma, `seasonal_period` deve ser um inteiro positivo e `y_train` precisa conter mais amostras do que esse período. Se o erro ingênuo for zero, a implementação retorna 0 para uma predição perfeita, `inf` para uma imperfeita e preserva `NaN`.

### MSLE e SMAPE

O erro logarítmico quadrático médio é

$$
\mathrm{MSLE}=\frac{1}{N}\sum_{k=1}^{N}
\left[\log(1+y_k)-\log(1+\hat{y}_k)\right]^2.
\tag{9.11}
$$

O MSLE reduz a influência de diferenças absolutas em valores grandes e enfatiza diferenças relativas. Na implementação do SysIdentPy, todos os valores de `y` e `yhat` devem ser estritamente maiores que -1. Caso contrário, a função lança `ValueError` porque `log1p` não está definido no domínio real.

O erro percentual absoluto médio simétrico é calculado por

$$
\mathrm{SMAPE}=\frac{100}{N}\sum_{k=1}^{N}
\frac{2|y_k-\hat{y}_k|}{|y_k|+|\hat{y}_k|}.
\tag{9.12}
$$

Para uma única saída e valores finitos, o resultado varia de 0% a 200%. Quando $y_k=\hat{y}_k=0$, a contribuição daquela amostra é definida como zero. Apesar do nome, superestimações e subestimações de mesma magnitude não são necessariamente penalizadas da mesma forma.

### Variância Explicada e R²

O `explained_variance_score` calcula

$$
\mathrm{EVS}=1-\frac{\mathrm{Var}(y-\hat{y})}{\mathrm{Var}(y)},
\tag{9.13}
$$

enquanto o coeficiente de determinação é

$$
R^2=1-
\frac{\sum_{k=1}^{N}(y_k-\hat{y}_k)^2}
{\sum_{k=1}^{N}(y_k-\bar{y})^2}.
\tag{9.14}
$$

O melhor valor de ambos é 1, e resultados negativos são possíveis. A diferença principal aparece quando existe um deslocamento constante. Se $\hat{y}=y+c$, o resíduo tem variância zero e o EVS pode ser 1 mesmo com $c\neq0$. O $R^2$ inclui esse viés na soma dos erros quadráticos e será menor que 1.

Para uma saída constante, o $R^2$ retorna 1 quando a predição é perfeita e 0 quando há erro. No EVS, uma predição com deslocamento constante ainda retorna 1 porque a variância dos resíduos é zero; se os resíduos variarem, o resultado será 0. Esse comportamento é coerente com a diferença entre as duas definições, mas reforça por que o EVS não deve ser usado sozinho para detectar viés.

### Formas, Agregação e Valores Ausentes

Com exceção do MASE, as métricas não fazem uma validação comum de forma. Na prática, `y` e `yhat` devem ter a mesma forma e o mesmo alinhamento temporal; caso contrário, as regras de *broadcasting* do backend podem produzir um resultado diferente do pretendido ou lançar um erro. As métricas escalares geralmente agregam todos os elementos fornecidos. Duas exceções merecem atenção: `r2_score` calcula o $R^2$ de cada coluna de saída e retorna a média; o SMAPE soma todas as colunas, mas divide apenas pelo número de amostras, `y.shape[0]`. Portanto, para $q$ saídas, sua faixa teórica passa a ser de 0% a $200q$%. O MASE rejeita explicitamente múltiplas saídas.

As funções também não removem nem imputam valores ausentes. Para um alvo finito e não constante, um `NaN` em `yhat` aparece na posição correspondente de `forecast_error` e se propaga como `NaN` nas métricas escalares. NRMSE, RRSE e MASE também preservam `NaN` quando seus respectivos divisores são zero. EVS e $R^2$ têm uma exceção definida pela função interna usada para alvos constantes: se a variância do alvo é zero e o numerador é diferente de zero ou `NaN`, o resultado retornado é 0. Isso não representa tratamento do dado ausente; é apenas o resultado da regra de segurança implementada para o divisor nulo.

### Exemplo de Uso

```python
import numpy as np

from sysidentpy.metrics import (
    explained_variance_score,
    forecast_error,
    mean_absolute_error,
    mean_absolute_scaled_error,
    mean_forecast_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    normalized_root_mean_squared_error,
    r2_score,
    root_mean_squared_error,
    root_relative_squared_error,
    symmetric_mean_absolute_percentage_error,
)

y = np.array([3.0, -0.5, 2.0, 7.0])
yhat = np.array([2.5, 0.0, 2.0, 8.0])
y_train = np.array([1.0, 2.0, 3.0, 4.0])

errors = forecast_error(y, yhat)
bias = mean_forecast_error(y, yhat)
mse = mean_squared_error(y, yhat)
rmse = root_mean_squared_error(y, yhat)
nrmse = normalized_root_mean_squared_error(y, yhat)
rrse = root_relative_squared_error(y, yhat)
mae = mean_absolute_error(y, yhat)
median_ae = median_absolute_error(y, yhat)
mase = mean_absolute_scaled_error(y, yhat, y_train)
msle = mean_squared_log_error(y, yhat)
smape = symmetric_mean_absolute_percentage_error(y, yhat)
evs = explained_variance_score(y, yhat)
r2 = r2_score(y, yhat)
```

Essas métricas também participam do suporte à Array API do SysIdentPy quando o despacho é habilitado. Independentemente do backend, a escolha da métrica deve ser guiada pela pergunta de avaliação, e não apenas pela conveniência de obter um único número.

## Estudo de Caso: Sistema Eletromecânico

Vamos retomar o sistema eletromecânico apresentado no Capítulo 4. O objetivo aqui não é encontrar uma configuração ótima, mas mostrar como uma conclusão muda quando observamos a simulação livre, a predição de um passo e as correlações dos resíduos.

Os dados são carregados a partir de um commit específico do repositório `sysidentpy-data`, garantindo que o exemplo use sempre a mesma versão dos sinais.

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from sysidentpy.basis_function import Polynomial
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import (
    LeastSquares,
    RecursiveLeastSquares,
)
from sysidentpy.residues.residues_correlation import (
    compute_cross_correlation,
    compute_residues_autocorrelation,
)
from sysidentpy.utils.plotting import (
    plot_residues_correlation,
    plot_results,
)

data_url = (
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/"
    "4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/generator/"
)
df1 = pd.read_csv(f"{data_url}x_cc.csv", header=None)
df2 = pd.read_csv(f"{data_url}y_cc.csv", header=None)

x_train, x_valid = np.split(df1.iloc[::500].values, 2)
y_train, y_valid = np.split(df2.iloc[::500].values, 2)

x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_valid_scaled = x_scaler.transform(x_valid)
y_train_scaled = y_scaler.fit_transform(y_train)
y_valid_scaled = y_scaler.transform(y_valid)
```

Para evitar repetir o alinhamento, vamos criar uma pequena função de avaliação:

```python
def evaluate(model, *, steps_ahead=None):
    yhat = model.predict(
        X=x_valid,
        y=y_valid,
        steps_ahead=steps_ahead,
    )
    start = model.max_lag
    y_eval = y_valid[start:]
    yhat_eval = yhat[start:]
    x_eval = x_valid[start:]

    rrse = root_relative_squared_error(y_eval, yhat_eval)
    ee = compute_residues_autocorrelation(y_eval, yhat_eval)
    x1e = compute_cross_correlation(y_eval, yhat_eval, x_eval)
    return y_eval, yhat_eval, rrse, ee, x1e
```

Primeiro, usamos Least Squares, dois lags e seleção de ordem pelo BIC:

```python
basis_function = Polynomial(degree=2)
model = FROLS(
    order_selection=True,
    n_info_values=15,
    ylag=2,
    xlag=2,
    info_criteria="bic",
    estimator=LeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)

y_eval, yhat_eval, rrse, ee, x1e = evaluate(model)
print(rrse)

plot_results(y=y_eval, yhat=yhat_eval, n=100)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
668.9962328881088
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-ls-free-run.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-ls-residual-autocorrelation.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-ls-input-residual-correlation.png?raw=true)

O RRSE muito maior que 1 mostra que essa configuração é inadequada em simulação livre. A saída simulada se afasta rapidamente da saída medida, e a autocorrelação dos resíduos permanece elevada por muitos lags. Um gráfico limitado às primeiras amostras pode esconder a dimensão dessa divergência; por isso, a métrica e o gráfico devem ser analisados juntos.

Agora aumentamos os lags e usamos Recursive Least Squares:

```python
model = FROLS(
    order_selection=True,
    n_info_values=50,
    ylag=5,
    xlag=5,
    info_criteria="bic",
    estimator=RecursiveLeastSquares(unbiased=False),
    basis_function=basis_function,
)
model.fit(X=x_train, y=y_train)

y_eval, yhat_eval, rrse, ee, x1e = evaluate(model)
print(rrse)

plot_results(y=y_eval, yhat=yhat_eval, n=100)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
256.51064230974333
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-free-run.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-residual-autocorrelation.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-input-residual-correlation.png?raw=true)

O valor diminui em relação ao primeiro modelo, mas continua muito acima de 1. Portanto, seria incorreto chamar esse resultado de bom apenas porque houve uma melhora relativa. A simulação livre ainda é instável e os resíduos continuam apresentando dependência temporal.

Por fim, avaliamos exatamente o mesmo modelo em predição de um passo:

```python
y_eval, yhat_eval, rrse, ee, x1e = evaluate(
    model,
    steps_ahead=1,
)
print(rrse)

plot_results(
    y=y_eval,
    yhat=yhat_eval,
    n=100,
    title="One-step-ahead prediction",
)
plot_residues_correlation(
    data=ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
0.020984834884319806
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-one-step-ahead.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-one-step-residual-autocorrelation.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-one-step-input-residual-correlation.png?raw=true)

O RRSE agora parece excelente, embora o modelo seja o mesmo que apresentou RRSE de aproximadamente 257 em simulação livre. A diferença vem da realimentação da saída medida a cada passo, que impede a propagação do erro. A autocorrelação dos resíduos diminui de forma acentuada, mas a correlação com a entrada ainda apresenta lags fora dos limites aproximados. Esses resultados devem ser inspecionados, e não resumidos pelo RRSE.

### Condicionamento com Padronização Ajustada no Treino

Uma segunda avaliação mantém a classe do modelo, os atrasos, a base polinomial,
o estimador e a configuração de busca. A entrada e a saída são padronizadas por
transformações ajustadas exclusivamente no conjunto de treino. A validação usa
somente as médias e os desvios aprendidos nesse conjunto, evitando vazamento de
informação.

```python
scaled_model = FROLS(
    order_selection=True,
    n_info_values=50,
    ylag=5,
    xlag=5,
    info_criteria="bic",
    estimator=RecursiveLeastSquares(unbiased=False),
    basis_function=basis_function,
)
scaled_model.fit(X=x_train_scaled, y=y_train_scaled)

scaled_free_run_prediction = scaled_model.predict(
    X=x_valid_scaled,
    y=y_valid_scaled,
)
scaled_one_step_prediction = scaled_model.predict(
    X=x_valid_scaled,
    y=y_valid_scaled,
    steps_ahead=1,
)

if not (
    np.isfinite(scaled_free_run_prediction).all()
    and np.isfinite(scaled_one_step_prediction).all()
):
    raise RuntimeError("The scaled model produced a non-finite prediction.")

free_run_prediction = y_scaler.inverse_transform(scaled_free_run_prediction)
one_step_prediction = y_scaler.inverse_transform(scaled_one_step_prediction)

start = scaled_model.max_lag
scaled_y_eval = y_valid[start:]
scaled_x_eval = x_valid[start:]
scaled_free_run_eval = free_run_prediction[start:]
scaled_one_step_eval = one_step_prediction[start:]

scaled_free_run_rrse = root_relative_squared_error(
    scaled_y_eval,
    scaled_free_run_eval,
)
scaled_one_step_rrse = root_relative_squared_error(
    scaled_y_eval,
    scaled_one_step_eval,
)
scaled_ee = compute_residues_autocorrelation(
    scaled_y_eval,
    scaled_free_run_eval,
)
scaled_x1e = compute_cross_correlation(
    scaled_y_eval,
    scaled_free_run_eval,
    scaled_x_eval,
)

print(f"Free-run RRSE: {scaled_free_run_rrse}")
print(f"One-step-ahead RRSE: {scaled_one_step_rrse}")

plot_results(
    y=scaled_y_eval,
    yhat=scaled_free_run_eval,
    n=100,
    title="Free run simulation — standardized data",
)
plot_residues_correlation(
    data=scaled_ee,
    title="Residual autocorrelation",
    ylabel="$r_{ee}$",
)
plot_residues_correlation(
    data=scaled_x1e,
    title="Input-residual cross-correlation",
    ylabel="$r_{xe}$",
)
```

```text
Free-run RRSE: 0.08025565522199729
One-step-ahead RRSE: 0.04219066805522397
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-scaled-free-run.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-scaled-residual-autocorrelation.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/electromechanical-rls-scaled-input-residual-correlation.png?raw=true)

A simulação livre padronizada permanece finita e acompanha bem a saída, com RRSE
aproximadamente igual a 0,0803. Isso não decorre apenas de expressar a métrica em
outra unidade: as previsões retornam à unidade física antes da avaliação. Em uma
base polinomial, a escala altera o condicionamento numérico dos regressores e
pode afetar a estrutura escolhida e os parâmetros estimados.

O diagnóstico de resíduos impede, entretanto, uma conclusão excessiva. A
autocorrelação permanece fora dos limites aproximados por muitos atrasos, e a
correlação entrada-resíduo apresenta alguns picos fora desses limites. A
padronização resolve a instabilidade observada na simulação, mas o modelo ainda
não constitui uma descrição dinâmica completa do sistema.

Este exemplo mostra por que uma afirmação isolada como “o modelo tem RRSE igual a
0,02” é incompleta. É preciso informar o conjunto de dados, o alinhamento, o modo
de predição e o pré-processamento, inclusive o conjunto usado para ajustar suas
transformações. Em aplicações nas quais o modelo evolui sem acesso contínuo à
saída real, a simulação livre continua sendo o diagnóstico decisivo, sempre
interpretada em conjunto com a análise de resíduos.
