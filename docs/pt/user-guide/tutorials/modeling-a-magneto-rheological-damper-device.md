# Modelagem de um Dispositivo Amortecedor Magneto-Reológico

## Reprodutibilidade

Este tutorial foi verificado com SysIdentPy 0.9.0 no Python 3.12.12, pandas 2.3.3
e scikit-learn 1.7.2. Os dados hospedados em `sysidentpy-data` usam uma URL de
commit imutável, e os exemplos aleatórios definem semente explícita.

Nota: O exemplo mostrado neste notebook é retirado do livro complementar [Nonlinear System Identification and Forecasting: Theory and Practice with SysIdentPy](https://sysidentpy.org/book/0-Preface/).

Os efeitos de memória entre entrada e saída quase-estáticas tornam a modelagem de sistemas histeréticos muito difícil. Modelos baseados em física são frequentemente usados para descrever os loops de histerese, mas esses modelos geralmente carecem da simplicidade e eficiência requeridas em aplicações práticas envolvendo caracterização, identificação e controle de sistemas. Como detalhado em [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), modelos NARX provaram ser uma escolha viável para descrever os loops de histerese. Veja o Capítulo 8 para um background detalhado. No entanto, mesmo considerando as condições suficientes para representação de histerese independente de taxa, algoritmos clássicos de seleção de estrutura falham em retornar um modelo com desempenho decente e o usuário precisa definir uma função multi-valorada para garantir a ocorrência da estrutura limitante $\mathcal{H}$ ([Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)).

Embora algum progresso tenha sido feito, trabalhos anteriores foram limitados a modelos com um único ponto de equilíbrio. O presente estudo de caso visa apresentar novas perspectivas na seleção de estrutura de modelos de sistemas histeréticos considerando os casos onde os modelos têm múltiplas entradas e não é restrito quanto ao número de pontos de equilíbrio. Para isso, o algoritmo MetaMSS será usado para construir um modelo para um amortecedor magneto-reológico (MRD) considerando as condições suficientes mencionadas.

### Uma Breve descrição do modelo Bouc-Wen de dispositivo amortecedor magneto-reológico

Os dados usados neste estudo de caso são do modelo Bouc-Wen ([Bouc, R - Forced Vibrations of a Mechanical System with Hysteresis](https://www.scirp.org/reference/referencespapers?referenceid=726819)), ([Wen, Y. X. - Method for Random Vibration of Hysteretic Systems](https://ascelibrary.org/doi/10.1061/JMCEA3.0002106)) de um MRD cujo diagrama esquemático é mostrado na figura abaixo.


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/bouc_wen.png?raw=true)
> O modelo para um amortecedor magneto-reológico proposto por [Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972).

A forma geral do modelo Bouc-Wen pode ser descrita como ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)):

$$
\begin{equation}
\dfrac{dz}{dt} = g\left[x,z,sign\left(\dfrac{dx}{dt}\right)\right]\dfrac{dx}{dt},
\end{equation}
$$

onde $z$ é a saída do modelo histerético, $x$ a entrada e $g[\cdot]$ uma função não linear de $x$, $z$ e $sign (dx/dt)$. ([Spencer, B. F. and Sain, M. K. - Controlling buildings: a new frontier in feedback](https://ieeexplore.ieee.org/document/642972)) propuseram o seguinte modelo fenomenológico para o dispositivo mencionado:

$$
\begin{aligned}
f&= c_1\dot{\rho}+k_1(x-x_0),\nonumber\\
\dot{\rho}&=\dfrac{1}{c_0+c_1}[\alpha z+c_0\dot{x}+k_0(x-\rho)],\nonumber\\
\dot{z}&=-\gamma|\dot{x}-\dot{\rho}|z|z|^{n-1}-\beta(\dot{x}-\dot{\rho})|z|^n+A(\dot{x}-\dot{\rho}),\nonumber\\
\alpha&=\alpha_a+\alpha_bu_{bw},\nonumber\\
c_1&=c_{1a}+c_{1b}u_{bw},\nonumber\\
c_0&=c_{0a}+c_{0b}u_{bw},\nonumber\\
\dot{u}_{bw}&=-\eta(u_{bw}-E).
\end{aligned}
$$

onde $f$ é a força de amortecimento, $c_1$ e $c_0$ representam os coeficientes viscosos, $E$ é a tensão de entrada, $x$ é o deslocamento e $\dot{x}$ é a velocidade do modelo. Os parâmetros do sistema (veja a tabela abaixo) foram retirados de [Leva, A. and Piroddi, L. - NARX-based technique for the modelling of magneto-rheological damping devices](https://iopscience.iop.org/article/10.1088/0964-1726/11/1/309).

| Parâmetro  | Valor          | Parâmetro | Valor        |
|------------|----------------|-----------|--------------|
| $c_{0_a}$  | $20.2 \, N \, s/cm$  | $\alpha_{a}$  | $44.9 \, N/cm$  |
| $c_{0_b}$  | $2.68 \, N \, s/cm \, V$ | $\alpha_{b}$  | $638 \, N/cm$   |
| $c_{1_a}$  | $350 \, N \, s/cm$   | $\gamma$      | $39.3 \, cm^{-2}$ |
| $c_{1_b}$  | $70.7 \, N \, s/cm \, V$  | $\beta$       | $39.3 \, cm^{-2}$ |
| $k_{0}$    | $15 \, N/cm$    | $n$           | $2$          |
| $k_{1}$    | $5.37 \, N/cm$   | $\eta$       | $251 \, s^{-1}$ |
| $x_{0}$    | $0 \, cm$      | $A$           | $47.2$       |

Para este estudo particular, tanto as entradas de deslocamento quanto de tensão, $x$ e $E$, respectivamente, foram geradas filtrando uma sequência de ruído gaussiano branco usando um filtro FIR Blackman-Harris com frequência de corte de $6$Hz. O tamanho do passo de integração foi definido como $h = 0.002$, seguindo os procedimentos descritos em [Martins, S. A. M. and Aguirre, L. A. - Sufficient conditions for rate-independent hysteresis in autoregressive identified models](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968). Estes procedimentos são apenas para fins de identificação, já que as entradas de um MRD podem ter várias características diferentes.

Os dados usados neste exemplo são fornecidos pelo Professor Samir Angelo Milani Martins.

Os desafios são:

- possui uma não linearidade com memória, ou seja, uma não linearidade dinâmica;
- a não linearidade é governada por uma variável interna z(t), que não é mensurável;
- a forma funcional não linear na equação de Bouc Wen é não linear no parâmetro;
- a forma funcional não linear na equação de Bouc Wen não admite uma expansão de série de Taylor finita devido à presença de valores absolutos

### Pacotes Necessários e Versões

Este estudo de caso foi verificado com o SysIdentPy 0.9.0 no Python 3.12.12,
`pandas==2.3.3` e `scikit-learn==1.7.2`. Instale explicitamente o checkout do
repositório e os dois pacotes de preparação dos dados:

```bash
python -m pip install -e .
python -m pip install pandas==2.3.3 scikit-learn==1.7.2
```

O conjunto de dados é carregado por uma URL imutável do `sysidentpy-data`. Os
exemplos aleatórios usam a semente 42.

### Configuração do SysIdentPy


```python
from warnings import catch_warnings, simplefilter
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt

from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.display_results import results
from sysidentpy.parameter_estimation import LeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.utils.plotting import plot_results

df = pd.read_csv(
    "https://raw.githubusercontent.com/wilsonrljr/sysidentpy-data/4085901293ba5ed5674bb2911ef4d1fa20f3438d/datasets/bouc_wen/boucwen_histeretic_system.csv"
)
scaler_x = MaxAbsScaler()
scaler_y = MaxAbsScaler()

init = 400
x_train = df[["E", "v"]].iloc[init : df.shape[0] // 2, :]
x_train["sign_v"] = np.sign(df["v"])
x_train = scaler_x.fit_transform(x_train)

x_test = df[["E", "v"]].iloc[df.shape[0] // 2 + 1 : df.shape[0] - init, :]
x_test["sign_v"] = np.sign(df["v"])
x_test = scaler_x.transform(x_test)

y_train = df[["f"]].iloc[init : df.shape[0] // 2, :].values.reshape(-1, 1)
y_train = scaler_y.fit_transform(y_train)

y_test = (
    df[["f"]].iloc[df.shape[0] // 2 + 1 : df.shape[0] - init, :].values.reshape(-1, 1)
)
y_test = scaler_y.transform(y_test)

# Plotting the data
plt.figure(figsize=(10, 8))
plt.suptitle("Identification (training) data", fontsize=16)

plt.subplot(221)
plt.plot(y_train, "k")
plt.ylabel("Force - Output")
plt.xlabel("Samples")
plt.title("y")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.subplot(222)
plt.plot(x_train[:, 0], "k")
plt.ylabel("Control Voltage")
plt.xlabel("Samples")
plt.title("x_1")
plt.grid()
plt.axis([0, 1500, 0, 1])

plt.subplot(223)
plt.plot(x_train[:, 1], "k")
plt.ylabel("Velocity")
plt.xlabel("Samples")
plt.title("x_2")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.subplot(224)
plt.plot(x_train[:, 2], "k")
plt.ylabel("sign(Velocity)")
plt.xlabel("Samples")
plt.title("x_3")
plt.grid()
plt.axis([0, 1500, -1.5, 1.5])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-01.png?raw=true)


Os gráficos aos pares ajudam a revelar a relação multivalorada entre a força e
cada entrada candidata.


```python
plt.figure()
plt.plot(x_train[:, 0], y_train)
plt.xlabel("x1 - Voltage")
plt.ylabel("y - Force")

plt.figure()
plt.plot(x_train[:, 1], y_train)
plt.xlabel("x2 - Velocity")
plt.ylabel("y - Force")

plt.figure()
plt.plot(x_train[:, 2], y_train)
plt.xlabel("u3 - sign(Velocity)")
plt.ylabel("y - Force")
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-02.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-03.png?raw=true)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-04.png?raw=true)


Agora podemos construir um modelo NARX. Com as três entradas e o
`MaxAbsScaler`, o RRSE em simulação livre é $0.045104$:


```python
basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=[[1], [1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
)

model.fit(X=x_train, y=y_train)
yhat = model.predict(X=x_test, y=y_test[: model.max_lag :, :])
rrse = root_relative_squared_error(y_test[model.max_lag :], yhat[model.max_lag :])
print(rrse)
plot_results(
    y=y_test[model.max_lag :],
    yhat=yhat[model.max_lag :],
    n=10000,
    title="FROLS: sign(v) and MaxAbsScaler",
)
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-05.png?raw=true)


Se removermos a entrada `sign(v)` e usarmos a mesma configuração, a simulação
livre diverge na amostra 203, como mostra a figura a seguir:


```python
basis_function = Polynomial(degree=3)
model = FROLS(
    xlag=[[1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    info_criteria="aic",
)

model.fit(X=x_train[:, :2], y=y_train)
with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    yhat = model.predict(
        X=x_test[:, :2], y=y_test[: model.max_lag]
    )
if np.isfinite(yhat).all():
    rrse = root_relative_squared_error(
        y_test[model.max_lag :], yhat[model.max_lag :]
    )
    print(rrse)
    plot_results(
        y=y_test[model.max_lag :],
        yhat=yhat[model.max_lag :],
        n=10000,
        title="FROLS without sign(v)",
    )
else:
    finite_mask = np.isfinite(yhat[:, 0])
    finite_stop = int(np.flatnonzero(~finite_mask)[0])
    print(f"Free-run simulation diverged at sample {finite_stop}.")
    plot_results(
        y=y_test[model.max_lag : finite_stop],
        yhat=yhat[model.max_lag : finite_stop],
        n=max(1, finite_stop - model.max_lag),
        title="FROLS without sign(v): trajectory before divergence",
    )
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-06.png?raw=true)


Usar o MetaMSS sem `sign(v)` adia a perda de estabilidade numérica, mas não a
elimina: esta trajetória em simulação livre diverge na amostra 1153.


```python
from sysidentpy.model_structure_selection import MetaMSS

basis_function = Polynomial(degree=3)
model = MetaMSS(
    xlag=[[1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    random_state=42,
)

with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train[:, :2], y=y_train)
    yhat = model.predict(
        X=x_test[:, :2], y=y_test[: model.max_lag]
    )
if np.isfinite(yhat).all():
    rrse = root_relative_squared_error(
        y_test[model.max_lag :], yhat[model.max_lag :]
    )
    print(rrse)
    finite_stop = len(yhat)
    plot_results(
        y=y_test[model.max_lag :],
        yhat=yhat[model.max_lag :],
        n=10000,
        title="MetaMSS without sign(v)",
    )
else:
    finite_mask = np.isfinite(yhat[:, 0])
    finite_stop = int(np.flatnonzero(~finite_mask)[0])
    print(f"Free-run simulation diverged at sample {finite_stop}.")
    plot_results(
        y=y_test[model.max_lag : finite_stop],
        yhat=yhat[model.max_lag : finite_stop],
        n=max(1, finite_stop - model.max_lag),
        title="MetaMSS without sign(v): trajectory before divergence",
    )
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-07.png?raw=true)


Antes da divergência, o comportamento oscilatório se torna visível quando a
saída se aproxima de seu valor mínimo.


```python
window_stop = finite_stop
window_start = max(model.max_lag, window_stop - 100)
plot_results(
    y=y_test[window_start:window_stop],
    yhat=yhat[window_start:window_stop],
    n=100,
    title="MetaMSS without sign(v): last finite window",
)
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-08.png?raw=true)


Ao adicionar novamente a entrada `sign(v)` e usar o MetaMSS, a simulação livre
permanece finita e produz RRSE $0.055707$, próximo ao resultado do FROLS com
todas as entradas.


```python
basis_function = Polynomial(degree=3)
model = MetaMSS(
    xlag=[[1], [1], [1]],
    ylag=1,
    basis_function=basis_function,
    estimator=LeastSquares(),
    random_state=42,
)

with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
    simplefilter("ignore", RuntimeWarning)
    simplefilter("ignore", UserWarning)
    model.fit(X=x_train, y=y_train)
    yhat = model.predict(X=x_test, y=y_test[: model.max_lag :, :])
rrse = root_relative_squared_error(y_test[model.max_lag :], yhat[model.max_lag :])
print(rrse)
plot_results(
    y=y_test[model.max_lag :],
    yhat=yhat[model.max_lag :],
    n=10000,
    title="MetaMSS: sign(v) and MaxAbsScaler",
)
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-09.png?raw=true)


Este caso também destacará a importância do escalonamento de dados. Anteriormente, usamos o método `MaxAbsScaler`, que resultou em ótimos modelos ao usar as entradas `sign(v)`, mas também resultou em modelos instáveis ao remover essa feature de entrada. Quando o escalonamento é aplicado usando `MinMaxScaler`, no entanto, a estabilidade geral dos resultados melhora, e o modelo não diverge, mesmo quando a entrada `sign(v)` é removida, usando o algoritmo `FROLS`.

O usuário pode obter os resultados abaixo apenas alterando o método de escalonamento de dados usando


```python
minmax_scaler_x = MinMaxScaler()
minmax_scaler_y = MinMaxScaler()
midpoint = df.shape[0] // 2

x_train_minmax_frame = df[["E", "v"]].iloc[init:midpoint].copy()
x_train_minmax_frame["sign_v"] = np.sign(x_train_minmax_frame["v"])
x_test_minmax_frame = df[["E", "v"]].iloc[midpoint + 1 : df.shape[0] - init].copy()
x_test_minmax_frame["sign_v"] = np.sign(x_test_minmax_frame["v"])
x_train_minmax = minmax_scaler_x.fit_transform(x_train_minmax_frame)
x_test_minmax = minmax_scaler_x.transform(x_test_minmax_frame)

y_train_minmax = minmax_scaler_y.fit_transform(
    df[["f"]].iloc[init:midpoint].to_numpy()
)
y_test_minmax = minmax_scaler_y.transform(
    df[["f"]].iloc[midpoint + 1 : df.shape[0] - init].to_numpy()
)

def run_minmax_experiment(name, selector, use_sign):
    n_inputs = 3 if use_sign else 2
    x_train_variant = x_train_minmax[:, :n_inputs]
    x_test_variant = x_test_minmax[:, :n_inputs]
    if selector == "FROLS":
        candidate = FROLS(
            xlag=[[1]] * n_inputs,
            ylag=1,
            basis_function=Polynomial(degree=3),
            estimator=LeastSquares(),
            info_criteria="aic",
        )
    else:
        candidate = MetaMSS(
            xlag=[[1]] * n_inputs,
            ylag=1,
            basis_function=Polynomial(degree=3),
            estimator=LeastSquares(),
            random_state=42,
        )

    with catch_warnings(), np.errstate(over="ignore", invalid="ignore"):
        simplefilter("ignore", RuntimeWarning)
        simplefilter("ignore", UserWarning)
        candidate.fit(X=x_train_variant, y=y_train_minmax)
        prediction = candidate.predict(
            X=x_test_variant,
            y=y_test_minmax[: candidate.max_lag],
        )

    finite_mask = np.isfinite(prediction[:, 0])
    if finite_mask.all():
        score = root_relative_squared_error(
            y_test_minmax[candidate.max_lag :],
            prediction[candidate.max_lag :],
        )
        print(f"{name}: RRSE={score:.6f}")
        stop = len(prediction)
    else:
        score = np.nan
        stop = int(np.flatnonzero(~finite_mask)[0])
        print(f"{name}: free-run simulation diverged at sample {stop}")

    plot_results(
        y=y_test_minmax[candidate.max_lag : stop],
        yhat=prediction[candidate.max_lag : stop],
        n=max(1, stop - candidate.max_lag),
        title=f"{name} with MinMaxScaler",
    )
    return prediction, score

minmax_results = {
    "FROLS with sign(v)": run_minmax_experiment(
        "FROLS with sign(v)", "FROLS", True
    ),
    "FROLS without sign(v)": run_minmax_experiment(
        "FROLS without sign(v)", "FROLS", False
    ),
    "MetaMSS without sign(v)": run_minmax_experiment(
        "MetaMSS without sign(v)", "MetaMSS", False
    ),
    "MetaMSS with sign(v)": run_minmax_experiment(
        "MetaMSS with sign(v)", "MetaMSS", True
    ),
}
yhat = minmax_results["MetaMSS with sign(v)"][0]
x_test = x_test_minmax
y_test = y_test_minmax
```

e executando cada modelo novamente. Essa mudança torna finitas todas as
trajetórias em simulação livre, mas não melhora as configurações que usam todas
as entradas.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-10.png?raw=true)
> FROLS com `sign(v)` e `MinMaxScaler`: RRSE 0.115986.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-11.png?raw=true)
> FROLS sem `sign(v)` e com `MinMaxScaler`: RRSE 0.163944.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-12.png?raw=true)
> MetaMSS sem `sign(v)` e com `MinMaxScaler`: RRSE 0.185607.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-13.png?raw=true)
> MetaMSS com `sign(v)` e `MinMaxScaler`: RRSE 0.104511.

A comparação completa é:

| Escalonamento | Seletor | Entradas | Resultado em simulação livre |
| --- | --- | --- | ---: |
| MaxAbs | FROLS | com `sign(v)` | RRSE 0.045104 |
| MaxAbs | FROLS | sem `sign(v)` | divergiu na amostra 203 |
| MaxAbs | MetaMSS | sem `sign(v)` | divergiu na amostra 1153 |
| MaxAbs | MetaMSS | com `sign(v)` | RRSE 0.055707 |
| MinMax | FROLS | com `sign(v)` | RRSE 0.115986 |
| MinMax | FROLS | sem `sign(v)` | RRSE 0.163944 |
| MinMax | MetaMSS | sem `sign(v)` | RRSE 0.185607 |
| MinMax | MetaMSS | com `sign(v)` | RRSE 0.104511 |

O MetaMSS com `sign(v)` é o melhor entre as configurações com MinMax. O melhor
resultado geral continua sendo o FROLS com `sign(v)` e `MaxAbsScaler`. Uma saída
não finita em simulação livre é registrada como divergência, em vez de ser
convertida em uma métrica escalar.

Aqui está o loop histerético predito:


```python
plt.figure(figsize=(8, 6))
plt.plot(x_test[:, 1], yhat)
plt.xlabel("Scaled velocity")
plt.ylabel("Predicted scaled force")
plt.title("MetaMSS with sign(v) and MinMaxScaler")
plt.show()
```


![](https://github.com/wilsonrljr/sysidentpy-data/blob/f38f95efb02194bf2ab116d63982305e2ec09213/book/assets/modeling-a-magneto-rheological-damper-device-14.png?raw=true)
