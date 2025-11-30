Até agora, categorizamos os sistemas em duas classes distintas: **sistemas lineares** e sistemas **não lineares**. Como mencionado, **sistemas lineares** foram extensivamente estudados, com diversos métodos bem estabelecidos disponíveis, enquanto **sistemas não lineares** é uma área muito ativa, com vários problemas ainda em aberto para pesquisa. Além de sistemas lineares e não lineares, existem os chamados **Sistemas Severamente Não Lineares** (Severely Nonlinear Systems). Sistemas Severamente Não Lineares são aqueles que exibem comportamentos dinâmicos altamente complexos e exóticos, como sub-harmônicos, comportamento caótico e histerese. Por enquanto, focaremos em sistemas com histerese.

## Modelagem de Histerese com Modelos Polinomiais NARX

A não linearidade de histerese é um comportamento severamente não linear comumente encontrado em dispositivos eletromagnéticos, sensores, semicondutores, materiais inteligentes, entre outros, que possuem efeitos de memória entre entrada e saída quase-estáticas ([Visintin, A., "Differential Models of Hysteresis"](https://link.springer.com/book/10.1007/978-3-662-11557-2)), ([Ahmad, I., "Two Degree-of-Freedom Robust Digital Controller Design With Bouc-Wen Hysteresis Compensator for Piezoelectric Positioning Stage"](https://ieeexplore.ieee.org/document/8316821)). Um sistema histerético é aquele que exibe um comportamento dependente do caminho, o que significa que sua resposta depende não apenas de seu estado atual, mas também de seu histórico. Em um sistema histerético, quando você aplica uma entrada, a resposta do sistema (como deslocamento ou tensão) não segue o mesmo caminho de volta ao ponto de partida quando você remove a entrada. Em vez disso, ela forma um padrão em formato de laço chamado *hysteresis loop*. Isso ocorre porque o sistema possui a *capacidade* de preservar uma deformação causada por uma entrada, caracterizando um efeito de memória.

A identificação de sistemas histeréticos utilizando modelos polinomiais NARX é tipicamente uma tarefa intrigante, pois os algoritmos tradicionais de Seleção de Estrutura de Modelo (Model Structure Selection) não funcionam adequadamente ([Martins, S. A. M. and Aguirre, L. A., "Sufficient conditions for rate-independent hysteresis in autoregressive identified models"](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), [Leva, A. and Piroddi, L., "NARX-based technique for the modelling of magneto-rheological damping devices"](https://iopscience.iop.org/article/10.1088/0964-1726/11/1/309)). [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) apresentaram as condições suficientes para descrever histerese usando modelos polinomiais, fornecendo o conceito de estrutura limitante (*bounding structure*) $\mathcal{H}$. Modelos polinomiais NARX com um único equilíbrio podem ser usados para uma caracterização completa do comportamento de histerese adotando o conceito de estrutura limitante.

A seguir, são apresentados alguns dos conceitos essenciais e definições formais para entender como modelos NARX podem ser usados para descrever sistemas com histerese.

### Sinal quase-estático de carregamento-descarregamento em tempo contínuo

Uma característica importante para modelar sistemas histeréticos é o sinal de entrada. Um sinal quase-estático de carregamento-descarregamento (*loading-unloading quasi-static signal*) é um sinal periódico em tempo contínuo $x_t$ com período $T = (t_f - t_i)$ e frequência $\omega = 2\pi f$, onde $x_t$ aumenta monotonicamente de $x_{min}$ para $x_{max}$, considerando $t_i \leq t \leq t_m$ (carregamento) e diminui monotonicamente de $x_{max}$ para $x_{min}$, considerando $t_m \leq t \leq t_f$ (descarregamento). Se o sinal de carregamento-descarregamento varia com $\omega \rightarrow 0$, o sinal também é chamado de sinal quase-estático. Visualmente, isso é muito mais simples de entender. A imagem a seguir mostra um sinal quase-estático de carregamento-descarregamento em tempo contínuo.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/load_unloading_signal.png?raw=true)
> Figura 1. Sinal quase-estático de carregamento-descarregamento em tempo contínuo, demonstrando o aumento e diminuição periódicos do sinal de entrada.


Nesse sentido, [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) também apresentaram a ideia de transformar as entradas do sistema usando funções multivaloradas.

> Funções multivaloradas - Seja $\phi (\Delta x_{k}): \mathbb{R} \rightarrow \mathbb{R}$. Se $\Delta x_{k}=x_k-x_{k-1}$, $\phi (\Delta x_{k})$ é uma função multivalorada se:

$$
\begin{equation}
    \phi (\Delta x_{k})=
	\begin{cases}
		\phi_1, & se \ \Delta x_{k} > \epsilon; \\
		\phi_2, & se \ \Delta x_{k} < \epsilon; \\
		\phi_3, & se \ \Delta x_{k} = \epsilon; \\
	\end{cases}
\end{equation}
\tag{1}
$$

onde $\epsilon \in \mathbb{R}$, $\phi_1 \neq \phi_2 \neq \phi_3$. Para algumas entradas $\Delta x_{k}\neq \epsilon, \ \forall{k} \in \mathbb{N}$, e o último valor na equação acima não é utilizado.

Uma função multivalorada frequentemente usada é a sign$(\cdot): \mathbb{R} \rightarrow \mathbb{R}$:

$$
 \begin{equation}
 sign(x)=
	\begin{cases}
		1, & se \ x > 0; \\
		-1, & se \ x < 0; \\
		0, & se \ x = 0. \\
	\end{cases}
\end{equation}
\tag{2}
$$


### Hysteresis loops em tempo contínuo $\mathcal{H}_t(\omega)$

Seja $x_t$ um sinal quase-estático de carregamento-descarregamento em tempo contínuo aplicado a um sistema em tempo contínuo e $y_t$ a saída do sistema. $\mathcal{H}_t(\omega)$ denota um laço fechado no plano $x_t - y_t$, cuja forma depende de $\omega$. Se o sistema apresenta não linearidade histerética, $\mathcal{H}_t(\omega)$ é denotado como:

$$
\begin{equation}
\mathcal{H}_t(\omega) =
	\begin{cases}
		\mathcal{H}_t(\omega)^{+}, \ para \ t_i \ \leq \ t \ \leq \ t_m, \\
		\mathcal{H}_t(\omega)^{-}; \ para \ t_m \ \leq \ t \ \leq \ t_f, \\
	\end{cases}
\end{equation}
\tag{3}
$$

onde $\mathcal{H}_t(\omega)^{+} \neq \mathcal{H}_t(\omega)^{-}$, $\forall t \neq t_m$. $t_i \leq t \leq t_m$ e $t_m \leq t \leq t_f$ correspondem ao regime quando $x_t$ está em carregamento e descarregamento, respectivamente. $\mathcal{H}_t(\omega)^{+}$ corresponde à parte do laço formada no plano $x_t - y_t$, enquanto $t_i \leq t \leq t_m$ (quando $x_t$ está em carregamento), enquanto $\mathcal{H}_t(\omega)^{-}$ é a parte do laço formada no plano $x_t - y_t$ para $t_m \leq t \leq t_f$ (quando $x_t$ está em descarregamento), como mostrado na Figura 2:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/hysteresis_loop.png?raw=true)
> Figura 2. Exemplo de uma curva de histerese.


> Rate Independent Hysteresis (RIH) ([Visintin, A., "Differential Models of Hysteresis"](https://link.springer.com/book/10.1007/978-3-662-11557-2)) - O comportamento de histerese é chamado de *rate independent* se o caminho $ABCD$, que depende do par $x(t), y(t)$, é invariante em relação a qualquer difeomorfismo crescente $\varphi : [0,T] \rightarrow [0,T]$, ou seja:

$$
\begin{align}
        F(u \ o \ \varphi, y^{0}) = F(u,y^0)\ o \ \varphi & \ em \ [0,T].
\end{align}
\tag{4}
$$

Isso significa que, em qualquer instante $t$, $y(t)$ depende apenas de $u:[0,T] \rightarrow \mathbb{R}$ e da ordem em que os valores foram atingidos antes de $t$. Em outras palavras, o efeito de memória não é afetado pela frequência da entrada.

## Rate Independent Hysteresis em modelos polinomiais NARX

[Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968) apresentaram as condições suficientes para que modelos NARX representem histerese. Um dos conceitos desenvolvidos é a Estrutura Limitante (*Bounding Structure*) $\mathcal{H}$.

> Estrutura Limitante $\mathcal{H}$ ([Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968)) - Seja $\mathcal{H}_t(\omega)$ a histerese do sistema. $\mathcal{H}= \lim_{\omega \to 0} \mathcal{H}_t(\omega)$ é definida como a estrutura limitante que delimita $\mathcal{H}_t(\omega)$.

Agora, considere um modelo polinomial NARX excitado por um sinal quase-estático de carregamento-descarregamento. Se o modelo possui um ponto de equilíbrio real e estável, cuja localização depende da entrada e do regime de carregamento/descarregamento, o polinômio exibirá um *hysteresis loop* Rate Independent $\mathcal{H}_t(\omega)$ no plano $x-y$.

Aqui está um exemplo. Seja $y_k  =  0.8y_{k-1} + 0.4\phi_{k-1} + 0.2x_{k-1}$, onde $\phi_{k} = \rm{sign}(\Delta(x_{k}))$ e $x_{k} = sin(\omega k)$ e $\omega$ é a frequência do sinal de entrada $x$. Os equilíbrios deste modelo são dados por:

$$
\begin{equation}
    \overline{y}(\overline{\phi},\overline{x})=
	\begin{cases}
		\frac{0.6+0.2\overline{x}}{1-0.8} \ = 3 \ + \ \overline{x} \ , & para \ carregamento; \\
		\frac{-0.6+0.2\overline{x}}{1-0.8} \ = -3 \ + \ \overline{x} \ , & para \ descarregamento; \\
	\end{cases}
\end{equation}
\tag{5}
$$

onde $\overline {x}$ é um sinal de entrada quase-estático de carregamento-descarregamento. Como os pontos de equilíbrio são assintoticamente estáveis, a saída converge para $\mathcal{H}_k (w)$ no plano $x-y$. Note que, para um valor de entrada constante $x ~ = ~ 1 ~ = ~ \overline{x}$, o equilíbrio está em $\overline{y} ~ = ~ 3$ para o regime de carregamento e $\overline {y} ~ = ~ -1$ para o regime de descarregamento. Analogamente, para $\overline {x} ~ = ~ -1$, o equilíbrio está em $\overline {y} ~ = ~ 1$ para o regime de carregamento e $\overline {y} ~ = ~ -3$ para o regime de descarregamento, como mostrado na figura abaixo:

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bounded_structure_example.png?raw=true)
> Figura 3. Exemplo de uma estrutura limitante $\mathcal{H}$. Os pontos pretos estão em $\mathcal{H}_{k}(\omega)$ para o modelo $y_k  =  0.8y_{k-1} + 0.4\phi_{k-1} + 0.2x_{k-1}$. A estrutura limitante $\mathcal{H}$, em vermelho, confina $\mathcal{H}_{k}(\omega)$.

Como pode ser observado na Figura 3, se garantirmos as condições suficientes propostas por [Martins, S. A. M. and Aguirre, L. A.](https://www.sciencedirect.com/science/article/abs/pii/S0888327015005968), um modelo NARX pode reproduzir um comportamento histerético. O Capítulo 10 apresenta um estudo de caso de um sistema com histerese.

O código a seguir pode ser usado para reproduzir o comportamento mostrado na Figura 3. Altere `w` de $1$ para $0.1$ para ver como a estrutura limitante $\mathcal{H}$ converge para os equilíbrios do sistema.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
w = 1
t = np.arange(0, 60.1, 0.1)
y = np.zeros(len(t))
x = np.sin(w * t)

# Initialize y and fi
fi = np.zeros(len(t))
# Iterate over the time array to calculate y
for k in range(1, len(t)):
    fi[k] = np.sign(x[k] - x[k-1])
    y[k] = 0.8 * y[k-1] + 0.2 * x[k-1] + 0.4 * fi[k-1]

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Example')
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/bounded_structure_example_python.png?raw=true)
> Figura 4. Reprodução de uma estrutura limitante $\mathcal{H}$ usando Python.
