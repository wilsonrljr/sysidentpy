---
template: overrides/main.html
title: Introdução aos Modelos NARMAX
---

# Introdução

> Autor: Wilson Rocha Lacerda Junior

Este é o primeiro de uma série de publicações explicando um pouco sobre modelos NARMAX[^1]. Espero que o conteúdo dessas publicações ajude aqueles que usam ou gostariam de usar a biblioteca SysIdentPy.

> **Procurando mais detalhes sobre modelos NARMAX?**
> Para informações completas sobre modelos, métodos e uma ampla gama de exemplos e benchmarks implementados no SysIdentPy, confira nosso livro:
> [*Nonlinear System Identification and Forecasting: Theory and Practice With SysIdentPy*](https://sysidentpy.org/book/0%20-%20Preface/)
>
> Este livro oferece orientação detalhada para auxiliar seu trabalho com o SysIdentPy.
>
> Você também pode explorar os [tutoriais na documentação](https://sysidentpy.org/examples/basic_steps/) para exemplos práticos.


### Identificação de Sistemas

Como usarei o termo **Identificação de Sistemas** aqui e ali, deixe-me fazer uma breve definição.

<br>
Identificação de sistemas é uma das principais áreas que lida com a modelagem de processos baseados em dados. Neste contexto, o termo "sistema" pode ser interpretado como qualquer conjunto de operações que processam uma ou mais entradas e retornam uma ou mais saídas. Exemplos incluem sistemas elétricos, sistemas mecânicos, sistemas biológicos, sistemas financeiros, sistemas químicos... literalmente qualquer coisa que você possa relacionar a dados de entrada e saída. A demanda de eletricidade é parte de um sistema cujas entradas podem ser, por exemplo, tamanho da população, quantidade de água nos reservatórios, estação do ano, eventos. O preço de um imóvel é a saída de um sistema cujas entradas podem ser a cidade, renda per capita, bairro, número de quartos, idade do imóvel, entre muitos outros. Você entendeu a ideia.

<br>
Embora existam muitas coisas relacionadas com Machine Learning, Statistical Learning e outros campos, cada área tem suas particularidades.


### Então, o que é um modelo NARMAX?

Você deve ter notado a semelhança entre o acrônimo NARMAX com os conhecidos modelos ARX, ARMAX, etc., que são amplamente utilizados para previsão de séries temporais. E essa semelhança não é por acaso. Os modelos Autorregressivos com Média Móvel e Entrada Exógena (ARMAX) e suas variações AR, ARX, ARMA (para citar apenas alguns) são uma das representações matemáticas mais utilizadas para identificação de sistemas lineares.

<br>
Vamos voltar ao modelo. Eu disse que a família de modelos **ARX** é comumente usada para modelar sistemas lineares. Linear é a palavra-chave aqui. Para cenários não lineares, temos a classe **NARMAX**. Como relatado por Billings (um dos criadores do modelo NARMAX) no livro [Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains], NARMAX começou como um nome de modelo, mas logo se tornou uma filosofia quando se trata de identificar sistemas não lineares. Obter modelos NARMAX consiste em realizar as seguintes etapas:

  [^1]:
    Modelos Autorregressivos Não Lineares com Média Móvel e Entrada Exógena.

  [Nonlinear System Identification: NARMAX Methods in the Time, Frequency, and Spatio-Temporal Domains](https://www.wiley.com/en-us/Nonlinear+System+Identification:+NARMAX+Methods+in+the+Time,+Frequency,+and+Spatio+Temporal+Domains-p-9781119943594)

- Testes dinâmicos e coleta de dados;
- Escolha da representação matemática;
- Detecção da estrutura do modelo;
- Estimação de parâmetros;
- Validação;
- Análise do modelo.

Abordaremos cada uma dessas etapas em publicações futuras. A ideia deste texto é apresentar uma visão geral dos modelos NARMAX.

<br>
Modelos NARMAX **não são**, entretanto, uma simples extensão dos modelos ARMAX. Modelos NARMAX são capazes de representar os mais diferentes e complexos sistemas não lineares. Introduzidos em 1981 pelo Engenheiro Eletricista Stephen A. Billings, os modelos NARMAX podem ser descritos como:

$$
    y_k= F^\ell[y_{k-1}, \dotsc, y_{k-n_y},x_{k-d}, x_{k-d-1}, \dotsc, x_{k-d-n_x}, e_{k-1}, \dotsc, e_{k-n_e}] + e_k
$$

onde $n_y\in \mathbb{N}$, $n_x \in \mathbb{N}$, $n_e \in \mathbb{N}$ são os atrasos máximos para a saída e entrada do sistema, respectivamente; $x_k \in \mathbb{R}^{n_x}$ é a entrada do sistema e $y_k \in \mathbb{R}^{n_y}$ é a saída do sistema no tempo discreto $k \in \mathbb{N}^n$; $e_k \in \mathbb{R}^{n_e}$ representa incertezas e possível ruído no tempo discreto $k$. Neste caso, $\mathcal{F}^\ell$ é alguma função não linear dos regressores de entrada e saída com grau de não linearidade $\ell \in \mathbb{N}$ e $d$ é um atraso de tempo tipicamente definido como $d=1$.

Se não incluirmos os termos de ruído, $e_{k-n_e}$, temos modelos NARX. Se definirmos $\ell = 1$, lidamos com modelos ARMAX; se $\ell = 1$ e não incluirmos termos de entrada e ruído, torna-se um modelo AR (ARX se incluirmos entradas, ARMA se incluirmos termos de ruído); se $\ell>1$ e não há termos de entrada, temos o NARMA. Se não há termos de entrada ou ruído, temos NAR. Existem várias variantes, mas isso é suficiente por enquanto.

### Representação NARMAX

Existem várias representações de funções não lineares para aproximar o mapeamento desconhecido $\mathrm{f}[\cdot]$ nos métodos NARMAX, por exemplo:

- redes neurais;
- modelos baseados em lógica fuzzy;
- funções de base radial;
- base wavelet;
- **base polinomial**;
- modelos aditivos generalizados;

O restante deste texto contempla métodos relacionados aos modelos polinomiais na forma de potência, que é a representação mais comumente utilizada. O NARMAX Polinomial é um modelo matemático baseado em equações de diferença e relaciona a saída atual como uma função de entradas e saídas passadas.

### NARMAX Polinomial

O modelo NARMAX polinomial com pontos de equilíbrio assintoticamente estáveis pode ser descrito como:

\begin{align}
    y_k =& \sum_{0} + \sum_{i=1}^{p}\Theta_{y}^{i}y_{k-i} + \sum_{j=1}^{q}\Theta_{e}^{j}e_{k-j} + \sum_{m=1}^{r}\Theta_{x}^{m}x_{k-m}\\
    &+ \sum_{i=1}^{p}\sum_{j=1}^{q}\Theta_{ye}^{ij}y_{k-i} e_{k-j} + \sum_{i=1}^{p}\sum_{m=1}^{r}\Theta_{yx}^{im}y_{k-i} x_{k-m} \\
    &+ \sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{e x}^{jm}e_{k-j} x_{k-m} \\
    &+ \sum_{i=1}^{p}\sum_{j=1}^{q}\sum_{m=1}^{r}\Theta_{y e x}^{ijm}y_{k-i} e_{k-j} x_{k-m} \\
    &+ \sum_{m_1=1}^{r} \sum_{m_2=m_1}^{r}\Theta_{x^2}^{m_1 m_2} x_{k-m_1} x_{k-m_2} \dotsc \\
    &+ \sum_{m_1=1}^{r} \dotsc \sum_{m_l=m_{l-1}}^{r} \Theta_{x^l}^{m_1, \dotsc, m_2} x_{k-m_1} x_{k-m_l}
\end{align}

onde $\sum\nolimits_{0}$, $c_{y}^{i}$, $c_{e}^{j}$, $c_{x}^{m}$, $c_{y\e}^{ij}$, $c_{yx}^{im}$, $c_{e x}^{jm}$, $c_{y e x}^{ijm}$, $c_{x^2}^{m_1 m_2} \dotsc c_{x^l}^{m_1, \dotsc, ml}$ são parâmetros constantes.

<br>
Vamos dar uma olhada em um exemplo de modelo NARMAX para facilitar o entendimento. O seguinte é um modelo NARMAX de grau~$2$, identificado a partir de dados experimentais de um motor/gerador CC sem conhecimento prévio da forma do modelo. Se você quiser mais informações sobre o processo de identificação, escrevi um artigo comparando um NARMAX polinomial com um modelo NARX neural usando esses dados (EM PORTUGUÊS: Identificação de um motor/gerador CC por meio de modelos polinomiais autorregressivos e redes neurais artificiais)

\begin{align}
    y_k =& 1.7813y_{k-1}-0,7962y_{k-2}+0,0339x_{k-1} -0,1597x_{k-1} y_{k-1} +0,0338x_{k-2} \\
    & + 0,1297x_{k-1}y_{k-2} - 0,1396x_{k-2}y_{k-1}+ 0,1086x_{k-2}y_{k-2}+0,0085y_{k-2}^2 + 0.1938e_{k-1}e_{k-2}
\end{align}

Mas como esses termos foram selecionados? Como os parâmetros foram estimados? Essas perguntas nos levarão aos tópicos de seleção de estrutura do modelo e estimação de parâmetros, mas, por enquanto, vamos discutir esses tópicos de maneira mais simples.

<br>
Primeiro, a "estrutura" de um modelo é o conjunto de termos (também chamados de regressores) incluídos no modelo final. Os parâmetros são os valores que multiplicam cada um desses termos. E olhando para o exemplo acima, podemos notar algo realmente importante sobre os modelos NARMAX polinomiais tratados neste texto: eles têm uma estrutura não linear, mas são lineares nos parâmetros. Você verá como essa observação é importante no post sobre estimação de parâmetros.

<br>
Nesse sentido, considere o caso onde temos os dados de entrada e saída de algum sistema. Por simplicidade, suponha uma entrada e uma saída. Temos os dados, mas não sabemos quais atrasos escolher para a entrada ou a saída. Além disso, não sabemos nada sobre a não linearidade do sistema. Então, temos que definir alguns valores para os atrasos máximos da entrada, saída e dos termos de ruído, além da escolha do valor de $\ell$. Vale notar que muitas suposições feitas para casos lineares não são válidas no cenário não linear e, portanto, selecionar os atrasos máximos não é trivial. Então, como esses valores podem tornar a modelagem mais difícil?

<br>
Então temos uma entrada e uma saída (desconsidere os termos de ruído por enquanto). E se escolhermos $n_y = n_x = \ell = 2$? Com esses valores, temos as seguintes possibilidades para compor o modelo final:

\begin{align}
    & constant, y_{k-1}, y_{k-2}, y_{k-1}^2, y_{k-2}^2, x_{k-1}, x_{k-2}, x_{k-1}^2, x_{k-2}^2,y_{k-1}y_{k-2},\\
    & y_{k-1}x_{k-1}, y_{k-1}x_{k-2}, y_{k-2}x_{k-1}, y_{k-2}x_{k-2}, x_{k-1}x_{k-2} .
\end{align}

Então temos $15$ termos candidatos para compor o modelo final.

<br>
Novamente, não sabemos quais desses termos são significativos para compor o modelo. Alguém poderia decidir usar todos os termos porque são apenas $15$. Isso, mesmo em um cenário simples como este, pode levar a uma representação muito errada do sistema que você está tentando modelar. Ok, e se executarmos um algoritmo de força bruta para testar os regressores candidatos e selecionar apenas os significativos? Neste caso, temos $2^{15} = 32768$ possíveis estruturas de modelo para serem testadas.

<br>
Você pode pensar que está tudo bem, temos poder computacional para isso. Mas este caso é muito simples e o sistema pode ter atrasos iguais a $10$ para entrada e saída. Se definirmos $n_y = n_x = 10$ e $\ell=2$, o número de modelos possíveis a serem testados aumenta para $2^{231}=3.4508732\times10^{69}$. Se a não linearidade for definida como $3$, então temos $2^{1771} = 1.3308291989700907535925992... \times 10^{533}$ modelos candidatos.

<br>
Agora, pense no caso quando temos não 1, mas 5, 10 ou mais entradas... e temos que incluir termos para o ruído, e os atrasos máximos são maiores que 10... e a não linearidade é maior que 3...

<br>
E o problema não é resolvido apenas identificando os termos mais significativos. Como você escolhe o número de termos para incluir no modelo final? Não se trata apenas de verificar a relevância de cada regressor, temos que pensar no impacto de incluir $5$, $10$ ou $50$ regressores no modelo. E não esqueça: após selecionar os termos, temos que estimar seus parâmetros.

<br>
Como você pode ver, selecionar os termos mais significativos de um enorme dicionário de termos possíveis não é uma tarefa fácil. E é difícil não apenas por causa do complexo problema combinatório e da incerteza sobre a ordem do modelo. Identificar os termos mais significativos em um cenário não linear é muito difícil porque depende do tipo de não linearidade (singularidade esparsa ou comportamento quase singular, efeitos de memória ou amortecimento e muitos outros), resposta dinâmica (sistemas espaço-temporais, dependentes do tempo), resposta em regime permanente, frequência dos dados, o ruído...

<br>
Apesar de toda essa complexidade, os modelos NARMAX são amplamente utilizados porque são capazes de representar sistemas complexos com modelos simples e transparentes, cujos termos são selecionados usando algoritmos robustos para seleção de estrutura do modelo. A seleção de estrutura do modelo é o núcleo dos métodos NARMAX e a comunidade científica é muito ativa em melhorar métodos clássicos e desenvolver novos. Como eu disse, apresentarei alguns desses métodos em outro post.

<br>
Espero que esta publicação tenha servido como uma breve introdução aos modelos NARMAX. Além disso, espero ter despertado seu interesse nessa classe de modelos. Os links para os outros textos serão disponibilizados em breve, mas sinta-se à vontade para entrar em contato conosco se estiver interessado em colaborar com a biblioteca SysIdentPy ou se quiser esclarecer qualquer dúvida.
