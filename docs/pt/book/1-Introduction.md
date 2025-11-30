# Introdução

O conceito de modelo matemático é fundamental em muitas áreas da ciência. Da engenharia à sociologia, modelos desempenham um papel central no estudo de sistemas complexos, pois permitem simular o que acontecerá em diferentes cenários e condições, prever a saída para uma determinada entrada, analisar suas propriedades e explorar diferentes esquemas de projeto. Para alcançar esses objetivos, no entanto, é crucial que o modelo seja uma representação adequada do sistema em estudo. A modelagem de comportamentos dinâmicos e de regime permanente é, portanto, fundamental para esse tipo de análise e depende de procedimentos de Identificação de Sistemas (SI).

## Modelos

A modelagem matemática é uma excelente maneira de entender e analisar diferentes partes do nosso mundo. Ela nos fornece uma estrutura clara para compreender sistemas complexos e seu comportamento. Seja para tarefas cotidianas ou questões de grande escala como controle de doenças, modelos são uma parte essencial de como lidamos com diversos desafios.

Digitar eficientemente em um layout de teclado QWERTY convencional é resultado de um modelo bem aprendido do teclado QWERTY incorporado nos processos cognitivos individuais. No entanto, se você se deparar com um layout de teclado diferente, como Dvorak ou AZERTY, provavelmente terá dificuldades para se adaptar ao novo modelo. O sistema mudou, então você terá que atualizar seu *modelo*.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/QWERTY.png?raw=true)
> [QWERTY - Wikipedia](https://en.wikipedia.org/wiki/QWERTY) - Layout de teclado [ANSI](https://en.wikipedia.org/wiki/ANSI "ANSI") QWERTY (US)

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/AZERTY.png?raw=true)
> Layout [AZERTY](https://en.wikipedia.org/wiki/AZERTY) usado em um teclado

A modelagem matemática está presente em muitas partes de nossas vidas. Seja analisando tendências econômicas, rastreando como doenças se propagam, ou compreendendo o comportamento do consumidor, modelos são ferramentas essenciais para adquirir conhecimento, tomar decisões informadas e assumir controle sobre sistemas complexos.

Em essência, modelos matemáticos nos ajudam a dar sentido ao mundo. Eles nos permitem entender o comportamento humano e os sistemas com os quais lidamos todos os dias. Ao usar esses modelos, podemos aprender, adaptar e ajustar nossas estratégias para acompanhar as mudanças ao nosso redor.

## Identificação de Sistemas

Identificação de sistemas é uma estrutura orientada a dados para modelar sistemas dinâmicos. Inicialmente, cientistas focavam na identificação de sistemas lineares, mas isso tem mudado nas últimas décadas com maior ênfase em sistemas não lineares. A identificação de sistemas não lineares é amplamente considerada um dos tópicos mais importantes relacionados à modelagem de diversos sistemas dinâmicos, desde séries temporais até comportamentos dinâmicos severamente não lineares.

Recursos extensivos, incluindo excelentes livros-texto cobrindo identificação de sistemas lineares e previsão de séries temporais, estão prontamente disponíveis. Neste livro, revisitamos alguns tópicos conhecidos, mas também tentamos abordar tais assuntos de uma forma diferente e complementar. Exploraremos a modelagem de sistemas dinâmicos não lineares usando métodos NARMAX (Nonlinear AutoRegressive Moving Average model with eXogenous inputs), que foram introduzidos por [Stephen A. Billings] em [1981](https://pdf.sciencedirectassets.com/314898/1-s2.0-S1474667082X74355/1-s2.0-S1474667017630398/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEK7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIAlDD9s3TmWmj7vi6jUiyGu3%2B4wOlhUltouuMtDCf7DdAiBibaBn42D8EkLzeKS6NhEc2E5PPjz%2BpNf7fxe7GuuZ1Cq7BQiW%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIMft9sy4HmUQsgz1JTKo8FyQuLbLHlwSW1p7EBeDgywpc0moBrT8CdqIjV2ucAEJ%2Bxf6PZVgMRTE0KPuxx6tNksRk827UBbXjvWD2b7FCIdNMczpYDcD9LPL0xM3SojpHYLUN9nMOVqssnts1C0efyJrowQbn6Jd6LGGHuF3%2BCnaLsxMTyf8pt%2FlLeYyFzLSe8ins0NcC%2BBWR476hcCSY5fbwU2JxQWLFZPv2xAS7WUge0YiMlc3W%2FmZY8Zx3yTgvnQOVm7qwlq7HM9QVc8hoaoMvPmJH0ZzIAxSbqxuRWwCTE712FOW1CQ1upVRksesVdDX3Tv%2BItXKAp%2Blv2ijKpPDn2z8F1zt1Om%2FutokMZzJZZ5w07PtDgkq23px%2BU6CpXlEZXtAyJRyxXChffGK6Ac7QaBt4vMTuHmD8kqJDqEln2qJYZghUn%2FZx0%2B6NaxkpbdV8u5iG2PnHEwn2FHGO1JKIaewUAV%2BXA92v%2FJjVVVkoqLdR4j4BQOSa2%2F69Scc%2BZq5a29zOHX46lXbXtONYoskQP69GJlLHgfEV3tPoDEe0P%2F3r31muBK4a3qeXqnaLS1TzoBjHqEwiDBlhFbxpIsjDhctWxEo84jGDuyjyz8ByvF%2FcRQ6U73Q%2Bre0SmQABoniognhfL40RXVua0si7CASO2I1y6vmQvR4yGUsG2g5%2FizxKZqWuTeJRMIQqrmTjzgK1EOjpn8B4og68x8hcxVGzd1Bb0i%2FZ0HsXyBZ2DVG7YykR6I8NGQk7pNFeF3PcF5r446wc8vgVYvDy4yq1GkyaGKsI32TQSnYk5aDKOo%2Fx05HHhE1juw9bROiKYrJV%2BDmUj1ToH3AT1YOW4U%2FYyoGoLl2q0QdUi9zRW%2FA%2FCaWIZ3XYtB%2BY7xEFa7YZiSjCmlLW1BjqyAZDOCbxeC9NMZd8ZbHQNkV1tGJLpsuFvFKwPplQw4w3ZFd1F0KEcbH5NqECMYDqFER%2Bnvlpxl%2FYoBODHrVfxUvM7bv3PL5Jhdb8DdaoygIUFgMADVdeGquP0F7FYUeXbJ1s0kJSO5TkXvkyEalxA7Hg1DpvpTXhE7Had9exiuzBPC4A7pLISpdguQkKVbt4gxarxiLPihb5rpueXLp4g2rTcLgls%2BHr5jO4C7wQRbY8QI48%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240802T220345Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY3JGFOCJZ%2F20240802%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=82ab64cae9cae307cbfaf00a3ab07182268c43e6a05052f6172128e80bd86e65&hash=ba5f80609dd1f2a175c54cebdbf1c92094ee30786c7ffd01a10468d28f059112&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1474667017630398&tid=spdf-0f070471-6549-4439-ba88-fc62fb637581&sid=e715fa3e9ec8c847157af269cf9c0ee0d69cgxrqa&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=181658050b5351005659&rr=8ad15c165cd77e01&cc=br).

## Identificação de Sistemas Lineares ou Não Lineares

### Modelos Lineares

Embora a maioria dos sistemas do mundo real seja não linear, você provavelmente deveria tentar modelos lineares primeiro. Modelos lineares geralmente servem como uma forte baseline e podem ser suficientes para o seu caso, proporcionando desempenho satisfatório. [Astron e Murray](https://www.cds.caltech.edu/~murray/books/AM05/pdf/am06-complete_16Sep06.pdf) e [Glad e Ljung](https://www.taylorfrancis.com/books/mono/10.1201/9781315274737/control-theory-lennart-ljung-torkel-glad) mostraram que muitos sistemas não lineares podem ser bem descritos por modelos localmente lineares. Além disso, modelos lineares são fáceis de ajustar, fáceis de interpretar e requerem menos recursos computacionais do que modelos não lineares, permitindo que você experimente rapidamente e obtenha insights antes de pensar em modelos gray box ou modelos não lineares complexos.

Modelos lineares podem ser muito úteis, mesmo na presença de fortes não linearidades, porque é muito mais fácil lidar com eles. Além disso, o desenvolvimento de algoritmos de identificação linear ainda é um campo de pesquisa muito ativo e saudável, com muitos artigos sendo lançados todos os anos [Sai Li, Linjun Zhang, T. Tony Cai & Hongzhe Li](https://www.tandfonline.com/doi/abs/10.1080/01621459.2023.2184373), [Maria Jaenada, Leandro Pardo](https://www.mdpi.com/1099-4300/24/1/123), [Xing Liu; Lin Qiu, Youtong Fang; Kui Wang; Yongdong Li, Jose Rodríguez](https://ieeexplore.ieee.org/abstract/document/10296948), [Alessandro D'Innocenzo and Francesco Smarra](https://www.paperhost.org/proceedings/controls/ECC24/files/0026.pdf). Modelos lineares funcionam bem na maioria das vezes e devem ser a primeira escolha para muitas aplicações. No entanto, ao lidar com sistemas complexos onde as suposições lineares não se aplicam, modelos não lineares tornam-se essenciais.

### Modelos Não Lineares

Quando modelos lineares não apresentam desempenho suficientemente bom, você deve considerar modelos não lineares. É importante notar, no entanto, que mudar de um modelo linear para um não linear nem sempre é uma tarefa simples. Para usuários inexperientes, é comum construir modelos não lineares que apresentam desempenho pior do que os lineares. Para trabalhar com modelos não lineares, você deve considerar que características como erros estruturais, ruído, ponto de operação, sinais de excitação e muitos outros aspectos do seu sistema em estudo impactam sua abordagem e estratégia de modelagem.

> Como sugerido por Johan Schoukens e Lennart Ljung em "[Nonlinear System Identiﬁcation - A User-Oriented Roadmap](https://arxiv.org/pdf/1902.00683)", só comece a trabalhar com modelos não lineares se houver evidência suficiente de que modelos lineares não resolverão o problema.

Modelos não lineares são mais flexíveis do que modelos lineares e podem ser construídos usando muitas representações matemáticas diferentes, como polinomial, aditivo generalizado, redes neurais, wavelet e muitas outras ([Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594)). Tal flexibilidade, no entanto, torna a identificação de sistemas não lineares muito mais complexa do que a linear, desde o projeto do experimento até a seleção do modelo. O usuário deve considerar que, além da complexidade da modelagem, a transição para modelos não lineares exigirá uma revisão no roteiro e nos recursos computacionais definidos ao lidar com modelos lineares. Nesse sentido, sempre se pergunte se os benefícios potenciais dos modelos não lineares valem o esforço.

## Métodos NARMAX

O modelo NARMAX é uma das representações de modelos não lineares mais frequentemente empregadas e é amplamente utilizado para representar uma ampla classe de sistemas não lineares. Os métodos NARMAX foram aplicados com sucesso em muitos cenários, que incluem processos industriais, sistemas de controle, sistemas estruturais, sistemas econômicos e financeiros, biologia, medicina, sistemas sociais e muito mais. A representação do modelo NARMAX e a classe de sistemas que podem ser representados por ele serão discutidas posteriormente no livro.

Os principais passos envolvidos na construção de modelos NARMAX são [(Billings, 2013)](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594):

1. Representação do Modelo: definir a representação matemática do modelo.
2. Seleção de Estrutura do Modelo: definir quais termos estão no modelo final.
3. Estimação de Parâmetros: estimar os coeficientes de cada termo do modelo selecionado no passo 1.
4. Validação do Modelo: garantir que o modelo seja não polarizado e preciso;
5. Predição/Simulação do Modelo: prever saídas futuras ou simular o comportamento do sistema dadas diferentes entradas.
6. Análise: compreender as propriedades dinâmicas do sistema em estudo.
7. Controle: desenvolver esquemas de projeto de controle baseados no modelo obtido.

Model Structure Selection (MSS) é o aspecto mais importante dos métodos NARMAX e também o mais complexo. Selecionar os termos do modelo é fundamental se o objetivo da identificação é obter modelos que possam reproduzir a dinâmica do sistema original e impacta todos os outros aspectos do processo de identificação. Problemas relacionados à sobreparametrização e mal-condicionamento numérico são típicos devido às limitações dos algoritmos de identificação em selecionar os termos apropriados que devem compor o modelo final ([L. A. Aguirre e S. A. Billings](https://www.sciencedirect.com/science/article/abs/pii/0167278995900535), [L. Piroddi e W. Spinelli](https://www.tandfonline.com/doi/abs/10.1080/00207170310001635419)).

> No SysIdentPy, você pode interagir diretamente com cada item descrito nos 7 passos, exceto o de controle. O SysIdentPy foca em modelagem, não em projeto de controle. Você terá que usar parte do código abaixo em toda tarefa de modelagem usando SysIdentPy. Você aprenderá os detalhes ao longo do livro, então não se preocupe se ainda não estiver familiarizado com esses métodos.

```python
from sysidentpy.basis_function import Polynomial
from sysidentpy.neural_network import NARXNN
from sysidentpy.general_estimators import NARX
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import RecursiveLeastSquares
from sysidentpy.metrics import root_relative_squared_error
from sysidentpy.simulation import SimulateNARMAX

```

## Qual é o Propósito da Identificação de Sistemas?

Por causa do problema de Model Structure Selection, [Billings, S. A.](https://www.wiley.com/en-us/Nonlinear+System+Identification%3A+NARMAX+Methods+in+the+Time%2C+Frequency%2C+and+Spatio-Temporal+Domains-p-9781119943594) afirma que o objetivo da Identificação de Sistemas usando métodos NARMAX é duplo: desempenho e parcimônia.

O primeiro objetivo geralmente é sobre aproximação. Aqui, o foco principal é construir um modelo que faça previsões com o menor erro possível. Essa abordagem é comum em aplicações como previsão do tempo, previsão de demanda, previsão de preços de ações, reconhecimento de fala, rastreamento de alvos e classificação de padrões. Nesses casos, a forma específica do modelo não é tão crítica. Em outras palavras, como os termos interagem (em modelos paramétricos), a representação matemática, o comportamento estático e assim por diante não são tão importantes; o que mais importa é encontrar uma maneira de minimizar os erros de previsão.

Mas a identificação de sistemas não é apenas sobre minimizar erros de previsão. Um dos principais objetivos da Identificação de Sistemas é construir modelos que ajudem o usuário a entender e interpretar o sistema sendo modelado. Além de apenas fazer previsões precisas, o objetivo é desenvolver modelos que realmente capturem o comportamento dinâmico do sistema em estudo, idealmente na forma mais simples possível. Ciência e engenharia tratam de entender sistemas decompondo comportamentos complexos em outros mais simples que podemos entender e controlar. Por exemplo, se o comportamento do sistema pode ser descrito por um modelo dinâmico simples de primeira ordem com um termo não linear cúbico na entrada, a identificação de sistemas deve ajudar a descobrir isso.

## Identificação de Sistemas é Machine Learning?

Primeiro, vamos ter uma visão geral de sistemas estáticos e dinâmicos. Imagine que você tem uma guitarra elétrica conectada a um processador de efeitos que pode aplicar vários efeitos de áudio, como reverb ou distorção. O efeito é controlado por um interruptor que alterna entre "ligado" e "desligado". Vamos considerar isso da perspectiva de sinais. O sinal de entrada representa o estado do interruptor de efeito: interruptor desligado (nível baixo), interruptor ligado (nível alto). Se representarmos o sinal da guitarra, temos uma condição binária: efeito desligado (som original da guitarra), efeito ligado (som modificado da guitarra). Este é um exemplo de sistema estático: a saída (som da guitarra) segue diretamente a entrada (estado do interruptor de efeito).

Quando o interruptor de efeito está desligado, a saída é apenas o sinal limpo e inalterado da guitarra. Quando o interruptor de efeito está ligado, a saída é o sinal da guitarra com o efeito aplicado, como amplificação ou distorção. Neste sistema, o efeito estar ligado ou desligado influencia diretamente o sinal da guitarra sem qualquer atraso ou processamento adicional.

Este exemplo ilustra como um sistema estático opera com entradas de controle binárias, onde a saída reflete diretamente o estado da entrada, fornecendo um mapeamento direto entre o sinal de controle e a resposta do sistema.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t = np.linspace(0, 30, 500, endpoint=False)
u = signal.square(0.2*np.pi * t)
u[u < 0] = 0
# In a static system, the output y directly follows the input u
y = u

# Plot the input and output
plt.figure(figsize=(15, 3))
plt.plot(t, u, label='Input (State of the Switch)', color="grey", linewidth=10, alpha=0.5)
plt.plot(t, y, label='Output (Static System Response)', color='k', linewidth=0.5)
plt.title('Static System Response to the Input')
plt.xlabel('Time [s]')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/static_example.png?raw=true)
> Representação de resposta estática. O sinal de entrada representando o estado do interruptor (interruptor desligado (nível baixo), interruptor ligado (nível alto)), e a resposta estática: som original (nível baixo), som processado (nível alto).

Agora, vamos considerar um sistema dinâmico: usar um ar-condicionado para baixar a temperatura do ambiente. Este exemplo ilustra efetivamente os conceitos de sistemas dinâmicos e como sua saída responde ao longo do tempo.

Vamos imaginar isso da perspectiva de sinais. O sinal de entrada representa o estado do controle do ar-condicionado: ligar o ar-condicionado (nível alto) ou desligá-lo (nível baixo). Quando o ar-condicionado é ligado, ele começa a resfriar o ambiente. No entanto, a temperatura do ambiente não cai instantaneamente para o nível mais frio desejado. Leva tempo para o ar-condicionado afetar a temperatura, e a taxa na qual a temperatura diminui pode variar com base em fatores como o tamanho do ambiente e o isolamento.

Por outro lado, quando o ar-condicionado é desligado, a temperatura do ambiente não retorna imediatamente à sua temperatura ambiente original. Em vez disso, ela gradualmente esquenta conforme o efeito de resfriamento diminui.

![](https://github.com/wilsonrljr/sysidentpy-data/blob/4085901293ba5ed5674bb2911ef4d1fa20f3438d/book/assets/dynamic_example.png?raw=true)
> Usar um ar-condicionado para baixar a temperatura do ambiente como representação de sistema dinâmico.

Neste sistema dinâmico, a saída (temperatura do ambiente) não segue instantaneamente a entrada (estado do ar-condicionado) porque há um atraso de tempo envolvido tanto nos processos de resfriamento quanto de aquecimento. O sistema tem memória, o que significa que a temperatura atual do ambiente depende não apenas do estado atual do ar-condicionado, mas também de quanto tempo ele esteve ligado ou desligado, e quanto ele já resfriou ou permitiu que o ambiente esquentasse.

Este exemplo destaca a natureza dos sistemas dinâmicos: a resposta a uma entrada é gradual e afetada pela dinâmica interna do sistema. O efeito do ar-condicionado na temperatura do ambiente exemplifica como sistemas dinâmicos têm uma resposta dependente do tempo, onde a saída muda ao longo do tempo e não corresponde imediatamente ao sinal de entrada.

Para sistemas estáticos, a saída é uma função direta da entrada, representada por uma equação algébrica:

$$
y(t) = G \cdot u(t)
$$

Para sistemas dinâmicos, a saída depende da entrada e da taxa de variação da entrada, representada por uma equação diferencial. Por exemplo, a saída $y(t)$ pode ser modelada como:

$$
y(t) = G \cdot u(t) - \tau \cdot \frac{dy(t)}{dt}
$$

Aqui, $G$ é o ganho, e $\tau$ é uma constante que incorpora a memória do sistema. Para sistemas em tempo discreto, consideramos sinais em intervalos específicos e espaçados. A equação diferencial é discretizada, e a derivada é aproximada por uma diferença finita:

$$
y[k] = \alpha y[k-1] + \beta u[k]
$$

onde $\alpha$ e $\beta$ são constantes que determinam a resposta do sistema. A transformada z pode ser usada para obter a função de transferência no domínio z.

Em resumo, sistemas estáticos são modelados por equações algébricas, enquanto sistemas dinâmicos são modelados por equações diferenciais.

> Como Luis Antonio Aguirre afirma em uma de suas [aulas no YouTube](https://www.youtube.com/watch?v=OVs0p2jem1Q), **todos os sistemas físicos são dinâmicos, mas dependendo da escala de tempo, podem ser modelados como estáticos para simplificação**. Por exemplo, a transição entre os efeitos no som da guitarra, se considerada em segundos (como fizemos no exemplo), poderia ser tratada como estática dependendo da sua análise. No entanto, a pedaleira tem componentes como capacitores, que são componentes elétricos dinâmicos, tornando-a um sistema dinâmico. A resposta, no entanto, é tão rápida que a tratamos como um sistema estático. Portanto, representar um sistema como estático é uma **decisão de modelagem**.

A Tabela 1 mostra como este campo pode ser categorizado em relação a sistemas lineares/não lineares e estáticos/dinâmicos.

| Características do Sistema | Modelo Linear                      | Modelo Não Linear                      |
| -------------------------- | ---------------------------------- | -------------------------------------- |
| Estático                   | Regressão Linear                   | Machine Learning                       |
| Dinâmico                   | Identificação de Sistemas Lineares | Identificação de Sistemas Não Lineares |
> Tabela 1: Convenções de nomenclatura no campo de Identificação de Sistemas. Adaptado de [Oliver Nelles](https://link.springer.com/book/10.1007/978-3-662-04323-3#author-0-0)

## Aplicações de Identificação de Sistemas Não Lineares e Previsão: Estudos de Caso

Há muita pesquisa sobre identificação de sistemas não lineares, incluindo métodos NARMAX. No entanto, há um número relativamente pequeno de livros e artigos mostrando como aplicar esses métodos a sistemas da vida real de uma forma fácil de entender. Nosso objetivo com este livro é mudar isso. Queremos tornar esses métodos práticos e acessíveis. Embora cobriremos a matemática e os algoritmos necessários, manteremos as coisas o mais claras e simples possível, facilitando para leitores de todas as formações aprenderem como modelar sistemas dinâmicos não lineares usando o **SysIdentPy**.

Portanto, este livro visa preencher uma lacuna na literatura existente. No Capítulo 10, apresentamos estudos de caso do mundo real para mostrar como os métodos NARMAX podem ser aplicados a uma variedade de sistemas complexos. Seja modelando um sistema altamente não linear como o modelo de Bouc-Wen, modelando um comportamento dinâmico em uma aeronave F-16 em escala real, ou trabalhando com o dataset M4 para benchmarking, guiaremos você na construção de modelos NARMAX usando o **SysIdentPy**.

Os estudos de caso que selecionamos vêm de uma ampla gama de campos, não apenas os típicos exemplos de séries temporais ou industriais que você poderia esperar de livros tradicionais de identificação de sistemas ou séries temporais. Nosso objetivo é mostrar a versatilidade dos algoritmos NARMAX e do **SysIdentPy** e ilustrar o tipo de análise aprofundada que você pode alcançar com essas ferramentas.

## Abreviações

| Abreviação | Nome Completo                                                |
| ---------- | ------------------------------------------------------------ |
| AIC        | Akaike Information Criterion                                 |
| AICC       | Corrected Akaike Information Criterion                       |
| AOLS       | Accelerated Orthogonal Least Squares                         |
| ANN        | Artificial Neural Network                                    |
| AR         | AutoRegressive                                               |
| ARMAX      | AutoRegressive Moving Average with eXogenous Input           |
| ARARX      | AutoRegressive AutoRegressive with eXogenous Input           |
| ARX        | AutoRegressive with eXogenous Input                          |
| BIC        | Bayesian Information Criterion                               |
| ELS        | Extended Least Squares                                       |
| ER         | Entropic Regression                                          |
| ERR        | Error Reduction Ratio                                        |
| FIR        | Finite Impulse Response                                      |
| FPE        | Final Prediction Error                                       |
| FROLS      | Forward Regression Orthogonal Least Squares                  |
| GLS        | Generalized Least Squares                                    |
| LMS        | Least Mean Square                                            |
| LS         | Least Squares                                                |
| LSTM       | Long Short-Term Memory                                       |
| MA         | Moving Average                                               |
| MetaMSS    | Meta Model Structure Selection                               |
| MIMO       | Multiple Input Multiple Output                               |
| MISO       | Multiple Input Single Output                                 |
| MLP        | Multilayer Perceptron                                        |
| MSE        | Mean Squared Error                                           |
| MSS        | Model Structure Selection                                    |
| NARMAX     | Nonlinear AutoRegressive Moving Average with eXogenous Input |
| NARX       | Nonlinear AutoRegressive with eXogenous Input                |
| NFIR       | Nonlinear Finite Impulse Response                            |
| NIIR       | Nonlinear Infinite Impulse Response                          |
| NLS        | Nonlinear Least Squares                                      |
| NN         | Neural Network                                               |
| OBF        | Orthonormal Basis Function                                   |
| OE         | Output Error                                                 |
| OLS        | Orthogonal Least Squares                                     |
| RBF        | Radial Basis Function                                        |
| RELS       | Recursive Extended Least Squares                             |
| RLS        | Recursive Least Squares                                      |
| RMSE       | Root Mean Squared Error                                      |
| SI         | System Identification                                        |
| SISO       | Single Input Single Output                                   |
| SVD        | Singular Value Decomposition                                 |
| WLS        | Weighted Least Squares                                       |

## Variáveis

| Nome da Variável        | Descrição                                                          |
| ----------------------- | ------------------------------------------------------------------ |
| $f(\cdot)$              | função a ser aproximada                                            |
| $k$                     | tempo discreto                                                     |
| $m$                     | ordem dinâmica                                                     |
| $x$                     | entradas do sistema                                                |
| $y$                     | saída do sistema                                                   |
| $\hat{y}$               | saída predita do modelo                                            |
| $\lambda$               | força de regularização                                             |
| $\sigma$                | desvio padrão                                                      |
| $\theta$                | vetor de parâmetros                                                |
| $N$                     | número de pontos de dados                                          |
| $\Psi(\cdot)$           | Matriz de Informação                                               |
| $n_{m^r}$               | Número de regressores potenciais para modelos MIMO                 |
| $\mathcal{F}$           | Representação matemática arbitrária                                |
| $\Omega_{y^p x^m}$      | Cluster de termos do NARX polinomial                               |
| $\ell$                  | grau de não linearidade do modelo                                  |
| $\hat{\Theta}$          | Vetor de Parâmetros Estimados                                      |
| $\hat{y}_k$             | saída predita do modelo no tempo discreto $k$                      |
| $\mathbf{X}_k$          | Vetor coluna de múltiplas entradas do sistema no tempo discreto $k$|
| $\mathbf{Y}_k$          | Vetor coluna de múltiplas saídas do sistema no tempo discreto $k$  |
| $\mathcal{H}_t(\omega)$ | Loop de histerese do sistema em tempo contínuo                     |
| $\mathcal{H}$           | Estrutura delimitadora que delimita o loop de histerese do sistema |
| $\rho$                  | Valor de tolerância                                                |
| $\sum_{y^p x^m}$        | Coeficientes de cluster do NARX polinomial                         |
| $e_k$                   | vetor de erro no tempo discreto $k$                                |
| $n_r$                   | Número de regressores potenciais para modelos SISO                 |
| $n_x$                   | lag máximo do regressor de entrada                                 |
| $n_y$                   | lag máximo do regressor de saída                                   |
| $n$                     | número de observações em uma amostra                               |
| $x_k$                   | entrada do sistema no tempo discreto $k$                           |
| $y_k$                   | saída do sistema no tempo discreto $k$                             |

## Organização do Livro

Este livro foca em tornar os conceitos fáceis de entender, enfatizando explicações claras e conexões práticas entre diferentes métodos. Evitamos formalismo excessivo e equações complexas, optando em vez disso por ilustrar ideias centrais com muitos exemplos práticos. Escrito com uma perspectiva de Identificação de Sistemas, o livro oferece detalhes de implementação prática ao longo dos capítulos.

Os objetivos deste livro são ajudá-lo a:

- Entender as vantagens, desvantagens e áreas de aplicação de diferentes modelos e algoritmos NARMAX.
- Escolher a abordagem correta para o seu problema específico.
- Ajustar todos os hiperparâmetros adequadamente.
- Interpretar e compreender os resultados obtidos.
- Avaliar a confiabilidade e limitações dos seus modelos.

Muitos capítulos incluem exemplos e dados do mundo real, guiando você sobre como aplicar esses métodos usando o SysIdentPy na prática.
