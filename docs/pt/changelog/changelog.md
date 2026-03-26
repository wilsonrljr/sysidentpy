---
template: overrides/main.html
title: Histórico de Alterações
---

# Alterações no SysIdentPy

## v0.8.0
### COLABORADORES
- Wilson Rocha Lacerda Junior (wilsonrljr)

### MUDANÇAS
Esta versão adiciona duas novas famílias de algoritmos de seleção de estrutura de modelo, correções no UOFR e atualizações abrangentes de documentação e configuração.

#### Novos Recursos
- Adicionado `RMSS` (Robust Model Structure Selection), um novo algoritmo para seleção de estrutura de modelo usando o critério OMAE (Overall Mean Absolute Error) com reamostragem leave-one-out por padrão. Projetado para robustez em amostras pequenas e múltiplos conjuntos de dados. Baseado no artigo de Gu, Y., & Wei, H.-L., "A Robust Model Structure Selection Method for Small Sample Size and Multiple Datasets Problems."
- Adicionada a família de algoritmos Orthogonal Floating Search: `OSF` (Orthogonal Sequential Floating Forward), `OIF` (Orthogonal Insertion-removal Floating search), `OOS` (Orthogonal Oscillating Search) e `O2S` (alias para OOS). Esses algoritmos combinam projeções ortogonais com o critério ERR (Error Reduction Ratio) e estratégias de busca flutuante.

#### Correções / Melhorias
- Corrigida a aumentação de Sobolev no UOFR para corresponder às equações do artigo original.
- Melhorado o desempenho do UOFR com operações em bloco BLAS e einsum.

#### Testes
- Adicionados conjuntos de testes abrangentes para RMSS e algoritmos OFS.
- Expandidos testes do UOFR incluindo cobertura baseada em profundidade para falhas de swing repetidas.

#### Documentação & Configuração
- Adicionadas páginas de documentação da API para RMSS e Orthogonal Floating Search.
- Atualizados guias de início rápido (EN, PT, ES) para listar todos os algoritmos MSS disponíveis.
- Corrigidos requisitos de versão desatualizados de Python e NumPy no README.
- Corrigida URL do changelog no pyproject.toml (master → main).
- Atualizado Black target_version para corresponder às versões suportadas de Python (3.10–3.14).
- Removida configuração deprecada `ignore-init-module-imports` do Ruff.
- Ampliados limites de versão de dependências de desenvolvimento para pytest e pytest-cov.
- Adicionado guard de importação para dependência opcional do PyTorch.
- Atualizado `actions/setup-python` para v5 no workflow de CI.
- Atualizado ano de copyright para 2026.

### IMPACTO
Duas novas famílias de algoritmos de seleção de estrutura de modelo expandem o conjunto de ferramentas disponíveis para pesquisadores e profissionais. O RMSS aborda o importante problema de seleção robusta de modelos com conjuntos de dados pequenos, enquanto a família OFS fornece estratégias flexíveis de busca flutuante para seleção de termos do modelo.

### TESTES
Conjuntos de testes expandidos cobrem todos os novos algoritmos. A matriz de CI continua testando contra Python 3.10–3.14.

## v0.7.0
### COLABORADORES
- Wilson Rocha Lacerda Junior (wilsonrljr)
- aryan

### ALTERAÇÕES
Esta versão traz exportação de equações em formato legível, fluxos neurais reprodutíveis, ganhos significativos de desempenho e a maior atualização de documentação até agora, incluindo sites localizados.

#### Novos Recursos
- Adicionados `sysidentpy.utils.equation_formatter` e utilitários como `format_equation`, permitindo que qualquer modelo ajustado gere uma equação simbólica que respeita a função base escolhida, os lags e a ordenação `pivv`.
- `NeuralNarx` agora aceita `random_state`, normaliza parâmetros do construtor, reutiliza tensores fixados entre dispositivos e emite métricas verbosas em uma única passagem, mantendo os experimentos determinísticos e leves em hardware.
- O formatter expõe objetos estruturados `EquationItem`, permitindo que docs, notebooks e ferramentas externas incorporem exatamente os regressores e coeficientes que definem o modelo treinado.

#### Melhorias de Desempenho
- Funções base polinomiais agora armazenam índices de combinação e constroem termos com multiplicações vetorizadas e buffers pré-alocados, deixando os ajustes NARX/NARMAX de altos graus várias vezes mais rápidos.
- `predict` e construtores de regressores reutilizam matrizes de expoentes e buffers, reduzindo drasticamente alocações durante simulações e previsões longas.
- UOFR, ERR, `shift_column` e Accelerated OLS agora utilizam slices/einsum amigáveis a BLAS, reduzindo significativamente o tempo de execução em grandes conjuntos de dados.

#### Alterações na API
- Suporte oficial agora contempla Python 3.10–3.14; Python 3.8/3.9 foram removidos para alinhar com NumPy ≥2.0 e as rodas mais recentes do PyTorch, e o CI acompanha a nova matriz.
- Métricas de informação mútua (`mutual_information_knn`, `conditional_mutual_information`) agora ordenam explicitamente seleções de índices, mantendo os cálculos de epsilon determinísticos entre versões do NumPy.
- Workflows do GitHub foram atualizados: dependências do PyTorch foram elevadas e artefatos de cobertura passam a ser publicados automaticamente a cada pull request.

#### Documentação e Site
- A documentação foi reorganizada em `docs/en`, `docs/es` e `docs/pt`, agora alimentadas por `mkdocs-static-i18n`; o livro completo, o guia do desenvolvedor, o quickstart e as landing pages foram traduzidos para português, e as páginas iniciais/landing em espanhol ganharam seus próprios overrides.
- Todos os tutoriais/how-to em notebook foram migrados para Markdown, garantindo renderização consistente no mobile e permitindo que tradutores trabalhem diretamente com diffs de texto.
- Adicionados novos how-tos (funções base customizadas, Neural NARX, salvar/carregar modelos, simular modelos existentes, least squares estendido), além de páginas expandidas de suporte à comunidade e atualizações no README.
- Landing pages receberam CSS inédito, logos de patrocinadores responsivos à cor, blocos “Trusted by” polidos e templates dedicados para cada idioma.
- Hooks customizados do MkDocs buscam métricas em tempo real de downloads no PePy e a versão mais recente do PyPI, mantendo o site sempre atualizado.

#### Testes, Ferramentas e CI
- Centenas de novos testes cobrem MetaMSS, OFRBase, AOLS, ERR, ramos do Neural NARX, utilitários de simulação, estimadores gerais e o formatter de equações, garantindo fixtures determinísticos.
- Resultados de cobertura agora são enviados pelo CI, e os segredos necessários para capturar métricas de download foram integrados ao pipeline de docs.
- Atualizações de tooling (.gitignore, suporte a uv/venv, linters renovados) simplificam o desenvolvimento local.

### IMPACTO
- Equações de modelo podem ser auditadas e incorporadas em qualquer lugar, experimentos com Neural NARX ficam reprodutíveis e regressores vetorizados reduzem drasticamente o tempo de treino e simulação.
- Documentação localizada (inglês, espanhol, português) e o livro traduzido diminuem a barreira de entrada para novos usuários ao redor do mundo.

### TESTES
- A matriz de CI cobre Python 3.10–3.14, publica resumos de cobertura e o conjunto ampliado de pytest valida o formatter, MetaMSS, fluxos OFR/AOLS, utilidades de simulação, ramos neurais e os hooks do MkDocs.

## v0.6.0
### COLABORADORES
- wilsonrljr
- oliveira-mark

### ALTERAÇÕES
Esta versão traz melhorias significativas focadas em organização de código, legibilidade e conformidade com PEP8. Inclui ainda uma nova classe base para algoritmos baseados em Error Reduction Ratio (ERR), mais testes e o fim do suporte ao Python 3.7.

#### Novos Recursos
  - Introdução da classe `OFRBase`, que encapsula métodos comuns essenciais para algoritmos baseados em ERR.
  - Refatoração da classe `FROLS` para herdar de `OFRBase`, concentrando-se apenas no algoritmo ERR.
  - Implementação do algoritmo Ultra Orthogonal Forward Regression (UOFR), também herdando de `OFRBase`.

#### Alterações na API
  - **BREAKING CHANGE**: Correção de um typo na função base Bernstein (antes `Bersntein`).
  - **Refatoração e Modularização:**
    - A classe `InformationMatrix` foi extraída de `narmax_base` para o novo módulo `utils.information_matrix`.
    - Métodos específicos foram movidos para os novos módulos `utils.lags` e `utils.simulation`, promovendo melhor separação de responsabilidades.
    - Adicionada mensagem de depreciação para o argumento `solver` do RidgeRegression.

#### Melhorias de Qualidade de Código
  - Renomeação de variáveis e métodos para melhor legibilidade e conformidade com PEP8, incluindo a troca de variáveis maiúsculas por minúsculas.
  - Imports atualizados para usar os novos módulos utilitários, reduzindo redundância e melhorando a manutenção.
  - Remoção de imports não usados e parênteses redundantes, simplificando a base de código.
  - Atualização da versão do Python no workflow de deploy.

#### Aprimoramentos em Testes
  - Inclusão de testes abrangentes para as funções base Bernstein, Bilinear, Hermite (normalizada), Laguerre e Legendre.
  - Novos testes para métodos utilitários, incluindo `narmax_tools`, `save_load` e as novas utilidades de simulação.
  - Cobertura de testes aumentada para **92%**, garantindo robustez e confiabilidade.

#### Validação e Tratamento de Erros
  - Implementada validação para `train_percentage`, levantando erro para valores acima de 100%.
  - Métodos e testes adaptados após a remoção da classe `InformationMatrix`, mantendo consistência em toda a base.

#### Atualizações de Documentação
  - Lançado um frontend redesenhado, com UI moderna e melhor responsividade.
  - Diversas seções foram reestruturadas para maior organização e clareza.
  - Reformulados guias como `quick_start`, `developer_guide` e `user_guide`.
  - Novos exemplos adicionados, incluindo o Sistema de Lorenz e o Mapa Caótico.
  - Melhoria de gramática e legibilidade em toda a documentação.
  - Dependências relacionadas a `mkdocs` atualizadas para melhor performance e compatibilidade.
  - Integração do Google Analytics aprimorada.
  - Correção de links quebrados (blogs do Nubank e Estatidados) e ajustes de formatação no livro.
  - Docstrings atualizadas para refletir as mudanças recentes.
  - Padronização de docstrings e assinaturas de métodos com variáveis minúsculas, seguindo PEP8.
  - Exemplos de contribuição revisados para refletir a versão atual do `sysidentpy`.
  - Exemplos do livro integrados à documentação tradicional, com links diretos para a seção do livro.
  - Estrutura, títulos e links ajustados em vários docs e exemplos para melhor navegação.
  - Arquivos de dataset removidos; agora estão no repositório dedicado `sysidentpy-data`.
  - Apoio da JetBrains mencionado no README e na página de patrocinadores.
  - Correção do `edit uri` ao clicar para editar páginas no site de docs.
  - Agora todo exemplo carrega os dados do repositório `sysidentpy-data`.

#### Atualização de Suporte a Versões do Python
  - **Suporte ao Python 3.7 foi descontinuado.** A medida acompanha o fim oficial do suporte à versão e resolve incompatibilidades com dependências mais novas.
  - Alguns algoritmos de estimação de parâmetros, como Bounded Variable Least Squares, exigem versões recentes do SciPy que não suportam Python 3.7.
  - Usuários ainda podem rodar o SysIdentPy em Python 3.7, mas determinados recursos, incluindo algumas funcionalidades de estimação de parâmetros, ficarão indisponíveis.

#### IMPACTO
- As mudanças melhoram a modularidade, legibilidade e manutenção da base. A introdução de `OFRBase` simplifica a implementação de algoritmos baseados em ERR e facilita extensões futuras. Testes abrangentes garantem confiabilidade para recursos novos e existentes.

#### TESTES
- Todos os testes novos e existentes foram executados, atingindo **92% de cobertura**.

## v0.5.3

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

IMPORTANTE! Esta atualização corrige um bug relacionado à função base Bilinear para modelos com mais de 2 entradas. A versão continua preparando o terreno para futuras evoluções do SysIdentPy, facilitando novas funcionalidades e melhorias rumo a um lançamento 1.0.0 robusto.

#### Alterações na API
- Correção da função base Bilinear para modelos com mais de 2 entradas. A correção ajusta o método `get_max_xlag` em `basis_function_base` e também a criação de `combination_xlag`.

## v0.5.2

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

IMPORTANTE! Esta atualização corrige um bug crítico nas funções base Polynomial e Bilinear para modelos com mais de 3 entradas. O problema surgiu nas mudanças da versão v0.5.0 e agora foi resolvido. A release mantém o foco em preparar o futuro do SysIdentPy.

#### Alterações na API
- Correção da função base Polynomial e Bilinear para modelos com mais de 3 entradas.

## v0.5.1

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

Esta atualização corrige um bug crítico no estimador não enviesado. O problema afetava todas as funções base e foi solucionado. A versão segue preparando a base para recursos futuros rumo ao 1.0.0.

#### Documentação
- Remoção de código desnecessário ao importar funções base em vários exemplos.

#### Alterações na API
- Correção do estimador não enviesado para todas as funções base.


## v0.5.0

### COLABORADORES

- wilsonrljr
- nataliakeles
- LeWerner42
- Suyash Gaikwad


### ALTERAÇÕES

Esta atualização apresenta novos recursos importantes e correções essenciais. A release fornece a base para evoluções futuras do SysIdentPy rumo a um 1.0.0 completo.


#### Novos Recursos
- **MAJOR**: Adicionada a função base Bilinear (obrigado, nataliakeles). Agora é possível usar modelos NARX bilineares em previsões.
- **MAJOR**: Adicionada a função base polinomial Legendre. Agora é possível usar modelos NARX Legendre.
- **MAJOR**: Adicionada a função base polinomial Hermite. Agora é possível usar modelos NARX Hermite.
**MAJOR**: Adicionada a função base polinomial Hermite Normalizada. Agora é possível usar modelos NARX Hermite Normalizada.
**MAJOR**: Adicionada a função base polinomial Laguerre. Agora é possível usar modelos NARX Laguerre.

#### Documentação
- Adicionada visão geral das funções base.
- Arquivos relacionados à documentação v.3.* removidos.
- Melhorias de formatação em equações matemáticas.
- Correções de typos e gramática no README.md (obrigado Suyash Gaikwad e LeWerner42).
- Pequenas adições e correções gramaticais.
- Removidos os assets do livro do repositório principal (movidos para `sysidentpy-data`).
- Correção do link na capa do livro e ajuste de `x2_val` para `x_valid` em exemplos no README.
- Adicionado Pix como método alternativo para patrocinadores brasileiros.
- Correção da documentação de código das funções base (não aparecia nos docs antes).
- Remoção do `pip install` da lista de dependências necessárias no capítulo.

#### Datasets
- Datasets agora estão disponíveis em repositório separado.

#### Alterações na API
- Adicionadas mensagens de depreciação para `bias` e `n` na função base Bernstein. Ambos os parâmetros serão removidos na v0.6.0; use `include_bias` e `degree`.
- Deploy-docs.yml: opção alterada para gerar build limpo da documentação.
- Deploy-docs.yml: versão do Python ajustada para deploy de docs.
- Suporte planejado para Python 3.13 dependendo do lançamento do PyTorch 2.6. Todos os métodos funcionam em Python 3.13, exceto Neural NARX.
- Atualização da versão da dependência `mkdocstrings`.
- Troca do check polinomial de nome de classe para `isinstance` em todas as classes.
- Remoção do suporte a `torch==2.4.0` devido a erro no pip do PyTorch. Será reavaliado antes de liberar versões futuras.
- `main` se torna a nova branch padrão. `master` removida.
- Actions ajustadas de `master` para `main`.
- Classes de funções base divididas em múltiplos arquivos (um por função).
- Correção da verificação redundante de bias em Bernstein.
- Correção da notação matemática nas docstrings de funções base.
- Remoção do arquivo requirements.txt.
- Grande refatoração de código: melhorias em type hints, docstrings, remoção de código não usado e outras mudanças para suportar novos recursos.
- Adicionada `model_type` no `fit` e `predict` da base de funções base.
- Variável `combinations` renomeada para `combination_list` para evitar conflitos com `itertools.combinations`.
- Remoção do arquivo requirements.txt.

## v0.4.0

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

Esta atualização traz diversos recursos importantes e mudanças, incluindo breaking changes. Há um guia para ajudar a atualizar o código. Dependendo da definição do modelo, nenhuma alteração pode ser necessária. Optei por ir direto para a versão v0.4.0 porque as mudanças são fáceis de aplicar e os novos recursos são bastante úteis. A release prepara terreno para evoluções futuras rumo ao 1.0.0.

#### Novos Recursos
- **MAJOR**: Algoritmo NonNegative Least Squares para estimação de parâmetros.
- **MAJOR**: Algoritmo Bounded Variables Least Squares para estimação de parâmetros.
- **MAJOR**: Algoritmo Least Squares Minimal Residual para estimação de parâmetros.
- **MAJOR**: Aprimoramento do algoritmo Error Reduction Ratio no FROLS. Agora é possível definir `err_tol` para interromper quando a soma do ERR atingir o limite, oferecendo alternativa mais rápida aos critérios de informação. Novo exemplo disponível na docs.
- **MAJOR**: Nova função base Bernstein, além das opções Polynomial e Fourier.
- **MAJOR**: v0.1 do livro "Nonlinear System Identification: Theory and Practice With SysIdentPy". O livro open source serve como documentação robusta do pacote e introdução amigável à identificação de sistemas não lineares e previsão de séries temporais. Há estudos de caso adicionais no livro que não estavam na documentação na época.

#### Documentação
- Todos os exemplos atualizados para refletir as mudanças da v0.4.0.
- Adicionado guia sobre como definir um método customizado de estimação de parâmetros e integrá-lo ao SysIdentPy.
- Documentação movida para a branch `gh-pages`.
- Definida GitHub Action para construir os docs automaticamente ao fazer push na main.
- Remoção de código não utilizado de modo geral.

#### Datasets
- Datasets agora estão disponíveis em repositório separado.

#### Alterações na API
- **BREAKING CHANGE**: O método de estimação precisa ser importado e passado na definição do modelo, substituindo a string anterior. Exemplo: use `from sysidentpy.parameter_estimation import LeastSquares` em vez de `"least_squares"`. Isso dá mais flexibilidade e facilita estimadores customizados. Há página específica guiando a migração da v0.3.4 para a v0.4.0.
- **BREAKING CHANGE**: O método `fit` do MetaMSS agora requer apenas `X` e `y`, sem necessidade de `fit(X=, y=, X_test=, y_test=)`.
- Adicionado suporte ao Python 3.12.
- Introduzido o hiperparâmetro `test_size` para definir a proporção de dados de treino usada no ajuste.
- Ampla refatoração com melhorias em type hints, docstrings, remoção de código não usado e outras mudanças para suportar novos recursos.

## v0.3.4

### COLABORADORES

- wilsonrljr
- dj-gauthier
- mtsousa

### ALTERAÇÕES

#### Novos Recursos
- **MAJOR**: Estimação de Parâmetros via Ridge Regression:
  - Introdução do algoritmo Ridge para estimação de parâmetros de modelos (Issue #104). Defina `estimator="ridge_regression"` e controle a regularização com `alpha`. Obrigado a @dj-gauthier e @mtsousa pela contribuição. Veja https://www.researchgate.net/publication/380429918_Controlling_chaos_using_edge_computing_hardware para saber como @dj-gauthier utilizou o SysIdentPy em sua pesquisa.

#### Alterações na API
- Código de `plotting.py` aprimorado com type hints e novas opções de visualização.
- Métodos refatorados para resolver avisos futuros do NumPy.
- Código refatorado seguindo PEP8.
- Estilo "default" definido como padrão para evitar erros em novas versões do matplotlib.

#### Datasets
- Adicionados os datasets `buck_id.csv` e `buck_valid.csv` ao repositório.

#### Documentação
- Adicionado exemplo NFIR (Issue #103) mostrando como construir modelos sem regressores de saída passados.
- Exemplo de uso do MetaMSS aprimorado.
- Continuidade na adição de type hints.
- Docstrings melhoradas em toda a base.
- Pequenas correções e ajustes gramaticais.
- @dj-gauthier sugeriu melhorias extras para a documentação; elas estão em refinamento e serão disponibilizadas em breve.

#### Ferramentas de Desenvolvimento
- Adicionados hooks de pre-commit.
- `pyproject.toml` aprimorado para ajudar contribuidores a configurarem o ambiente.

## v0.3.3

### COLABORADORES

- wilsonrljr
- GabrielBuenoLeandro
- samirmartins

### ALTERAÇÕES

- A versão **v0.3.3** foi lançada com novos recursos, mudanças de API e correções.

#### Alterações na API
- MAJOR: Framework Multiobjetivo — Affine Information Least Squares (AILS)
    - Agora é possível usar AILS para estimar parâmetros de modelos NARMAX (e variantes) com abordagem multiobjetivo.
    - AILS pode ser importado via `from sysidentpy.multiobjective_parameter_estimation import AILS`.
    - Veja a documentação para detalhes.
    - Recurso relacionado ao Issue #101. Trabalho fruto de pesquisa de graduação de Gabriel Bueno Leandro sob supervisão de Samir Milani Martins e Wilson Rocha Lacerda Junior.
    - Diversos métodos novos foram implementados para suportar o recurso (veja `sysidentpy -> multiobjective_parameter_estimation`).
- Mudança de API: variável `regressor_code` renomeada para `enconding` para evitar usar o mesmo nome do método `regressor_code` em `narmax_tool`.

#### Datasets
- DATASET: adicionados `buck_id.csv` e `buck_valid.csv` ao repositório.

#### Documentação
- DOC: Notebook de Otimização Multiobjetivo mostrando como usar o novo AILS.
- DOC: Pequenas adições e correções gramaticais.

## v0.3.2

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.3.2** foi lançada com mudanças de API e correções.

#### Alterações na API
- Major:
    - Adicionado Akaike Information Criterion corrigido (AICc) no FROLS. Agora é possível usar `aicc` como critério de informação para selecionar a ordem do modelo.

- FIX: Issue #114. Substituição de `yhat` por `y` no root relative squared error (obrigado @miroder).
- TESTES: Pequenas mudanças removendo cargas de dados desnecessárias.
- Remoção de código e comentários não usados.

#### Documentação
- Docs: Pequenas alterações em notebooks. Método AICc adicionado no exemplo de critérios de informação.

## v0.3.1

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.3.1** foi lançada com mudanças de API e correções.

#### Alterações na API
- MetaMSS retornava o máximo lag do modelo final em vez do lag máximo relacionado a xlag e ylag. Embora não estivesse incorreto (Issue #55), a mudança foi aplicada em todos os métodos. Assim, voltamos a retornar o lag máximo de xlag/ylag.
- Mudança de API: método `build_matrix` adicionado em BaseMSS, melhorando legibilidade ao eliminar blocos if/elif/else em cada algoritmo de estrutura.
- Mudança de API: adicionados métodos `bic`, `aic`, `fpe` e `lilc` no FROLS. A escolha é feita via dicionário pré-definido, melhorando legibilidade e reduzindo if/elif/else.
- TESTES: Adicionados testes para a classe Neural NARX. O problema com PyTorch foi solucionado e agora existem testes para todas as classes de modelos.
- Remoção de código e comentários não usados.


## v0.3.0

### COLABORADORES

- wilsonrljr
- gamcorn
- Gabo-Tor

### ALTERAÇÕES

- A versão **v0.3.0** foi lançada com novos recursos, mudanças de API e correções.

#### Alterações na API
- MAJOR: Suporte a estimadores no AOLS
    - Agora é possível usar qualquer estimador do SysIdentPy no AOLS.

- Refatoração da classe base de seleção de estrutura de modelo, preparando o pacote para novos recursos futuros (multiobjetivo, novas funções base, etc.).

  Diversos métodos foram reescritos para melhorar funcionalidade e desempenho, facilitando a incorporação de técnicas avançadas de seleção.
  - Evita herança desnecessária em cada algoritmo de MSS e melhora legibilidade.
  - Métodos reescritos para eliminar duplicação.
  - Melhoria geral de legibilidade ao reescrever blocos if/elif/else.

- Breaking Change: `X_train`/`y_train` substituídos por `X`/`y` no `fit` do MetaMSS. `X_test`/`y_test` substituídos por `X`/`y` no `predict`.

- Mudança de API: adicionada classe BaseBasisFunction (classe abstrata para funções base).
- Melhoria: suporte ao Python 3.11.
- Aviso de depreciação futura: o usuário deverá definir o estimador e passá-lo para cada algoritmo de seleção em vez de strings. Atualmente `estimator="least_squares"`; na versão 0.4.0 será `estimator=LeastSquares()`.
- FIX: Issue #96. Correção com numpy 1.24.* (obrigado @gamcorn).
- FIX: Issue #91. Correção na métrica r2_score com arrays 2D.
- FIX: Issue #90.
- FIX: Issue #88. Correção do erro de previsão one-step-ahead em `SimulateNARMAX` (obrigado, Lalith).
- FIX: Correção na seleção de regressores em AOLS.
- Fix: Correção na previsão n passos à frente ao passar apenas condição inicial.
- FIX: Correção de Visible Deprecation Warning no método `get_max_lag`.
- FIX: Correção de aviso de depreciação no exemplo de Extended Least Squares.

#### Datasets
- DATASET: Adicionado dataset de passageiros aéreos ao repositório.
- DATASET: Adicionado dataset de carga hospitalar de São Francisco.
- DATASET: Adicionado dataset PV GHI de São Francisco.

#### Documentação
- DOC: Documentação aprimorada na página Setting Specific Lags, com exemplo para modelos MISO.
- DOC: Pequenas adições e correções gramaticais.
- DOC: Melhoria de visualização de imagens com mkdocs-glightbox.
- Pacotes de desenvolvimento atualizados.

## v0.2.1

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.2.1** foi lançada com novos recursos, pequenas mudanças de API e correções.

#### Alterações na API
- MAJOR: Neural NARX agora suporta CUDA
    - É possível construir modelos Neural NARX com suporte a CUDA, basta adicionar `device='cuda'` para aproveitar a GPU.
    - Documentação atualizada com instruções do novo recurso.
- Testes:
    - Agora há testes para quase todas as funções.
    - Testes de Neural NARX estavam gerando issues com numpy; seriam corrigidos na próxima atualização.
- FIX: Modelos NFIR em General Estimators
    - Correção do suporte a modelos NFIR usando estimadores do sklearn.
- O setup agora é gerenciado pelo arquivo pyproject.toml.
- Remoção de código não utilizado.

#### Documentação
- MAJOR: Novo site de documentação
    - Agora toda a documentação é baseada em Markdown (sem rst).
    - Uso de MkDocs com o tema Material for MkDocs.
    - O site possui tema escuro.
    - Página Contribute com mais detalhes para quem quer colaborar.
    - Novas seções (Blog, Sponsors etc.).
    - Muitas melhorias internas.

- MAJOR: GitHub Sponsor
    - Agora é possível apoiar o SysIdentPy tornando-se patrocinador: https://github.com/sponsors/wilsonrljr

- Correção de variáveis em docstrings.
- Correções de formatação de código.
- Ajustes gramaticais menores.
- Correções em HTML dos notebooks na documentação.
- README atualizado.


## v0.2.0

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.2.0** foi lançada com novos recursos, pequenas mudanças de API e correções.

#### Alterações na API
- MAJOR: Muitos novos recursos para General Estimators
    - Agora é possível construir modelos General NARX com função base Fourier.
    - É possível escolher a função base importando de `sysidentpy.basis_function`. Consulte os notebooks de exemplo.
    - Agora é possível construir modelos General NAR; basta passar `model_type="NAR"`.
    - Agora é possível construir modelos General NFIR; basta `model_type="NFIR"`.
    - Agora é possível rodar previsão n-passos à frente usando General Estimators. Antes apenas infinito-passos era permitido.
    - Polynomial e Fourier são suportadas por enquanto. Novas funções base virão em releases futuras.
    - Não há mais necessidade de informar o número de entradas.
    - Docstrings melhoradas.
    - Correções gramaticais menores.
    - Várias mudanças internas.

- MAJOR: Muitos novos recursos para NARX Neural Network
    - Agora é possível construir modelos Neural NARX com função base Fourier.
    - Escolha de função base via `sysidentpy.basis_function`.
    - Possibilidade de construir modelos Neural NAR.
    - Possibilidade de construir modelos Neural NFIR.
    - Agora é possível rodar previsão n-passos à frente usando Neural NARX (antes apenas infinito-passos).
    - Polynomial e Fourier suportadas inicialmente; novas funções base chegarão em breve.
    - Não é mais necessário passar o número de entradas.
    - Docstrings melhoradas.
    - Correções gramaticais menores.
    - Muitas mudanças internas.

- Major: Suporte a métodos antigos removido.
    - `sysidentpy.PolynomialNarmax` antigo foi removido. Todos os recursos antigos foram incluídos na nova API, com várias melhorias.

- Mudança de API (nova): `sysidentpy.general_estimators.ModelPrediction`
    - Classe adaptada para suportar General Estimators como classe independente.
    - `predict`: método base de previsão. Suporta infinity, one-step e n-step ahead com qualquer função base.
    - `_one_step_ahead_prediction`: previsão 1 passo para qualquer função base.
    - `_n_step_ahead_prediction`: previsão n-passos para função polinomial.
    - `_model_prediction`: previsão infinity-step para função polinomial.
    - `_narmax_predict`: wrapper para modelos NARMAX e NAR.
    - `_nfir_predict`: wrapper para modelos NFIR.
    - `_basis_function_predict`: previsão infinity-step para funções base diferentes de polinomial.
    - `basis_function_n_step_prediction`: previsão n-passos para funções base diferentes de polinomial.

- Mudança de API (nova): `sysidentpy.neural_network.ModelPrediction`
    - Classe adaptada para suportar Neural NARX como classe independente.
    - Métodos equivalentes aos descritos acima para General Estimators.

- Mudança de API: método `fit` do Neural NARX reformulado.
    - Não é mais necessário converter os dados para tensor antes de chamar `fit`.

Mudança de API: argumentos posicionais vs nomeados
    - Agora todos os parâmetros devem ser passados por nome (keyword arguments) em todas as classes de modelo.

- Mudança de API (nova): `sysidentpy.utils.narmax_tools`
    - Funções auxiliares para obter informações úteis ao construir modelos. Inclui `regressor_code` para ajudar a montar Neural NARX.

#### Documentação
- DOC: Notebook de Passos Básicos aprimorado com novos detalhes sobre a função de previsão.
- DOC: Notebook de NARX Neural Network atualizado com a nova API e recursos.
- DOC: Notebook de General Estimators atualizado com a nova API e recursos.
- DOC: Correções gramaticais menores, incluindo Issues #77 e #78.
- DOC: Correções em HTML dos notebooks na documentação.


## v0.1.9

### COLABORADORES

- wilsonrljr
- samirmartins

### ALTERAÇÕES

- A versão **v0.1.9** foi lançada com novos recursos, pequenas mudanças de API e correções das novidades da v0.1.7.

#### Alterações na API
- MAJOR: Algoritmo de Regressão Entrópica
    - Nova classe ER para construir modelos NARX usando o algoritmo de regressão entrópica.
    - Apenas Mutual Information KNN implementado nesta versão; pode levar bastante tempo com muitos regressores, portanto atenção ao número de candidatos.
- API: `save_load`
    - Adicionada função para salvar/carregar modelos em arquivo.
- API: Adicionados testes para Python 3.9.
- Fix: alteração na condição `n_info_values` do FROLS. Agora o valor definido pelo usuário é comparado contra o formato da matriz X em vez do espaço de regressores, corrigindo o uso de Fourier com mais de 15 regressores no FROLS.

#### Documentação
- DOC: Salvar e Carregar modelos
    - Adicionado notebook mostrando como usar `save_load`.
- DOC: Exemplo de Regressão Entrópica
    - Notebook simples mostrando o uso de AOLS.
- DOC: Exemplo de Função Base Fourier
    - Notebook mostrando o uso da função Fourier.
- DOC: Benchmark de previsão PV
    - Correção da previsão do AOLS (o exemplo usava `meta_mss`).
- DOC: Correções gramaticais menores.
- DOC: Correções em HTML nos notebooks da documentação.

## v0.1.8

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.1.8** foi lançada com novos recursos e pequenas mudanças de API corrigindo as novidades da v0.1.7.

#### Alterações na API
- MAJOR: Funções Base em Ensemble
    - Agora é possível usar diferentes funções base juntas. Por ora, Fourier pode ser combinada com Polynomial de graus distintos.
- Mudança de API: parâmetro `ensemble` adicionado às funções base para combinar recursos de diferentes funções.
- Fix: previsão n-passos para `model_type="NAR"` funciona corretamente com diferentes horizontes.

#### Documentação
- DOC: Benchmark de passageiros aéreos
    - Removido código não utilizado.
    - Uso dos hiperparâmetros padrão nos modelos SysIdentPy.
- DOC: Benchmark de previsão de carga
    - Removido código não utilizado.
    - Uso de hiperparâmetros padrão.
- DOC: Benchmark de previsão PV
    - Removido código não utilizado.
    - Uso de hiperparâmetros padrão.

## v0.1.7

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

- A versão **v0.1.7** foi lançada com grandes mudanças e novos recursos. Há diversas modificações na API e será necessário ajustar o código para aproveitar os novos (e futuros) recursos. Todas as mudanças foram feitas para facilitar expansões futuras.
- Do ponto de vista do usuário, as mudanças não são tão disruptivas, mas internamente houve muitas alterações que permitiram novos recursos e correções que seriam difíceis sem isso. Consulte a `página de documentação <http://sysidentpy.org/notebooks.html>`__.
- Muitas classes foram praticamente reescritas, então recomenda-se olhar os novos exemplos de uso.
- A seguir, os principais destaques e depois todas as mudanças de API.

#### Alterações na API
- MAJOR: Modelos NARX com função base Fourier `Issue63 <https://github.com/wilsonrljr/sysidentpy/issues/63>`__, `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - É possível escolher a função base importando de `sysidentpy.basis_function`. Veja os notebooks de exemplo.
    - Polynomial e Fourier são suportadas inicialmente. Novas funções virão nas próximas versões.
- MAJOR: Modelos NAR `Issue58 <https://github.com/wilsonrljr/sysidentpy/issues/58>`__
    - Já era possível construir NAR polinomiais com alguns hacks. Agora basta definir `model_type="NAR"`.
    - Não é mais necessário passar vetor de zeros como entrada.
    - Funciona com qualquer algoritmo de seleção de estrutura (FROLS, AOLS, MetaMSS).
- Major: Modelos NFIR `Issue59 <https://github.com/wilsonrljr/sysidentpy/issues/59>`__
    - Modelos onde a saída depende apenas das entradas passadas. Antes exigia muito código manual; agora basta `model_type="NFIR"`.
    - Funciona com qualquer algoritmo de seleção de estrutura.
- Major: Selecionar ordem dos lags de resíduos no Extended Least Squares (elag)
    - Usuários podem selecionar o lag máximo dos resíduos usados no algoritmo, seguindo o grau da função base.
- Major: Métodos de análise de resíduos `Issue60 <https://github.com/wilsonrljr/sysidentpy/issues/60>`__
    - Funções específicas agora calculam autocorrelação dos resíduos e correlação cruzada, superando limitações anteriores.
- Major: Métodos de plotagem `Issue61 <https://github.com/wilsonrljr/sysidentpy/issues/61>`__
    - Funções de plot foram separadas dos objetos de modelo, oferecendo mais flexibilidade.
    - Gráficos de resíduos separados do gráfico de previsão.
- Mudança de API: `sysidentpy.polynomial_basis.PolynomialNarmax` está deprecated. Use `sysidentpy.model_structure_selection.FROLS` `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/62>`__
    - Não é mais necessário informar o número de entradas.
    - Parâmetro `elag` adicionado ao estimador não enviesado. Agora dá para definir lags dos resíduos no Extended Least Squares.
    - Parâmetro `model_type` permite escolher entre "NARMAX", "NAR" e "NFIR" (padrão "NARMAX").
- Mudança de API: `sysidentpy.polynomial_basis.MetaMSS` está deprecated. Use `sysidentpy.model_structure_selection.MetaMSS` `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Não é necessário informar número de entradas.
    - `elag` adicionado para estimativa não enviesada.
- Mudança de API: `sysidentpy.polynomial_basis.AOLS` está deprecated. Use `sysidentpy.model_structure_selection.AOLS` `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
- Mudança de API: `sysidentpy.polynomial_basis.SimulatePolynomialNarmax` está deprecated. Use `sysidentpy.simulation.SimulateNARMAX`.
- Mudança de API: introdução de `sysidentpy.basis_function`. Como modelos NARMAX podem usar bases diferentes, um módulo novo facilita a implementação de futuras funções `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__.
    - Cada função base deve possuir métodos `fit` e `predict` para treino e previsão.
- Mudança de API: método `unbiased_estimator` movido para Estimators.
    - Adicionado parâmetro `elag`.
    - `build_information_matrix` renomeado para `build_output_matrix`.
- Mudança de API (nova): `sysidentpy.narmax_base`
    - Nova base para construção de modelos NARMAX. Classes reescritas para facilitar expansões.
- Mudança de API (nova): `sysidentpy.narmax_base.GenerateRegressors`
    - `create_narmax_code`: cria a codificação base para representar modelos NARMAX, NAR e NFIR.
    - `regressor_space`: cria a representação codificada.
- Mudança de API (nova): `sysidentpy.narmax_base.ModelInformation`
    - `_get_index_from_regressor_code`: obtém índice do código do modelo no espaço de regressores.
    - `_list_output_regressor_code`: cria array flatten de regressores de saída.
    - `_list_input_regressor_code`: idem para entradas.
    - `_get_lag_from_regressor_code`: obtém lag máximo de um array de regressores.
    - `_get_max_lag_from_model_code`: idem para um código de modelo.
    - `_get_max_lag`: obtém lag máximo de ylag e xlag.
- Mudança de API (nova): `sysidentpy.narmax_base.InformationMatrix`
    - `_create_lagged_X`: cria matriz defasada de entradas sem combinações.
    - `_create_lagged_y`: cria matriz defasada da saída sem combinações.
    - `build_output_matrix`: constrói a matriz de informação de valores de saída.
    - `build_input_matrix`: constrói a matriz de informação de entradas.
    - `build_input_output_matrix`: matriz de informação de entrada e saída.
- Mudança de API (nova): `sysidentpy.narmax_base.ModelPrediction`
    - `predict`: método base de previsão. Suporta infinity, one-step e n-step para qualquer função base.
    - `_one_step_ahead_prediction`: previsão 1 passo para qualquer função base.
    - `_n_step_ahead_prediction`: previsão n-passos para base polinomial.
    - `_model_prediction`: previsão infinity-step para base polinomial.
    - `_narmax_predict`: wrapper para modelos NARMAX e NAR.
    - `_nfir_predict`: wrapper para NFIR.
    - `_basis_function_predict`: previsão infinity-step para bases não polinomiais.
    - `basis_function_n_step_prediction`: previsão n-passos para bases não polinomiais.
- Mudança de API (nova): `sysidentpy.model_structure_selection.FROLS` `Issue62 <https://github.com/wilsonrljr/sysidentpy/issues/62>`__, `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Baseada na antiga `PolynomialNARMAX`. A classe foi reconstruída com novas funções e código otimizado.
    - Argumentos apenas nomeados, promovendo uso claro.
    - Suporte a novas funções base.
    - Possibilidade de escolher lags residuais.
    - Não é necessário informar número de entradas.
    - Docstring aprimorada.
    - Correções gramaticais menores.
    - Novo método de previsão.
    - Muitas mudanças internas.
- Mudança de API (nova): `sysidentpy.model_structure_selection.MetaMSS` `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Baseada na antiga `Polynomial_basis.MetaMSS`. Reconstruída com novas funções e código otimizado.
    - Argumentos apenas nomeados.
    - Possibilidade de escolher lags residuais.
    - Suporte a Extended Least Squares.
    - Suporte a novas bases.
    - Não é necessário informar número de entradas.
    - Docstring aprimorada.
    - Correções gramaticais menores.
    - Novo método de previsão.
    - Muitas mudanças internas.
- Mudança de API (nova): `sysidentpy.model_structure_selection.AOLS` `Issue64 <https://github.com/wilsonrljr/sysidentpy/issues/64>`__
    - Baseada na antiga `AOLS`. Reconstruída com novas funções e código otimizado.
    - Argumentos apenas nomeados.
    - Suporte a novas bases.
    - Não é necessário informar número de entradas.
    - Docstring aprimorada.
    - Parâmetro "l" renomeado para "L".
    - Correções gramaticais menores.
    - Novo método de previsão.
    - Muitas mudanças internas.
- Mudança de API (nova): `sysidentpy.simulation.SimulateNARMAX`
    - Baseada na antiga `SimulatePolynomialNarmax`. Reconstruída com novas funções e código otimizado.
    - Correção do suporte ao Extended Least Squares.
    - Correção da previsão n-passos e 1-passo à frente.
    - Argumentos apenas nomeados.
    - Possibilidade de escolher lags residuais.
    - Docstring aprimorada.
    - Correções gramaticais menores.
    - Novo método de previsão.
    - Não herda mais do algoritmo de seleção, apenas de `narmax_base`, evitando importações circulares.
    - Muitas mudanças internas.
- Mudança de API (nova): `sysidentpy.residues`
    - `compute_residues_autocorrelation`: calcula autocorrelação dos resíduos.
    - `calculate_residues`: obtém resíduos a partir de y e yhat.
    - `get_unnormalized_e_acf`: autocorrelação não normalizada dos resíduos.
    - `compute_cross_correlation`: correlação cruzada entre duas séries.
    - `_input_ccf`
    - `_normalized_correlation`: correlação normalizada entre dois sinais.
- Mudança de API (nova): `sysidentpy.utils.plotting`
    - `plot_results`: plota previsão.
    - `plot_residues_correlation`: autocorrelação/correlação cruzada.
- Mudança de API (nova): `sysidentpy.utils.display_results`
    - `results`: retorna regressores do modelo, parâmetros estimados e índice ERR em uma tabela.

#### Documentação
- DOC: Benchmark de passageiros aéreos `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Notebook adicionado comparando SysIdentPy a prophet, neuralprophet, autoarima, tbats e outros.
- DOC: Benchmark de previsão de carga `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Notebook adicionado.
- DOC: Benchmark de previsão PV `Issue65 <https://github.com/wilsonrljr/sysidentpy/issues/65>`__
    - Notebook adicionado.
- DOC: Apresentação das funcionalidades principais
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Uso com múltiplas entradas
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Critérios de Informação — Exemplos
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Notas importantes e exemplos de como usar Extended Least Squares
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Definindo lags específicos
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Estimação de Parâmetros
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Uso do MetaMSS para construir modelos NARX polinomiais
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Uso do AOLS para construir modelos NARX polinomiais
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Exemplo: benchmark de vibração do F-16
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Construindo Neural NARX com o SysIdentPy
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Construindo modelos NARX usando estimadores gerais
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Simular um modelo predefinido
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Identificação de sistemas usando filtros adaptativos
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Identificação de um sistema eletromecânico
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Exemplo: previsão n-passos — benchmark F-16
    - Exemplo reescrito conforme a nova API.
    - Correções gramaticais menores.
- DOC: Introdução aos modelos NARMAX
    - Correções gramaticais e ortográficas.


## v0.1.6

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

#### Alterações na API
- MAJOR: Algoritmo Meta-Model Structure Selection (Meta-MSS).
    - Novo método para construir modelos NARMAX com base em metaheurísticas. O algoritmo usa um híbrido binário PSO/GSA com nova função de custo para gerar modelos parcimoniosos.
    - Nova classe para o algoritmo BPSOGSA. Outros algoritmos podem ser adaptados ao framework Meta-MSS.
	- Atualizações futuras incluirão modelos NARX para classificação e seleção multiobjetivo.
- MAJOR: Algoritmo Accelerated Orthogonal Least-Squares.
    - Nova classe AOLS para construir modelos NARX usando o algoritmo AOLS.
    - Pelo que se sabe, é a primeira aplicação do algoritmo no framework NARMAX. Os testes preliminares são promissores, mas recomenda-se cautela até formalização em artigo.

#### Documentação
- Notebook adicionado com exemplo simples do MetaMSS e comparação no sistema eletromecânico.
- Notebook adicionado com exemplo simples do AOLS.
- Adicionada classe ModelInformation. Ela possui métodos para retornar informações do modelo, como `max_lag`.
    - `_list_output_regressor_code`
    - `_list_input_regressor_code`
    - `_get_lag_from_regressor_code`
    - `_get_max_lag_from_model_code`
- Pequena melhoria de performance: argumento "predefined_regressors" adicionado em `build_information_matrix` no `base.py` para acelerar o método de simulação.
- Pytorch agora é dependência opcional. Use `pip install sysidentpy['full']`.
- Correções de formatação de código.
- Correções gramaticais menores.
- Correções em HTML nos notebooks da documentação.
- README atualizado com exemplos.
- Descrições e comentários aprimorados.
- `metaheuristics.bpsogsa` (detalhes nas docstrings)
    - `evaluate_objective_function`
    - `optimize`
    - `generate_random_population`
    - `mass_calculation`
    - `calculate_gravitational_constant`
    - `calculate_acceleration`
    - `update_velocity_position`
- FIX issue #52


## v0.1.5

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

#### Alterações na API
- MAJOR: Previsão n-passos à frente.
    - Agora é possível definir o número de passos à frente no método `predict`.
	- Disponível para modelos polinomiais por enquanto. Próxima atualização trará o recurso para Neural NARX e General Estimators.
- MAJOR: Simulação de modelos predefinidos.
    - Nova classe `SimulatePolynomialNarmax` para simular estruturas conhecidas.
    - Agora é possível simular modelos predefinidos apenas passando a codificação da estrutura. Veja os notebooks.
- Correções de formatação de código.
- Novos testes para `SimulatePolynomialNarmax` e `generate_data`.
- Iniciadas mudanças relacionadas ao numpy 1.19.4. Ainda restam alguns avisos de depreciação a serem corrigidos.

#### Documentação
- Adicionados 4 novos notebooks na seção de exemplos.
- Adicionados notebooks iterativos. Agora é possível rodá-los no Colab direto da documentação.
- Correções em HTML nos notebooks da documentação.
- README atualizado com exemplos.

## v0.1.4

### COLABORADORES

- wilsonrljr

### ALTERAÇÕES

#### Alterações na API
- MAJOR: Introdução da NARX Neural Network no SysIdentPy.
    - Agora é possível construir redes Neural NARX no SysIdentPy.
    - O recurso é baseado em PyTorch. Veja os docs para detalhes e exemplos.
- MAJOR: Introdução de estimadores gerais no SysIdentPy.
    - Agora é possível usar qualquer estimador com métodos Fit/Predict (Sklearn, CatBoost etc.) para construir modelos NARX.
    - Aproveitamos as funções centrais do SysIdentPy mantendo a interface Fit/Predict para facilitar o uso.
    - Mais estimadores virão em breve, como XGBoost.
- Parâmetros padrão da função `plot_results` foram alterados.

#### Documentação
- Adicionados notebooks mostrando como construir Neural NARX.
- Adicionados notebooks mostrando como construir modelos NARX com estimadores gerais.
- Novo template para o site de documentação.
- Correções em HTML nos notebooks da documentação.
- README atualizado com exemplos.

- NOTA: Continuaremos aprimorando os modelos NARX polinomiais (novos algoritmos de seleção e identificação multiobjetivo estão no roadmap). As modificações recentes permitem introduzir novos modelos como PWARX em breve.

## v0.1.3

### COLABORADORES

- wilsonrljr
- renard162

### ALTERAÇÕES

#### Alterações na API
- Correção de bug relacionado a `xlag` e `ylag` em cenários com múltiplas entradas.
- Função `predict` refatorada. Desempenho melhorado em até 87% conforme o número de regressores.
- Agora é possível definir lags de tamanhos diferentes para cada entrada.
- Adicionada função para obter o valor máximo de `xlag` e `ylag`. Funciona com int, lista e listas aninhadas.
- Correção de testes para critérios de informação.
- Código de todas as classes refatorado seguindo PEP8 para melhorar legibilidade.
- Ajustes nos testes de critérios de informação.
- Adicionado workflow para rodar testes ao fazer merge na master.

#### Documentação
- Adicionado o logo do SysIdentPy.
- Inserida informação de citação no README.
- Novo domínio do site adicionado.
- Documentação atualizada.
