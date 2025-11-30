# Proposta de Reestruturação da Documentação do SysIdentPy

Este documento descreve uma reorganização da documentação do SysIdentPy para melhorar a descoberta de conteúdo, reduzir a fricção para iniciantes e alinhar com padrões modernos de documentação. A estrutura seguirá quatro categorias principais: **Tutoriais**, **How-Tos**, **Explicações** e **Referência da API**, com seções adicionais para contribuidores e exemplos do mundo real.

> Agradecimentos: Esta reestruturação da documentação se inspira no [NEP 44 do NumPy](https://numpy.org/neps/nep-0044-restructuring-numpy-docs.html#nep44), adaptando seus princípios de clareza e organização lógica às necessidades específicas do SysIdentPy no domínio de identificação de sistemas e previsão de séries temporais, enquanto enfatiza tutoriais e reprodutibilidade.

## Motivação e Escopo

A documentação atual do SysIdentPy (como muitos pacotes Python científicos) mistura explicações conceituais, exemplos de código e referências de API, o que pode sobrecarregar novos usuários. Ao adotar uma estrutura centrada no usuário inspirada no [Diátaxis](https://diataxis.fr/), pretendemos:

- Separar caminhos de aprendizado para **iniciantes** (Tutoriais) e **praticantes** (How-Tos).
- Melhorar o material para entendimento conceitual (Explicações).
- Manter um **Guia de Referência** limpo e pesquisável.
- Destacar os recursos do SysIdentPy.

Uma estrutura de documentação bem organizada pode melhorar significativamente a experiência da nossa comunidade, fornecendo recursos específicos para diferentes grupos de usuários:

- Para Iniciantes: Um caminho claro e guiado com tutoriais e instruções passo a passo ajuda novos usuários a superar a curva de aprendizado.

- Para Pesquisadores: Recursos como funções base personalizadas e configurações de modelo podem ser facilmente descobertos e compreendidos. Com seções claramente definidas, pesquisadores podem localizar rapidamente as informações necessárias para experimentar novos métodos.

- Para Usuários Corporativos/Industriais: Guias de benchmarking e exemplos de comparação de modelos são facilmente acessíveis, facilitando para profissionais da indústria avaliar e escolher as ferramentas certas para suas necessidades específicas.

O objetivo é estruturar a documentação para atender às necessidades específicas desses diversos grupos de usuários, tornando o processo de aprendizado mais rápido e eficiente para todos na comunidade.

## Estrutura Proposta

Aqui está uma visão geral das principais seções da documentação, descrevendo o propósito e o conteúdo proposto para cada uma:

- Primeiros Passos
- Guia do Usuário
- Guia do Desenvolvedor
- Comunidade & Suporte
- Sobre

### Guia do Usuário

A seção Guia do Usuário é projetada para fornecer uma compreensão abrangente do SysIdentPy, cobrindo conceitos essenciais, exemplos práticos e recursos avançados. A estrutura proposta inclui:


#### Tutoriais

Público: Novos usuários com experiência mínima em identificação de sistemas.

Conteúdo Sugerido:

<div class="grid cards" markdown>
- :material-book-open-variant: __Guia do Iniciante__
  Comece do zero com guias fáceis de seguir projetados para aqueles novos no SysIdentPy e modelos NARMAX.
- :material-application-cog: __Tutoriais Específicos por Domínio__
  Exemplos e casos de uso para áreas como engenharia, saúde, finanças e outras.
</div>

Formato: Jupyter Notebooks com explicações narrativas e código.


#### How-Tos

Público: Praticantes resolvendo problemas específicos.

Conteúdo Sugerido:

<div class="grid cards" markdown>
- :material-tune: __Otimização de Modelos__
- :material-rocket-launch: __Customizações Avançadas__
- :material-chart-box: __Análise de Erros__
- :material-repeat: __Reprodutibilidade__
</div>

Formato: Arquivos markdown curtos e focados em tarefas com snippets de código.

#### Explicações

Público: Usuários buscando fundamentos matemáticos rigorosos.

<div class="grid cards" markdown>
- :material-book-open-page-variant: __Livro__
  [Nonlinear System Identification and Forecasting: Theory and Practice with SysIdentPy](https://sysidentpy.org/book/0%20-%20Preface/). Oferece contexto teórico para os algoritmos do SysIdentPy através de um livro.
</div>

#### Referência da API

Público: Usuários avançados precisando de detalhes da API.

<div class="grid cards" markdown>
- :material-code-tags: __Referência da API__
  Acesse o código-fonte completo do **SysIdentPy** com módulos e métodos bem documentados.
</div>

Formato: Documentação de API gerada automaticamente com seções "Veja Também" com links cruzados.

### Guia do Desenvolvedor

A seção Guia do Desenvolvedor visa fornecer informações claras sobre a estrutura interna do SysIdentPy, focando em detalhes de implementação, exemplos de código e opções de customização. A estrutura proposta inclui:

#### Como Contribuir

Público: Mantenedores e contribuidores de código aberto.

<div class="grid cards" markdown>
- :material-account-plus: __Guia do Contribuidor__
</div>

#### Guia de Documentação

Público: Mantenedores e contribuidores de código aberto.


<div class="grid cards" markdown>
- :material-book-edit: __Escrevendo um tutorial__
- :material-book-edit: __Criando um guia how-to__
- :material-book-edit: __Criando conteúdo para o livro__
</div>

### Comunidade & Suporte

Público: Indivíduos de todos os níveis de experiência, de iniciantes a especialistas, com interesse em Python e SysIdentPy.

<div class="grid cards" markdown>
- :material-lifebuoy: __Obter Ajuda__
- :material-video: __Workshops__
- :material-book-open-page-variant: __Sugestões de Leitura__
- :material-forum: __Discussões da Comunidade__
</div>
