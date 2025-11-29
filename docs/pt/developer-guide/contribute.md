---
template: overrides/main.html
title: Contribuir
---

# Contribuindo

O SysIdentPy é um projeto comunitário, portanto todas as contribuições são bem-vindas! Existem muitos casos de uso possíveis na área de Identificação de Sistemas e não podemos testar todos os cenários sem a sua ajuda! Se você encontrar algum bug ou tiver sugestões, por favor reporte-os no [issue tracker] no GitHub.

> Recebemos novos contribuidores de todos os níveis de experiência. Os objetivos da comunidade SysIdentPy são ser prestativa, acolhedora e eficaz.

  [issue tracker]: https://github.com/wilsonrljr/sysidentpy/issues

## Ajude outros com issues no GitHub

Você pode ver as <a href="https://github.com/wilsonrljr/sysidentpy/issues" class="external-link" target="_blank">issues existentes</a> e tentar ajudar outros, na maioria das vezes são perguntas para as quais você já pode saber a resposta.

## Acompanhe o repositório do GitHub

Você pode [acompanhar] o SysIdentPy no GitHub (clicando no botão "watch" no canto superior direito):
  [acompanhar]: https://github.com/wilsonrljr/sysidentpy

Se você selecionar "Watching" em vez de "Releases only", receberá notificações quando alguém criar uma nova issue.

Assim você pode tentar ajudá-los a resolver essas issues.

## Documentação

A documentação é tão importante quanto a própria biblioteca. O inglês não é a língua principal dos autores, então se você encontrar algum erro de digitação ou algo errado, não hesite em nos avisar.

## Criar um Pull Request

Você pode [contribuir](contribute.md){.internal-link target=_blank} com o código-fonte através de Pull Requests, por exemplo:

* Para corrigir um erro de digitação que você encontrou na documentação.
* Para compartilhar um artigo, vídeo ou podcast que você criou ou encontrou sobre o SysIdentPy.
* Para propor novas seções de documentação.
* Para corrigir uma issue/bug existente.
* Para adicionar um novo recurso.

## Ambiente de desenvolvimento

Estes são alguns passos básicos para nos ajudar com o código:

- [x] Instalar e configurar o Git no seu computador.
- [x] [Fork] o SysIdentPy.
- [x] [Clone] o fork na sua máquina local.
- [x] Criar uma nova branch.
- [x] Fazer alterações seguindo o estilo de codificação do projeto (ou sugerindo melhorias).
- [x] Executar os testes.
- [x] Escrever e/ou adaptar testes existentes se necessário.
- [x] Adicionar documentação se necessário.
- [x] Commit.
- [x] [Push] para o seu fork.
- [x] Abrir um [pull_request].

  [Fork]: https://help.github.com/articles/fork-a-repo/
  [Clone]: https://help.github.com/articles/cloning-a-repository/
  [Push]: https://help.github.com/articles/pushing-to-a-remote/
  [pull_request]: https://help.github.com/articles/creating-a-pull-request/


### Ambiente

Clone o repositório usando

```console
git clone https://github.com/wilsonrljr/sysidentpy.git
```

Se você já clonou o repositório e sabe que precisa mergulhar fundo no código, aqui estão algumas diretrizes para configurar seu ambiente.

#### Ambiente virtual com `venv`

Você pode criar um ambiente virtual em um diretório usando o módulo `venv` do Python ou Conda:

=== "venv"

    ```console
    $ python -m venv env
    ```

=== "conda"

    ```console
    conda create -n env
    ```


Isso criará um diretório `./env/` com os binários do Python e então você poderá instalar pacotes para esse ambiente isolado.

#### Ativar o ambiente

Se você criou o ambiente usando o módulo `venv` do Python, ative-o com:

=== "Linux, macOS"

    ```console
    source ./env/bin/activate
    ```

=== "Windows PowerShell"

    ```console
    .\env\Scripts\Activate.ps1
    ```

=== "Windows Bash"

    Ou se você usa Bash no Windows (ex: <a href="https://gitforwindows.org/" class="external-link" target="_blank">Git Bash</a>):

    ```console
    source ./env/Scripts/activate
    ```

Se você criou o ambiente usando Conda, ative-o com:

=== "Conda Bash"

    ```console
    conda activate env
    ```

Para verificar se funcionou, use:

=== "Linux, macOS, Windows Bash"

    ```console
    $ which pip

    some/directory/sysidentpy/env/Scripts/pip
    ```

=== "Windows PowerShell"

    ```console
    $ Get-Command pip

    some/directory/sysidentpy/env/Scripts/pip
    ```

Se mostrar o binário `pip` em `env/bin/pip`, então funcionou.



!!! tip
    Toda vez que você instalar um novo pacote com `pip` nesse ambiente, ative o ambiente novamente.



!!! note
    Usamos o pacote `pytest` para testes. As funções de teste estão localizadas em subdiretórios de testes em cada pasta dentro do SysIdentPy, que verificam a validade dos algoritmos.

#### Dependências

Instale o SysIdentPy com as opções `dev` e `docs` para obter todas as dependências necessárias para executar os testes

=== "Dependências Dev e Docs"

    ``` sh
    pip install "sysidentpy[dev, docs]"
    ```

## Documentação

Primeiro, certifique-se de configurar seu ambiente conforme descrito acima, isso instalará todos os requisitos.

A documentação usa <a href="https://www.mkdocs.org/" class="external-link" target="_blank">MkDocs</a> e <a href="https://squidfunk.github.io/mkdocs-material/" class="external-link" target="_blank">Material for MKDocs</a>.

Toda a documentação está em formato Markdown no diretório `./docs/`.

### Verificar as alterações

Durante o desenvolvimento local, você pode servir o site localmente e verificar quaisquer alterações. Isso ajuda a garantir que:

* Todas as suas modificações foram aplicadas.
* Os arquivos não modificados estão sendo exibidos conforme esperado.


```console
$ mkdocs serve

INFO     -  [13:25:00] Browser connected: http://127.0.0.1:8000
```

Isso servirá a documentação em `http://127.0.0.1:8008`.

Dessa forma, você pode continuar editando os arquivos fonte e ver as alterações ao vivo.

!!! warning
  Se alguma modificação quebrar o build, você terá que servir o site novamente. Sempre verifique seu `console` para garantir que está servindo o site.


## Executar testes localmente

É sempre bom verificar se suas implementações/modificações não quebram nenhuma outra parte do pacote. Você pode executar os testes do SysIdentPy localmente usando `pytest` na respectiva pasta para realizar todos os testes dos sub-pacotes correspondentes.

#### Exemplo de como executar os testes:

Abra um emulador de terminal de sua escolha e vá para o diretório principal, ex:

	\sysidentpy\

Basta digitar `pytest` no emulador de terminal

```console
pytest
```

e você obtém um resultado como:

```console
========== test session starts ==========

platform linux -- Python 3.7.6, pytest-5.4.2, py-1.8.1, pluggy-0.13.1

rootdir: ~/sysidentpy

plugins: cov-2.8.1

collected 12 items

tests/test_regression.py ............ [100%]

========== 12 passed in 2.45s ==================
```
