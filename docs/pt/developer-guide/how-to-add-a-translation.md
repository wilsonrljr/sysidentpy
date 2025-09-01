---
template: overrides/main.html
title: Como Adicionar uma Tradução
---

# Como Adicionar uma Tradução

Guia para criar ou melhorar traduções da documentação em **qualquer** idioma.

Usamos MkDocs + Material + `mkdocs-static-i18n`. Inglês é o fallback. Qualquer idioma novo replica a estrutura de pastas. Se faltar uma página traduzida, aparece a versão inglesa.

---
## 1. Visão geral

Três cenários comuns:

1. Melhorar uma página já traduzida.
2. Traduzir uma página que só existe em inglês.
3. Adicionar um idioma totalmente novo.

Tudo abaixo cobre esses casos.

---
## 2. Estrutura de pastas

```
/docs
  en/
    developer-guide/
    getting-started/
    user-guide/
  <locale>/
    (mesmos caminhos relativos)
```

`<locale>` exemplos: `pt`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`. Use códigos curtos (BCP‑47). Evite variantes regionais salvo necessidade (`pt-BR`, `pt-PT`).

Os caminhos relativos devem ser idênticos:

```
Inglês: docs/en/developer-guide/how-to-add-a-translation.md
Espanhol: docs/es/developer-guide/how-to-add-a-translation.md
Francês:  docs/fr/developer-guide/how-to-add-a-translation.md
```

---
## 3. Início rápido (traduzir ou melhorar)

1. Fork e clone.
2. Crie / ative ambiente virtual.
3. Instale extras de docs:
    ```console
    pip install -e ".[docs]"
    ```
4. Servidor local:
    ```console
    mkdocs serve
    ```
5. Abra a URL e use o seletor de idioma.

Reinicie se arquivos novos não aparecerem.

---
## 4. Adicionando um novo idioma (setup inicial)

Se a pasta já existe (ex: `pt/`), pule.

1. Escolha código (ex: `es`).
2. Crie `docs/es/`.
3. Copie `docs/en/index.md` para `docs/es/index.md` e traduza.
4. (Opcional) Comece só com index + páginas principais para PR menor.
5. Edite `mkdocs.yml` em `i18n.languages`:
    ```yaml
    - locale: es
      name: Español
      build: true
      site_description: <slogan traduzido>
      theme:
        docs_dir: docs/es/
        custom_dir: docs/es/
        site_dir: site/es/
        logo: overrides/assets/img/logotype-sysidentpy.svg
    ```
6. Não duplique a navegação; o plugin mapeia automaticamente.
7. Rode `mkdocs serve` e confirme o idioma no seletor.

Para variantes regionais (ex: `pt-BR`) mantenha consistência no nome da pasta e no `locale`.

---
## 5. Nova página em inglês (fonte)

1. Crie em `docs/en/...`.
2. Front matter:
    ```md
    ---
    template: overrides/main.html
    title: Título
    ---
    ```
3. Adicione no `nav` do `mkdocs.yml` (apenas uma vez).
4. Verifique build.
5. (Opcional) Comentário para tradutores:
    ```md
    <!-- Nota para tradução: manter "NARMAX" em inglês. -->
    ```

---
## 6. Criando o arquivo traduzido

1. Caminho espelhado: `docs/<locale>/<mesmo>.md`.

2. Copie o original.

3. Traduza só texto natural. Preserve:
    - Blocos de código (comentários apenas se ajudar)
    - Identificadores (funções, classes, imports)
    - Caminhos, chaves de config, URLs
    - Alvos de links relativos

4. Mantenha hierarquia de títulos.

5. Preserve tipos de admonitions (`!!! note`, etc.). Título interno pode ser traduzido.

6. Parte pendente? Use:
   ```md
   !!! note "Tradução pendente"
       Este parágrafo ainda será traduzido.
   ```

7. Remova notas pendentes antes de finalizar (se concluir).

---
## 7. Links internos

Use links relativos:
```md
Veja o [guia de contribuição](contribute.md).
```
O plugin resolve por idioma. Evite hardcode `/en/` ou outro prefixo.

Âncoras: se traduzir título, o slug muda; ajuste referências `(#ancora)`.

---
## 8. Imagens e mídia

Se a imagem contém texto:

- **Opção A**: localizar imagem dentro de `docs/<locale>/assets/` com mesmo nome.
- **Opção B**: reutilizar imagem inglesa se o texto não atrapalha.

SVG: manter símbolos ou termos técnicos; traduzir rótulos descritivos.

---
## 9. Formatação & estilo

| Aspecto | Regra |
|---------|-------|
| Números | Mantenha precisão; separador decimal local é opcional. |
| Unidades | Não traduzir (ms, Hz, etc.). |
| APIs | Nunca traduzir identificadores. |
| Aspas | Use padrão local sem quebrar Markdown. |
| Capitalização | Igual só para nomes próprios / APIs. |
| Tom | Neutro, direto. |

Evite blocos não revisados de tradução automática. Prefira frases curtas.

---
## 10. Glossário de tradução

Mantenha estes termos consistentes. Adicione equivalentes para outros idiomas conforme necessário:

| Termo em Inglês | Português (pt) |
|------------------|----------------|
| **Conceitos centrais** |
| model structure | estrutura do modelo |
| parameter estimation | estimação de parâmetros |
| residual analysis | análise dos resíduos |
| time series | série temporal |
| identification | identificação |
| **Termos técnicos** |
| basis function | função de base |
| regression | regressão |
| algorithm | algoritmo |
| validation | validação |
| simulation | simulação |
| **Termos de desenvolvimento** |
| feature | funcionalidade |
| pull request (PR) | pull request (PR) |
| branch | branch |
| commit | commit |
| documentation | documentação |

Para outros idiomas, siga padrões similares. Prefira clareza à tradução literal.

---
## 11. Checklist de revisão (arquivo traduzido)

- [ ] Build local OK.
- [ ] Caminho espelhado correto.
- [ ] Links relativos sem `/en/` fixo.
- [ ] Blocos de código intactos (comentários revisados).
- [ ] Terminologia consistente.
- [ ] Sem notas pendentes (ou marcadas claramente se parcial).
- [ ] Front matter com `title:` traduzido.

---
## 12. Commit & PR

Inclua arquivo inglês + traduzido se a página é nova; caso contrário só o traduzido.

Exemplo:
```console
git add docs/en/developer-guide/new-topic.md docs/es/developer-guide/new-topic.md
git commit -m "docs: adicionar tradução em espanhol de new-topic"
```

Template de descrição de PR:

- **Idioma**: `<locale>`
- **Páginas**: lista
- **Trechos ainda em inglês**: (se houver)
- **Termos novos de glossário**: (se houver)
- **Notas para revisão**: contexto, termos difíceis

Traduções parciais são aceitáveis — marque claramente.

---
## 13. Atualizando traduções

Quando o inglês mudar:

1. Veja o diff.
2. Aplique mudanças equivalentes.
3. Sem tempo para traduzir? Deixe em inglês + nota temporária.
4. Remova a nota ao finalizar.

Prefira PRs menores.

---
## 14. Problemas comuns

| Sintoma | Causa | Correção |
|---------|-------|----------|
| Página só em inglês | Falta arquivo no locale | Criar arquivo espelhado |
| Erro de build | Entrada `i18n` incorreta | Corrigir `locale` / indentação |
| Link 404 | Caminho diferente do inglês | Sincronizar caminho |
| Âncora quebrada | Título mudou | Ajustar slug / título |
| Idioma não aparece | Faltou adicionar em `mkdocs.yml` | Adicionar e reiniciar |

---
## 15. Automação (opcional)

Scripts podem copiar estrutura base, mas revise manualmente termos técnicos. Não sobrescreva traduções existentes.

---
## 16. Dúvidas

Abra Issue ou Discussion para confirmar termos antes de traduzir grandes trechos. Feedback cedo evita retrabalho.

Obrigado por tornar a documentação acessível a mais pessoas.
