---
template: overrides/main.html
title: How to Add a Translation
---

# How to Add a Translation

This page explains how to add or improve a translation of the docs for **any** language.

The site uses MkDocs + Material + `mkdocs-static-i18n`. English is the reference (fallback). Any other language mirrors its folder structure. If a page is missing in a target language, the English version is shown automatically.

---
## 1. Overview

There are three typical scenarios:

1. Improve an existing translated page.
2. Translate an English page that currently has no localized version.
3. Add an entirely new language to the project.

Everything below covers all three.

---
## 2. Folder layout

```
/docs
  en/
    developer-guide/
    getting-started/
    user-guide/
  <locale>/
    (mirrors the structure you translate)
```

Examples of `<locale>`: `pt`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`. Use short IETF / BCP‑47 language tags (no country code unless needed, e.g. `pt` vs `pt-BR` only if a regional variant is required). Keep it consistent with `mkdocs.yml`.

Relative paths must match. Example:

```
English: docs/en/developer-guide/how-to-add-a-translation.md
Spanish: docs/es/developer-guide/how-to-add-a-translation.md
French:  docs/fr/developer-guide/how-to-add-a-translation.md
```

---
## 3. Quick start (improving or adding a translation)

1. Fork and clone your fork.
2. Create / activate a virtual environment.
3. Install docs extras:
    ```console
    pip install -e ".[docs]"
    ```
4. Run the local server:
    ```console
    mkdocs serve
    ```
5. Open the local URL and switch language with the selector.

Restart the server if new files do not show up (MkDocs sometimes needs a restart after new file additions).

---
## 4. Adding a new language (one-time setup)

If the language already exists (e.g. `pt/`), skip this section.

1. Pick a locale code (e.g. `es`).
2. Create the folder: `docs/es/`.
3. Copy the English `index.md` to `docs/es/index.md` and translate its visible text.
4. (Optional) Start by copying only a minimal subset (index + key getting-started pages) to keep the first PR reviewable.
5. Update `mkdocs.yml` `i18n.languages` section:
    ```yaml
    - locale: es
      name: Español
      build: true
      site_description: <translated site tagline>
      theme:
        docs_dir: docs/es/
        custom_dir: docs/es/
        site_dir: site/es/
        logo: overrides/assets/img/logotype-sysidentpy.svg
    ```
6. Do **not** duplicate nav entries for the new language: the plugin maps them automatically based on folder structure.
7. Run `mkdocs serve` and confirm the new language appears in the selector.

If you need a regional variant (e.g. `pt-BR`), use that code consistently in both folder name and `mkdocs.yml`.

---
## 5. Adding a new English page (source language)

1. Create the file under `docs/en/...` in the right section.
2. Add front matter:
    ```md
    ---
    template: overrides/main.html
    title: My Page Title
    ---
    ```
3. Add its path to the `nav` tree in `mkdocs.yml` (English only).
4. Verify the site builds.
5. (Optional) Add a translator note (HTML comment) for tricky concepts:
    ```md
    <!-- Translator note: keep the term "NARMAX" in English. -->
    ```

---
## 6. Creating the translated file

1. Mirror the folder path: `docs/<locale>/<same relative path>.md`.

2. Copy the English file.

3. Translate only human-readable prose. Keep intact:
    - Code blocks (except inline comments if clarity improves)
    - Identifiers: function/class names, parameters, imports
    - File paths, config keys, URLs
    - Markdown link targets (unless they refer to a language-specific external resource)

4. Preserve heading hierarchy (`#`, `##`, etc.) but translate heading text itself.

5. Keep admonition types (`!!! note`, `!!! warning`, etc.). You may translate the title string after the admonition type.

6. For unfinished parts, you can temporarily keep English or add:
   ```md
   !!! note "Pending translation"
       This paragraph still needs translation.
   ```

7. Remove all "Pending" notes before final review if completed.

---
## 7. Internal links

Use relative links without language prefixes:
```md
See the [Contribute guide](contribute.md).
```
The i18n plugin rewrites them per language. Avoid hardcoding `/en/` or another locale path unless you intentionally want a fallback link.

Anchor links: if you translate a heading, Material generates a localized slug. Keep anchor usage consistent or adjust any `(#anchor)` references accordingly.

---
## 8. Images and media

If screenshots contain embedded language text:

- **Option A**: replicate and localize the image inside `docs/<locale>/assets/` and keep the same filename (per-locale path isolates it).
- **Option B**: reuse the English image if text is minimal or language-agnostic.

SVG diagrams: prefer keeping labels English if they are code or model symbols; translate UI captions where helpful.

---
## 9. Formatting & style guidelines

| Aspect | Guideline |
|--------|-----------|
| Numbers | Keep numeric precision; adapt decimal separators only if standard for the target audience (optional). |
| Units | Do not translate SI units (e.g. `ms`, `Hz`). |
| API names | Never translate identifiers. |
| Quotes | Follow local typographic conventions only if they do not break Markdown. |
| Capitalization | Mirror English capitalization only where it's a proper noun or API name. |
| Tone | Neutral, concise, instructional. |

Avoid machine‑translated chunks without revision. Prefer shorter, unambiguous sentences.

---
## 10. Translation glossary

Keep these terms consistent across languages. Add target language equivalents as needed:

| English Term | Portuguese (pt) |
|--------------|-----------------|
| **Core concepts** |
| model structure | estrutura do modelo |
| parameter estimation | estimação de parâmetros |
| residual analysis | análise dos resíduos |
| time series | série temporal |
| identification | identificação |
| **Technical terms** |
| basis function | função de base |
| regression | regressão |
| algorithm | algoritmo |
| validation | validação |
| simulation | simulação |
| **Development terms** |
| feature | funcionalidade |
| pull request (PR) | pull request (PR) |
| branch | branch |
| commit | commit |
| documentation | documentação |

For other languages, follow similar patterns. Prefer clarity over literal translation.

---
## 11. Review checklist (per translated file)

- [ ] Builds locally (`mkdocs serve`).
- [ ] File path mirrors English.
- [ ] Relative links work; no `/en/` hardcoding.
- [ ] Code blocks untouched (except explanatory comments if needed).
- [ ] Terminology consistent with existing pages.
- [ ] No leftover placeholder notes (unless intentionally partial PR).
- [ ] Front matter present and `title:` localized.

---
## 12. Commit & Pull Request

Add both English and translated files if you introduce a page + its translation; otherwise just the translated file.

Example:
```console
git add docs/en/developer-guide/new-topic.md docs/es/developer-guide/new-topic.md
git commit -m "docs: add Spanish translation for new-topic"
```

PR description template:

- **Language**: `<locale>` (e.g. `es`)
- **Pages added/updated**: list paths
- **Sections intentionally left untranslated**: (if any)
- **Glossary additions**: (if any)
- **Notes for reviewer**: context, tricky terms

Partial translations are acceptable—label them clearly so others can help.

---
## 13. Updating existing translations

When an English page changes:

1. Open the diff to see what changed.
2. Apply equivalent edits to the translated file.
3. If you cannot translate immediately, keep the English change and add a temporary "Pending translation" note.
4. Remove the note once updated.

Aim for incremental PRs rather than large rewrites.

---
## 14. Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Page shows in English only | Missing localized file or wrong path | Mirror path under new locale folder |
| Build error after adding language | Misconfigured `i18n.languages` entry | Re-check `locale` key + indentation |
| 404 on internal link | Link path differs from English | Match relative path to English version |
| Broken anchor | Heading translated; old anchor used | Update anchor or keep similar heading slug |
| Language not in selector | Forgot to add language in `mkdocs.yml` | Add `locale` entry and restart server |

---
## 15. Automation (optional)

You may use external CAT / translation tools, but always manually review technical terms. Avoid committing raw machine output.

If you script copying English files to a new locale, exclude already translated ones to prevent overwriting.

---
## 16. Questions & support

If you are unsure about terminology, open an Issue or Discussion before translating large sections. Early feedback prevents rework.

Thanks for helping make the documentation accessible to more people.
