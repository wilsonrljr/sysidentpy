---
template: overrides/main.html
title: Cómo Añadir una Traducción
---

# Cómo Añadir una Traducción

Guía para crear o mejorar traducciones de la documentación en **cualquier** idioma.

Usamos MkDocs + Material + `mkdocs-static-i18n`. El inglés es el fallback. Cualquier idioma nuevo replica la estructura de carpetas. Si falta una página traducida, se muestra la versión en inglés.

---

## 1. Resumen

Tres escenarios comunes:

1. Mejorar una página ya traducida.
2. Traducir una página que sólo existe en inglés.
3. Añadir un idioma completamente nuevo.

Todo lo siguiente cubre esos casos.

---

## 2. Estructura de carpetas

```
/docs
  en/
    developer-guide/
    getting-started/
    user-guide/
  <locale>/
    (mismos caminos relativos)
```

Ejemplos de `<locale>`: `pt`, `es`, `fr`, `de`, `it`, `ja`, `zh`, `ru`. Use códigos cortos (BCP47). Evite variantes regionales salvo necesidad (`pt-BR`, `pt-PT`).

---

## 3. Cómo traducir una página existente

1. Busca la página en `docs/en/`.
2. Copia la página idéntica a `docs/<locale>/`.
3. Traduce el contenido entre `---` front-matter y footer sin tocar código o rutas.
4. Mantén todos los `code fences`, URLs y nombres de paquetes sin traducir.

---

## 4. Recomendaciones y reglas

- Mantén la propiedad técnica: no traduzcas código, identificadores, parámetros o nombres de paquetes.
- Preserva front matter (delimitadores `---`).
- Traduce los títulos de admoniciones (Nota, Atención) y contenido textual.
- Usa el glosario del proyecto cuando exista.
- Si una cadena es ambigua, deja una nota tipo `<!-- TODO: revisar traducción -->`.

---

## 5. Pruebas locales

Para comprobar tu traducción localmente:

```powershell
conda activate syspyenv
mkdocs serve
```

Abre `http://127.0.0.1:8000` y selecciona el idioma en la esquina superior.

---

## 6. Pull request

1. Añade tu rama con sólo los archivos traducidos.
2. Incluye una nota en la PR describiendo las decisiones de traducción.
3. Una revisión técnica y lingüística es deseable antes de merge.

---

Gracias por ayudar a traducir la documentación.
