"""Render chunk text for embedding and lexical search."""

from __future__ import annotations

import re

from ..config import Settings
from ..text import humanise

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


_IDENT_CAP = 25


def embed_text(*, path: str, lang: str, node_kind: str,
               leading_comments, body: str, settings: Settings) -> str:
    """The string that gets embedded.

    The raw span, unless `distill_docs` is on and `lang` is a doc
    language -- then a `path :: kind :: comments :: identifiers`
    summary. That branch is the only route by which a definition's
    leading comments reach the index at all.
    """
    if not (settings.distill_docs and lang in settings.doc_languages):
        return body
    idents: list[str] = []
    seen: set[str] = set()
    for raw in _IDENT.findall(body):
        word = humanise(raw)
        if word and word not in seen:
            seen.add(word)
            idents.append(word)
        if len(idents) >= _IDENT_CAP:
            break
    parts = [humanise(path), node_kind, " ".join(leading_comments),
             " ".join(idents)]
    return " :: ".join(p for p in parts if p)


def lexical_override(*, path: str, body: str, settings: Settings) -> str | None:
    """The FTS text to store, or None when the view can derive it.

    The `chunk_lexical` view falls back to `file.path_words` followed by
    the chunk's text. Anything the settings do beyond that -- identifier
    splitting, stem/directory enrichment, a cap -- has to be stored, so
    what decides is a comparison against that fallback: a new setting
    cannot quietly reshape the FTS text without being written down.
    """
    rendered = lexical(path=path, body=body, settings=settings)
    return None if rendered == f"{humanise(path)}\n{body}" else rendered


def lexical(*, path: str, body: str, settings: Settings) -> str:
    """Build the text that FTS5 indexes.

    Humanised path followed by the raw span, so keyword search reaches
    string literals and error messages no distilled form would keep.
    `lexical_mode`, `lexical_enrich` and `split_identifiers` each
    reshape it from there.
    """
    if settings.lexical_mode == "capped":
        body = body[: settings.lexical_cap_chars]
    if settings.lexical_enrich:
        body = _enriched(path, body)
    else:
        body = f"{humanise(path)}\n{body}"
    if settings.split_identifiers == "inline":
        return _split_inline(body)
    if settings.split_identifiers == "append":
        return f"{body}\n{_split_identifiers(body)}"
    return body


_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def _split_inline(text: str) -> str:
    """Rewrite each compound identifier as `compound part part` in place.

    Differs from `_split_identifiers`, which appends one de-duplicated
    blob at the end: inline keeps each occurrence's term frequency, so
    a file that uses `HandlerStack` ten times scores ten hits on
    "handler", not one.
    """
    def expand(m: "re.Match[str]") -> str:
        raw = m.group(0)
        parts = ([p for p in raw.lower().split("_") if p] if "_" in raw
                 else [x.lower() for x in _CAMEL.findall(raw)])
        if len(parts) < 2:
            return raw
        return raw + " " + " ".join(p for p in parts if len(p) > 2)
    return _IDENT.sub(expand, text)


def _split_identifiers(text: str) -> str:
    """The sub-words of every compound identifier, appended for FTS5.

    FTS5's `unicode61` tokenizer treats `HandlerStack` as one token,
    while `tokenize_query` splits it into ['handlerstack', 'handler',
    'stack']. Splitting only the query side is worse than splitting
    neither: it looks like it works and silently matches nothing.

    The compound stays in the text, so an exact match on the whole
    identifier still scores; these are additions, not replacements.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in _IDENT.findall(text):
        parts = ([p for p in raw.lower().split("_") if p] if "_" in raw
                 else [m.lower() for m in _CAMEL.findall(raw)])
        if len(parts) < 2:
            continue
        for part in parts:
            if len(part) > 2 and part not in seen:
                seen.add(part)
                out.append(part)
    return " ".join(out)


def _enriched(path: str, body: str) -> str:
    """FTS input: span, then the bare stem twice, then dirs.

    The stem is doubled deliberately -- a file named `suppression.py`
    should be reachable by the word "suppression" even when the body
    never uses it. Lives here rather than in search/rank.py because it
    builds indexed text, and corpus may not import search.
    """
    from pathlib import PurePosixPath

    p = PurePosixPath(path)
    dirs = [part for part in p.parent.parts if part not in (".", "/")]
    return f"{body} {p.stem} {p.stem} {' '.join(dirs[-3:])}"
