"""The boosts reranking applies before path penalties and saturation."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .query_shape import is_symbol_query
from .tokens import split_identifier, stem_keywords

#: camelCase or PascalCase appearing inside an otherwise prose query.
_EMBEDDED_SYMBOL_RE = re.compile(
    r"\b(?:[A-Z][a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*"
    r"|[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]+)\b"
)


_DEFINITION_KEYWORDS = (
    "class", "module", "defmodule", "def", "interface", "struct", "enum",
    "trait", "type", "func", "function", "object", "abstract class",
    "data class", "fn", "fun", "package", "namespace", "protocol",
    "record", "typedef",
)
_SQL_KEYWORDS = ("CREATE TABLE", "CREATE VIEW", "CREATE PROCEDURE",
                 "CREATE FUNCTION")
_KEYWORD_PREFIX = r"(?:^|(?<=\s))(?:"
_DEF_BODY = "|".join(re.escape(k) for k in _DEFINITION_KEYWORDS)
_SQL_BODY = "|".join(re.escape(k) for k in _SQL_KEYWORDS)


@functools.lru_cache(maxsize=256)
def _definition_pattern(name: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    escaped = re.escape(name)
    ns = r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\.|::))*"
    suffix = r")\s+" + ns + escaped + r"(?:\s|[<({:\[;]|$)"
    return (
        # Case-sensitive: IGNORECASE matches "Module" in prose and
        # "Class" used as a method name.
        re.compile(_KEYWORD_PREFIX + _DEF_BODY + suffix, re.MULTILINE),
        re.compile(_KEYWORD_PREFIX + _SQL_BODY + suffix,
                   re.MULTILINE | re.IGNORECASE),
    )


def defines_symbol(text: str, name: str) -> bool:
    """Whether `text` declares `name`, by keyword match rather than parse.

    A chunk may be a fragment no grammar accepts, so this reads the
    definition keywords directly. A namespace qualifier ahead of the name
    is allowed: `class foo.Bar` defines `Bar`.
    """
    general, sql = _definition_pattern(name)
    return general.search(text) is not None or sql.search(text) is not None


def _stem_matches(stem: str, name: str) -> bool:
    flat = stem.replace("_", "")
    return name in (stem, flat, stem.rstrip("s"), flat.rstrip("s"))


@dataclass
class Candidate:
    """One scored chunk, with the text and path re-ranking needs."""

    symbol_id: int
    path: str
    text: str
    score: float


#: `(names, stems) -> extra candidates`. Chunks retrieval never returned
#: but whose file stem matches the queried symbol. This is the only way
#: a chunk absent from every tier can reach the results.
NonCandidateLoader = Callable[[set[str]], Iterable[Candidate]]


def _boost_coherence(cands: list[Candidate], settings) -> None:
    """Add to each file's best chunk, in proportion to that file's total.

    Only the best chunk: boosting all of them would fill the result list
    with one file, which the saturation decay in `_select_top` exists to
    prevent.
    """
    top = max(c.score for c in cands)
    if top <= 0.0:
        return
    by_file: dict[str, float] = {}
    best: dict[str, Candidate] = {}
    for c in cands:
        by_file[c.path] = by_file.get(c.path, 0.0) + c.score
        if c.path not in best or c.score > best[c.path].score:
            best[c.path] = c
    scale = max(by_file.values())
    if scale <= 0.0:
        return
    unit = top * settings.file_coherence
    for path, c in best.items():
        c.score += unit * by_file[path] / scale


def _boost_query(cands: list[Candidate], query: str, settings, *,
                 load_non_candidates: NonCandidateLoader | None) -> None:
    top = max(c.score for c in cands)
    if top <= 0.0:
        return
    if is_symbol_query(query):
        _boost_definitions(cands, query, top, settings, load_non_candidates)
    else:
        _boost_stems(cands, query, top, settings)
        _boost_embedded(cands, query, top, settings, load_non_candidates)


def _definition_bonus(c: Candidate, names: set[str], unit: float) -> float:
    """Half again when the file is named for the symbol it defines.

    A definition of `HandlerStack` in `handler_stack.py` is the canonical
    one; the same definition elsewhere is more often a re-export or a
    local shadow.
    """
    if not any(defines_symbol(c.text, n) for n in names):
        return 0.0
    stem = Path(c.path).stem.lower()
    matched = any(_stem_matches(stem, n.lower()) for n in names)
    return unit * (1.5 if matched else 1.0)


def _boost_definitions(cands, query, top, settings, loader) -> None:
    """Boost definitions of the queried symbol, unqualified name included.

    `pkg::Widget` also boosts definitions of `Widget`, since the chunk
    holding the definition rarely repeats the qualifier.
    """
    name = query.strip()
    for sep in ("::", "\\", "->", "."):
        if sep in name:
            name = name.rsplit(sep, 1)[-1]
            break
    names = {name, query.strip()}
    unit = top * settings.definition_boost
    for c in cands:
        c.score += _definition_bonus(c, names, unit)
    if loader is None:
        return
    seen = {c.symbol_id for c in cands}
    for extra in loader(names):
        if extra.symbol_id in seen:
            continue
        bonus = _definition_bonus(extra, names, unit)
        if bonus:
            extra.score = bonus
            cands.append(extra)


def _boost_embedded(cands, query, top, settings, loader) -> None:
    """Boost definitions of CamelCase words found inside a prose query.

    Half the strength of a bare symbol query: a capitalised word in a
    sentence may be incidental.
    """
    names = set(_EMBEDDED_SYMBOL_RE.findall(query))
    if not names:
        return
    unit = top * settings.definition_boost * 0.5
    for c in cands:
        c.score += _definition_bonus(c, names, unit)
    if loader is None:
        return
    seen = {c.symbol_id for c in cands}
    for extra in loader(names):
        if extra.symbol_id in seen:
            continue
        bonus = _definition_bonus(extra, names, unit)
        if bonus:
            extra.score = bonus
            cands.append(extra)


def _count_matches(keywords: set[str], parts: set[str]) -> int:
    """Exact matches, then prefix overlap of at least 3 characters.

    Prefixes carry the morphology: "dependency" should match
    "dependencies".
    """
    exact = keywords & parts
    if len(exact) == len(keywords):
        return len(exact)
    n = len(exact)
    for kw in keywords - exact:
        for part in parts:
            short, long = (kw, part) if len(kw) <= len(part) else (part, kw)
            if len(short) >= 3 and long.startswith(short):
                n += 1
                break
    return n


def _boost_stems(cands: list[Candidate], query: str, top: float,
                 settings) -> None:
    """Match query words against the file stem and its parent directory.

    Scaled by what fraction of the query the path accounts for, so one
    incidental word shared with a directory name cannot lift a whole
    subtree.
    """
    keywords = stem_keywords(query)
    if not keywords:
        return
    boost = top * settings.stem_boost
    cache: dict[str, set[str]] = {}
    for c in cands:
        if c.path not in cache:
            p = Path(c.path)
            parts = set(split_identifier(p.stem))
            if p.parent.name not in (".", "/", "..", ""):
                parts.update(split_identifier(p.parent.name))
            cache[c.path] = parts
        n = _count_matches(keywords, cache[c.path])
        if n:
            ratio = n / len(keywords)
            if ratio >= 0.10:
                c.score += boost * ratio
