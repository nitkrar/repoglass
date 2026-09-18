"""Exact symbol lookup and FTS5/BM25 keyword search.

Tiers 1 and 2. Exact is the right answer when the caller knows the
identifier -- the case embeddings handle worst.
"""

from __future__ import annotations

import re

from ..models import SearchMode, Symbol
from ..store import Store
from .query_shape import looks_like_prose

#: Words carrying no retrieval signal in a natural-language query.
_STOP = frozenset({
    "how", "does", "do", "the", "where", "is", "are", "in", "to", "a", "an",
    "of", "and", "with", "it", "we", "for", "from", "on", "this", "that",
    "what", "when", "which", "work", "works", "done", "use", "used", "app",
})


def exact(store: Store, name: str, *, lang: str | None = None) -> list[Symbol]:
    """Tier 1: symbol.name equality."""
    return store.definitions(name, lang=lang)


def keyword(
    store: Store, query: str, *, limit: int, mode: SearchMode, lang=None, include=None, exclude=None
) -> list[tuple[int, float]]:
    """Tier 2: FTS5 MATCH ranked by bm25().

    Returns (symbol_id, raw_score). Raw bm25 is negative and ascending-best;
    normalisation to [0,1] happens in fuse.py, not here.
    """
    terms = tokenize_query(query)
    if not terms:
        return []
    return store.fts_search(" OR ".join(terms), limit=limit, mode=mode,
                            lang=lang, include=include,
                            exclude=exclude)


def tokenize_query(query: str) -> list[str]:
    """Content words for an FTS5 MATCH expression.

    Identifiers are split as well as kept: a query naming `decryptVault`
    should also match text containing `decrypt`.
    """
    words: list[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query):
        low = raw.lower()
        if len(low) > 2 and low not in _STOP:
            words.append(low)
        for part in re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+", raw):
            p = part.lower()
            if len(p) > 2 and p not in _STOP and p not in words:
                words.append(p)
    return words

