"""Query-shape tests used by different retrieval stages.

These are not complements: a short semantic phrase can fail both. Callers
pick the test that matches their routing decision rather than treating one
as the inverse of the other.
"""

from __future__ import annotations

import re

#: A bare symbol, a namespace-qualified name, or anything carrying an
#: uppercase letter or underscore. A plain lowercase word is prose.
_SYMBOL_QUERY_RE = re.compile(
    r"^(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:(?:::|\\|->|\.)[A-Za-z_][A-Za-z0-9_]*)+"
    r"|_[A-Za-z0-9_]*"
    r"|[A-Za-z][A-Za-z0-9]*[A-Z_][A-Za-z0-9_]*"
    r"|[A-Z][A-Za-z0-9]*"
    r")$"
)


def is_symbol_query(query: str) -> bool:
    """Whether the query reads as an identifier rather than a sentence."""
    return _SYMBOL_QUERY_RE.match(query.strip()) is not None


def looks_like_prose(query: str, *, min_words: int) -> bool:
    """Whether a query reads as a question rather than an identifier."""
    return len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)) >= min_words
