"""Identifier splitting for the reranking layer.

This stays separate from lexical.tokenize_query: reranking wants raw
repeated words, while the FTS builder applies different stop-word and
length rules.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by do does for from has have how if in is it not"
    " of on or the to was what when where which who why with".split()
)


def split_identifier(token: str) -> list[str]:
    """`HandlerStack` -> `[handlerstack, handler, stack]`.

    The compound form is kept alongside the parts so an exact match on
    the whole identifier still outranks a match on one component.
    """
    lower = token.lower()
    if "_" in token:
        parts = [p for p in lower.split("_") if p]
    else:
        parts = [m.lower() for m in _CAMEL_RE.findall(token)]
    return [lower, *parts] if len(parts) >= 2 else [lower]


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text):
        out.extend(split_identifier(tok))
    return out


def stem_keywords(query: str) -> set[str]:
    return {
        word.lower() for word in _TOKEN_RE.findall(query)
        if len(word) > 2 and word.lower() not in _STOPWORDS
    }
