"""Identifier and path normalisation.

Peer to `config` and `models`: both the extractor and the store need it,
and the store may not import corpus.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|[0-9]+")


def humanise(text: str) -> str:
    """Split camelCase, snake_case and paths into space-separated words."""
    return " ".join(_WORD.findall(text.replace("/", " ").replace(".", " ")))
