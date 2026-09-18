"""Post-retrieval re-ranking.

Retrieval gets a pool of plausible chunks; this decides their order.

Order matters:

    coherence boost  ->  query boost  ->  penalties over the top-k

The stages carry weights an order of magnitude apart -- `file_coherence`,
`stem_boost` and `definition_boost` in Settings -- so none of them is
representative of the layer on its own.

Chunks live in SQLite, not in memory, so the non-candidate scan takes a
loader callback that fetches only chunks whose file stem could match,
rather than scanning the whole corpus per query.
"""

from __future__ import annotations

from .boosting import (Candidate, NonCandidateLoader, _boost_coherence,
                       _boost_query, defines_symbol)
from .penalties import _select_top, path_penalty
from .query_shape import is_symbol_query
from .tokens import split_identifier, tokenize

__all__ = [
    "Candidate",
    "NonCandidateLoader",
    "defines_symbol",
    "is_symbol_query",
    "path_penalty",
    "rerank",
    "split_identifier",
    "tokenize",
]


def rerank(
    candidates: list[Candidate],
    query: str,
    settings,
    *,
    load_non_candidates: NonCandidateLoader | None = None,
    penalise_paths: bool = True,
    limit: int = 10,
) -> list[Candidate]:
    """Boost, then penalise, then take the top `limit`.

    Mutates `candidates` in place: scores are adjusted on the objects
    passed in, and the definition boost appends chunks that retrieval
    never returned.
    """
    if not candidates:
        return []
    if settings.file_coherence > 0.0:
        _boost_coherence(candidates, settings)
    _boost_query(candidates, query, settings,
                 load_non_candidates=load_non_candidates)
    return _select_top(candidates, settings, limit=limit,
                       penalise_paths=penalise_paths)
