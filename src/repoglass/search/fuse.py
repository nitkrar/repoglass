"""Tier weighting and the configured ranker.

Fusion itself is semsift's; this module turns repoglass's tier results
into ranked lists and decides how much each tier counts.
"""

from __future__ import annotations

from typing import Sequence

from semsift.fuse import Fused, RankedList, Scored, blend, first, rrf

from ..config import Settings


def ranked(tier: str, pairs, *, lower_is_better: bool = False,
           preserve_order: bool = False) -> RankedList:
    """(id, raw score) pairs as a best-first list; raw is kept for display.

    bm25() is lower-is-better, so the lexical tier negates it for the
    score fusion reads. `preserve_order` represents a ranked source whose
    raw values do not distinguish its items, such as exact matches.
    """
    pairs = tuple(pairs)
    if preserve_order:
        items = tuple(Scored(i, float(len(pairs) - rank), raw)
                      for rank, (i, raw) in enumerate(pairs))
    else:
        sign = -1.0 if lower_is_better else 1.0
        items = tuple(sorted(
            (Scored(i, sign * raw, raw) for i, raw in pairs),
            key=lambda s: (-s.score, s.id),
        ))
    return RankedList(tier, tuple(items))


def _normalised(fused: Fused) -> list[tuple[int, float]]:
    """Fused scores mapped onto [0, 1], best first."""
    rl = RankedList("fused", tuple(Scored(i, s, s) for i, s in fused.items))
    return list(blend([rl], normalise="min-max").items)


def tier_weights(settings: Settings, *, is_prose: bool) -> dict[str, float]:
    """How much each tier counts, given the shape of the query.

    An identifier-shaped query is a job for BM25 and exact matching; a
    sentence is a job for the embedder.

    `exact` is never down-weighted: it fires only on a literal symbol
    name match, which is the one signal that should not be diluted.
    """
    alpha = settings.alpha_prose if is_prose else settings.alpha_symbol
    return {"exact": 1.0, "vector": alpha, "lexical": 1.0 - alpha}


def merge(lists: Sequence[RankedList], settings: Settings, *,
          is_prose: bool = False) -> list[tuple[int, float]]:
    """Apply the configured ranker: RRF, or the first non-empty tier."""
    populated = [rl for rl in lists if rl.items]
    if not populated:
        return []
    if settings.ranker == "none":
        fused = first(populated)
        if len({s.raw for s in populated[0].items}) == 1:
            return [(i, 1.0) for i, _ in fused.items]
        return _normalised(fused)
    weights = (tier_weights(settings, is_prose=is_prose)
               if settings.adaptive_alpha else None)
    return _normalised(rrf(populated, weights=weights))
