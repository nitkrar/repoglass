"""Rank merging: RankedList, RRF, tier weighting, tier orchestration.

All merging lives here. lexical.py and vector.py return raw per-tier scores
and never combine them, so the two cannot drift into separate notions of
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..config import Settings
from ..config import RRF_K


@dataclass(frozen=True)
class RankedList:
    """One tier's output: symbol ids in rank order, with raw scores."""

    tier: str
    items: tuple[tuple[int, float], ...]

    def ranks(self) -> dict[int, int]:
        """symbol_id -> 1-based rank, for RRF."""
        return {sid: i + 1 for i, (sid, _) in enumerate(self.items)}


def rrf(lists: Sequence[RankedList], *, k: int = RRF_K,
        weights: dict[str, float] | None = None) -> list[tuple[int, float]]:
    """Reciprocal rank fusion across tiers, optionally weighted per tier.

    Unweighted RRF gives every tier the same say regardless of how good
    its list is: a tier that answers a query badly still contributes a
    guaranteed 1/(k+1) for its rank-1 item, diluting a tier that
    answered it well. Weighting is the answer to that rather than
    dropping a tier outright, which loses the queries it does win.
    """
    scores: dict[int, float] = {}
    for rl in lists:
        w = 1.0 if weights is None else weights.get(rl.tier, 1.0)
        if w == 0.0:
            continue
        for sid, rank in rl.ranks().items():
            scores[sid] = scores.get(sid, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def tier_weights(settings: Settings, *, is_prose: bool) -> dict[str, float]:
    """How much each tier counts, given the shape of the query.

    An identifier-shaped query is a job for BM25 and exact matching; a
    sentence is a job for the embedder.

    `exact` is never down-weighted: it fires only on a literal symbol
    name match, which is the one signal that should not be diluted.
    """
    alpha = settings.alpha_prose if is_prose else settings.alpha_symbol
    return {"exact": 1.0, "vector": alpha, "lexical": 1.0 - alpha}


def normalise(scores: Sequence[tuple[int, float]]) -> list[tuple[int, float]]:
    """Map raw tier scores onto [0,1], higher better.

    Hit.score is comparable only within one call; this is where that
    guarantee is established. Raw inputs differ in direction -- bm25() is
    negative and ascending-best, cosine positive and descending-best -- so
    callers must never see them.
    """
    if not scores:
        return []
    values = [s for _, s in scores]
    low, high = min(values), max(values)
    if high == low:
        return [(sid, 1.0) for sid, _ in scores]
    span = high - low
    return sorted(
        ((sid, (s - low) / span) for sid, s in scores),
        key=lambda kv: (-kv[1], kv[0]),
    )


def merge(lists: Sequence[RankedList], settings: Settings, *,
          is_prose: bool = False) -> list[tuple[int, float]]:
    """Apply the configured ranker: RRF, or the first non-empty tier."""
    populated = [rl for rl in lists if rl.items]
    if not populated:
        return []
    if settings.ranker == "none":
        return normalise(list(populated[0].items))
    weights = (tier_weights(settings, is_prose=is_prose)
               if settings.adaptive_alpha else None)
    return normalise(rrf(populated, weights=weights))
