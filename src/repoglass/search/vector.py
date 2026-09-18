"""Cosine similarity over the vector matrix.

numpy only. There is no stdlib fallback: a pure-Python path accumulates in
float64 while numpy accumulates in float32, and the two disagree on the
ordering of near-identical pairs. A size-threshold switch between them
would change top-1 results purely because the corpus grew.
"""

from __future__ import annotations

from typing import Sequence

from ..models import SearchMode
from ..store import Store


def search(
    store: Store, query_vec: Sequence[float], *, limit: int,
    mode: SearchMode, lang=None, include=None, exclude=None
) -> list[tuple[int, float]]:
    """Tier 3: M @ q over pre-normalised vectors.

    Results sort by (-score, symbol_id). The secondary key is required:
    float ties are common among near-duplicate chunks, and rowid order is
    not stable across refreshes.

    The matrix is widened to float32 for the product: float16 accumulates
    error across the row, and the saving that dtype buys is in the stored
    blob, not in the arithmetic.
    """
    import numpy as np

    from ..config import VECTOR_DTYPE

    ids, blob = store.vectors(mode=mode, lang=lang, include=include, exclude=exclude)
    if not ids or not blob:
        return []
    stride = np.dtype(VECTOR_DTYPE).itemsize
    dims = len(blob) // stride // len(ids)
    if dims == 0:
        return []
    matrix = np.frombuffer(blob, dtype=VECTOR_DTYPE).reshape(len(ids), dims)
    q = np.asarray(query_vec, dtype="float32")
    norm = float(np.linalg.norm(q))
    if norm:
        q = q / norm
    scores = matrix.astype("float32") @ q
    id_col = np.asarray(ids, dtype="int64")

    # A full sort of the corpus costs more than the product it is
    # ordering. Partition to the top `limit` first, then order only those.
    if limit < len(ids) - 1:
        cut = np.argpartition(-scores, limit)[:limit + 1]
        # Everything tied with the weakest of that set, so the
        # tiebreak below sees the same candidates a full sort would.
        # Without it a tie straddling the cut resolves by whichever
        # side of the partition numpy happened to place it.
        pool = np.flatnonzero(scores >= scores[cut].min())
    else:
        pool = np.arange(len(ids))
    # lexsort's last key is primary, so this is (-score, id).
    order = pool[np.lexsort((id_col[pool], -scores[pool]))][:limit]
    return [(int(id_col[i]), float(scores[i])) for i in order]
