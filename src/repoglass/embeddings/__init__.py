"""Text to vectors.

The same model must embed chunks and queries; meta.embed_model enforces it.
Tests use FakeEmbedder, so no test reaches the network -- model2vec
downloads on first use.
"""

from __future__ import annotations

from typing import Sequence

from ..config import Paths, Settings, VECTOR_DTYPE
from .backends import (Embedder, FakeEmbedder, HttpEmbedder, OnnxEmbedder,
                       StaticEmbedder, _pool, _providers,
                       resolve_model_source)
from .policy import (default_doc_prefix, default_pooling,
                     default_query_prefix, resolve_doc_prefix,
                     resolve_query_prefix)

__all__ = [
    "VECTOR_DTYPE",
    "Embedder",
    "StaticEmbedder",
    "HttpEmbedder",
    "FakeEmbedder",
    "OnnxEmbedder",
    "default_doc_prefix",
    "default_pooling",
    "default_query_prefix",
    "resolve_doc_prefix",
    "resolve_model_source",
    "resolve_query_prefix",
    "build",
    "pack",
    "_pool",
    "_providers",
]


def build(settings: Settings, paths: Paths) -> Embedder | None:
    """Construct the configured embedder, or None when embed_backend='none'.

    Every backend is named explicitly and an unknown value raises. A
    trailing `return StaticEmbedder(...)` would turn a typo in
    `embed_backend` into a silent fallback that indexes the whole corpus
    with the wrong model.
    """
    backend = settings.embed_backend
    if backend == "none":
        return None
    if backend == "http":
        if not settings.embed_endpoint:
            raise ValueError("embed_backend='http' requires embed_endpoint")
        emb: Embedder = HttpEmbedder(
            settings.embed_endpoint, settings.embed_model,
            api_key=settings.embed_api_key,
        )
    elif backend == "onnx":
        emb = OnnxEmbedder(settings.embed_model,
                           providers=settings.embed_providers,
                           filename=settings.embed_onnx_file)
    elif backend == "static":
        emb = StaticEmbedder(settings.embed_model, paths.models)
    else:
        raise ValueError(
            f"unknown embed_backend {backend!r};"
            " expected 'static', 'onnx', 'http' or 'none'"
        )
    emb.query_prefix = resolve_query_prefix(settings)
    emb.doc_prefix = resolve_doc_prefix(settings)
    return emb


def pack(vector: Sequence[float]) -> bytes:
    """Normalise to unit length, so query time is a bare dot product."""
    import numpy as np

    v = np.asarray(vector, dtype="float32")
    norm = float(np.linalg.norm(v))
    if norm:
        v = v / norm
    return v.astype(VECTOR_DTYPE).tobytes()
