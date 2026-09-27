"""repoglass settings to a semsift encoder.

The same model must embed chunks and queries; the stored vector space
(`meta` in the index) enforces it.
"""

from __future__ import annotations

from typing import Never

from semsift.embed import (Encoder, HttpEncoder, OnnxEncoder, StaticEncoder,
                           VectorSpace)
from semsift.embed.policy import resolve_prefix

from .config import Settings

#: Query instructions for models trained to take a task. These name code
#: search, so they are repoglass's rather than the model's defaults.
_CODE_QUERY_PREFIXES = {
    "qwen3-embedding": "Instruct: Given a question, retrieve code that"
                       " answers it\nQuery: ",
    "coderankembed": "Represent this query for searching relevant code: ",
}


def query_prefix(settings: Settings) -> str:
    return resolve_prefix(settings.embed_query_prefix, settings.embed_model,
                          "query", _CODE_QUERY_PREFIXES)


def doc_prefix(settings: Settings) -> str:
    return resolve_prefix(settings.embed_doc_prefix, settings.embed_model, "doc")


def space(settings: Settings, dims: int) -> VectorSpace:
    """The space `build(settings)` writes into, without loading a model."""
    variant = {"onnx": settings.embed_onnx_file,
               "http": settings.embed_endpoint}.get(settings.embed_backend, "")
    return VectorSpace.of(settings.embed_model, settings.embed_backend,
                          variant=variant, doc_prefix=doc_prefix(settings),
                          dims=dims)


def _raise_onnx_error(exc: ImportError) -> Never:
    """Name repoglass's extra rather than semsift's for a missing package."""
    cause = exc.__cause__
    missing = cause.name if isinstance(cause, ImportError) else None
    if missing == "onnxruntime":
        raise ValueError(
            "embed_backend='onnx' requires onnxruntime: "
            "pip install 'repoglass[onnx]'"
        ) from exc
    if missing == "onnxruntime_ep_webgpu":
        raise ValueError(
            "embed_providers='webgpu' requires the plugin: "
            "pip install 'repoglass[webgpu]', or set "
            "embed_providers='cpu'"
        ) from exc
    raise exc


def build(settings: Settings) -> Encoder | None:
    """Construct the configured embedder, or None when embed_backend='none'.

    Every backend is named explicitly and an unknown value raises. A
    trailing `return StaticEncoder(...)` would turn a typo in
    `embed_backend` into a silent fallback that indexes the whole corpus
    with the wrong model.
    """
    backend = settings.embed_backend
    if backend == "none":
        return None
    prefixes = {"query_prefix": query_prefix(settings),
                "doc_prefix": doc_prefix(settings)}
    if backend == "http":
        if not settings.embed_endpoint:
            raise ValueError("embed_backend='http' requires embed_endpoint")
        return HttpEncoder(settings.embed_endpoint, settings.embed_model,
                           api_key=settings.embed_api_key, **prefixes)
    if backend == "onnx":
        try:
            return OnnxEncoder(settings.embed_model,
                               providers=settings.embed_providers,
                               filename=settings.embed_onnx_file, **prefixes)
        except ImportError as exc:
            _raise_onnx_error(exc)
    if backend == "static":
        return StaticEncoder(settings.embed_model, **prefixes)
    raise ValueError(
        f"unknown embed_backend {backend!r};"
        " expected 'static', 'onnx', 'http' or 'none'"
    )
