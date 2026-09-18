"""Model-family defaults for prefixes and pooling."""

from __future__ import annotations

from ..config import Settings

#: Models that were trained with an instruction on the query side only.
#: Encoding a query the same way as a document costs recall silently --
#: nothing errors, results are just worse -- so the recommendation ships
#: with the model name rather than waiting to be configured.
_QUERY_PREFIXES = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
    "qwen3-embedding": "Instruct: Given a question, retrieve code that"
                       " answers it\nQuery: ",
    "coderankembed": "Represent this query for searching relevant code: ",
    "nomic-embed": "search_query: ",
}

#: Prefixes for the DOCUMENT side. Most models want nothing here, but
#: e5 and nomic were trained with a marker on both sides and omitting
#: the document half is not neutral -- it puts queries and documents in
#: different regions of the space. Nothing errors; recall just drops.
_DOC_PREFIXES = {
    "e5": "passage: ",
    "nomic-embed": "search_document: ",
}

#: How a model turns token vectors into one sentence vector. Getting this
#: wrong produces plausible-looking vectors that rank badly, and nothing
#: errors -- so it is a named setting rather than a constant.
_POOLING = {
    "bge": "cls",
    "e5": "mean",
    "gte": "mean",
    "qwen3-embedding": "last",     # last non-pad token, not CLS
    "coderankembed": "mean",       # community ONNX card; base model is CLS
    "all-minilm": "mean",
    "nomic-embed": "mean",
}


def _family(model: str) -> str:
    stem = model.rsplit("/", 1)[-1].lower()
    known = set(_POOLING) | set(_QUERY_PREFIXES) | set(_DOC_PREFIXES)
    for family in sorted(known, key=len, reverse=True):
        if family in stem:
            return family
    return ""


def default_doc_prefix(model: str) -> str:
    """The prefix a model family expects on documents, or '' if none."""
    return _DOC_PREFIXES.get(_family(model), "")


def default_pooling(model: str) -> str:
    """The pooling a model was trained with, or 'cls' when unknown."""
    return _POOLING.get(_family(model), "cls")


def default_query_prefix(model: str) -> str:
    """The prefix a model family expects on queries, or '' if none."""
    return _QUERY_PREFIXES.get(_family(model), "")


def resolve_query_prefix(settings: Settings) -> str:
    """Setting wins over recommendation; `''` is a real choice.

    `embed_query_prefix=None` means "whatever the model wants".
    `embed_query_prefix=""` means "none, I know what I am doing".
    Collapsing the two would make the recommendation impossible to turn
    off.
    """
    if settings.embed_query_prefix is None:
        return default_query_prefix(settings.embed_model)
    return settings.embed_query_prefix


def resolve_doc_prefix(settings: Settings) -> str:
    """As `resolve_query_prefix`, for the document side."""
    if settings.embed_doc_prefix is None:
        return default_doc_prefix(settings.embed_model)
    return settings.embed_doc_prefix
