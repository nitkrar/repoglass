"""Records exchanged across module boundaries.

Leaf module: imports nothing from repoglass. Everything may depend on it.

Chunk construction lives here rather than in each extractor, so code and prose
extraction cannot drift into two incompatible notions of a chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

Tag = Literal["def", "ref"]

#: Which category of file a search may return. Resolved at walk time and
#: stored on the file row, so filtered search is an indexed equality
#: test any SQLite connection can run.
#:
#: Language decides 'docs', 'config' and 'data'; path substrings decide
#: 'tests'; 'code' is the remainder. A markdown file under tests/ is
#: documentation, not a test.
ContentType = Literal["code", "tests", "docs", "config", "data"]

#: What a caller may pass: one category, several, or nothing at all.
#: Nothing means no filter -- "everything" is the absence of a
#: restriction, not a fifth category to choose.
ContentFilter = ContentType | "Sequence[ContentType]" | None

#: Public alias for ContentFilter, exported from the package. Both of
#: `search()`'s content arguments are typed with it.
SearchMode = ContentFilter

#: One language, several, or nothing. Shaped like ContentFilter for the
#: same reason: a caller should not have to wrap a single value in a
#: list, and several is the common case once you have two dialects of
#: one thing in a repository.
LanguageFilter = str | Sequence[str] | None


@dataclass(frozen=True)
class SourceFile:
    """A file selected for indexing.

    Carries no ignore flag: gitignore membership is decided at walk
    time, by simply not yielding the file.

    It does carry `content_type`, resolved at walk time and persisted so
    filtered queries match on a stored value rather than generated SQL
    calling back into Python.
    """

    path: str          # relative to index root, / separated
    mtime_ns: int
    size: int
    lang: str
    content_type: ContentType = "code"


@dataclass(frozen=True)
class Symbol:
    name: str
    tag: Tag
    path: str
    start_line: int    # 1-indexed
    end_line: int
    lang: str
    signature: str | None = None    # source line; None for refs
    #: The file's category. Most useful on references, where "who calls
    #: this" and "who calls this from a test" are different questions.
    content_type: ContentType = "code"
    #: For a reference, the name of the definition whose span contains
    #: it; None at module scope, which is the majority case in some
    #: languages. Always None on a definition.
    enclosing: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One embeddable unit: a definition span and the text to embed.

    What FTS5 indexes is derived from `path` and `text` by the
    `chunk_lexical` view, not carried here.
    """

    path: str
    name: str
    start_line: int
    end_line: int
    text: str
    content_hash: str
    #: Set only when the FTS view cannot derive the lexical form. See
    #: `extract.lexical_override`.
    lexical_override: str | None = None

    @classmethod
    def build(
        cls,
        *,
        path: str,
        name: str,
        start_line: int,
        end_line: int,
        source: str,
        leading_comments: Sequence[str],
        node_kind: str,
        text: str,
        lexical_override: str | None = None,
    ) -> "Chunk":
        """Assemble a chunk from a span and its rendered text.

        `text` is supplied by the extractor rather than computed here:
        rendering needs language knowledge and settings, which models must
        not depend on. This constructor owns identity — the span and the
        content hash — so both extraction paths produce the same notion of
        a chunk.
        """
        import hashlib

        return cls(
            path=path,
            name=name,
            start_line=start_line,
            end_line=end_line,
            text=text,
            lexical_override=lexical_override,
            content_hash=hashlib.blake2b(
                source.encode(), digest_size=16
            ).hexdigest(),
        )


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int
    name: str
    score: float       # normalised [0,1], higher better, comparable within one call only
    code: str          # real source, read via the span
    #: The header of the definition this chunk holds, cut at its body.
    #: A window chunk owns no definition, so it falls back to the first
    #: non-blank line of its span: a path and a line range alone are not
    #: enough to decide whether to open a file. `name` is empty in that
    #: case, which is how a caller tells the two apart.
    signature: str | None = None
    #: Which tiers matched, and their raw scores: ("vector", 0.31),
    #: ("lexical", -7.4), ("exact", 1.0). `score` is divided by the top
    #: hit, so it cannot distinguish a confident match from the best of
    #: a bad set; these are the untransformed numbers. bm25() is
    #: negative, ascending-best; cosine is [-1,1], descending-best.
    tiers: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class RefreshReport:
    added: int
    changed: int
    deleted: int
    elapsed_s: float


#: Reference kinds that mean an invocation. Upstream query authors did
#: not agree on a word: Java captures `method_invocation` as `method`,
#: C# captures `invocation_expression` as `send`, after Smalltalk. The
#: alternative is renaming the capture in each vendored file, which
#: captures the node twice and yields two reference rows for one call.
CALL_KINDS = frozenset({"call", "send", "method"})


@dataclass(frozen=True)
class LanguageCapability:
    """What a language's tag queries actually provide.

    Upstream queries are authored per grammar and are not uniform. A language
    without reference captures yields no edges, so `references()` returning
    empty means "not supported here", not "no matches".
    """

    lang: str
    definitions: bool
    #: The `name.reference.<kind>` kinds the query captures -- "call",
    #: "type", "class", whatever the grammar's authors chose. Held as
    #: the set rather than a bool because the kinds are not
    #: interchangeable: a query capturing type annotations and nothing
    #: else supports `references()` and cannot answer "who calls this".
    reference_kinds: frozenset[str] = frozenset()

    @property
    def references(self) -> bool:
        """Any reference capture at all."""
        return bool(self.reference_kinds)

    @property
    def calls(self) -> bool:
        """Call sites specifically, which is what `refs` is asked for."""
        return bool(self.reference_kinds & CALL_KINDS)
