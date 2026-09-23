"""Parse a file into symbols, spans and chunks.

`Chunk.text` is what gets embedded: the raw span, unless `distill_docs`
says otherwise. FTS5 indexes the same span with the humanised path in
front, derived through a view rather than stored a second time -- see
`render.lexical_override` for when that derivation does not hold.

One path for all languages. Markdown is a language whose tag query captures
sections rather than functions; there is no separate prose pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS
from ..models import Chunk, SourceFile, Symbol
from .render import _enriched, embed_text, lexical, lexical_override
from .windows import _bounded, _line_spans, window_chunks


@dataclass(frozen=True)
class Extraction:
    symbols: tuple[Symbol, ...]
    chunks: tuple[Chunk, ...]
    refs: tuple["RawRef", ...]


#: A signature longer than this is not a signature. A minified file
#: puts a whole bundle on one line, and the body boundary does not
#: help when there are no line breaks to stop at.
MAX_SIGNATURE_CHARS = 500


def _signature(node, raw: bytes) -> str | None:
    """A definition's header: everything before its body.

    `body` is a named field in every grammar that has one, so the cut
    is the same in Python, Go, Rust and JavaScript without a rule per
    language. A definition with no body -- a markdown heading, an
    interface member -- has no header to take, so its first line
    stands in.
    """
    if node is None:
        return None
    body = node.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    text = raw[node.start_byte:end].decode(errors="ignore").strip()
    if body is None:
        text = text.splitlines()[0].strip() if text else ""
    return text[:MAX_SIGNATURE_CHARS] or None


def _span(node) -> tuple[int, int, int, int]:
    """(start_byte, end_byte, start_line, end_line) for a definition.

    A markdown `section` holds its subsections, so the node for `# Top`
    runs to the end of the last `###` beneath it. Taken whole, every
    heading's chunk repeats all the prose under it and a nested
    paragraph is indexed once per level above it. The span therefore
    ends where the first nested section begins.

    Only `section` nests inside itself in the queries shipped here. A
    class holding a method is composition -- the method is part of what
    the class is -- and its span stays whole.
    """
    end, end_line = node.end_byte, node.end_point[0] + 1
    if node.type == "section":
        for child in node.children:
            if child.type == "section":
                end, end_line = child.start_byte, child.start_point[0]
                break
    return node.start_byte, end, node.start_point[0] + 1, end_line


@dataclass(frozen=True)
class RawRef:
    """A reference before edge resolution.

    `enclosing` is the innermost definition whose span contains it, or None
    for module scope -- a top-level call, an import, a decorator.
    """

    name: str
    line: int
    enclosing: Symbol | None


def extract(file: SourceFile, source: str, settings: Settings) -> Extraction:
    """Run the language's tag query over one file.

    A chunk's span is the `@definition.*` node, capped at
    `max_chunk_lines` and, for a markdown section, cut where its first
    subsection begins. See `_span`.
    """
    from grep_ast.tsl import get_parser
    from tree_sitter import QueryCursor

    from . import languages

    # No query means no symbols and no references, not no chunks: the
    # coverage function still runs, and under hybrid the file is covered
    # by windows. Navigation needs the query; retrieval does not.
    query = languages.compiled_query(file.lang)
    captures: dict = {}
    if query is not None:
        tree = get_parser(file.lang).parse(source.encode())
        captures = QueryCursor(query).captures(tree.root_node)

    # Definition spans, smallest first so containment lookups find the
    # innermost match.
    spans = sorted(
        (
            _span(n)
            for key, nodes in captures.items()
            if key.startswith("definition.")
            for n in nodes
        ),
        key=lambda s: s[1] - s[0],
    )
    # Kept so a definition's signature can be cut at its body. The
    # spans above carry only offsets, and the boundary is a field on
    # the node.
    def_nodes = {
        _span(n)[:2]: n
        for key, nodes in captures.items()
        if key.startswith("definition.")
        for n in nodes
    }
    raw = source.encode()

    def innermost(start: int, end: int):
        for s in spans:
            if s[0] <= start and end <= s[1]:
                return s
        return None

    lines = source.splitlines()
    symbols: list[Symbol] = []
    # Definitions worth a chunk, collected here and turned into chunks
    # afterwards by the chosen coverage function. Symbols are extracted
    # either way: navigation and the exact tier read the symbol table,
    # not chunks, so coverage changes what is retrievable without
    # changing what is navigable.
    defs: list[_Def] = []
    by_span: dict[tuple[int, int], Symbol] = {}
    #: Byte ranges of the nodes that name a definition. A grammar can
    #: match one of these under a reference pattern too -- Go captures
    #: every `type_identifier`, including the one naming the type --
    #: and a definition is not a use of itself.
    named_here: set[tuple[int, int]] = set()
    #: Nodes a query marked `@ignore`. A pattern cannot cancel the match
    #: another pattern makes on the same node, so a query ruling an
    #: identifier out has to do it from a pattern of its own.
    ignored: set[tuple[int, int]] = {
        (n.start_byte, n.end_byte) for n in captures.get("ignore", ())
    }
    # Sorted, so which capture claims a node shared by two patterns is
    # the same on every run rather than whatever order the query
    # happened to return.
    for key in sorted(captures):
        if not key.startswith("name.definition."):
            continue
        node_kind = key.rsplit(".", 1)[-1]
        for n in captures[key]:
            # One node, one definition. A grammar can match the same
            # node under two patterns -- Go captures a struct as both
            # a type and a class -- and a second symbol for it is a
            # definition that does not exist.
            if (n.start_byte, n.end_byte) in named_here:
                continue
            span = innermost(n.start_byte, n.end_byte)
            if span is None:
                continue
            start, end = span[2], span[3]
            sym = Symbol(
                name=n.text.decode(),
                tag="def",
                path=file.path,
                start_line=start,
                end_line=end,
                lang=file.lang,
                signature=_signature(def_nodes.get((span[0], span[1])), raw),
            )
            symbols.append(sym)
            by_span[(span[0], span[1])] = sym
            named_here.add((n.start_byte, n.end_byte))

            # The chunk's span is capped; the symbol's is not. Store the
            # capped end, or readback returns source that was never indexed.
            chunk_end = min(end, start - 1 + settings.max_chunk_lines)
            # Both bounds are decided here rather than on the finished
            # chunks: hybrid coverage treats a definition's lines as
            # owned, so a definition dropped afterwards takes the gap
            # windows over those lines with it and leaves nothing.
            if MIN_CHUNK_CHARS <= len(
                "\n".join(lines[start - 1 : chunk_end])
            ) <= MAX_CHUNK_CHARS:
                defs.append(_Def(sym.name, node_kind, start, chunk_end))
            # else: navigable, not retrievable

    refs: list[RawRef] = []
    for key, nodes in captures.items():
        if not key.startswith("name.reference."):
            continue
        for n in nodes:
            if (n.start_byte, n.end_byte) in named_here | ignored:
                continue
            span = innermost(n.start_byte, n.end_byte)
            # The innermost definition containing this reference, which
            # is what `enclosing` reports.
            owner = by_span.get((span[0], span[1])) if span else None
            refs.append(
                RawRef(
                    name=n.text.decode(),
                    line=n.start_point[0] + 1,
                    enclosing=owner,
                )
            )
            symbols.append(
                Symbol(
                    name=n.text.decode(),
                    tag="ref",
                    path=file.path,
                    start_line=n.start_point[0] + 1,
                    end_line=n.end_point[0] + 1,
                    lang=file.lang,
                    enclosing=owner.name if owner else None,
                )
            )

    chunks = COVERAGE[settings.coverage](file, source, settings, lines, defs)
    return Extraction(symbols=tuple(symbols), chunks=tuple(chunks), refs=tuple(refs))


@dataclass(frozen=True)
class _Def:
    """A definition the tags query found, before it becomes a chunk.

    What every coverage function in `COVERAGE` takes, so the choice of
    strategy does not reach back into the symbol loop.
    """

    name: str
    node_kind: str
    start_line: int
    end_line: int      # capped at max_chunk_lines


def definition_chunks(file: SourceFile, source: str, settings: Settings,
                      lines: list[str], defs: list[_Def]) -> list[Chunk]:
    """One chunk per definition. Leaves module-level code in no chunk."""
    out: list[Chunk] = []
    for d in defs:
        body = "\n".join(lines[d.start_line - 1 : d.end_line])
        comments = _leading_comments(lines, d.start_line)
        out.append(Chunk.build(
            path=file.path, name=d.name,
            start_line=d.start_line, end_line=d.end_line,
            source=body, leading_comments=comments, node_kind=d.node_kind,
            text=embed_text(path=file.path, lang=file.lang,
                            node_kind=d.node_kind, leading_comments=comments,
                            body=body, settings=settings),
            lexical_override=lexical_override(path=file.path, body=body,
                                              settings=settings),
        ))
    return out


def hybrid_chunks(file: SourceFile, source: str, settings: Settings,
                  lines: list[str], defs: list[_Def]) -> list[Chunk]:
    """Definition chunks, plus windows over the lines none of them owns.

    A window that mostly repeats a definition is dropped rather than
    kept: it would add no reachable text and one more candidate
    competing for the same slot.
    """
    definitions = definition_chunks(file, source, settings, lines, defs)
    covered: set[int] = set()
    for c in definitions:
        covered.update(range(c.start_line, c.end_line + 1))
    out = list(definitions)
    for w in window_chunks(file, source, settings):
        span = range(w.start_line, w.end_line + 1)
        if sum(1 for n in span if n not in covered) * 2 >= len(span):
            out.append(w)
    return out


#: The `coverage` setting selects one of these. Peers, same signature,
#: so they can be compared on the eval at the same bar.
COVERAGE = {
    "definition": definition_chunks,
    "hybrid": hybrid_chunks,
}


_COMMENT_PREFIXES = ("#", "//", "*", "/*", "///", ";", "--")


def _leading_comments(lines: list[str], start_line: int, limit: int = 6) -> list[str]:
    """Comment lines immediately above a definition.

    They sit above the definition's span, so they are not in the chunk
    body. `embed_text` is the only thing that reads them, and only under
    `distill_docs` for a doc language -- under any other settings, and
    so by default, they reach neither the embedding nor the FTS text.
    """
    out: list[str] = []
    for i in range(max(0, start_line - 1 - limit), start_line - 1):
        stripped = lines[i].strip()
        if stripped.startswith(_COMMENT_PREFIXES):
            out.append(stripped.lstrip("/*#;- ").strip())
    return out
