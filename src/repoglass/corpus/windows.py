"""Whole-file window chunking."""

from __future__ import annotations

from ..config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, Settings
from ..models import Chunk, SourceFile
from .render import lexical_override

#: Below this, a node is not worth descending into; emit it whole.
_MIN_WINDOW_NODE = 50
_MAX_DEPTH = 500


def window_chunks(file: SourceFile, source: str, settings: Settings) -> list[Chunk]:
    """Split a whole file into size-bounded, syntax-aligned windows.

    Walk the tree grouping adjacent siblings until the next one would
    exceed `window_chars`, recursing into any single child that is
    already too big, then merge neighbouring groups back up to the
    target. Falls back to grouping whole lines for a language with no
    tags query, and on a parse error.

    Two properties the definition chunker lacks, and both are the point:

    * **Every byte is covered.** Module docstrings, imports and
      top-level statements are owned by no definition, so without this
      they are unreachable by any query.
    * **Chunks are of comparable size.** The file-coherence boost sums
      a file's candidate scores, which misbehaves when one chunk is a
      3-line getter and the next a 200-line class.
    """
    from grep_ast.tsl import get_parser

    from . import languages

    if not source.strip():
        return []
    target = settings.window_chars
    spans: list[tuple[int, int]] | None = None
    if languages.compiled_query(file.lang) is not None:
        try:
            tree = get_parser(file.lang).parse(source.encode())
            spans = _merge_adjacent(_split_node(tree.root_node, target, 0), target)
        except Exception:       # no grammar for this platform, or a parse error
            spans = None
    data = source.encode()
    if spans is None:
        spans = _line_spans(data, target)
    spans = _bounded(spans)

    starts = [i for i, b in enumerate(data) if b == 0x0A]
    out: list[Chunk] = []
    for start, end in spans:
        body = data[start:end].decode(errors="ignore")
        if len(body.strip()) < MIN_CHUNK_CHARS:
            continue
        first = _line_of(starts, start)
        last = _line_of(starts, max(end - 1, start))
        out.append(
            Chunk.build(
                path=file.path, name="", start_line=first, end_line=last,
                source=body, leading_comments=[], node_kind="window",
                text=body,
                lexical_override=lexical_override(
                    path=file.path, body=body, settings=settings),
            )
        )
    return out


def _bounded(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Cut any span still over MAX_CHUNK_CHARS into pieces that fit.

    `window_chars` is what the tree walk aims for, not a bound it can
    promise: a node it cannot recurse into, or a single line longer
    than the target, comes back whole -- and a generated or minified
    file can make that arbitrarily large.

    Splitting rather than dropping, because a window is the coverage
    fallback and has no name to lose. Byte offsets, so a cut can land
    mid-character; the caller decodes with errors="ignore", which drops
    the partial rather than raising.
    """
    out: list[tuple[int, int]] = []
    for start, end in spans:
        while end - start > MAX_CHUNK_CHARS:
            out.append((start, start + MAX_CHUNK_CHARS))
            start += MAX_CHUNK_CHARS
        out.append((start, end))
    return out


def _line_of(newline_offsets: list[int], offset: int) -> int:
    from bisect import bisect_left

    return bisect_left(newline_offsets, offset) + 1


def _split_node(node, target: int, depth: int) -> list[tuple[int, int]]:
    """Group a node's children into spans, aiming at `target`.

    A node with no children, or one hit at `_MAX_DEPTH`, comes back
    whole however large it is -- which is why `_bounded` runs after.
    """
    if not node.children or depth > _MAX_DEPTH:
        return [(node.start_byte, node.end_byte)]
    if node.end_byte - node.start_byte < _MIN_WINDOW_NODE:
        return [(node.start_byte, node.end_byte)]

    groups: list[tuple[int, int]] = []
    children = node.children
    i = 0
    while i < len(children):
        child = children[i]
        start, end = child.start_byte, child.end_byte
        size = end - start
        i += 1
        if size > target:
            # Too big to be one window: split it instead of emitting it.
            groups.extend(_split_node(child, target, depth + 1))
            continue
        while i < len(children):
            nxt = children[i]
            if size + (nxt.end_byte - nxt.start_byte) > target:
                break
            end = nxt.end_byte
            size += nxt.end_byte - nxt.start_byte
            i += 1
        groups.append((start, end))
    return groups


def _merge_adjacent(spans: list[tuple[int, int]], target: int) -> list[tuple[int, int]]:
    """Coalesce neighbouring spans back up to the target size.

    Splitting alone leaves many small spans, because a node's children
    are often individually tiny. Merging afterwards is what makes the
    output uniform rather than merely bounded.
    """
    if not spans:
        return []
    out: list[tuple[int, int]] = []
    start, end = spans[0]
    for nxt_start, nxt_end in spans[1:]:
        if (end - start) + (nxt_end - nxt_start) > target:
            out.append((start, end))
            start, end = nxt_start, nxt_end
            continue
        end = nxt_end
    out.append((start, end))
    return out


def _line_spans(data: bytes, target: int) -> list[tuple[int, int]]:
    """Fallback when the tree walk is unavailable: group whole lines."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i, b in enumerate(data):
        if b != 0x0A:
            continue
        if i + 1 - start >= target:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(data):
        spans.append((start, len(data)))
    return spans
