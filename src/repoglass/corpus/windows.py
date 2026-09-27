"""Whole-file window chunking."""

from __future__ import annotations

from ..config import MAX_CHUNK_CHARS, MIN_CHUNK_CHARS, Settings
from ..models import Chunk, SourceFile
from .render import lexical_override



def window_chunks(file: SourceFile, source: str, settings: Settings) -> list[Chunk]:
    """Split a whole file into size-bounded, syntax-aligned windows.

    A language with a tags query is chunked by semsift's
    LanguagePackChunker, which walks the parse tree grouping siblings up
    to `window_chars` and merging neighbours back towards it. Any other
    language groups whole lines.

    Two properties the definition chunker lacks, and both are the point:

    * **Every byte is covered.** Module docstrings, imports and
      top-level statements are owned by no definition, so without this
      they are unreachable by any query.
    * **Chunks are of comparable size.** The file-coherence boost sums
      a file's candidate scores, which misbehaves when one chunk is a
      3-line getter and the next a 200-line class.
    """
    from semsift.chunk import LanguagePackChunker

    from . import languages

    if not source.strip():
        return []
    if languages.compiled_query(file.lang) is not None:
        chunker = LanguagePackChunker(target_bytes=settings.window_chars,
                                      max_bytes=MAX_CHUNK_CHARS, min_chars=MIN_CHUNK_CHARS)
        return [_window(file, c.text, c.start_line, c.end_line, settings)
                for c in chunker.chunk(source, file.lang)]
    data = source.encode()
    starts = [i for i, b in enumerate(data) if b == 0x0A]
    out: list[Chunk] = []
    for start, end in _bounded(_line_spans(data, settings.window_chars)):
        body = data[start:end].decode(errors="ignore")
        if len(body.strip()) < MIN_CHUNK_CHARS:
            continue
        out.append(_window(file, body, _line_of(starts, start),
                           _line_of(starts, max(end - 1, start)), settings))
    return out


def _window(file: SourceFile, body: str, first: int, last: int,
            settings: Settings) -> Chunk:
    return Chunk.build(
        path=file.path, name="", start_line=first, end_line=last,
        source=body, leading_comments=[], node_kind="window", text=body,
        lexical_override=lexical_override(path=file.path, body=body, settings=settings),
    )


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
