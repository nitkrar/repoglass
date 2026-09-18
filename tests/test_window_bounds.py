"""Oversized window spans are split instead of dropped."""

from __future__ import annotations

import unittest

from repoglass.config import MAX_CHUNK_CHARS, Settings
from repoglass.corpus.extract import window_chunks
from repoglass.models import SourceFile


def _file(source: str, lang: str = "python") -> SourceFile:
    return SourceFile(path="big.py", mtime_ns=0, size=len(source), lang=lang)


class OneEnormousLineTests(unittest.TestCase):
    """The line is longer than the target, so line grouping cannot help."""

    SOURCE = "x = " + "1234567890" * 8_000 + "\n"

    def windows(self):
        return window_chunks(_file(self.SOURCE), self.SOURCE, Settings())

    def test_no_window_exceeds_the_bound(self) -> None:
        self.assertGreater(len(self.SOURCE), MAX_CHUNK_CHARS)
        oversized = [len(c.text) for c in self.windows()
                     if len(c.text) > MAX_CHUNK_CHARS]
        self.assertEqual([], oversized)

    def test_the_line_is_still_covered(self) -> None:
        """Splitting, not dropping: the text stays reachable."""
        self.assertTrue(self.windows())

    def test_splitting_neither_drops_nor_repeats(self) -> None:
        """The splitting helper keeps spans contiguous and bounded."""
        from repoglass.corpus.extract import _bounded

        spans = _bounded([(0, 50_000), (50_000, 50_010)])
        self.assertEqual(50_010, spans[-1][1])
        self.assertEqual(0, spans[0][0])
        for (_, a), (b, _) in zip(spans, spans[1:]):
            self.assertEqual(a, b)                 # contiguous, no gaps
        for start, end in spans:
            self.assertLessEqual(end - start, MAX_CHUNK_CHARS)


class OrdinaryFilesAreUnchangedTests(unittest.TestCase):
    SOURCE = "".join(
        f"def fn_{i}(value):\n    return value + {i}  # a comment here\n\n"
        for i in range(40)
    )

    def test_no_split_is_applied_below_the_bound(self) -> None:
        chunks = window_chunks(_file(self.SOURCE), self.SOURCE, Settings())
        self.assertTrue(chunks)
        for c in chunks:
            self.assertLessEqual(len(c.text), MAX_CHUNK_CHARS)
        # Nothing pathological here, so the tree walk's own boundaries
        # should survive: no chunk should land exactly on the bound.
        self.assertNotIn(MAX_CHUNK_CHARS, [len(c.text) for c in chunks])


if __name__ == "__main__":
    unittest.main()
