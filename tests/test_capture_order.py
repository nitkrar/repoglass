"""Extraction must not depend on the order tree-sitter returns captures in.

py-tree-sitter's `captures()` orders nodes by memory layout, which varies
between processes, so two builds of one tree disagreed on chunk ids,
`enclosing` names and exact-tie rankings.
"""

from __future__ import annotations

import unittest
from unittest import mock

import tree_sitter

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

#: Two definitions sharing one span (`let x = ..., y = ...`), nested
#: definitions, and references inside each.
SWIFT = """\
struct Point {
    let x = seed(1), y = seed(2)
    func length() -> Double {
        return hypot(Double(x), Double(y))
    }
    func scaled(by factor: Double) -> Point {
        return Point(x: scale(x, factor), y: scale(y, factor))
    }
}
"""

PYTHON = """\
def outer(a):
    def inner(b):
        return helper(b) + helper(a)
    return inner(a) + helper(a)

class Thing:
    def method(self):
        return outer(self.value) + helper(self.other)

def helper(value):
    return value * 2 + len(str(value))
"""


def _extract(lang: str, path: str, source: str):
    f = SourceFile(path=path, mtime_ns=1, size=len(source), lang=lang)
    result = extract.extract(f, source, Settings())
    symbols = [(s.name, s.tag, s.start_line, s.end_line, s.enclosing, s.signature)
               for s in result.symbols]
    chunks = [(c.name, c.start_line, c.end_line, c.content_hash) for c in result.chunks]
    return symbols, chunks


class _Reversed:
    """A QueryCursor whose captures come back in reverse: keys and nodes."""

    def __init__(self, query):
        self._inner = _REAL(query)

    def captures(self, node):
        got = self._inner.captures(node)
        return {k: list(reversed(got[k])) for k in reversed(list(got))}


_REAL = tree_sitter.QueryCursor


class CaptureOrderTests(unittest.TestCase):
    def test_extraction_ignores_capture_order(self) -> None:
        for lang, path, source in (("swift", "a.swift", SWIFT), ("python", "a.py", PYTHON)):
            with self.subTest(lang=lang):
                natural = _extract(lang, path, source)
                with mock.patch.object(tree_sitter, "QueryCursor", _Reversed):
                    shuffled = _extract(lang, path, source)
                self.assertEqual(natural, shuffled)


if __name__ == "__main__":
    unittest.main()
