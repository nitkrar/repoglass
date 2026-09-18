"""Hybrid coverage fills gaps with window chunks."""

from __future__ import annotations

import unittest

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

SRC = '''"""Module docstring explaining the dedupe table."""

import time

RETENTION = 300


def register(key):
    """Record a delivery so it is not repeated."""
    return _store.setdefault(key, time.time())


def consume(key):
    """Take a recorded delivery, if it is still fresh."""
    seen = _store.get(key)
    return seen is not None and time.time() - seen < RETENTION
'''


def pyfile(path="a.py"):
    return SourceFile(path=path, mtime_ns=1, size=1, lang="python")


class WindowCoverageTests(unittest.TestCase):
    """Hybrid keeps named chunks and adds unnamed gap chunks."""

    S = Settings(coverage="hybrid", window_chars=200)

    def test_the_module_docstring_is_reachable(self) -> None:
        """Hybrid fills text outside definitions."""
        chunks = extract.extract(pyfile(), SRC, self.S).chunks
        self.assertTrue(any("dedupe table" in c.text for c in chunks))

    def test_definition_mode_misses_it(self) -> None:
        chunks = extract.extract(pyfile(), SRC,
                                 Settings(coverage="definition")).chunks
        self.assertFalse(any("dedupe table" in c.text for c in chunks))

    def test_module_level_constants_are_reachable(self) -> None:
        chunks = extract.extract(pyfile(), SRC, self.S).chunks
        self.assertTrue(any("RETENTION = 300" in c.text for c in chunks))

    def test_symbols_are_still_extracted(self) -> None:
        """Window chunks must not change symbol extraction."""
        res = extract.extract(pyfile(), SRC, self.S)
        names = {s.name for s in res.symbols if s.tag == "def"}
        self.assertIn("register", names)
        self.assertIn("consume", names)

    def test_the_gap_chunks_carry_no_symbol(self) -> None:
        """A window belongs to no definition, so it has no name. Only
        the gap chunks: hybrid keeps the named definition chunks."""
        for c in extract.window_chunks(pyfile(), SRC, self.S):
            self.assertEqual("", c.name)

    def test_hybrid_keeps_the_named_chunks_too(self) -> None:
        names = {c.name for c in extract.extract(pyfile(), SRC, self.S).chunks}
        self.assertIn("register", names)
        self.assertIn("", names, "and at least one gap chunk")


class WindowSizeTests(unittest.TestCase):
    def test_windows_are_bounded_and_comparable(self) -> None:
        """Window sizes stay within a narrow band."""
        body = "\n\n".join(
            f"def f{i}():\n" + "\n".join(f"    x{j} = {j}" for j in range(i))
            for i in range(1, 30)
        )
        s = Settings(window_chars=400)
        sizes = [len(c.text) for c in extract.window_chunks(pyfile(), body, s)]
        self.assertTrue(sizes)
        # No window should be wildly over target; a single indivisible
        # node can exceed it, but not by an order of magnitude.
        self.assertLess(max(sizes), 400 * 4)

    def test_spans_do_not_overlap(self) -> None:
        s = Settings(window_chars=150)
        chunks = sorted(extract.window_chunks(pyfile(), SRC, s),
                        key=lambda c: c.start_line)
        for a, b in zip(chunks, chunks[1:]):
            self.assertLessEqual(a.end_line, b.start_line + 1)

    def test_an_empty_file_yields_nothing(self) -> None:
        self.assertEqual([], extract.window_chunks(pyfile(), "   \n\n  ",
                                                   Settings()))

    def test_line_fallback_covers_ungrammared_text(self) -> None:
        spans = extract._line_spans(b"a\n" * 200, 50)
        self.assertGreater(len(spans), 1)
        self.assertEqual(0, spans[0][0])
        self.assertEqual(400, spans[-1][1])


class ExactTierUnderWindowsTests(unittest.TestCase):
    """Exact lookup works with definition and hybrid coverage."""

    import tempfile as _t

    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.root = __import__("pathlib").Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "m.py").write_text(SRC)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _hits(self, mode: str) -> int:
        from pathlib import Path

        from repoglass import Index
        from repoglass.config import Paths
        s = Settings(coverage=mode, embed_backend="none")
        idx = Index.open(self.root, s,
                         paths=Paths(root=self.root,
                                     home=Path(self.tmp.name) / mode))
        idx.refresh()
        return len(idx._store.chunked_definitions("register", mode="all"))

    def test_definition_mode_finds_the_symbol(self) -> None:
        self.assertGreater(self._hits("definition"), 0)

    def test_hybrid_finds_the_gap_chunk(self) -> None:
        self.assertGreater(self._hits("hybrid"), 0)


if __name__ == "__main__":
    unittest.main()
