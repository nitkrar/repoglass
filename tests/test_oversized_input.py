"""Discovery and extraction bound pathological input sizes."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from repoglass.config import MAX_CHUNK_CHARS, Paths, Settings
from repoglass.corpus import discovery, extract
from repoglass.models import SourceFile


class MaxFileBytesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "small.py").write_text("def a(): pass\n")
        self.paths = Paths(root=self.root, home=Path(self.tmp.name) / "home")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def walked(self, settings: Settings | None = None) -> set[str]:
        return {f.path for f in discovery.walk(self.paths, settings or Settings())}

    def test_file_over_the_limit_is_not_walked(self) -> None:
        (self.root / "bundle.js").write_text("var x=1;\n" * 200_000)
        self.assertEqual({"small.py"}, self.walked())

    def test_file_under_the_limit_is_walked(self) -> None:
        (self.root / "bundle.js").write_text("var x=1;\n")
        self.assertEqual({"small.py", "bundle.js"}, self.walked())

    def test_limit_is_configurable(self) -> None:
        (self.root / "mid.py").write_text("x = 1\n" * 1000)
        settings = replace(Settings(), max_file_bytes=100)
        self.assertEqual({"small.py"}, self.walked(settings))


class MaxChunkCharsTests(unittest.TestCase):
    """A file under the size limit whose lines are still pathological."""

    def _extract(self, source: str):
        file = SourceFile(path="min.js", mtime_ns=0,
                          size=len(source), lang="javascript")
        return extract.extract(file, source, Settings())

    def test_oversized_chunk_is_dropped(self) -> None:
        body = "a = 1; " * (MAX_CHUNK_CHARS // 4)
        source = f"function huge() {{ {body} }}\n"
        self.assertGreater(len(source), MAX_CHUNK_CHARS)
        result = self._extract(source)
        self.assertEqual([], [c for c in result.chunks
                              if len(c.text) > MAX_CHUNK_CHARS])

    def test_oversized_definition_stays_navigable(self) -> None:
        """Symmetric with MIN_CHUNK_CHARS: no chunk, but still a symbol."""
        body = "a = 1; " * (MAX_CHUNK_CHARS // 4)
        source = f"function huge() {{ {body} }}\n"
        result = self._extract(source)
        self.assertIn("huge", {s.name for s in result.symbols})

    def test_oversized_definition_leaves_no_hole_under_hybrid(self) -> None:
        """The bound must be applied before coverage, not after it.

        Applied after, `hybrid_chunks` has already suppressed the gap
        windows over the definition's lines as redundant, so dropping the
        definition leaves the region in no chunk at all.
        """
        body = "a = 1; " * (MAX_CHUNK_CHARS // 4)
        source = f"function huge() {{ {body} }}\n"
        result = self._extract(source)
        covered = {n for c in result.chunks
                   for n in range(c.start_line, c.end_line + 1)}
        self.assertEqual(set(range(1, len(source.splitlines()) + 1)), covered)

    def test_normal_definition_is_unaffected(self) -> None:
        source = ("function ordinary(first, second) {\n"
                  "  const total = compute(first, second);\n"
                  "  return format(total, {precision: 2});\n"
                  "}\n")
        result = self._extract(source)
        self.assertEqual(1, len(result.chunks))


if __name__ == "__main__":
    unittest.main()
