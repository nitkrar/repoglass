"""Extraction against a hand-counted fixture.

Counts are exact, not non-zero. A grammar or query upgrade that silently
stops capturing presents as success under a non-zero assertion.
See tests/fixtures/python/EXPECTED.md.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

FIXTURE = Path(__file__).parent / "fixtures" / "python" / "sample.py"

EXPECTED_DEFS = {
    "top_level",
    "outer",
    "inner",      # nested inside outer
    "helper",
    "Thing",
    "method",
    "tiny",       # short span: symbol, but no chunk
}


def _source_file() -> SourceFile:
    st = FIXTURE.stat()
    return SourceFile(
        path="sample.py",
        mtime_ns=st.st_mtime_ns,
        size=st.st_size,
        lang="python",
    )


class PythonExtractionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.result = extract.extract(
            _source_file(), FIXTURE.read_text(), Settings()
        )

    def test_captures_exactly_the_expected_definitions(self) -> None:
        names = {s.name for s in self.result.symbols if s.tag == "def"}
        self.assertEqual(EXPECTED_DEFS, names)

    def test_nested_function_is_captured(self) -> None:
        """`inner` is defined inside `outer`; both must appear."""
        names = {s.name for s in self.result.symbols if s.tag == "def"}
        self.assertIn("inner", names)
        self.assertIn("outer", names)

    def test_definition_spans_are_one_indexed_and_ordered(self) -> None:
        for s in self.result.symbols:
            self.assertGreaterEqual(s.start_line, 1, s.name)
            self.assertGreaterEqual(s.end_line, s.start_line, s.name)


class EnclosingScopeTests(unittest.TestCase):
    """The innermost enclosing definition owns a reference."""

    def setUp(self) -> None:
        self.result = extract.extract(
            _source_file(), FIXTURE.read_text(), Settings()
        )

    def test_reference_inside_nested_function_resolves_to_innermost(self) -> None:
        """`helper()` inside `inner` belongs to `inner`, not `outer`."""
        refs = [r for r in self.result.refs if r.name == "helper"]
        enclosing = {r.enclosing.name for r in refs if r.enclosing is not None}
        self.assertIn("inner", enclosing)
        self.assertNotIn("outer", enclosing)

    def test_reference_inside_method_resolves_to_method(self) -> None:
        """`helper()` inside `Thing.method` belongs to `method`, not `Thing`."""
        refs = [r for r in self.result.refs if r.name == "helper"]
        enclosing = {r.enclosing.name for r in refs if r.enclosing is not None}
        self.assertIn("method", enclosing)
        self.assertNotIn("Thing", enclosing)

    def test_module_scope_reference_has_no_enclosing(self) -> None:
        """`top_level()` on the last line sits inside no definition."""
        refs = [r for r in self.result.refs if r.name == "top_level"]
        self.assertTrue(refs, "expected a module-scope call to top_level")
        self.assertTrue(any(r.enclosing is None for r in refs))


if __name__ == "__main__":
    unittest.main()
