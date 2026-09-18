"""Tree-sitter queries are compiled once per language."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import tree_sitter

from repoglass.config import Paths, Settings
from repoglass.corpus import extract, languages
from repoglass.models import SourceFile


class QueryCompilationTests(unittest.TestCase):
    def setUp(self) -> None:
        languages.compiled_query.cache_clear()
        self.calls = 0
        self.real = tree_sitter.Query

        outer = self
        real = self.real

        def counting(*a, **k):
            # A wrapper, not a subclass: tree_sitter.Query is a C type
            # and cannot be subclassed.
            outer.calls += 1
            return real(*a, **k)

        tree_sitter.Query = counting

    def tearDown(self) -> None:
        tree_sitter.Query = self.real
        languages.compiled_query.cache_clear()

    @staticmethod
    def _file(path: str) -> SourceFile:
        return SourceFile(path=path, mtime_ns=1, size=1, lang="python")

    def test_many_files_of_one_language_compile_the_query_once(self) -> None:
        body = "def alpha():\n    return 'a value long enough to chunk here'\n"
        for i in range(5):
            extract.extract(self._file(f"m{i}.py"), body, Settings())
        self.assertEqual(1, self.calls)

    def test_a_second_language_compiles_its_own(self) -> None:
        extract.extract(self._file("a.py"),
                        "def alpha():\n    return 'long enough value here'\n",
                        Settings())
        md = SourceFile(path="a.md", mtime_ns=1, size=1, lang="markdown")
        extract.extract(md, "# Heading\n\nSome prose that is long enough.\n",
                        Settings())
        self.assertEqual(2, self.calls)

    def test_the_cache_returns_the_same_object(self) -> None:
        self.assertIs(languages.compiled_query("python"),
                      languages.compiled_query("python"))

    def test_an_unknown_language_yields_none_without_compiling(self) -> None:
        self.assertIsNone(languages.compiled_query("not-a-language"))
        self.assertEqual(0, self.calls)

class ExtractionStillWorks(unittest.TestCase):
    """The cache must not change what extraction produces.

    A shared Query is only safe because QueryCursor holds the per-run
    state; if that were not true, results would bleed between files and
    this is where it would show.
    """

    def test_repeated_extraction_is_identical(self) -> None:
        body = ("def first():\n    return 'a value long enough to chunk'\n\n"
                "def second():\n    return 'another value long enough'\n")
        f = SourceFile(path="a.py", mtime_ns=1, size=1, lang="python")
        runs = [extract.extract(f, body, Settings()) for _ in range(3)]
        names = [sorted(s.name for s in r.symbols) for r in runs]
        self.assertEqual(names[0], names[1])
        self.assertEqual(names[1], names[2])
        self.assertIn("first", names[0])

    def test_different_files_do_not_bleed(self) -> None:
        s = Settings()
        a = extract.extract(
            SourceFile(path="a.py", mtime_ns=1, size=1, lang="python"),
            "def only_in_a():\n    return 'a value long enough to chunk'\n", s)
        b = extract.extract(
            SourceFile(path="b.py", mtime_ns=1, size=1, lang="python"),
            "def only_in_b():\n    return 'a value long enough to chunk'\n", s)
        self.assertIn("only_in_a", {x.name for x in a.symbols})
        self.assertNotIn("only_in_a", {x.name for x in b.symbols})


if __name__ == "__main__":
    unittest.main()
