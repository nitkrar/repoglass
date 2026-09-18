"""Languages without tags queries stay searchable through window chunks."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from repoglass.config import Paths, Settings
from repoglass.corpus import discovery, extract
from repoglass.models import SourceFile

SQL = """\
-- Revenue by region for the quarterly board pack.
CREATE TABLE regional_revenue AS
SELECT region, sum(amount) AS total
  FROM orders
 WHERE placed_at >= '2026-01-01'
 GROUP BY region
 ORDER BY total DESC;
"""


def _sql_file(source: str = SQL) -> SourceFile:
    return SourceFile(path="report.sql", mtime_ns=0,
                      size=len(source), lang="sql")


class WalkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text("def a(): pass\n")
        (self.root / "report.sql").write_text(SQL)
        self.paths = Paths(root=self.root, home=Path(self.tmp.name) / "home")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_language_without_a_tags_query_is_walked(self) -> None:
        walked = {f.path for f in discovery.walk(self.paths, Settings())}
        self.assertEqual({"a.py", "report.sql"}, walked)

    def test_file_with_no_language_is_still_skipped(self) -> None:
        (self.root / "photo.png").write_bytes(b"\x89PNG\r\n")
        (self.root / ".env").write_text("SECRET=1\n")
        walked = {f.path for f in discovery.walk(self.paths, Settings())}
        self.assertEqual({"a.py", "report.sql"}, walked)

    def test_secret_bearing_languages_are_never_indexed(self) -> None:
        """Discovery can still refuse a language with a grammar."""
        (self.root / "key.pem").write_text(
            "-----BEGIN PRIVATE KEY-----\nMIIEv\n-----END PRIVATE KEY-----\n")
        walked = {f.path for f in discovery.walk(self.paths, Settings())}
        self.assertEqual({"a.py", "report.sql"}, walked)


class ExtractTests(unittest.TestCase):
    def test_hybrid_yields_windows(self) -> None:
        result = extract.extract(_sql_file(), SQL, Settings())
        self.assertTrue(result.chunks)
        # Windows carry no name; every named chunk comes from a definition.
        self.assertTrue(all(c.name == "" for c in result.chunks))

    def test_no_symbols_and_no_refs(self) -> None:
        result = extract.extract(_sql_file(), SQL, Settings())
        self.assertEqual((), result.symbols)
        self.assertEqual((), result.refs)

    def test_the_text_is_searchable(self) -> None:
        result = extract.extract(_sql_file(), SQL, Settings())
        self.assertIn("regional_revenue",
                      "\n".join(c.text for c in result.chunks))

    def test_definition_coverage_yields_nothing(self) -> None:
        """No definitions exist to chunk, and windows are hybrid's job."""
        settings = replace(Settings(), coverage="definition")
        self.assertEqual((), extract.extract(_sql_file(), SQL, settings).chunks)


if __name__ == "__main__":
    unittest.main()
