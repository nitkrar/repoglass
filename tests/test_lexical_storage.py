"""Default lexical text is derived, not stored per chunk."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from repoglass import Index
from repoglass.config import Paths, Settings
from repoglass.corpus import extract


class LexicalStorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "payments").mkdir(parents=True)
        (self.root / "payments" / "refundEngine.py").write_text(
            "def issue_refund(order):\n"
            "    # reverses a settled charge\n"
            "    return order.total\n"
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def open(self, settings: Settings | None = None) -> Index:
        idx = Index.open(
            self.root, settings or Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        idx.refresh()
        return idx


class DerivedByDefaultTests(LexicalStorageTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.idx = self.open()
        self.db = self.idx._paths.db

    def test_no_column_holds_the_whole_lexical_text(self) -> None:
        cols = {r[1] for r in
                self.idx._store.conn.execute("PRAGMA table_info(chunk)")}
        self.assertNotIn("lexical_text", cols)

    def test_the_path_words_are_stored_once_per_file(self) -> None:
        got = self.idx._store.conn.execute(
            "SELECT path_words FROM file WHERE path LIKE '%refundEngine%'"
        ).fetchone()[0]
        self.assertEqual("payments refund Engine py", got)

    def test_nothing_is_stored_per_chunk(self) -> None:
        n = self.idx._store.conn.execute(
            "SELECT count(*) FROM chunk WHERE lexical_override IS NOT NULL"
        ).fetchone()[0]
        self.assertEqual(0, n)

    def test_a_plain_connection_can_read_the_index(self) -> None:
        """No registered function, no import of this package."""
        self.idx._store.conn.close()
        conn = sqlite3.connect(self.db)
        rows = conn.execute(
            "SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'refund'"
        ).fetchone()[0]
        self.assertGreater(rows, 0)

    def test_the_index_agrees_with_its_content_source(self) -> None:
        """FTS5's own check: re-derive from the view and compare."""
        self.idx._store.conn.execute(
            "INSERT INTO chunk_fts(chunk_fts, rank) VALUES('integrity-check', 1)")

    def test_camel_case_in_the_path_is_searchable(self) -> None:
        """unicode61 would leave `refundEngine` as one token; the
        humanised path is what splits it."""
        hits = self.idx.search("refund engine", k=5)
        self.assertTrue(any("refundEngine" in h.path for h in hits))

    def test_the_body_is_searchable(self) -> None:
        hits = self.idx.search("reverses a settled charge", k=5)
        self.assertTrue(any("refundEngine" in h.path for h in hits))


class NonDerivableRenderingTests(LexicalStorageTestCase):
    """Settings that change lexical text store an override per chunk."""

    def test_an_enriched_rendering_is_stored(self) -> None:
        idx = self.open(Settings(embed_backend="none", lexical_enrich=True))
        stored = idx._store.conn.execute(
            "SELECT lexical_override FROM chunk WHERE lexical_override IS NOT NULL"
        ).fetchall()
        self.assertTrue(stored)

    def test_what_is_stored_is_what_the_renderer_produces(self) -> None:
        s = Settings(embed_backend="none", lexical_enrich=True)
        idx = self.open(s)
        path, text, override = idx._store.conn.execute(
            "SELECT f.path, c.text, c.lexical_override FROM chunk c"
            " JOIN file f ON f.id = c.file_id"
            " WHERE c.lexical_override IS NOT NULL LIMIT 1"
        ).fetchone()
        self.assertEqual(extract.lexical(path=path, body=text, settings=s),
                         override)

    def test_it_still_reads_without_a_registered_function(self) -> None:
        idx = self.open(Settings(embed_backend="none", lexical_enrich=True))
        db = idx._paths.db
        idx._store.conn.close()
        sqlite3.connect(db).execute(
            "SELECT count(*) FROM chunk_fts WHERE chunk_fts MATCH 'refund'")


if __name__ == "__main__":
    unittest.main()
