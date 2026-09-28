"""Chunk text is stored once; the lexical text is derived from it."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
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


def assert_fts_reads_rendering(case: unittest.TestCase, idx: Index,
                               settings: Settings) -> None:
    """For every chunk, the text FTS5 indexes is `extract.lexical`'s."""
    conn = idx._store.conn
    rows = conn.execute(
        "SELECT c.id, f.path, k.keyword_text FROM chunk c"
        " JOIN file f ON f.id = c.file_id JOIN rg_keyword k ON k.id = c.id").fetchall()
    case.assertTrue(rows)
    texts = idx._store.items.fetch([cid for cid, _, _ in rows], {"text"})
    for cid, path, indexed in rows:
        case.assertEqual(
            extract.lexical(path=path, body=texts[cid].text, settings=settings), indexed)


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

    def test_what_fts_reads_is_the_rendered_lexical_text(self) -> None:
        assert_fts_reads_rendering(self, self.idx, Settings(embed_backend="none"))

    def test_chunk_text_is_stored_once(self) -> None:
        import random

        rng = random.Random(0)
        words = [f"w{i}" for i in range(200)]
        lines = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(8000)]
        (self.root / "notes.md").write_text("\n".join(lines) + "\n")
        self.idx.refresh()
        self.idx.close()
        conn = sqlite3.connect(self.db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        size = (conn.execute("PRAGMA page_count").fetchone()[0]
                * conn.execute("PRAGMA page_size").fetchone()[0])
        conn.close()
        text = sum(len(line) + 1 for line in lines)
        # One copy plus the inverted index; a second copy would pass 2x.
        self.assertLess(size, 2 * text)

    def test_a_plain_connection_can_read_the_index(self) -> None:
        """No registered function, no import of this package."""
        self.idx.close()
        with closing(sqlite3.connect(self.db)) as conn:
            rows = conn.execute(
                "SELECT count(*) FROM rg_fts WHERE rg_fts MATCH 'refund'"
            ).fetchone()[0]
        self.assertGreater(rows, 0)

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

    def test_what_fts_reads_is_the_enriched_rendering(self) -> None:
        s = Settings(embed_backend="none", lexical_enrich=True)
        assert_fts_reads_rendering(self, self.open(s), s)

    def test_it_still_reads_without_a_registered_function(self) -> None:
        idx = self.open(Settings(embed_backend="none", lexical_enrich=True))
        db = idx._paths.db
        idx.close()
        with closing(sqlite3.connect(db)) as conn:
            conn.execute(
                "SELECT count(*) FROM rg_fts WHERE rg_fts MATCH 'refund'")


if __name__ == "__main__":
    unittest.main()
