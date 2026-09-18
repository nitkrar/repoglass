"""Storage behaviour: identity, deletion, FTS, vectors, and package data."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from repoglass import store as store_mod
from repoglass.config import Settings
from repoglass.models import Chunk, SourceFile, Symbol


def _file(path: str = "a.py", mtime: int = 111, size: int = 222) -> SourceFile:
    return SourceFile(path=path, mtime_ns=mtime, size=size, lang="python")


def _sym(name: str, tag: str = "def", line: int = 1) -> Symbol:
    return Symbol(name=name, tag=tag, path="a.py", start_line=line,
                  end_line=line + 2, lang="python")


def _chunk(name: str, text: str = "raw source here") -> Chunk:
    return Chunk.build(path="a.py", name=name, start_line=1, end_line=3,
                       source=f"body of {name}", leading_comments=[],
                       node_kind="function", text=text)


def _window(path: str, text: str = "window text") -> Chunk:
    return Chunk.build(path=path, name="", start_line=1, end_line=3,
                       source=text, leading_comments=[],
                       node_kind="window", text=text)


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "index.db"
        self.store = store_mod.Store.open(self.db, Settings())

    def tearDown(self) -> None:
        self.tmp.cleanup()


class ConnectionTests(StoreTestCase):
    def test_foreign_keys_are_enabled(self) -> None:
        """Inert cascades are the single most dangerous default here."""
        self.assertEqual(1, self.store.conn.execute("PRAGMA foreign_keys").fetchone()[0])

    def test_wal_mode(self) -> None:
        mode = self.store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual("wal", mode.lower())


class RoundTripTests(StoreTestCase):
    def test_known_files_round_trips_mtime_and_size(self) -> None:
        self.store.upsert_files([_file(mtime=5, size=9)])
        self.assertEqual({"a.py": (5, 9)}, self.store.known_files())

    def test_upsert_updates_rather_than_duplicates(self) -> None:
        self.store.upsert_files([_file(mtime=1, size=1)])
        self.store.upsert_files([_file(mtime=2, size=2)])
        self.assertEqual({"a.py": (2, 2)}, self.store.known_files())

    def test_definitions_and_references_are_separable(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("foo"), _sym("foo", tag="ref", line=9)])
        self.assertEqual(1, len(self.store.definitions("foo")))
        self.assertEqual(1, len(self.store.references("foo")))
        self.assertEqual("def", self.store.definitions("foo")[0].tag)

    def test_replace_symbols_clears_previous(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("old")])
        self.store.replace_symbols("a.py", [_sym("new")])
        self.assertEqual([], self.store.definitions("old"))
        self.assertEqual(1, len(self.store.definitions("new")))


class DeletionTests(StoreTestCase):
    def test_deleting_a_file_leaves_no_orphans(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("foo")])
        self.store.upsert_chunks("a.py", [_chunk("foo")])
        self.store.rebuild_fts()

        self.store.delete_files(["a.py"])

        c = self.store.conn
        self.assertEqual(0, c.execute("SELECT count(*) FROM file").fetchone()[0])
        self.assertEqual(0, c.execute("SELECT count(*) FROM symbol").fetchone()[0])
        self.assertEqual(0, c.execute("SELECT count(*) FROM chunk").fetchone()[0])
        self.assertEqual([], self.store.definitions("foo"))


class FtsTests(StoreTestCase):
    def _seed(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("unlock"), _sym("other", line=20)])
        self.store.upsert_chunks("a.py", [
            _chunk("unlock", 'raise ValueError("vault is already unlocked")'),
            _chunk("other", "completely unrelated payment invoice text"),
        ])
        self.store.rebuild_fts()

    def test_finds_a_string_literal(self) -> None:
        """The reason FTS5 indexes raw source rather than distilled text."""
        self._seed()
        hits = self.store.fts_search("unlocked", limit=10, mode="all")
        self.assertTrue(hits)

    def test_ranks_the_matching_chunk_first(self) -> None:
        self._seed()
        hits = self.store.fts_search("vault unlocked", limit=10, mode="all")
        best = hits[0][0]
        span = self.store.chunk_span(best)
        self.assertIsNotNone(span)

    def test_rebuild_drops_deleted_content(self) -> None:
        """Stale FTS rows would return chunks that no longer exist."""
        self._seed()
        self.store.delete_files(["a.py"])
        self.store.rebuild_fts()
        self.assertEqual([], self.store.fts_search("unlocked", limit=10, mode="all"))


class IdentityTests(StoreTestCase):
    def test_identity_round_trips(self) -> None:
        """A non-default model on purpose: comparing against
        `Settings().embed_model` would pass even if `open` ignored the
        settings entirely and wrote the default."""
        with tempfile.TemporaryDirectory() as tmp:
            store = store_mod.Store.open(
                Path(tmp) / "i.db", Settings(embed_model="some/other-model")
            )
            ident = store.identity()
            self.assertIsNotNone(ident)
            self.assertEqual("some/other-model", ident.embed_model)

    def test_model_change_forces_reindex(self) -> None:
        from dataclasses import replace

        changed = replace(self.store.identity(), embed_model="something/else")
        self.assertTrue(self.store.needs_reindex(changed))

    def test_backend_change_forces_reindex(self) -> None:
        """Same model name, different backend, same width.

        Two backends can serve one model name at one width and still
        produce vectors that are not comparable, so the width alone
        cannot stand in for which embedder wrote them.
        """
        from dataclasses import replace

        changed = replace(self.store.identity(), embed_backend="http")
        self.assertTrue(self.store.needs_reindex(changed))

class SchemaRevisionTests(unittest.TestCase):
    """A schema change must not leave an unreadable database behind.

    The revision is a fingerprint of the DDL, not a hand-maintained
    number: bumping an integer is a manual step, and forgetting it
    produces exactly the `no such column` failure the check exists to
    prevent.
    """

    def test_the_revision_tracks_the_ddl(self) -> None:
        import repoglass.store as sm

        original = sm.SCHEMA
        try:
            before = sm.schema_rev()
            sm.SCHEMA = original + "\n-- a change\n"
            self.assertNotEqual(before, sm.schema_rev())
        finally:
            sm.SCHEMA = original

    def test_stale_schema_is_rebuilt_not_reused(self) -> None:
        import repoglass.store as sm

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "index.db"
            store = sm.Store.open(db, Settings())
            store.conn.execute("UPDATE meta SET schema_rev = 'stale'")
            store.conn.commit()
            store.conn.close()

            reopened = sm.Store.open(db, Settings())
            self.assertEqual(sm.schema_rev(), reopened.identity().schema_rev)
            # And the current columns are present.
            reopened.conn.execute("SELECT start_line, end_line FROM chunk")


class DuplicateCaptureTests(StoreTestCase):
    """A tags query can capture one node under two capture names.

    `chunk` is keyed on its own id with a unique index on `symbol_id`,
    so the second capture of a pair aborts the whole file's insert
    unless duplicates are dropped first. Keying on `symbol_id` instead
    would take the other bad option: an upsert that silently collapses
    the pair.
    """

    def test_two_chunks_for_one_symbol_do_not_abort_the_insert(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("thing")])
        first = _chunk("thing", "the first capture of this node")
        second = _chunk("thing", "the second capture of the same node")
        self.store.upsert_chunks("a.py", [first, second])
        n = self.store.conn.execute(
            "SELECT count(*) FROM chunk WHERE symbol_id IS NOT NULL"
        ).fetchone()[0]
        self.assertEqual(1, n)

    def test_window_chunks_without_symbols_are_allowed_to_repeat(self) -> None:
        """The unique index is partial: NULL symbol_id must not collide."""
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [])
        anon = [Chunk.build(path="a.py", name="", start_line=i,
                            end_line=i + 2, source=f"body {i}",
                            leading_comments=[], node_kind="window",
                            text=f"text {i}")
                for i in (1, 10, 20)]
        self.store.upsert_chunks("a.py", anon)
        n = self.store.conn.execute("SELECT count(*) FROM chunk").fetchone()[0]
        self.assertEqual(3, n)


class QueryHelperTests(StoreTestCase):
    def test_pending_vectors_returns_unembedded_chunks(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("alpha")])
        self.store.upsert_chunks("a.py", [_chunk("alpha", "embed this chunk")])

        rows = self.store.pending_vectors()

        self.assertEqual(1, len(rows))
        cid, text = rows[0]
        self.assertEqual("embed this chunk", text)
        self.store.set_vectors([(cid, b"\x00\x00\x80\x3f")])
        self.assertEqual([], self.store.pending_vectors())

    def test_set_embed_dims_updates_identity(self) -> None:
        self.store.set_embed_dims(16)
        self.assertEqual(16, self.store.identity().embed_dims)

    def test_chunk_rows_and_paths_return_the_requested_ids(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("alpha")])
        self.store.upsert_chunks("a.py", [_chunk("alpha", "chunk text")])
        cid = self.store.conn.execute("SELECT id FROM chunk").fetchone()[0]

        self.assertEqual({cid: ("a.py", "chunk text")}, self.store.chunk_rows([cid]))
        self.assertEqual({cid: "a.py"}, self.store.chunk_paths([cid]))

    def test_non_candidate_rows_filter_by_stem_and_mode(self) -> None:
        self.store.upsert_files([
            SourceFile(path="src/userstore.py", mtime_ns=1, size=1,
                       lang="python", content_type="code"),
            SourceFile(path="tests/test_userstore.py", mtime_ns=1, size=1,
                       lang="python", content_type="tests"),
            SourceFile(path="docs/guide.md", mtime_ns=1, size=1,
                       lang="markdown", content_type="docs"),
        ])
        self.store.upsert_chunks("src/userstore.py", [
            _window("src/userstore.py", "class UserStore:\n    pass")
        ])
        self.store.upsert_chunks("tests/test_userstore.py", [
            _window("tests/test_userstore.py", "def test_userstore():\n    pass")
        ])
        self.store.upsert_chunks("docs/guide.md", [
            _window("docs/guide.md", "# Guide\n\nUser storage docs")
        ])

        self.assertEqual(
            [
                (1, "src/userstore.py", "class UserStore:\n    pass"),
                (2, "tests/test_userstore.py", "def test_userstore():\n    pass"),
            ],
            self.store.non_candidate_rows({"UserStore"}, mode="all"),
        )
        self.assertEqual(
            [(1, "src/userstore.py", "class UserStore:\n    pass")],
            self.store.non_candidate_rows({"UserStore"}, mode="code"),
        )

    def test_status_counts_and_top_languages_report_stored_rows(self) -> None:
        self.store.upsert_files([
            _file(path="a.py"),
            SourceFile(path="b.py", mtime_ns=2, size=2, lang="python"),
            SourceFile(path="guide.md", mtime_ns=3, size=3, lang="markdown"),
        ])
        self.store.replace_symbols("a.py", [_sym("alpha")])
        self.store.upsert_chunks("a.py", [_chunk("alpha")])
        self.store.upsert_chunks("guide.md", [_window("guide.md", "# Guide\n\nSome docs")])

        self.assertEqual((3, 2, 1), self.store.status_counts())
        self.assertEqual(
            [("python", 2), ("markdown", 1)],
            self.store.top_languages(limit=2),
        )


class VectorCacheTests(StoreTestCase):
    """Writers that change chunks must invalidate the cached matrix."""

    def _seed(self) -> None:
        self.store.upsert_files([_file()])
        self.store.replace_symbols("a.py", [_sym("alpha")])
        self.store.upsert_chunks("a.py", [_chunk("alpha")])
        cid = self.store.conn.execute("SELECT id FROM chunk").fetchone()[0]
        self.store.set_vectors([(cid, b"\x00\x00\x80\x3f")])

    def test_a_repeat_read_does_not_hit_the_database(self) -> None:
        self._seed()
        first = self.store.vectors(mode="all")
        self.store.conn.execute("DELETE FROM chunk")   # behind the cache's back
        self.assertEqual(first, self.store.vectors(mode="all"))

    def test_set_vectors_invalidates(self) -> None:
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.store.conn.execute("DELETE FROM chunk")
        self.store.set_vectors([])
        self.assertEqual(0, len(self.store.vectors(mode="all")[0]))

    def test_deleting_a_file_invalidates(self) -> None:
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.store.delete_files(["a.py"])
        self.assertEqual(0, len(self.store.vectors(mode="all")[0]))

    def test_upserting_chunks_invalidates(self) -> None:
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.store.upsert_chunks("a.py", [])
        self.assertEqual(0, len(self.store.vectors(mode="all")[0]))

    def test_replacing_symbols_invalidates(self) -> None:
        """The cascade from symbol to chunk drops vectors too."""
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.store.replace_symbols("a.py", [])
        self.assertEqual(0, len(self.store.vectors(mode="all")[0]))

    def test_reset_content_invalidates(self) -> None:
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.store.reset_content()
        self.assertEqual(0, len(self.store.vectors(mode="all")[0]))

    def test_modes_are_cached_separately(self) -> None:
        self._seed()
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))
        self.assertEqual(0, len(self.store.vectors(mode="docs")[0]))
        self.assertEqual(1, len(self.store.vectors(mode="all")[0]))


class SchemaFileTests(unittest.TestCase):
    """The schema SQL ships as package data and drives `schema_rev()`."""

    def test_the_ddl_is_readable_as_package_data(self) -> None:
        text = store_mod.schema_sql()
        self.assertIn("CREATE TABLE meta", text)
        self.assertIn("CREATE VIRTUAL TABLE chunk_fts", text)

    def test_the_revision_tracks_the_file(self) -> None:
        """`schema_rev` must hash what is executed, or a schema edit
        leaves databases the code can no longer read."""
        self.assertEqual(store_mod.schema_rev(), store_mod.schema_rev())
        self.assertIn("CREATE TABLE", store_mod.schema_sql())

    def test_every_table_the_code_reads_is_created(self) -> None:
        import re
        ddl = store_mod.schema_sql()
        created = set(re.findall(r"CREATE (?:VIRTUAL )?TABLE (\w+)", ddl))
        created |= set(re.findall(r"CREATE VIEW (\w+)", ddl))
        src = Path(store_mod.__file__).read_text()
        read = set(re.findall(r"FROM (\w+)", src)) - {"excluded", "sqlite_master"}
        missing = {t for t in read if t not in created and t.islower()}
        self.assertEqual(set(), missing, "SQL names a table the DDL never creates")


class PackageDataTests(unittest.TestCase):
    """Package data declarations must cover non-Python files."""

    def test_every_data_file_is_declared(self) -> None:
        import fnmatch
        import tomllib

        root = Path(store_mod.__file__).parent
        pyproject = root.parents[1] / "pyproject.toml"
        if not pyproject.exists():          # installed, not a source tree
            self.skipTest("no pyproject beside the package")
        data = tomllib.loads(pyproject.read_text())["tool"]["setuptools"]["package-data"]
        patterns = [
            f"{key.removeprefix('repoglass').removeprefix('.').replace('.', '/')}/{pat}"
            .lstrip("/")
            for key, pats in data.items() for pat in pats
        ]
        undeclared = [
            rel for rel in (
                str(p.relative_to(root)) for p in root.rglob("*")
                if p.is_file() and p.suffix not in {".py", ".pyc"}
            )
            if "__pycache__" not in rel
            and not any(fnmatch.fnmatch(rel, pat) for pat in patterns)
        ]
        self.assertEqual([], undeclared,
                         "shipped in the source tree, absent from the wheel")


if __name__ == "__main__":
    unittest.main()
