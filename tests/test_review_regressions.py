"""Regression tests for previously broken public behaviour."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repoglass import Index
from repoglass.config import Paths, Settings


class Fixture(unittest.TestCase):
    settings = Settings(embed_backend="none")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, rel: str, body: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
        return p

    def open_index(self, settings: Settings | None = None, home: str = "h") -> Index:
        paths = Paths(root=self.root, home=Path(self.tmp.name) / home)
        return Index.open(self.root, settings or self.settings, paths=paths)

    def edge_count(self, idx: Index) -> int:
        return idx._store.conn.execute("SELECT count(*) FROM edge").fetchone()[0]


class F2_IdentityChangeForcesReindex(Fixture):
    """Mixed-width vectors crash the reshape in vector.search."""

    def test_changing_the_embedder_does_not_leave_mixed_width_vectors(self) -> None:
        self.write("a.py", "def alpha():\n    return 'a value here to pass min chunk'\n")
        idx = self.open_index(Settings(embed_backend="none"))
        idx.refresh()

        # Reopen declaring a different model; stored rows must not survive.
        other = self.open_index(
            Settings(embed_backend="none", embed_model="different/model"), home="h"
        )
        other.refresh()
        widths = {
            len(v) for (v,) in other._store.conn.execute(
                "SELECT vec FROM rg_vectors"
            )
        }
        self.assertLessEqual(len(widths), 1, f"mixed vector widths: {widths}")


    def test_changing_the_document_prefix_re_embeds(self) -> None:
        """Same model, same width, different document prefix: the old
        vectors sit in another region of the space and must not stay."""
        from unittest import mock

        from semsift.embed import FakeEncoder
        from semsift.store.codec import pack

        body = ("def alpha():\n    # long enough to be a retrievable chunk\n"
                "    return 'a value here to pass the minimum chunk size'\n")
        self.write("a.py", body)

        def vectors_after(prefix: str) -> list[bytes]:
            settings = Settings(embed_backend="static", embed_doc_prefix=prefix)
            with mock.patch("repoglass.index.build_embedder",
                            return_value=FakeEncoder(dims=8, doc_prefix=prefix)):
                idx = self.open_index(settings)
                idx.refresh()
                rows = idx._store.conn.execute(
                    "SELECT i.text, v.vec FROM rg_vectors v JOIN rg_items i ON i.id = v.id").fetchall()
                idx.close()
            self.assertTrue(rows)
            fresh = FakeEncoder(dims=8, doc_prefix=prefix)
            for text, vec in rows:
                self.assertEqual(pack(fresh.encode([text])[0]), vec, prefix)
            return [vec for _, vec in rows]

        before = vectors_after("old: ")
        self.assertNotEqual(before, vectors_after("new: "))

    def test_a_changed_variant_is_probed_before_the_old_index_is_reset(self) -> None:
        """If the replacement backend cannot load, the last complete index
        remains usable rather than being replaced by unembedded rows."""
        from unittest import mock

        from semsift.embed import FakeEncoder

        body = ("def alpha():\n    # long enough to be a retrievable chunk\n"
                "    return 'a value here to pass the minimum chunk size'\n")
        self.write("a.py", body)
        before = Settings(embed_backend="http", embed_endpoint="http://old")
        with mock.patch("repoglass.index.build_embedder",
                        return_value=FakeEncoder(dims=8)):
            idx = self.open_index(before)
            idx.refresh()
            idx.close()

        after = Settings(embed_backend="http", embed_endpoint="http://new")
        with mock.patch("repoglass.index.build_embedder",
                        side_effect=OSError("replacement backend unavailable")):
            idx = self.open_index(after)
            with self.assertRaises(OSError):
                idx.refresh()
            stored = idx._store.identity()
            vectors = idx._store.conn.execute(
                "SELECT count(*) FROM rg_vectors").fetchone()[0]
            idx.close()
        self.assertEqual("http://old", stored.embed_variant)
        self.assertGreater(vectors, 0)


class EmbeddingRecovery(Fixture):
    """Chunks commit before they are embedded, so a failed embed must be
    retried by a later refresh even when no file has changed."""

    def test_a_failed_embed_is_retried_on_the_next_refresh(self) -> None:
        from unittest import mock

        from semsift.embed import FakeEncoder

        class Failing(FakeEncoder):
            def _encode(self, texts):
                raise OSError("embedding server down")

        self.write("a.py", "def alpha():\n    # long enough to be a retrievable chunk\n"
                           "    return 'a value here to pass the minimum chunk size'\n")
        settings = Settings(embed_backend="static")
        with mock.patch("repoglass.index.build_embedder", return_value=Failing()):
            idx = self.open_index(settings)
            with self.assertRaises(OSError):
                idx.refresh()
            idx.close()
        with mock.patch("repoglass.index.build_embedder", return_value=FakeEncoder()):
            idx = self.open_index(settings)
            idx.refresh()
            missing = idx._store.conn.execute(
                "SELECT count(*) FROM rg_items i LEFT JOIN rg_vectors v ON v.id = i.id WHERE v.id IS NULL").fetchone()[0]
            total = idx._store.conn.execute("SELECT count(*) FROM chunk").fetchone()[0]
            idx.close()
        self.assertGreater(total, 0)
        self.assertEqual(0, missing)

    def test_a_settled_index_does_not_build_the_embedder(self) -> None:
        """Checking for pending vectors must not load a model when there
        are none."""
        from unittest import mock

        from semsift.embed import FakeEncoder

        self.write("a.py", "def alpha():\n    # long enough to be a retrievable chunk\n"
                           "    return 'a value here to pass the minimum chunk size'\n")
        settings = Settings(embed_backend="static")
        with mock.patch("repoglass.index.build_embedder",
                        return_value=FakeEncoder()) as build:
            idx = self.open_index(settings)
            idx.refresh()
            idx.close()
            build.reset_mock()
            idx = self.open_index(settings)
            idx.refresh()
            idx.close()
        build.assert_not_called()

    def test_an_incomplete_embedding_batch_is_not_partly_attached(self) -> None:
        from unittest import mock

        from semsift.embed import FakeEncoder

        class Dropping(FakeEncoder):
            def _encode(self, texts):
                return super()._encode(texts)[:-1]

        for name in ("alpha", "beta"):
            self.write(
                f"{name}.py",
                f"def {name}():\n    # long enough to be a retrievable chunk\n"
                "    return 'a value here to pass the minimum chunk size'\n")
        with mock.patch("repoglass.index.build_embedder",
                        return_value=Dropping()):
            idx = self.open_index(Settings(embed_backend="static"))
            with self.assertRaises(ValueError):
                idx.refresh()
            attached = idx._store.conn.execute(
                "SELECT count(*) FROM rg_vectors").fetchone()[0]
            idx.close()
        self.assertEqual(0, attached)


class ChunkShapingSettingsReindex(Fixture):
    """Settings that reshape stored chunks must reach unchanged files."""

    BODY = "def long_function():\n" + "".join(
        f"    step_{i} = compute_value_number_{i}()\n" for i in range(12))

    def spans(self, settings: Settings) -> list[tuple[int, int]]:
        idx = self.open_index(settings)
        idx.refresh()
        rows = idx._store.conn.execute(
            "SELECT start_line, end_line FROM chunk ORDER BY start_line").fetchall()
        idx.close()
        return rows

    def test_changing_max_chunk_lines_rechunks_an_unchanged_file(self) -> None:
        self.write("a.py", self.BODY)
        wide = self.spans(Settings(embed_backend="none", max_chunk_lines=40))
        narrow = self.spans(Settings(embed_backend="none", max_chunk_lines=4))
        self.assertNotEqual(wide, narrow)
        fresh = Fixture.open_index(self, Settings(embed_backend="none",
                                                  max_chunk_lines=4), home="fresh")
        fresh.refresh()
        expected = fresh._store.conn.execute(
            "SELECT start_line, end_line FROM chunk ORDER BY start_line").fetchall()
        fresh.close()
        self.assertEqual(expected, narrow)

    def test_every_setting_the_chunker_reads_is_in_the_fingerprint(self) -> None:
        """Derived from the corpus source, so a new setting read while
        building chunks cannot be left out."""
        import ast

        from repoglass.config.schema import CHUNKING_FIELDS

        corpus = Path(__file__).resolve().parents[1] / "src" / "repoglass" / "corpus"
        read = set()
        for path in corpus.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "settings"):
                    read.add(node.attr)
        # Walk-time inputs: every refresh re-walks, so a change reaches
        # the file set without a fingerprint. Category lists have their
        # own, categories_rev.
        walk = {"hard_exclude", "gitignore", "max_file_bytes"}
        categories = {"doc_languages", "config_languages", "data_languages",
                      "test_markers", "index_excluded"}
        identity = {"coverage"}
        missing = read - walk - categories - identity - set(CHUNKING_FIELDS)
        self.assertEqual(set(), missing)


class F3_GitignoreChangePropagates(Fixture):
    """Editing .gitignore must take effect, and must not loop.

    Editing it changes no file's mtime, so nothing about the file
    itself tells the staleness comparison to look again. The guarantee
    is that a newly-ignored file leaves the walk and is deleted from
    the index.
    """

    def test_adding_a_file_to_gitignore_removes_it(self) -> None:
        self.write("secret.py", "def secret_thing():\n    return 'a value here ok'\n")
        idx = self.open_index()
        idx.refresh()
        self.assertTrue(self._indexed(idx, "secret.py"))

        self.write(".gitignore", "secret.py\n")
        idx.refresh()
        self.assertFalse(self._indexed(idx, "secret.py"))

    def test_an_unchanged_gitignore_does_no_work(self) -> None:
        self.write("a.py", "def alpha():\n    return 'a value here to pass min'\n")
        self.write(".gitignore", "nothing_here\n")
        idx = self.open_index()
        idx.refresh()
        idx.refresh()
        report = idx.refresh()
        self.assertEqual(0, report.added + report.changed + report.deleted)

    @staticmethod
    def _indexed(idx: Index, path: str) -> bool:
        return idx._store.conn.execute(
            "SELECT count(*) FROM file WHERE path=?", (path,)
        ).fetchone()[0] > 0


class F6_RetrievalIsChunkDriven(Fixture):
    def test_a_symbol_without_a_chunk_is_not_returned_by_search(self) -> None:
        """Short definitions are navigable, not retrievable."""
        self.write("a.py", "class Thing:\n    def tiny(self):\n        pass\n")
        idx = self.open_index()
        idx.refresh()
        self.assertTrue(idx.definitions("tiny"))          # navigable
        self.assertEqual([], idx.search("tiny", mode="all"))   # not retrievable

    def test_mode_filters_the_exact_tier(self) -> None:
        self.write("src/a.py", "def helper():\n    return 'source value long enough'\n")
        self.write("tests/test_a.py",
                   "def helper():\n    return 'test value long enough here'\n")
        idx = self.open_index()
        idx.refresh()
        code_paths = {h.path for h in idx.search("helper", mode="code")}
        self.assertNotIn("tests/test_a.py", code_paths)
        test_paths = {h.path for h in idx.search("helper", mode="tests")}
        self.assertNotIn("src/a.py", test_paths)

    def test_returned_code_respects_max_chunk_lines(self) -> None:
        """Lines must be long enough that the capped body still clears
        MIN_CHUNK_CHARS -- capping below it drops the chunk entirely."""
        body = "def wide():\n" + "".join(
            f"    variable_number_{i} = compute_something({i})\n" for i in range(20)
        )
        self.write("a.py", body)
        idx = self.open_index(Settings(embed_backend="none", max_chunk_lines=4))
        idx.refresh()
        hits = idx.search("wide", mode="all")
        self.assertTrue(hits)
        self.assertLessEqual(len(hits[0].code.splitlines()), 4)

    def test_capping_below_min_chunk_chars_drops_the_chunk(self) -> None:
        """max_chunk_lines and MIN_CHUNK_CHARS interact: a cap tight enough
        to take the body under the minimum yields no chunk at all."""
        self.write("a.py", "def wide():\n" + "".join(
            f"    x{i} = {i}\n" for i in range(20)))
        idx = self.open_index(Settings(embed_backend="none", max_chunk_lines=2,
                                       coverage="definition"))
        idx.refresh()
        self.assertTrue(idx.definitions("wide"))        # still navigable
        self.assertEqual([], idx.search("wide", mode="all"))

    def test_under_hybrid_the_gap_fill_makes_it_retrievable_again(self) -> None:
        """The same file and the same cap, under the default coverage.

        "Navigable but not retrievable" is a property of definition
        chunking rather than a guarantee: what no definition chunk
        covers is exactly what hybrid fills in."""
        self.write("a.py", "def wide():\n" + "".join(
            f"    x{i} = {i}\n" for i in range(20)))
        idx = self.open_index(Settings(embed_backend="none", max_chunk_lines=2))
        idx.refresh()
        self.assertTrue(idx.search("wide", mode="all"))


if __name__ == "__main__":
    unittest.main()
