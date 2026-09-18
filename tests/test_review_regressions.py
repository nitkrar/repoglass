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
                "SELECT vec FROM chunk WHERE vec IS NOT NULL"
            )
        }
        self.assertLessEqual(len(widths), 1, f"mixed vector widths: {widths}")


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
