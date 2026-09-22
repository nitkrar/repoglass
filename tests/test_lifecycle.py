"""Closing an index releases its database connection.

Nothing here downloads a model: embed_backend="none".
"""

from __future__ import annotations

import gc
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from repoglass import Index
from repoglass.config import Paths, Settings

FIXTURE = Path(__file__).parent / "fixtures" / "python" / "sample.py"


class LifecycleTests(unittest.TestCase):
    settings = Settings(embed_backend="none")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        shutil.copy(FIXTURE, self.root / "sample.py")
        self.paths = Paths(root=self.root, home=Path(self.tmp.name) / "home")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _open(self) -> Index:
        return Index.open(self.root, self.settings, paths=self.paths)

    def test_close_releases_the_connection(self) -> None:
        index = self._open()
        index.refresh()
        index.close()
        with self.assertRaises(sqlite3.ProgrammingError):
            index._store.conn.execute("SELECT 1")

    def test_context_manager_closes_on_exit(self) -> None:
        with self._open() as index:
            index.refresh()
        with self.assertRaises(sqlite3.ProgrammingError):
            index._store.conn.execute("SELECT 1")

    def test_context_manager_closes_when_the_body_raises(self) -> None:
        index = self._open()
        with self.assertRaises(ZeroDivisionError):
            with index:
                1 / 0
        with self.assertRaises(sqlite3.ProgrammingError):
            index._store.conn.execute("SELECT 1")

    def test_close_is_idempotent(self) -> None:
        index = self._open()
        index.close()
        index.close()

    def test_dropping_an_index_releases_its_connection(self) -> None:
        """Callers that never close must not leak the connection.

        Asserted on the connection rather than by catching a warning,
        which would also catch one raised by another test's garbage.
        """
        index = self._open()
        index.refresh()
        conn = index._store.conn
        del index
        gc.collect()
        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")


if __name__ == "__main__":
    unittest.main()
