"""What gets indexed, and what a search returns when you do not say.

Two separate gates with separate defaults:

    index_excluded  what never enters the index. Holds `data`,
                    because tabular rows are bulk that competes for
                    candidate slots and is rarely the answer.
    content_excluded what a search holds back unless named.
                    Only `data`; config is indexed and often
                    answers how something is wired.

Asking for a category the index does not hold raises rather than
returning an empty list, because "not indexed" and "no matches" are
different answers and a caller cannot tell them apart otherwise.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repoglass import Index
from repoglass.config import Paths, Settings

BODY = "def {name}():\n    return 'a value long enough to clear min chunk'\n"
ROWS = "region,total\nnorth,1200\nsouth,980\neast,1443\nwest,777\n" * 4
CONF = '{\n  "handle_payment": {"retries": 3, "timeout_seconds": 30}\n}\n'


class ContentFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        self.write("src/handler.py", BODY.format(name="handle_payment"))
        self.write("docs/guide.md", "# handle_payment\n\nHow payment handling"
                                    " works in practice, at some length.\n")
        self.write("data/handle_payment.csv", ROWS)
        self.write("settings/handle_payment.json", CONF)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, rel: str, body: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def index(self, settings: Settings | None = None, home: str = "h") -> Index:
        paths = Paths(root=self.root, home=Path(self.tmp.name) / home)
        idx = Index.open(self.root, settings or Settings(embed_backend="none"),
                         paths=paths)
        idx.refresh()
        return idx

    def paths_for(self, idx: Index, **kw) -> set[str]:
        return {h.path for h in idx.search("handle_payment", k=20, **kw)}

class QueryTimeDefaultTests(ContentFixture):
    def test_default_search_includes_code_docs_and_config(self) -> None:
        """Only `data` is held back. A question about how something is
        wired is often answered by a manifest."""
        got = self.paths_for(self.index())
        self.assertIn("src/handler.py", got)
        self.assertIn("docs/guide.md", got)
        self.assertIn("settings/handle_payment.json", got)

    def test_all_still_excludes_data(self) -> None:
        """`all` means everything a question about code could want.

        Data is reachable only by naming it: a caller that forgets the
        filter -- or a small model that does not know to set one --
        should never get spreadsheet rows back, and rows do not answer
        questions about code.
        """
        settings = Settings(
            embed_backend="none",
            index_excluded=())
        idx = self.index(settings)
        self.assertNotIn("data/handle_payment.csv",
                         self.paths_for(idx, content="all"))
        self.assertIn("data/handle_payment.csv",
                      self.paths_for(idx, content="data"))

    def test_config_is_reachable_when_asked_for(self) -> None:
        got = self.paths_for(self.index(), content="config")
        self.assertEqual({"settings/handle_payment.json"}, got)


class UnindexedCategoryRaisesTests(ContentFixture):
    def test_asking_for_data_raises_when_data_was_not_indexed(self) -> None:
        idx = self.index()
        with self.assertRaises(LookupError) as caught:
            idx.search("handle_payment", content="data")
        message = str(caught.exception)
        self.assertIn("data", message)
        self.assertIn("index_excluded", message)

    def test_no_raise_once_data_is_indexed(self) -> None:
        settings = Settings(
            embed_backend="none",
            index_excluded=())
        got = self.paths_for(self.index(settings), content="data")
        self.assertEqual({"data/handle_payment.csv"}, got)

    def test_empty_result_is_not_confused_with_unindexed(self) -> None:
        """A category that IS indexed but has no match returns empty."""
        idx = self.index()
        self.assertEqual(set(), self.paths_for(idx, content="tests"))


class IdentityTests(ContentFixture):
    def test_changing_index_excluded_invalidates_the_index(self) -> None:
        from repoglass.config import categories_rev

        a = Settings(embed_backend="none", index_excluded=("data",))
        b = Settings(embed_backend="none", index_excluded=())
        self.assertNotEqual(categories_rev(a), categories_rev(b))


if __name__ == "__main__":
    unittest.main()
