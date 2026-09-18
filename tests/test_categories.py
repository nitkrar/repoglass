"""File categories: docs, config, data, tests, code.

Decided by `corpus.classify` at walk time and stored on the file row,
so a filtered query is an equality test any `sqlite3` connection can
run. Editing a category list changes `categories_rev`, which is part of
the index identity, so the next refresh reclassifies.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repoglass import Index
from repoglass.config import Paths, Settings

BODY = "def {name}():\n    return 'a value long enough to clear min chunk'\n"


class CategoryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        self.write("src/handler.py", BODY.format(name="handle_payment"))
        self.write("tests/test_handler.py", BODY.format(name="handle_payment"))
        self.write("docs/guide.md", "# handle_payment\n\nHow payment handling"
                                    " works in practice, at some length.\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, rel: str, body: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)

    def index(self, settings: Settings | None = None, home: str = "h") -> Index:
        base = Settings(embed_backend="none")
        paths = Paths(root=self.root, home=Path(self.tmp.name) / home)
        idx = Index.open(self.root, settings or base, paths=paths)
        idx.refresh()
        return idx

    def paths_for(self, idx: Index, content) -> set[str]:
        return {h.path for h in idx.search("handle_payment", content=content, k=20)}


class ModeSelectsCategory(CategoryFixture):
    def test_code_excludes_tests_and_docs(self) -> None:
        got = self.paths_for(self.index(), "code")
        self.assertIn("src/handler.py", got)
        self.assertNotIn("tests/test_handler.py", got)
        self.assertNotIn("docs/guide.md", got)

    def test_tests_selects_only_test_paths(self) -> None:
        got = self.paths_for(self.index(), "tests")
        self.assertEqual({"tests/test_handler.py"}, got)

    def test_docs_selects_only_doc_languages(self) -> None:
        got = self.paths_for(self.index(), "docs")
        self.assertEqual({"docs/guide.md"}, got)

    def test_all_selects_everything(self) -> None:
        got = self.paths_for(self.index(), "all")
        self.assertEqual(
            {"src/handler.py", "tests/test_handler.py", "docs/guide.md"}, got
        )

    def test_omitting_content_searches_everything(self) -> None:
        """Omitting `content` narrows nothing: the caller opts in.

        Asserted through what a search returns rather than through
        `Settings().content`, because a default that is set but never
        consumed satisfies a check on the literal.
        """
        idx = self.index()
        self.assertEqual(self.paths_for(idx, "all"),
                         {h.path for h in idx.search("handle_payment", k=20)})


class CategoriesAreStoredAndReclassified(CategoryFixture):
    """The category is written at walk time, so editing a category list
    must reclassify rather than leave a stale value on disk.

    Resolving the category in SQL instead would take a registered
    Python function, and a filtered query then fails on any connection
    that has not registered one. Reindex is seconds; an unreadable
    database is not.
    """

    def reopen(self, settings: Settings) -> Index:
        """Same database, different settings, and a refresh."""
        idx = Index.open(
            self.root, settings,
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "h"),
        )
        idx.refresh()
        return idx

    def test_changing_doc_languages_reclassifies(self) -> None:
        idx = self.index(Settings(embed_backend="none",
                                  doc_languages=("markdown",)))
        self.assertIn("docs/guide.md", self.paths_for(idx, "docs"))

        changed = self.reopen(
            Settings(embed_backend="none", doc_languages=("rst",)))
        self.assertEqual(set(), self.paths_for(changed, "docs"))
        self.assertIn("docs/guide.md", self.paths_for(changed, "code"))

    def test_changing_test_markers_reclassifies(self) -> None:
        self.write("src/spec_runner.py", BODY.format(name="handle_payment"))
        idx = self.index(Settings(embed_backend="none", test_markers=("test",)))
        self.assertIn("src/spec_runner.py", self.paths_for(idx, "code"))

        widened = self.reopen(
            Settings(embed_backend="none", test_markers=("test", "spec")))
        self.assertIn("src/spec_runner.py", self.paths_for(widened, "tests"))
        self.assertNotIn("src/spec_runner.py", self.paths_for(widened, "code"))

    def test_editing_a_category_list_invalidates_the_index(self) -> None:
        """What makes the two tests above work: the lists feed
        `categories_rev`, which is part of the index identity."""
        from repoglass.config import categories_rev

        a = Settings(embed_backend="none", data_languages=("csv",))
        b = Settings(embed_backend="none", data_languages=("csv", "tsv"))
        self.assertNotEqual(categories_rev(a), categories_rev(b))

    def test_stored_category_is_the_one_filters_read(self) -> None:
        """The stored column has to be consumed, not merely written.

        Corrupting it must change what a filtered search returns. If
        it does not, something else is deciding the category and the
        column is dead weight.
        """
        idx = self.index()
        self.assertIn("docs/guide.md", self.paths_for(idx, "docs"))
        idx._store.conn.execute(
            "UPDATE file SET content_type='code' WHERE path='docs/guide.md'")
        idx._store.conn.commit()
        self.assertNotIn("docs/guide.md", self.paths_for(idx, "docs"))
        self.assertIn("docs/guide.md", self.paths_for(idx, "code"))


class TestMarkerBoundaryTests(unittest.TestCase):
    """Every form a test file takes, and every near-miss that is not one.

    The boundary is easy to draw wrong in three directions at once,
    and each has a counter-example on a real repository: a bare
    substring makes `inspection.py` a test, requiring a `_.-`
    delimiter drops Swift's `FooTests.swift` and its `Tests/`
    directory, and matching camelCase case-insensitively makes
    `latest.py` one.
    """

    @staticmethod
    def matcher(markers=("test", "spec")):
        from repoglass.corpus.classify import is_test

        return lambda p: is_test(p, markers)

    def test_every_test_form_matches(self) -> None:
        is_test = self.matcher()
        for path in ("tests/test_a.py", "Tests/FooTests.swift", "a_test.go",
                     "a.spec.ts", "spec/thing.rb", "__tests__/x.js",
                     "Sources/X/CaptureControllerSpec.swift",
                     "testing/helper.py", "TESTS/x.py", "pkg/foo_spec.rb"):
            self.assertTrue(is_test(path), path)

    def test_words_merely_containing_a_marker_do_not(self) -> None:
        is_test = self.matcher()
        for path in ("src/pkg/inspection.py", "src/core/attestation.py",
                     "app/latest.py", "src/respect.py", "src/protest.py",
                     "src/contest.py", "greatest.py", "src/handler.py"):
            self.assertFalse(is_test(path), path)

    def test_camel_case_matching_is_case_sensitive(self) -> None:
        """`latest.py` is `la` + `test`; only an uppercase T makes it a
        camelCase boundary."""
        is_test = self.matcher()
        self.assertTrue(is_test("FooTests.swift"))
        self.assertFalse(is_test("latest.py"))

    def test_no_markers_matches_nothing(self) -> None:
        is_test = self.matcher(markers=())
        self.assertFalse(is_test("tests/test_a.py"))

    def test_markers_are_configurable(self) -> None:
        is_test = self.matcher(markers=("check",))
        self.assertTrue(is_test("checks/thing.py"))
        self.assertFalse(is_test("tests/test_a.py"))


class ContentIsASetTests(CategoryFixture):
    """`content` takes one category, several, or nothing.

    A single-valued enum could not say "code and docs but not tests" --
    the combination a caller most often wants -- without a fifth value
    meaning everything. Nothing now means everything, so the absence of
    a filter is the absence of a restriction.
    """

    def test_one_category(self) -> None:
        self.assertEqual({"src/handler.py"}, self.paths_for(self.index(), "code"))

    def test_several_categories(self) -> None:
        got = self.paths_for(self.index(), ("code", "docs"))
        self.assertEqual({"src/handler.py", "docs/guide.md"}, got)
        self.assertNotIn("tests/test_handler.py", got)

    def test_none_means_no_restriction(self) -> None:
        idx = self.index()
        every = {"src/handler.py", "tests/test_handler.py", "docs/guide.md"}
        self.assertEqual(every, self.paths_for(idx, None))

    def test_listing_every_category_is_the_same_as_none(self) -> None:
        idx = self.index()
        self.assertEqual(
            self.paths_for(idx, None),
            self.paths_for(idx, ("code", "tests", "docs", "config")),
        )

    def test_all_is_still_accepted_as_a_spelling_of_no_filter(self) -> None:
        idx = self.index()
        self.assertEqual(self.paths_for(idx, None), self.paths_for(idx, "all"))

    def test_an_unknown_category_is_rejected_not_ignored(self) -> None:
        """Silently ignoring a typo would return everything and look
        like it worked."""
        with self.assertRaises(ValueError):
            self.index().search("handle_payment", content="cod")


if __name__ == "__main__":
    unittest.main()
