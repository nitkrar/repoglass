"""End-to-end through the public API.

Uses FakeEmbedder throughout: no test may download a model.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from repoglass import Index
from repoglass.config import Paths, Settings
from repoglass.corpus import discovery

FIXTURE = Path(__file__).parent / "fixtures" / "python" / "sample.py"


class IndexTestCase(unittest.TestCase):
    settings = Settings(embed_backend="none")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "src").mkdir(parents=True)
        shutil.copy(FIXTURE, self.root / "src" / "sample.py")
        (self.root / "README.md").write_text("# Vault\n\nHow unlocking works.\n")
        self.paths = Paths(root=self.root, home=Path(self.tmp.name) / "home")
        self.index = Index.open(self.root, self.settings, paths=self.paths)
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()


class RefreshTests(IndexTestCase):
    def test_first_refresh_reports_added(self) -> None:
        """A fresh root, not a fresh home: the db lives under the root."""
        other = Path(self.tmp.name) / "repo2"
        (other / "src").mkdir(parents=True)
        shutil.copy(FIXTURE, other / "src" / "sample.py")
        report = Index.open(
            other, self.settings,
            paths=Paths(root=other, home=Path(self.tmp.name) / "home2"),
        ).refresh()
        self.assertGreater(report.added, 0)

    def test_second_refresh_is_a_no_op(self) -> None:
        report = self.index.refresh()
        self.assertEqual(0, report.added)
        self.assertEqual(0, report.changed)
        self.assertEqual(0, report.deleted)

    def test_edit_is_detected(self) -> None:
        target = self.root / "src" / "sample.py"
        target.write_text(target.read_text() + "\n\ndef added_later():\n    return 1\n")
        report = self.index.refresh()
        self.assertEqual(1, report.changed)
        self.assertTrue(self.index.definitions("added_later"))

    def test_new_file_is_detected(self) -> None:
        (self.root / "src" / "extra.py").write_text("def brand_new():\n    return 2\n")
        report = self.index.refresh()
        self.assertEqual(1, report.added)
        self.assertTrue(self.index.definitions("brand_new"))

    def test_deleted_file_is_removed(self) -> None:
        (self.root / "src" / "sample.py").unlink()
        report = self.index.refresh()
        self.assertEqual(1, report.deleted)
        self.assertEqual([], self.index.definitions("top_level"))

    def test_mtime_moving_backwards_is_still_a_change(self) -> None:
        """cp -p and branch checkout move mtime backwards."""
        import os

        target = self.root / "src" / "sample.py"
        target.write_text("def only_this():\n    return 1\n")
        os.utime(target, (1_000_000, 1_000_000))
        report = self.index.refresh()
        self.assertEqual(1, report.changed)


class NavigationTests(IndexTestCase):
    def test_definitions_returns_the_span(self) -> None:
        found = self.index.definitions("helper")
        self.assertEqual(1, len(found))
        self.assertEqual("src/sample.py", found[0].path)
        self.assertGreaterEqual(found[0].start_line, 1)

    def test_definitions_of_unknown_name_is_empty(self) -> None:
        self.assertEqual([], self.index.definitions("no_such_symbol"))

    def test_references_finds_call_sites(self) -> None:
        found = self.index.references("helper")
        self.assertGreaterEqual(len(found), 2)   # called from inner and method

    def test_nested_definition_is_navigable(self) -> None:
        self.assertTrue(self.index.definitions("inner"))


class SearchTests(IndexTestCase):
    def test_exact_name_is_found(self) -> None:
        hits = self.index.search("helper", mode="all")
        self.assertTrue(hits)
        self.assertIn("helper", {h.name for h in hits})

    def test_hit_carries_real_source(self) -> None:
        hits = self.index.search("helper", mode="all")
        self.assertIn("def helper", hits[0].code)

    def test_scores_are_normalised(self) -> None:
        hits = self.index.search("helper", mode="all")
        self.assertTrue(all(0.0 <= h.score <= 1.0 for h in hits))

    def test_scores_are_descending(self) -> None:
        hits = self.index.search("def return", mode="all")
        scores = [h.score for h in hits]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unmatched_query_returns_empty(self) -> None:
        self.assertEqual([], self.index.search("zzzznomatchzzzz", mode="all"))


class ProseTests(IndexTestCase):
    def test_markdown_sections_are_indexed(self) -> None:
        self.assertTrue(self.index.definitions("Vault"))


class RescanThrottleTests(unittest.TestCase):
    """A read triggers a staleness walk at most every
    `rescan_after_seconds`.

    Without the throttle, every query repeats the walk even when the
    tree has not changed.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        # Must clear MIN_CHUNK_CHARS (60) or the symbol is navigable
        # but has no chunk, and search returns nothing.
        (self.root / "a.py").write_text(
            "def alpha():\n"
            "    return 'a value that is comfortably long enough to chunk'\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _index(self, seconds: int) -> Index:
        return Index.open(
            self.root,
            Settings(embed_backend="none", rescan_after_seconds=seconds),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "h"),
        )

    def test_a_second_read_inside_the_window_does_not_walk(self) -> None:
        idx = self._index(300)
        idx.refresh()
        calls = []
        real = discovery.walk
        discovery.walk = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            idx.search("alpha")
            idx.search("alpha")
        finally:
            discovery.walk = real
        self.assertEqual(0, len(calls))

    def test_zero_walks_every_time(self) -> None:
        idx = self._index(0)
        idx.refresh()
        calls = []
        real = discovery.walk
        discovery.walk = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
        try:
            idx.search("alpha")
            idx.search("alpha")
        finally:
            discovery.walk = real
        self.assertEqual(2, len(calls))

    def test_an_index_that_was_never_built_always_walks(self) -> None:
        """An empty result would be indistinguishable from no matches."""
        idx = self._index(300)
        self.assertTrue(idx.search("alpha"))


class IdentityPersistenceTests(unittest.TestCase):
    """What `set_identity` writes has to be what `needs_reindex` reads.

    A field in `_REINDEX_FIELDS` that the update statement omits can
    never converge: the stored value stays behind the wanted one, so
    every refresh reindexes the whole corpus and the next one still
    thinks it must.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text("def alpha():\n    return 1\n")
        self.home = Path(self.tmp.name) / "home"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _open(self, settings: Settings) -> Index:
        return Index.open(self.root, settings,
                          paths=Paths(root=self.root, home=self.home))

    def test_a_category_change_reindexes_once_not_forever(self) -> None:
        from dataclasses import replace

        before = Settings(embed_backend="none")
        self._open(before).refresh()

        after = replace(before, doc_languages=("markdown", "rst"))
        first = self._open(after).refresh()
        self.assertEqual(1, first.added, "the category change should reindex")

        second = self._open(after).refresh()
        self.assertEqual(0, second.added,
                         "settings unchanged since the last build, so there "
                         "is nothing to do")

    def test_every_reindex_field_is_written_back(self) -> None:
        """Each field, moved from its value at build time.

        Comparing a freshly-built index against itself passes whatever
        `set_identity` omits, because nothing has moved yet.
        """
        from dataclasses import replace

        from repoglass.store import _REINDEX_FIELDS

        before = Settings(embed_backend="none")
        self._open(before).refresh()
        # Each of these moves one field of the identity.
        after = replace(before, coverage="definition",
                        doc_languages=("markdown", "rst"))
        index = self._open(after)
        index.refresh()

        stored = index._store.identity()
        wanted = index._current_identity()
        for field in _REINDEX_FIELDS:
            self.assertEqual(getattr(wanted, field), getattr(stored, field),
                             f"{field} is compared but never persisted")


class ColdRefreshCostTests(unittest.TestCase):
    """A refresh that changes nothing must not construct an embedder.

    The embedder is wanted for one integer, `dims`, which the last
    build already persisted. Constructing it imports the model stack,
    which dominates a no-op refresh -- and the CLI runs one process
    per invocation, so it is paid every time.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text("def alpha():\n    return 1\n")
        self.home = Path(self.tmp.name) / "home"
        self.settings = Settings(embed_backend="none")
        Index.open(self.root, self.settings,
                   paths=Paths(root=self.root, home=self.home)).refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_noop_refresh_does_not_build_the_embedder(self) -> None:
        with mock.patch("repoglass.index.build_embedder", return_value=None) as build:
            index = Index.open(self.root, self.settings,
                               paths=Paths(root=self.root, home=self.home))
            report = index.refresh()
        self.assertEqual((0, 0, 0),
                         (report.added, report.changed, report.deleted))
        build.assert_not_called()

    def test_a_refresh_with_work_still_builds_it(self) -> None:
        (self.root / "b.py").write_text("def beta():\n    return 2\n")
        with mock.patch("repoglass.index.build_embedder", return_value=None) as build:
            index = Index.open(self.root, self.settings,
                               paths=Paths(root=self.root, home=self.home))
            self.assertEqual(1, index.refresh().added)
        build.assert_called_once()


class ReferenceContextTests(unittest.TestCase):
    """A reference knows which definition it sits inside.

    A bare line number makes the caller open the file to learn what
    "who uses this" actually answers; the enclosing definition is what
    the question means, and extraction already resolves it.
    """

    SOURCE = (
        "def target():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def caller():\n"
        "    return target()\n"
        "\n"
        "\n"
        "class Holder:\n"
        "    def method(self):\n"
        "        return target()\n"
        "\n"
        "\n"
        "AT_MODULE_SCOPE = target()\n"
    )

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text(self.SOURCE)
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_reference_names_the_definition_containing_it(self) -> None:
        enclosing = [r.enclosing for r in self.index.references("target")]
        self.assertIn("caller", enclosing)
        self.assertIn("method", enclosing)

    def test_a_module_scope_reference_has_no_enclosing_definition(self) -> None:
        refs = self.index.references("target")
        top = [r for r in refs if r.start_line == 14]
        self.assertEqual(1, len(top), "the module-scope call should be a ref")
        self.assertIsNone(top[0].enclosing)

    def test_definitions_carry_no_enclosing(self) -> None:
        for d in self.index.definitions("target"):
            self.assertIsNone(d.enclosing)


class AnnotationReferenceTests(unittest.TestCase):
    """A type annotation is a use of the type.

    For a symbol used only as a type, call captures find nothing and
    `references()` returns an empty list -- which reads as "unused"
    rather than "this query cannot see annotations".
    """

    SOURCE = (
        "class Widget:\n"
        "    pass\n"
        "\n"
        "\n"
        "class Gadget:\n"
        "    pass\n"
        "\n"
        "\n"
        "def build(w: Widget) -> Gadget:\n"
        "    return Gadget()\n"
        "\n"
        "\n"
        "def many(ws: list[Widget]) -> None:\n"
        "    held: Widget = ws[0]\n"
        "    return None\n"
    )

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text(self.SOURCE)
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _lines(self, name: str) -> set[int]:
        return {r.start_line for r in self.index.references(name)}

    def test_a_type_used_only_in_annotations_is_still_referenced(self) -> None:
        """Widget is never called, so call captures see nothing."""
        self.assertNotEqual(set(), self._lines("Widget"))

    def test_parameter_return_and_variable_annotations_all_count(self) -> None:
        found = self._lines("Widget")
        self.assertIn(9, found, "parameter annotation")
        self.assertIn(13, found, "annotation inside list[...]")
        self.assertIn(14, found, "variable annotation")

    def test_a_return_annotation_is_a_reference(self) -> None:
        self.assertIn(9, self._lines("Gadget"))

    def test_the_enclosing_definition_still_resolves(self) -> None:
        by_line = {r.start_line: r.enclosing
                   for r in self.index.references("Widget")}
        self.assertEqual("build", by_line.get(9))
        self.assertEqual("many", by_line.get(14))


class SignatureTests(unittest.TestCase):
    """Between the whole body and nothing there is the signature.

    A caller deciding whether to open a 200-line class needs its
    shape, not its implementation and not just its path.
    """

    SOURCE = (
        "class Widget:\n"
        "    def render(self, width: int,\n"
        "               height: int) -> str:\n"
        "        body = 'x' * width * height\n"
        "        return body\n"
        "\n"
        "\n"
        "def build(count: int = 3) -> Widget:\n"
        "    made = Widget()\n"
        "    return made\n"
    )

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.py").write_text(self.SOURCE)
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _sig(self, name: str) -> str | None:
        found = self.index.definitions(name)
        self.assertTrue(found, f"{name} should be defined")
        return found[0].signature

    def test_a_definition_carries_its_signature(self) -> None:
        self.assertEqual("def build(count: int = 3) -> Widget:",
                         self._sig("build"))

    def test_a_signature_spanning_lines_is_kept_whole(self) -> None:
        self.assertEqual("def render(self, width: int,\n"
                         "               height: int) -> str:",
                         self._sig("render"))

    def test_the_signature_stops_before_the_body(self) -> None:
        self.assertNotIn("body =", self._sig("render"))
        self.assertNotIn("return", self._sig("build"))

    def test_a_class_signature_is_its_header(self) -> None:
        self.assertEqual("class Widget:", self._sig("Widget"))

    def test_references_have_no_signature(self) -> None:
        for r in self.index.references("Widget"):
            self.assertIsNone(r.signature)


class CrashDuringIndexTests(unittest.TestCase):
    """A file's row and its chunks land together or not at all.

    A row written ahead of its chunks claims the file is indexed. The
    walk then finds nothing added and nothing changed -- the source
    file is fine, only the index is not -- so the gap is never
    noticed and every later search returns nothing.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        for i in range(4):
            (self.root / f"m{i}.py").write_text(
                f"def alpha{i}():\n"
                f"    return 'long enough to make a real chunk here'\n")
        self.home = Path(self.tmp.name) / "home"
        self.settings = Settings(embed_backend="none")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _index(self) -> Index:
        return Index.open(self.root, self.settings,
                          paths=Paths(root=self.root, home=self.home))

    def _counts(self, index: Index) -> tuple[int, int]:
        conn = index._store.conn
        return tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                     for t in ("file", "chunk"))

    def _crash_on(self, nth: int):
        """Make extraction raise partway through the run."""
        from repoglass.corpus import extract as extract_mod

        real = extract_mod.extract
        seen = []

        def faulty(file, source, settings):
            seen.append(file.path)
            if len(seen) == nth:
                raise RuntimeError("disk full")
            return real(file, source, settings)
        return real, faulty

    def test_a_crash_leaves_no_file_row_without_chunks(self) -> None:
        from repoglass.corpus import extract as extract_mod

        real, faulty = self._crash_on(3)
        extract_mod.extract = faulty
        try:
            with self.assertRaises(RuntimeError):
                self._index().refresh()
        finally:
            extract_mod.extract = real

        index = self._index()
        files, chunks = self._counts(index)
        self.assertEqual(
            0, files - chunks,
            f"{files} file rows against {chunks} chunks: a row was"
            " written for a file whose chunks never landed")

    def test_the_next_refresh_finishes_the_job(self) -> None:
        from repoglass.corpus import extract as extract_mod

        real, faulty = self._crash_on(3)
        extract_mod.extract = faulty
        try:
            with self.assertRaises(RuntimeError):
                self._index().refresh()
        finally:
            extract_mod.extract = real

        index = self._index()
        index.refresh()
        files, chunks = self._counts(index)
        self.assertEqual(4, files)
        self.assertEqual(4, chunks)
        self.assertTrue(index.search("alpha3"), "the whole tree is searchable")


class FtsRebuildRecoveryTests(unittest.TestCase):
    """The FTS index has to record that it is behind.

    Chunks are committed before the FTS rebuild runs. If the rebuild
    never happens the rows are all present and correct, so the next
    walk reports nothing added and nothing changed and the rebuild is
    never reached again. Only the lexical tier is affected, which
    makes it quieter than a missing chunk, not better: `COUNT(*)` on
    an external-content table reads through to the view, so the
    index looks full while matching nothing.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        for i in range(3):
            (self.root / f"m{i}.py").write_text(
                f"def alpha{i}():\n"
                f"    return 'a distinctive marker word zebrafish {i}'\n")
        self.home = Path(self.tmp.name) / "home"
        self.settings = Settings(embed_backend="none")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _index(self) -> Index:
        return Index.open(self.root, self.settings,
                          paths=Paths(root=self.root, home=self.home))

    def _crash_before_rebuild(self) -> None:
        from repoglass import store as store_mod

        real = store_mod.Store.rebuild_fts

        def boom(self):
            raise RuntimeError("died before the rebuild")
        store_mod.Store.rebuild_fts = boom
        try:
            with self.assertRaises(RuntimeError):
                self._index().refresh()
        finally:
            store_mod.Store.rebuild_fts = real

    def test_lexical_search_recovers_on_the_next_refresh(self) -> None:
        self._crash_before_rebuild()
        index = self._index()
        index.refresh()
        self.assertTrue(index.search("zebrafish"),
                        "a word in the chunk body, so only the lexical"
                        " tier can find it")

    def test_a_completed_refresh_leaves_nothing_outstanding(self) -> None:
        index = self._index()
        index.refresh()
        self.assertFalse(index._store.fts_dirty())

    def test_the_rebuild_is_not_repeated_when_nothing_changed(self) -> None:
        self._index().refresh()
        calls = []
        from repoglass import store as store_mod
        real = store_mod.Store.rebuild_fts
        store_mod.Store.rebuild_fts = lambda s: (calls.append(1), real(s))[1]
        try:
            self._index().refresh()
        finally:
            store_mod.Store.rebuild_fts = real
        self.assertEqual([], calls)


class SelfReferenceTests(unittest.TestCase):
    """A definition's own name is not a use of it.

    Some grammars capture an identifier under both a definition
    pattern and an unscoped reference pattern -- Go matches every
    `type_identifier`, including the one naming the type. The
    definition site then appears in its own reference list.
    """

    SOURCE = (
        "package m\n"
        "\n"
        "type Widget struct{ A int }\n"
        "\n"
        "func Build(w Widget) Widget {\n"
        "\treturn w\n"
        "}\n"
    )

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "a.go").write_text(self.SOURCE)
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_the_definition_line_is_not_a_reference(self) -> None:
        lines = {r.start_line for r in self.index.references("Widget")}
        self.assertNotIn(3, lines, "line 3 defines Widget, it does not use it")

    def test_real_uses_are_still_found(self) -> None:
        lines = {r.start_line for r in self.index.references("Widget")}
        self.assertIn(5, lines, "the parameter and return types are uses")

    def test_the_definition_is_still_navigable(self) -> None:
        found = self.index.definitions("Widget")
        self.assertIn(3, [d.start_line for d in found])


class DuplicateDefinitionTests(unittest.TestCase):
    """One definition, one row.

    A grammar can match the same node under two definition patterns --
    Go captures a struct as both `definition.type` and
    `definition.class` -- and each match became its own symbol, so
    `defs` reported a definition that does not exist.
    """

    CASES = {
        "a.go": ("package m\n\ntype Widget struct{ A int }\n", 3),
        "b.go": ("package m\n\ntype Reader interface{ Read() int }\n", 3),
        "c.ts": ("export interface Widget { a: number }\n", 1),
    }

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        for name, (src, _) in self.CASES.items():
            (self.root / name).write_text(src)
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_go_struct_is_defined_once(self) -> None:
        self.assertEqual(
            [3], [d.start_line for d in self.index.definitions("Widget")
                  if d.path == "a.go"])

    def test_a_go_interface_is_defined_once(self) -> None:
        self.assertEqual(
            [3], [d.start_line for d in self.index.definitions("Reader")])

    def test_a_typescript_interface_is_defined_once(self) -> None:
        self.assertEqual(
            [1], [d.start_line for d in self.index.definitions("Widget")
                  if d.path == "c.ts"])


class DuplicateChunkTests(unittest.TestCase):
    """The same text twice is one answer, not two.

    A tree with vendored copies or dated snapshots holds
    byte-identical spans in several files. `saturation_decay` only
    suppresses repeats within one file, so across files they each
    take a slot and the top-k narrows to one piece of text.
    """

    BODY = ("def render_invoice(order):\n"
            "    total = sum(line.amount for line in order.lines)\n"
            "    return f'invoice total {total} for {order.id}'\n")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        # Four byte-identical copies, as a snapshot directory produces.
        for i in range(4):
            d = self.root / f"snapshot-{i}"
            d.mkdir()
            (d / "billing.py").write_text(self.BODY)
        # A genuine competitor: different text, same subject.
        (self.root / "other.py").write_text(
            "def invoice_total(order):\n"
            "    lines = order.lines\n"
            "    return sum(line.amount for line in lines)\n")
        # Filler, so the competitor is not the weakest candidate.
        # Fusion normalises to [0,1], which pins the worst to zero,
        # and nothing can be demoted below that -- an artefact of a
        # tiny pool, not of the ranking.
        for i in range(6):
            (self.root / f"filler{i}.py").write_text(
                f"def unrelated_{i}(config):\n"
                f"    return config.get('setting_{i}', {i})\n")
        self.index = Index.open(
            self.root, Settings(embed_backend="none"),
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_copies_do_not_monopolise_the_top(self) -> None:
        """Without suppression one span takes every slot in the top
        four, and the whole result set is one piece of text."""
        hits = self.index.search("invoice total order lines", k=4)
        bodies = [h.code for h in hits]
        self.assertGreater(len(set(bodies)), 1,
                           "every one of the top four is the same text")

    def test_the_first_copy_still_wins_its_slot(self) -> None:
        hits = self.index.search("invoice total order lines", k=10)
        self.assertTrue(any("render_invoice" in h.code for h in hits))

    def test_the_later_copies_are_kept_not_dropped(self) -> None:
        """Demoted, not removed.

        Identical spans in different files are one answer to "what
        does this do" and several to "where is this".
        """
        hits = self.index.search("invoice total order lines", k=20)
        paths = {h.path for h in hits if "render_invoice" in h.code}
        self.assertEqual(4, len(paths), "all four locations are reachable")

    def test_distinct_content_is_untouched(self) -> None:
        hits = self.index.search("unrelated config setting", k=10)
        bodies = [h.code for h in hits]
        self.assertEqual(len(bodies), len(set(bodies)),
                         "nothing here repeats, so nothing should move")


class HitContentSourceTests(unittest.TestCase):
    """A hit's code comes from the index, not from the file.

    The span, the line numbers and the text are all recorded at index
    time. Reading the body back off disk mixes one view with another:
    the caller gets indexed line numbers against current content, and
    gets nothing at all when the tree has moved on.
    """

    BODY = ("def collect_totals(orders):\n"
            "    return sum(order.amount for order in orders if order.open)\n")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        (self.root / "billing.py").write_text(self.BODY)
        self.settings = Settings(embed_backend="none", rescan_after_seconds=3600)
        self.index = Index.open(
            self.root, self.settings,
            paths=Paths(root=self.root, home=Path(self.tmp.name) / "home"))
        self.index.refresh()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_the_body_is_returned(self) -> None:
        hits = self.index.search("collect totals orders", k=1)
        self.assertIn("def collect_totals", hits[0].code)

    def test_a_deleted_file_still_yields_its_indexed_body(self) -> None:
        """What `from_git` needs: the checkout is gone, the index is not."""
        (self.root / "billing.py").unlink()
        hits = self.index.search("collect totals orders", k=1)
        self.assertTrue(hits, "the chunk is still indexed")
        self.assertIn("def collect_totals", hits[0].code)

    def test_line_numbers_and_body_describe_the_same_thing(self) -> None:
        """Both come from the index, so they cannot disagree."""
        hits = self.index.search("collect totals orders", k=1)
        h = hits[0]
        self.assertEqual(h.end_line - h.start_line + 1,
                         len(h.code.splitlines()))


if __name__ == "__main__":
    unittest.main()
