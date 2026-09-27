"""repoglass's walk rules on top of semsift's discover: which settings and
ignore files it passes, and what it drops afterwards. Ignore-file and
symlink semantics are semsift's and tested there."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from repoglass.config import Paths, Settings
from repoglass.corpus import discovery


class WalkTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "a.py").write_text("def a(): pass\n")
        (self.root / "README.md").write_text("# Title\n")
        self.paths = Paths(root=self.root, home=Path(self.tmp.name) / "home")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def walked(self, settings: Settings | None = None) -> set[str]:
        return {f.path for f in discovery.walk(self.paths, settings or Settings())}


class BasicWalkTests(WalkTestCase):
    def test_finds_indexable_files(self) -> None:
        self.assertEqual({"src/a.py", "README.md"}, self.walked())

    def test_paths_are_relative_and_slash_separated(self) -> None:
        for p in self.walked():
            self.assertFalse(os.path.isabs(p))
            self.assertNotIn("\\", p)

    def test_skips_files_with_no_grammar(self) -> None:
        (self.root / ".env").write_text("SECRET=1\n")
        (self.root / "key.pem").write_text("-----BEGIN-----\n")
        self.assertEqual({"src/a.py", "README.md"}, self.walked())

    def test_hard_exclude_prunes_directories(self) -> None:
        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "node_modules" / "pkg" / "x.py").write_text("x=1\n")
        (self.root / "dist").mkdir()
        (self.root / "dist" / "bundle.js").write_text("var x=1\n")
        self.assertEqual({"src/a.py", "README.md"}, self.walked())

    def test_repoglassignore_excludes(self) -> None:
        (self.root / "gen").mkdir()
        (self.root / "gen" / "out.py").write_text("y=1\n")
        (self.root / ".repoglassignore").write_text("gen/\n")
        self.assertEqual({"src/a.py", "README.md"}, self.walked())


class ClassificationTests(WalkTestCase):
    """Discovery yields markdown and test files rather than filtering
    them out.

    The walk classifies each file and drops whatever falls in
    `index_excluded`, so a category mistake here keeps a file out of
    the index entirely rather than merely mis-filing it. Which
    category each one lands in is tests/test_categories.py.
    """

    def test_markdown_is_indexed(self) -> None:
        by_path = {f.path: f for f in discovery.walk(self.paths, Settings())}
        self.assertIn("README.md", by_path)
        self.assertEqual("markdown", by_path["README.md"].lang)

    def test_test_files_are_indexed_not_excluded(self) -> None:
        (self.root / "tests").mkdir()
        (self.root / "tests" / "test_a.py").write_text("def test_a(): pass\n")
        by_path = {f.path: f for f in discovery.walk(self.paths, Settings())}
        self.assertIn("tests/test_a.py", by_path)
        self.assertIn("src/a.py", by_path)

    def test_the_walk_carries_no_category(self) -> None:
        """`content_type` is the only category a walked file carries.

        A second attribute for the same concept is how one of them
        ends up written and never read.
        """
        f = next(iter(discovery.walk(self.paths, Settings())))
        self.assertFalse(hasattr(f, "kind"))
        self.assertFalse(hasattr(f, "role"))

    def test_stat_is_recorded(self) -> None:
        by_path = {f.path: f for f in discovery.walk(self.paths, Settings())}
        f = by_path["src/a.py"]
        st = (self.root / "src" / "a.py").stat()
        self.assertEqual(st.st_mtime_ns, f.mtime_ns)
        self.assertEqual(st.st_size, f.size)


class GitignoreTests(WalkTestCase):
    """`gitignore` is a boolean: honour the repo's .gitignore, or do not.

    There is no "index but flag" middle value. Flagging would need a
    stored column, a reclassification pass and a rules hash to keep
    that column honest when .gitignore changes; a newly-ignored file
    instead just stops appearing in the walk.
    """

    def setUp(self) -> None:
        super().setUp()
        (self.root / "build_out.py").write_text("z=1\n")
        (self.root / ".gitignore").write_text("build_out.py\n")

    def test_on_keeps_ignored_files_out_of_the_walk(self) -> None:
        got = self.walked(Settings(gitignore=True))
        self.assertNotIn("build_out.py", got)
        self.assertIn("src/a.py", got)

    def test_off_walks_them_like_any_other_file(self) -> None:
        self.assertIn("build_out.py", self.walked(Settings(gitignore=False)))

    def test_repoglassignore_applies_either_way(self) -> None:
        """Our own exclusion file is not the repo's .gitignore and is
        not switched off with it."""
        (self.root / ".repoglassignore").write_text("build_out.py\n")
        self.assertNotIn("build_out.py", self.walked(Settings(gitignore=False)))


class RepoglassignoreOverridesGitignoreTests(WalkTestCase):
    """`.gitignore` says "do not commit", not "do not search".

    A file can be legitimately untracked and still worth indexing --
    local notes, generated code you read, scratch config. A `!` pattern
    in .repoglassignore rescues it for that file alone, rather than
    forcing `gitignore=False` for the whole repo.
    """

    def setUp(self) -> None:
        super().setUp()
        (self.root / "generated.py").write_text("def gen(): pass\n")
        (self.root / "junk.py").write_text("def junk(): pass\n")
        (self.root / ".gitignore").write_text("generated.py\njunk.py\n")

    def test_a_negation_rescues_one_of_them(self) -> None:
        (self.root / ".repoglassignore").write_text("!generated.py\n")
        got = self.walked()
        self.assertIn("generated.py", got)
        self.assertNotIn("junk.py", got)

class GeneratedFileTests(unittest.TestCase):
    """A file whose typical line is enormous was not written by hand.

    Extension cannot tell a bundle from source -- both are `.js` --
    and the size cap does not catch it either, because a few hundred
    kilobytes is an ordinary file. What separates them is shape: real
    code and prose wrap, generated payloads do not.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = Paths(root=self.root, home=self.root / ".home")
        self.settings = Settings(embed_backend="none")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def walked(self) -> set[str]:
        return {f.path for f in discovery.walk(self.paths, self.settings)}

    def test_a_minified_bundle_is_skipped(self) -> None:
        (self.root / "bundle.js").write_text(
            "\n".join("var a=1," + "b=2," * 4000 + "c=3;" for _ in range(6)))
        self.assertNotIn("bundle.js", self.walked())

    def test_ordinary_source_is_kept(self) -> None:
        (self.root / "app.js").write_text(
            "\n".join(f"function handler{i}(req, res) {{ return res.send({i}); }}"
                      for i in range(2000)))
        self.assertIn("app.js", self.walked())

    def test_prose_with_long_paragraphs_is_kept(self) -> None:
        """A wrapped-off markdown line is long but nowhere near a bundle."""
        para = "This is an ordinary sentence of documentation prose. " * 8
        (self.root / "guide.md").write_text(
            "\n".join(para for _ in range(400)))
        self.assertIn("guide.md", self.walked())

    def test_a_small_odd_file_is_not_penalised(self) -> None:
        """Below the probe floor the shape test does not run at all."""
        (self.root / "tiny.json").write_text('{"k":"' + "x" * 3000 + '"}\n' * 3)
        self.assertIn("tiny.json", self.walked())


if __name__ == "__main__":
    unittest.main()
