"""Path derivation and settings resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from repoglass.config import Paths, Settings, as_toml, load


class PathsTests(unittest.TestCase):
    def test_all_paths_derive_from_two_roots(self) -> None:
        p = Paths(root=Path("/r"), home=Path("/h"))
        self.assertEqual(Path("/h/index"), p.data.parent)
        self.assertTrue(p.data.name.startswith("r-"))
        self.assertEqual(p.data / "index.db", p.db)
        self.assertEqual(Path("/r/.repoglassignore"), p.repo_ignore)
        self.assertEqual(Path("/r/repoglass.toml"), p.repo_config)
        self.assertEqual(Path("/h/models"), p.models)
        self.assertEqual(Path("/h/cache"), p.cache)
        self.assertEqual(Path("/h/repoglass.toml"), p.user_config)

    def test_for_root_defaults_home_under_dot_repoglass(self) -> None:
        p = Paths.for_root(Path("/some/repo"))
        self.assertEqual(Path.home() / ".repoglass", p.home)

    def test_env_overrides_home(self) -> None:
        old = os.environ.get("REPOGLASS_HOME")
        os.environ["REPOGLASS_HOME"] = "/tmp/rg-home"
        try:
            self.assertEqual(Path("/tmp/rg-home"), Paths.for_root(Path("/r")).home)
        finally:
            if old is None:
                del os.environ["REPOGLASS_HOME"]
            else:
                os.environ["REPOGLASS_HOME"] = old

    def test_root_is_resolved_to_absolute(self) -> None:
        self.assertTrue(Paths.for_root(Path(".")).root.is_absolute())


class SettingsLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_defaults_when_no_files(self) -> None:
        from unittest import mock

        missing = self.dir / "missing.toml"
        with mock.patch.dict(os.environ, {}, clear=True):
            s = load(user_config=missing, repo_config=missing)
        self.assertEqual("rrf", s.ranker)
        self.assertEqual("full", s.lexical_mode)

    def test_repo_config_overrides_defaults(self) -> None:
        cfg = self.dir / "repoglass.toml"
        cfg.write_text('ranker = "ppr"\ncandidate_depth = 42\n')
        s = load(repo_config=cfg)
        self.assertEqual("ppr", s.ranker)
        self.assertEqual(42, s.candidate_depth)

    def test_repo_config_beats_user_config(self) -> None:
        user = self.dir / "user.toml"
        repo = self.dir / "repo.toml"
        user.write_text('ranker = "none"\n')
        repo.write_text('ranker = "ppr"\n')
        self.assertEqual("ppr", load(user_config=user, repo_config=repo).ranker)

    def test_overrides_beat_files(self) -> None:
        repo = self.dir / "repo.toml"
        repo.write_text('ranker = "ppr"\n')
        s = load(repo_config=repo, overrides={"ranker": "none"})
        self.assertEqual("none", s.ranker)

    def test_lists_survive_as_tuples(self) -> None:
        cfg = self.dir / "c.toml"
        cfg.write_text('hard_exclude = ["a", "b"]\n')
        self.assertEqual(("a", "b"), load(repo_config=cfg).hard_exclude)

    def test_unknown_key_is_rejected(self) -> None:
        """A typo in a config file must not be silently ignored."""
        cfg = self.dir / "c.toml"
        cfg.write_text('rankr = "ppr"\n')
        with self.assertRaises(ValueError):
            load(repo_config=cfg)

    def test_nested_tables_are_flattened(self) -> None:
        cfg = self.dir / "c.toml"
        cfg.write_text('[retrieval]\nranker = "ppr"\n')
        self.assertEqual("ppr", load(repo_config=cfg).ranker)




class GeneratedConfigTests(unittest.TestCase):
    """`as_toml()` must produce a file this library can read back.

    Generated from the dataclass so an example config cannot drift from
    the code, which means the dataclass and TOML must agree on every
    shape: a constant is not a settable field, TOML has no null so an
    unset key must be absent rather than `""`, and a nested tuple comes
    back as a list unless it is converted.
    """

    def test_it_is_valid_toml(self) -> None:
        import tomllib
        tomllib.loads(as_toml())

    def test_it_round_trips_to_identical_settings(self) -> None:
        from dataclasses import fields
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "repoglass.toml"
            path.write_text(as_toml())
            back = load(repo_config=path)
            for f in fields(Settings):
                self.assertEqual(getattr(Settings(), f.name),
                                 getattr(back, f.name), f.name)

    def test_every_emittable_setting_appears(self) -> None:
        """Every setting except the secrets, which `load()` refuses in a
        repository config -- emitting one produces a file this tool
        writes and then will not read."""
        from dataclasses import fields

        from repoglass.config.load import _SECRET
        text = as_toml()
        for f in fields(Settings):
            if f.name in _SECRET:
                self.assertNotIn(f.name, text, f.name)
            else:
                self.assertIn(f.name, text, f.name)

    def test_unset_keys_are_commented_out(self) -> None:
        """An uncommented `key = ""` would read back as an empty string
        rather than None."""
        self.assertIn('# embed_endpoint = ""', as_toml())

    def test_it_reflects_the_settings_it_is_given(self) -> None:
        """Doubles as a diagnostic: what did this run actually resolve
        to? A flag whose default silently disagrees with the setting it
        shadows is invisible otherwise."""
        text = as_toml(Settings(alpha_prose=0.9, content=("code", "docs")))
        self.assertIn("alpha_prose = 0.9", text)
        self.assertIn('content = ["code", "docs"]', text)


class SecretsStayOutOfTheRepoConfigTests(unittest.TestCase):
    """The repo-level file is meant to be committed.

    `embed_api_key` in `<root>/repoglass.toml` would be shared with
    everyone who clones. Documenting "prefer the env var" does not stop
    it; refusing to load it does.
    """

    def _write(self, tmp, text):
        p = Path(tmp) / "repoglass.toml"
        p.write_text(text)
        return p

    def test_a_secret_in_the_repo_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, 'embed_api_key = "sk-committed"\n')
            with self.assertRaises(ValueError) as e:
                load(repo_config=p)
            self.assertIn("embed_api_key", str(e.exception))
            self.assertIn("REPOGLASS_EMBED_API_KEY", str(e.exception))

    def test_the_same_key_is_fine_in_the_user_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, 'embed_api_key = "sk-local"\n')
            self.assertEqual("sk-local", load(user_config=p).embed_api_key)

    def test_ordinary_keys_are_unaffected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = self._write(tmp, 'test_markers = ["spec"]\n')
            self.assertEqual(("spec",), load(repo_config=p).test_markers)

    def test_the_environment_still_works(self) -> None:
        old = os.environ.get("REPOGLASS_EMBED_API_KEY")
        os.environ["REPOGLASS_EMBED_API_KEY"] = "sk-env"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                missing = Path(tmp) / "missing.toml"
                self.assertEqual(
                    "sk-env",
                    load(user_config=missing, repo_config=missing).embed_api_key,
                )
        finally:
            if old is None:
                del os.environ["REPOGLASS_EMBED_API_KEY"]
            else:
                os.environ["REPOGLASS_EMBED_API_KEY"] = old


class IndexLocationTests(unittest.TestCase):
    """Where the index is written, and why it defaults away from the repo.

    In-repo storage cannot index a read-only checkout at all, and
    scatters state across every repository with no way to list or purge
    it.
    """

    def test_default_is_central_and_keyed_by_the_repo(self) -> None:
        a = Paths.for_root(Path("/tmp/alpha"))
        b = Paths.for_root(Path("/tmp/beta"))
        self.assertIn("index", a.db.parts)
        self.assertNotIn("alpha", str(a.db.parent.parent))
        self.assertNotEqual(a.db, b.db)

    def test_the_key_is_recognisable(self) -> None:
        """Hash for uniqueness, basename so a human can find it."""
        self.assertTrue(Paths.for_root(Path("/tmp/alpha")).db.parent.name
                        .startswith("alpha-"))

    def test_same_name_different_place_do_not_collide(self) -> None:
        self.assertNotEqual(Paths.for_root(Path("/one/app")).db,
                            Paths.for_root(Path("/two/app")).db)

    def test_a_relative_data_dir_puts_it_back_in_the_repo(self) -> None:
        p = Paths(root=Path("/tmp/alpha"), home=Path("/h"), data_dir=".repoglass")
        self.assertEqual(Path("/tmp/alpha/.repoglass/index.db"), p.db)

    def test_an_absolute_data_dir_is_used_as_given(self) -> None:
        p = Paths(root=Path("/tmp/alpha"), home=Path("/h"), data_dir="/mnt/idx")
        self.assertEqual(Path("/mnt/idx/index.db"), p.db)

    def test_a_read_only_tree_can_be_indexed(self) -> None:
        """The default index location must still work on a read-only tree."""
        import os

        from repoglass import Index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ro"
            root.mkdir()
            (root / "a.py").write_text(
                "def alpha():\n    return 'a value long enough to be chunked'\n")
            os.chmod(root, 0o555)
            try:
                idx = Index.open(root, Settings(embed_backend="none"),
                                 paths=Paths(root=root, home=Path(tmp) / "home"))
                idx.refresh()
                self.assertTrue(idx.definitions("alpha"))
            finally:
                os.chmod(root, 0o755)


if __name__ == "__main__":
    unittest.main()
