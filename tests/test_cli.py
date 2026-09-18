"""The command line, exercised end to end through main().

Tested at the boundary -- argv in, stdout and an exit code out --
because that is the whole contract. Asserting on the parser's namespace
would pass while the command did nothing.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from repoglass.cli import main

BODY = "def {name}():\n    return 'a value long enough to clear min chunk'\n"


class CliFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        (self.root / "src").mkdir(parents=True)
        (self.root / "src" / "handler.py").write_text(
            BODY.format(name="handle_payment"))
        # README.md must clear MIN_CHUNK_CHARS: otherwise `defs` can
        # navigate the heading but `search` has no docs chunk to return.
        (self.root / "README.md").write_text(
            "# handle_payment\n\nHow payment handling works in practice:"
            " authorisation, capture, retries and the refund path, at"
            " enough length to clear the minimum chunk size.\n")
        self.home = Path(self.tmp.name) / "home"
        self.cfg = Path(self.tmp.name) / "c.toml"
        self.cfg.write_text('embed_backend = "none"\n'
                            f'data_dir = "{self.home}"\n')

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_cli(self, *argv: str) -> tuple[int, dict | str, str]:
        out, err = io.StringIO(), io.StringIO()
        full = [*argv, "--repo", str(self.root), "--config", str(self.cfg)]
        with redirect_stdout(out), redirect_stderr(err):
            code = main(full)
        text = out.getvalue()
        # `config` emits TOML, not JSON -- see cmd_config.
        if "--text" in full or "-t" in full or argv[0] == "config":
            return code, text, err.getvalue()
        return code, (json.loads(text) if text.strip() else {}), err.getvalue()


class SearchTests(CliFixture):
    def test_returns_json_by_default(self) -> None:
        code, payload, _ = self.run_cli("search", "handle_payment")
        self.assertEqual(0, code)
        self.assertIn("results", payload)
        self.assertTrue(payload["results"])

    def test_results_carry_location_and_code(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment")
        first = payload["results"][0]
        for key in ("path", "start_line", "end_line", "score", "code"):
            self.assertIn(key, first)

    def test_no_code_omits_the_source(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment", "--no-code")
        self.assertNotIn("code", payload["results"][0])

    def test_k_limits_results(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment", "-k", "1")
        self.assertLessEqual(len(payload["results"]), 1)

    def test_no_matches_is_success_not_failure(self) -> None:
        code, payload, _ = self.run_cli("search", "zzzznotpresentzzz")
        self.assertEqual(0, code)
        self.assertIn("results", payload)

    def test_unknown_category_exits_two_without_a_traceback(self) -> None:
        code, _, err = self.run_cli("search", "x", "--content", "cod")
        self.assertEqual(2, code)
        self.assertIn("cod", err)
        self.assertNotIn("Traceback", err)

    def test_a_valid_content_filter_actually_works(self) -> None:
        """The success path. Every other --content test raises before
        reaching retrieval, so this is the only one that carries the
        list argparse builds from nargs="+" as far as the vector
        cache."""
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--content", "code")
        self.assertEqual(0, code)
        got = {r["path"] for r in payload["results"]}
        self.assertIn("src/handler.py", got)
        self.assertNotIn("README.md", got)      # markdown is docs

    def test_two_categories_at_once(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--content", "code", "docs")
        self.assertEqual(0, code)
        got = {r["path"] for r in payload["results"]}
        self.assertIn("src/handler.py", got)
        self.assertIn("README.md", got)

    def test_unindexed_category_exits_two_with_the_remedy(self) -> None:
        code, _, err = self.run_cli("search", "x", "--content", "data")
        self.assertEqual(2, code)
        self.assertIn("index_excluded", err)


class OutputChannelTests(CliFixture):
    def test_first_run_notice_goes_to_stderr_not_stdout(self) -> None:
        """stdout must stay parseable as JSON on the very first call."""
        code, payload, err = self.run_cli("search", "handle_payment")
        self.assertEqual(0, code)
        self.assertIsInstance(payload, dict)
        self.assertIn("building the index", err)

    def test_text_format_is_not_json(self) -> None:
        code, text, _ = self.run_cli(
            "search", "handle_payment", "--text")
        self.assertEqual(0, code)
        self.assertIn("src/handler.py", text)


class IncludeGlobTests(CliFixture):
    """SQLite GLOB, so `*` crosses `/` and `src/*` is recursive."""

    def test_include_narrows_to_a_directory(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--include", "src/*")
        self.assertEqual(0, code)
        self.assertEqual({"src/handler.py"},
                         {r["path"] for r in payload["results"]})

    def test_include_matches_by_extension_anywhere(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--include", "*.md")
        self.assertEqual({"README.md"},
                         {r["path"] for r in payload["results"]})

    def test_several_globs_are_ored(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code",
            "--include", "src/*", "*.md")
        self.assertEqual({"src/handler.py", "README.md"},
                         {r["path"] for r in payload["results"]})

    def test_a_glob_matching_nothing_is_empty_not_an_error(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--include", "nope/*")
        self.assertEqual(0, code)
        self.assertEqual([], payload["results"])

    def test_include_composes_with_lang(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code",
            "--include", "src/*", "-l", "markdown")
        self.assertEqual([], payload["results"])   # src/ has no markdown


class ExcludeGlobTests(CliFixture):
    def test_exclude_removes_a_directory(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--exclude", "src/*")
        self.assertEqual(0, code)
        self.assertNotIn("src/handler.py",
                         {r["path"] for r in payload["results"]})

    def test_exclude_wins_over_include(self) -> None:
        """Both clauses are ANDed, so the pair reads as "these, but not
        those" rather than the reverse."""
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code",
            "--include", "src/*", "--exclude", "src/*")
        self.assertEqual([], payload["results"])

    def test_several_excludes_all_apply(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code",
            "--exclude", "src/*", "*.md")
        self.assertEqual([], payload["results"])


class ResultSummaryTests(CliFixture):
    def test_refs_report_a_count(self) -> None:
        _, payload, _ = self.run_cli("refs", "handle_payment")
        self.assertEqual(len(payload["symbols"]), payload["count"])

    def test_defs_break_down_by_content_type(self) -> None:
        """A bare total says nothing for a name that is defined in both
        code and prose; the split is the answer, and tallying it by
        hand is not."""
        _, payload, _ = self.run_cli("defs", "handle_payment")
        split = payload["by_content_type"]
        self.assertEqual(sum(split.values()), payload["count"])
        self.assertIn("code", split)
        self.assertIn("docs", split)        # README.md defines the heading

    def test_text_shows_the_split_only_when_mixed(self) -> None:
        _, text, _ = self.run_cli("defs", "handle_payment", "--text")
        self.assertRegex(text, r"# 2 result\(s\): (code 1, docs 1|docs 1, code 1)")

    def test_search_reports_a_count(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment", "--no-code")
        self.assertEqual(len(payload["results"]), payload["count"])


class SearchLangTests(CliFixture):
    """Pre-filter, not post-filter: k results in that language, rather
    than k retrieved and thinned to whatever survived."""

    def test_search_narrows_to_one_language(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "-l", "python")
        self.assertEqual(0, code)
        got = {r["path"] for r in payload["results"]}
        self.assertIn("src/handler.py", got)
        self.assertNotIn("README.md", got)

    def test_search_narrows_to_markdown(self) -> None:
        code, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "-l", "markdown")
        self.assertEqual(0, code)
        self.assertEqual({"README.md"},
                         {r["path"] for r in payload["results"]})

    def test_search_accepts_several(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "-l", "python", "markdown")
        self.assertEqual({"src/handler.py", "README.md"},
                         {r["path"] for r in payload["results"]})

    def test_search_rejects_a_typo(self) -> None:
        code, _, err = self.run_cli(
            "search", "handle_payment", "--no-code", "-l", "pyton")
        self.assertEqual(2, code)
        self.assertIn("did you mean 'python'?", err)


class LangValidationTests(CliFixture):
    def test_a_near_miss_suggests_the_right_one(self) -> None:
        code, _, err = self.run_cli("defs", "handle_payment", "-l", "pyton")
        self.assertEqual(2, code)
        self.assertIn("did you mean 'python'?", err)

    def test_an_unrecognisable_value_lists_what_exists(self) -> None:
        """No near match, so fall back to the set -- a suggestion would
        be a guess."""
        code, _, err = self.run_cli("defs", "handle_payment", "-l", "zzzzz")
        self.assertEqual(2, code)
        self.assertIn("this index has:", err)
        self.assertIn("python", err)

    def test_a_typo_beside_a_valid_language_is_not_swallowed(self) -> None:
        """`-l python pyton` matches on python, so an empty-result-only
        check would never notice the second value."""
        code, _, err = self.run_cli(
            "defs", "handle_payment", "-l", "python", "pyton")
        self.assertEqual(2, code)
        self.assertIn("pyton", err)

    def test_several_languages_are_accepted(self) -> None:
        code, payload, _ = self.run_cli(
            "defs", "handle_payment", "-l", "python", "markdown")
        self.assertEqual(0, code)
        self.assertEqual({"python", "markdown"},
                         {s["lang"] for s in payload["symbols"]})

    def test_a_valid_language_is_not_rejected(self) -> None:
        code, payload, _ = self.run_cli("defs", "handle_payment", "-l", "python")
        self.assertEqual(0, code)
        self.assertTrue(payload["symbols"])


class FormatParityTests(CliFixture):
    """text and json render one payload; neither may drop a field."""

    def test_text_shows_the_score_and_tiers_json_carries(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment", "--no-code")
        first = payload["results"][0]
        code, text, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--text")
        self.assertEqual(0, code)
        self.assertIn(str(first["score"]), text)
        for metric in first["tiers"]:
            self.assertIn(metric, text)

    def test_signature_level_returns_something_for_every_result(self) -> None:
        """A window chunk holds no definition and so has no header, but
        a path and a line range alone are not enough for a caller to
        decide whether to open it."""
        (self.root / "src" / "settings.py").write_text(
            "# module level only, no definitions anywhere in this file\n"
            + "".join(f"RETRY_OPTION_{i} = {i}\n" for i in range(40)))
        _, payload, _ = self.run_cli(
            "search", "RETRY_OPTION_7", "--code", "signature")
        self.assertTrue(payload["results"])
        for r in payload["results"]:
            self.assertTrue(r.get("signature"),
                            f"no signature on {r['path']}:{r['start_line']}")

    def test_a_definition_still_reports_its_real_header(self) -> None:
        """The fallback must not shadow the header where one exists."""
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--code", "signature")
        top = payload["results"][0]
        self.assertEqual("def handle_payment():", top["signature"])

    def test_no_scale_legend_in_either_format(self) -> None:
        """The same four lines on every response, for a caller that is
        usually a program. `search --help` carries them instead."""
        _, text, _ = self.run_cli(
            "search", "handle_payment", "--no-code", "--text")
        _, payload, _ = self.run_cli("search", "handle_payment", "--no-code")
        self.assertIn("bm25", text)          # the score itself stays
        self.assertNotIn("more negative is better", text)
        self.assertNotIn("scales", payload)


class SearchHelpTests(CliFixture):
    def test_help_describes_the_output(self) -> None:
        """An agent reading `tiers: {bm25: -5.08}` cannot guess what
        that is or which way it runs."""
        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit):
                main(["search", "--help"])
        # Whitespace-normalised on both sides: the epilog wraps, and a
        # test that pins the indent fails on rewording that changed
        # nothing a reader would notice.
        flat = " ".join(out.getvalue().split())
        for expected in ("tiers", "more negative is better",
                         "cosine similarity in [-1, 1]",
                         "does NOT measure how good", "exit 0"):
            self.assertIn(expected, flat)


class NavigationTests(CliFixture):
    def test_defs_finds_the_definition(self) -> None:
        code, payload, _ = self.run_cli("defs", "handle_payment")
        self.assertEqual(0, code)
        paths = [s["path"] for s in payload["symbols"]]
        self.assertIn("src/handler.py", paths)

    def test_a_markdown_heading_is_a_definition(self) -> None:
        """Not an accident: markdown-tags.scm captures sections, so
        `# handle_payment` in README.md is a definition of that name
        alongside the Python function."""
        _, payload, _ = self.run_cli("defs", "handle_payment")
        by_lang = {s["lang"]: s["path"] for s in payload["symbols"]}
        self.assertEqual("README.md", by_lang.get("markdown"))
        self.assertEqual("src/handler.py", by_lang.get("python"))

    def test_refs_runs_and_reports_symbols(self) -> None:
        code, payload, _ = self.run_cli("refs", "handle_payment")
        self.assertEqual(0, code)
        self.assertIn("symbols", payload)


class MaintenanceTests(CliFixture):
    def test_index_reports_what_it_did(self) -> None:
        code, payload, _ = self.run_cli("index")
        self.assertEqual(0, code)
        self.assertGreater(payload["status"]["added"], 0)

    def test_status_counts_what_is_stored(self) -> None:
        self.run_cli("index")
        code, payload, _ = self.run_cli("status")
        self.assertEqual(0, code)
        self.assertGreater(payload["status"]["files"], 0)
        self.assertGreater(payload["status"]["chunks"], 0)

    def test_clear_dry_run_removes_nothing(self) -> None:
        self.run_cli("index")
        self.assertTrue((self.home / "index.db").exists())
        code, payload, _ = self.run_cli("clear", "--dry-run")
        self.assertEqual(0, code)
        self.assertTrue((self.home / "index.db").exists())
        self.assertTrue(any("would remove" in line
                            for line in payload["cleared"]))

    def test_clear_removes_only_this_index(self) -> None:
        """`data_dir` puts it outside the central store; --this still
        finds it."""
        self.run_cli("index")
        self.assertTrue((self.home / "index.db").exists())
        code, _, _ = self.run_cli("clear")
        self.assertEqual(0, code)
        self.assertFalse((self.home / "index.db").exists())


    def test_clear_reports_failure_instead_of_claiming_success(self) -> None:
        """`rmtree(..., ignore_errors=True)` swallows the error, so the
        directory has to be checked afterwards -- a symlinked index
        directory, which rmtree refuses outright, is the real case."""
        from unittest import mock

        self.run_cli("index")
        with mock.patch("shutil.rmtree"):            # silently does nothing
            code, payload, _ = self.run_cli("clear")
        self.assertEqual(1, code)
        self.assertTrue(any("could NOT remove" in line
                            for line in payload["cleared"]))
        self.assertTrue((self.home / "index.db").exists())


class InitTests(CliFixture):
    """`init` scaffolds a config; `config` only reports one."""

    def toml_path(self) -> Path:
        return self.root / "repoglass.toml"

    def test_init_writes_a_config_with_resolved_values(self) -> None:
        code, payload, _ = self.run_cli("init")
        self.assertEqual(0, code)
        text = self.toml_path().read_text()
        # Resolved, not a commented template: the key is live.
        self.assertRegex(text, r"(?m)^coverage = ")

    def test_init_says_it_pins_settings(self) -> None:
        """Writing resolved values freezes them against future default
        changes. The file has to admit that; it is the only warning the
        reader gets."""
        self.run_cli("init")
        self.assertIn("pins", self.toml_path().read_text().lower())

    def test_init_refuses_to_clobber(self) -> None:
        self.toml_path().write_text("# mine\n")
        code, _, err = self.run_cli("init")
        self.assertEqual(1, code)
        self.assertIn("exists", err)
        self.assertEqual("# mine\n", self.toml_path().read_text())

    def test_force_overwrites(self) -> None:
        self.toml_path().write_text("# mine\n")
        code, _, _ = self.run_cli("init", "--force")
        self.assertEqual(0, code)
        self.assertNotEqual("# mine\n", self.toml_path().read_text())

    def test_the_written_config_is_loadable(self) -> None:
        """A config this tool writes must be one it can read back.

        Loaded directly rather than through run_cli, which appends its
        own --config and so would never read the written file at all.
        """
        from repoglass.config import load

        self.run_cli("init")
        load(repo_config=self.toml_path())          # must not raise

    def test_secrets_are_never_written(self) -> None:
        self.run_cli("init")
        self.assertNotIn("embed_api_key", self.toml_path().read_text())


class ConfigTests(CliFixture):
    def test_config_prints_and_writes_nothing(self) -> None:
        code, text, _ = self.run_cli("config")
        self.assertEqual(0, code)
        self.assertRegex(text, r"(?m)^coverage = ")
        self.assertFalse((self.root / "repoglass.toml").exists())

    def test_config_reflects_an_override(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        cfg = Path(self.tmp.name) / "over.toml"
        cfg.write_text('embed_backend = "none"\ncandidate_depth = 77\n')
        with redirect_stdout(out), redirect_stderr(err):
            main(["config", "--repo", str(self.root), "--config", str(cfg)])
        self.assertIn("candidate_depth = 77", out.getvalue())


class UsageTests(CliFixture):
    def test_a_missing_root_fails_cleanly(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            with self.assertRaises(SystemExit):
                main(["status", "--repo", str(self.root / "nope")])
        self.assertIn("not a directory", err.getvalue())

    def test_no_subcommand_is_a_usage_error(self) -> None:
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main([])
        self.assertNotEqual(0, caught.exception.code)


class CodeLevelTests(CliFixture):
    """Three levels of body, not two.

    `--no-code` and the full span are the ends of a range; the useful
    middle is the definition's header, which is what a caller reads to
    decide whether the whole thing is worth fetching.
    """

    def test_full_is_the_default(self) -> None:
        _, payload, _ = self.run_cli("search", "handle_payment")
        first = payload["results"][0]
        self.assertIn("code", first)
        self.assertIn("return", first["code"])

    def test_signature_replaces_the_body(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--code", "signature")
        first = payload["results"][0]
        self.assertNotIn("code", first)
        self.assertEqual("def handle_payment():", first["signature"])

    def test_none_carries_neither(self) -> None:
        _, payload, _ = self.run_cli(
            "search", "handle_payment", "--code", "none")
        first = payload["results"][0]
        self.assertNotIn("code", first)
        self.assertNotIn("signature", first)

    def test_no_code_still_means_none(self) -> None:
        _, old, _ = self.run_cli("search", "handle_payment", "--no-code")
        _, new, _ = self.run_cli("search", "handle_payment", "--code", "none")
        self.assertEqual(old["results"], new["results"])

    def test_signature_is_much_smaller_than_the_body(self) -> None:
        _, full, _ = self.run_cli("search", "handle_payment")
        _, sig, _ = self.run_cli(
            "search", "handle_payment", "--code", "signature")
        self.assertLess(len(json.dumps(sig)), len(json.dumps(full)))

    def test_defs_reports_the_signature(self) -> None:
        _, payload, _ = self.run_cli("defs", "handle_payment")
        by_path = {s["path"]: s.get("signature") for s in payload["symbols"]}
        self.assertEqual("def handle_payment():",
                         by_path["src/handler.py"])

    def test_a_heading_signature_is_the_heading(self) -> None:
        """Markdown has no body field, so the first line stands in."""
        _, payload, _ = self.run_cli("defs", "handle_payment")
        by_path = {s["path"]: s.get("signature") for s in payload["symbols"]}
        self.assertEqual("# handle_payment", by_path["README.md"])


class TextFormatParityTests(CliFixture):
    """Text and JSON must agree on what a result is.

    Anything JSON carries that text drops is invisible to whoever
    reads the human format, and the two then describe different
    products.
    """

    def test_text_shows_the_signature(self) -> None:
        _, text, _ = self.run_cli(
            "search", "handle_payment", "--code", "signature", "--text")
        self.assertIn("def handle_payment():", text)

    def test_text_omits_the_body_at_signature_level(self) -> None:
        _, text, _ = self.run_cli(
            "search", "handle_payment", "--code", "signature", "--text")
        self.assertNotIn("a value long enough", text)

    def test_text_refs_show_the_enclosing_definition(self) -> None:
        (self.root / "src" / "caller.py").write_text(
            "from handler import handle_payment\n\n\n"
            "def checkout():\n    return handle_payment()\n")
        _, text, _ = self.run_cli("refs", "handle_payment", "--text")
        self.assertIn("checkout", text)

    def test_text_defs_show_the_signature(self) -> None:
        _, text, _ = self.run_cli("defs", "handle_payment", "--text")
        self.assertIn("def handle_payment():", text)


class SymbolsTests(CliFixture):
    """`symbols` enumerates and counts; `search` ranks and truncates.

    The distinction is the reason the command exists. `search` answers
    "what is the best match", which cannot answer "how many are there"
    or "is there one at all" -- and an agent that cannot ask those
    shells out to grep instead.
    """

    def test_lists_every_symbol_not_a_ranked_subset(self) -> None:
        _, payload, _ = self.run_cli("symbols")
        names = {s["name"] for s in payload["symbols"]}
        self.assertIn("handle_payment", names)
        self.assertEqual(payload["count"], len(payload["symbols"]))

    def test_a_glob_narrows_by_name(self) -> None:
        _, payload, _ = self.run_cli("symbols", "handle_*")
        self.assertTrue(payload["symbols"])
        for s in payload["symbols"]:
            self.assertTrue(s["name"].startswith("handle_"))

    def test_absence_is_provable(self) -> None:
        """The thing `search` cannot say. An empty ranked list means
        'nothing scored well'; count 0 here means 'not in the index'."""
        code, payload, _ = self.run_cli("symbols", "zzznotpresentzzz")
        self.assertEqual(0, code)
        self.assertEqual(0, payload["count"])
        self.assertEqual([], payload["symbols"])

    def test_count_by_groups_instead_of_listing(self) -> None:
        _, payload, _ = self.run_cli("symbols", "--count-by", "lang")
        self.assertIn("counts", payload)
        self.assertNotIn("symbols", payload)
        self.assertIn("python", payload["counts"])

    def test_count_by_name_pools_one_name_across_files(self) -> None:
        """`handle_payment` is a definition in handler.py and a heading
        in README.md, so grouping by name must reach 2 -- a per-file
        tally would report 1 twice."""
        _, payload, _ = self.run_cli("symbols", "--count-by", "name")
        self.assertGreaterEqual(payload["counts"]["handle_payment"], 2)
        self.assertEqual(payload["total"], sum(payload["counts"].values()))

    def test_a_limit_does_not_shrink_the_reported_total(self) -> None:
        """`--limit` caps what is shown, not what was counted. Summing
        the shown groups reports a total that is not the total, and the
        caller cannot tell it was truncated."""
        _, all_, _ = self.run_cli("symbols", "--count-by", "lang")
        _, one, _ = self.run_cli("symbols", "--count-by", "lang", "--limit", "1")
        self.assertEqual(1, len(one["counts"]))
        self.assertEqual(all_["total"], one["total"])
        self.assertEqual(all_["groups"], one["groups"])
        self.assertGreater(one["total"], sum(one["counts"].values()))

    def test_counts_are_ordered_largest_first(self) -> None:
        (self.root / "src" / "busy.py").write_text(
            "def handle_payment():\n    return 1\n")
        _, payload, _ = self.run_cli("symbols", "--count-by", "name")
        values = list(payload["counts"].values())
        self.assertEqual(values, sorted(values, reverse=True))

    def test_tag_selects_definitions_or_references(self) -> None:
        (self.root / "src" / "caller.py").write_text(
            "def checkout():\n    return handle_payment()\n")
        _, defs, _ = self.run_cli("symbols", "handle_payment", "--tag", "def")
        _, refs, _ = self.run_cli("symbols", "handle_payment", "--tag", "ref")
        self.assertTrue(all(s["tag"] == "def" for s in defs["symbols"]))
        self.assertTrue(all(s["tag"] == "ref" for s in refs["symbols"]))
        self.assertTrue(refs["symbols"])

    def test_it_counts_symbols_search_can_never_return(self) -> None:
        """A definition below MIN_CHUNK_CHARS is navigable and not
        retrievable. Enumeration must see it, or the count is a count
        of chunks wearing a symbol's name."""
        (self.root / "src" / "tiny.py").write_text("def wee():\n    pass\n")
        _, payload, _ = self.run_cli("symbols", "wee")
        _, hits, _ = self.run_cli("search", "wee")
        self.assertEqual(1, payload["count"])
        self.assertNotIn("wee", {r["name"] for r in hits["results"]})

    def test_lang_filter_applies(self) -> None:
        _, payload, _ = self.run_cli("symbols", "--lang", "markdown")
        self.assertTrue(payload["symbols"])
        self.assertTrue(all(s["lang"] == "markdown" for s in payload["symbols"]))


if __name__ == "__main__":
    unittest.main()
