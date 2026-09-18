"""Classification of a file into one content category.

Deliberately not a table of language -> category assertions: those
restate `doc_languages` and friends in a second place, break when the
tuples are edited legitimately, and pass when the logic breaks. What is
tested here is what the tuples cannot say -- the order categories are
tried in, where a marker boundary falls, that the lists are read rather
than baked, and that every language lands somewhere.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from repoglass.config import Settings
from repoglass.corpus.classify import classify


class PrecedenceTests(unittest.TestCase):
    """Language decides before path does, and the order is load-bearing."""

    def test_markdown_under_tests_is_docs_not_tests(self) -> None:
        self.assertEqual(
            "docs", classify("markdown", "tests/README.md", Settings()))

    def test_csv_under_tests_is_data_not_tests(self) -> None:
        self.assertEqual(
            "data", classify("csv", "tests/fixtures/rows.csv", Settings()))

    def test_config_language_under_tests_is_config_not_tests(self) -> None:
        self.assertEqual(
            "config", classify("json", "tests/fixtures/case.json", Settings()))

    def test_python_under_tests_is_tests(self) -> None:
        self.assertEqual(
            "tests", classify("python", "tests/test_thing.py", Settings()))

    def test_python_outside_tests_is_code(self) -> None:
        self.assertEqual(
            "code", classify("python", "src/thing.py", Settings()))


class ConfigIsReadTests(unittest.TestCase):
    """The lists are consulted, not inlined. Fails if someone hardcodes."""

    def test_adding_a_language_moves_its_category(self) -> None:
        """`rust` is in no list, so it is `code` by fallthrough until a
        list claims it."""
        self.assertEqual("code", classify("rust", "a.rs", Settings()))
        settings = replace(Settings(), data_languages=("rust",))
        self.assertEqual("data", classify("rust", "a.rs", settings))

    def test_earlier_category_wins_when_a_language_is_in_two_lists(self) -> None:
        """json ships in config_languages; listing it as data too must
        not make classification depend on dict ordering."""
        settings = replace(Settings(), data_languages=("json",))
        self.assertEqual("config", classify("json", "a.json", settings))

    def test_emptying_a_list_falls_through(self) -> None:
        settings = replace(Settings(), doc_languages=())
        self.assertEqual("code", classify("markdown", "a.md", settings))


class TotalityTests(unittest.TestCase):
    """Ungating took the walk from 20 languages to 150, and the set grows
    without us touching anything."""

    def test_every_known_language_gets_a_category(self) -> None:
        from grep_ast.parsers import PARSERS

        valid = {"code", "tests", "docs", "config", "data"}
        for lang in sorted(set(PARSERS.values())):
            with self.subTest(lang=lang):
                self.assertIn(classify(lang, f"src/a.{lang}", Settings()),
                              valid)

    def test_unknown_language_is_code(self) -> None:
        self.assertEqual(
            "code", classify("nosuchlang", "src/a.xyz", Settings()))


if __name__ == "__main__":
    unittest.main()
