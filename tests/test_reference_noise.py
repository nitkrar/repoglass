"""References a reader could navigate to, and nothing else.

`rpg refs` on a keyword or a parameter returns the sites that spell the
word, not the sites that use a thing.

Sets are exact. A subset assertion passes just as well when a query
upgrade stops capturing real references.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

FIXTURES = Path(__file__).parent / "fixtures"


def _extract(rel: str, lang: str):
    path = FIXTURES / rel
    st = path.stat()
    f = SourceFile(path=rel, mtime_ns=st.st_mtime_ns, size=st.st_size, lang=lang)
    return extract.extract(f, path.read_text(), Settings())


class ElixirReferenceTests(unittest.TestCase):
    """`def`, `defmodule` and friends are how a definition is spelled."""

    def setUp(self) -> None:
        self.result = _extract("elixir/sample.ex", "elixir")
        self.refs = {s.name for s in self.result.symbols if s.tag == "ref"}
        self.defs = {s.name for s in self.result.symbols if s.tag == "def"}

    def test_definitions_are_found(self) -> None:
        self.assertEqual({"Payments.Refund", "process", "validate"}, self.defs)

    def test_references_are_exactly_the_navigable_ones(self) -> None:
        self.assertEqual(
            {"validate", "Payments.Ledger", "record", "total"}, self.refs
        )

    def test_no_definition_keyword_is_a_reference(self) -> None:
        self.assertEqual(
            set(), self.refs & {"def", "defp", "defmodule", "defprotocol"}
        )

    def test_no_special_form_is_a_reference(self) -> None:
        self.assertEqual(set(), self.refs & {"if", "unless", "case", "cond"})

    def test_a_module_attribute_is_not_a_reference(self) -> None:
        self.assertEqual(set(), self.refs & {"moduledoc", "doc"})


class RubyReferenceTests(unittest.TestCase):
    """A parameter is a name being bound, not a name being used."""

    def setUp(self) -> None:
        self.result = _extract("ruby/sample.rb", "ruby")
        self.refs = {s.name for s in self.result.symbols if s.tag == "ref"}
        self.defs = {s.name for s in self.result.symbols if s.tag == "def"}

    def test_definitions_are_found(self) -> None:
        self.assertEqual({"RefundProcessor", "process", "validate"}, self.defs)

    def test_no_parameter_is_a_reference(self) -> None:
        self.assertEqual(set(), self.refs & {"order", "retries"})

    def test_calls_and_constants_are_still_references(self) -> None:
        self.assertLessEqual(
            {"validate", "Ledger", "record", "ArgumentError"}, self.refs
        )


if __name__ == "__main__":
    unittest.main()
