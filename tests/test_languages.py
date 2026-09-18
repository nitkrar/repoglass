"""Language detection and query capability metadata."""

from __future__ import annotations

import unittest

from repoglass.corpus import languages
from repoglass.models import LanguageCapability


class DetectTests(unittest.TestCase):
    def test_detects_python(self) -> None:
        self.assertEqual("python", languages.detect("src/thing.py"))

    def test_detects_markdown(self) -> None:
        self.assertEqual("markdown", languages.detect("docs/README.md"))

    def test_unknown_extension_is_none(self) -> None:
        self.assertIsNone(languages.detect("secrets/.env"))
        self.assertIsNone(languages.detect("image.png"))


class TagQueryTests(unittest.TestCase):
    def test_python_query_compiles_against_its_grammar(self) -> None:
        from grep_ast.tsl import get_language
        from tree_sitter import Query

        src = languages.tag_query("python")
        self.assertIsNotNone(src)
        Query(get_language("python"), src)   # raises if the query is invalid

    def test_unknown_language_has_no_query(self) -> None:
        self.assertIsNone(languages.tag_query("klingon"))

    def test_swift_query_includes_supplemental_references(self) -> None:
        """Upstream swift tags provide definitions only; refs are ours."""
        src = languages.tag_query("swift")
        self.assertIn("name.definition.", src)
        self.assertIn("name.reference.", src)


class CapabilityTests(unittest.TestCase):
    def test_python_has_definitions_and_references(self) -> None:
        cap = languages.capability("python")
        self.assertTrue(cap.definitions)
        self.assertTrue(cap.references)

    def test_swift_gains_references_from_supplement(self) -> None:
        cap = languages.capability("swift")
        self.assertTrue(cap.definitions)
        self.assertTrue(cap.references)

    def test_unknown_language_has_neither(self) -> None:
        cap = languages.capability("klingon")
        self.assertFalse(cap.definitions)
        self.assertFalse(cap.references)

    def test_capability_names_the_kinds_captured(self) -> None:
        """Not a bool. Which kinds is the question a caller has."""
        kinds = languages.capability("python").reference_kinds
        self.assertIn("call", kinds)
        self.assertIn("type", kinds)

    def test_references_without_calls_is_representable(self) -> None:
        """Type-only captures must not imply call coverage."""
        cap = LanguageCapability(lang="x", definitions=True,
                                 reference_kinds=frozenset({"type"}))
        self.assertTrue(cap.references)
        self.assertFalse(cap.calls)


#: Derived from the query files on disk.
SUPPORTED = sorted(
    p.name.removesuffix("-tags.scm")
    for p in languages.QUERY_DIR.glob("*-tags.scm")
)


class VendoredQueryTests(unittest.TestCase):
    """Every shipped query must compile against the loaded grammar."""

    def test_every_supported_query_compiles(self) -> None:
        from grep_ast.tsl import get_language
        from tree_sitter import Query

        broken = []
        for lang in SUPPORTED:
            src = languages.tag_query(lang)
            if src is None:
                broken.append(f"{lang}: no query file")
                continue
            try:
                Query(get_language(lang), src)
            except Exception as exc:
                broken.append(f"{lang}: {type(exc).__name__}: {exc}")
        self.assertEqual([], broken, "\n".join(broken))

    def test_every_supported_language_captures_both(self) -> None:
        """Every supported language except markdown exposes both kinds."""
        missing = [
            lang
            for lang in SUPPORTED
            if lang != "markdown"
            and not (languages.capability(lang).definitions
                     and languages.capability(lang).references)
        ]
        self.assertEqual([], missing)

    def test_every_supported_language_can_answer_who_calls_this(self) -> None:
        """Call-site queries are stronger than generic references."""
        missing = [lang for lang in SUPPORTED if lang != "markdown"
                   and not languages.capability(lang).calls]
        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
