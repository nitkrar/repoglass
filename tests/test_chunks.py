"""Chunk construction.

`text` is what gets embedded and what FTS5 reads through the
`chunk_lexical` view. `lexical()` renders the FTS5 form on demand.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import Chunk, SourceFile

FIXTURE = Path(__file__).parent / "fixtures" / "python" / "sample.py"


def _source_file() -> SourceFile:
    st = FIXTURE.stat()
    return SourceFile(
        path="src/thing/sample.py",
        mtime_ns=st.st_mtime_ns,
        size=st.st_size,
        lang="python",
    )


class EmbeddedTextTests(unittest.TestCase):
    """What gets embedded is the raw span, verbatim.

    A model trained on code should be given code, so nothing distils,
    summarises or reorders the span on the way to the embedder.
    """

    def test_chunk_text_is_the_source_span(self) -> None:
        body = ("def handle_payment(amount):\n"
                "    # settle against the ledger\n"
                "    return ledger.settle(amount)\n")
        f = SourceFile(path="pay.py", mtime_ns=1, size=1, lang="python")
        chunk = extract.extract(f, body, Settings()).chunks[0]
        self.assertEqual(body.rstrip("\n"), chunk.text)
        self.assertIn("# settle against the ledger", chunk.text)

    def test_the_path_reaches_fts_but_not_the_embedder(self) -> None:
        """BM25 benefits from path tokens; the embedder does not. The
        path is prepended by `lexical()`, never by the stored text."""
        body = ("def handle_payment(amount):\n"
                "    return ledger.settle(amount)  # long enough to chunk\n")
        f = SourceFile(path="src/billing/pay.py", mtime_ns=1, size=1,
                       lang="python")
        chunk = extract.extract(f, body, Settings()).chunks[0]
        self.assertNotIn("billing", chunk.text)
        self.assertIn("billing", extract.lexical(
            path=f.path, body=chunk.text, settings=Settings()))



class LexicalTests(unittest.TestCase):
    def test_preserves_string_literals(self) -> None:
        """The whole point: keyword search must reach error messages."""
        body = 'def f():\n    raise ValueError("vault is already unlocked")'
        out = extract.lexical(path="a.py", body=body, settings=Settings())
        self.assertIn("vault is already unlocked", out)

    def test_includes_humanised_path(self) -> None:
        out = extract.lexical(
            path="src/core/Mailer.php", body="x", settings=Settings()
        )
        self.assertIn("Mailer", out)

    def test_capped_mode_truncates(self) -> None:
        body = "x" * 5000
        out = extract.lexical(
            path="a.py", body=body,
            settings=Settings(lexical_mode="capped", lexical_cap_chars=100),
        )
        self.assertLess(len(out), 400)

    def test_full_mode_does_not_truncate(self) -> None:
        body = "y" * 5000
        out = extract.lexical(
            path="a.py", body=body, settings=Settings(lexical_mode="full")
        )
        self.assertIn("y" * 5000, out)


class ChunkBuildTests(unittest.TestCase):
    """build() owns identity, not rendering."""

    def test_content_hash_is_stable_and_span_sensitive(self) -> None:
        kw = dict(path="a.py", name="f", start_line=1, end_line=2,
                  leading_comments=[], node_kind="function",
                  text="t")
        a = Chunk.build(source="def f(): return 1", **kw)
        b = Chunk.build(source="def f(): return 1", **kw)
        c = Chunk.build(source="def f(): return 2", **kw)
        self.assertEqual(a.content_hash, b.content_hash)
        self.assertNotEqual(a.content_hash, c.content_hash)

    def test_hash_tracks_source_not_rendered_text(self) -> None:
        """Re-embedding is skipped on hash match, so the hash must follow
        the source span, not the text rendered from it."""
        a = Chunk.build(path="a.py", name="f", start_line=1, end_line=2,
                        source="def f(): return 1", leading_comments=[],
                        node_kind="function", text="one")
        b = Chunk.build(path="a.py", name="f", start_line=1, end_line=2,
                        source="def f(): return 1", leading_comments=[],
                        node_kind="function", text="TWO")
        self.assertEqual(a.content_hash, b.content_hash)


class ExtractionProducesChunksTests(unittest.TestCase):
    def test_short_definition_yields_symbol_but_no_chunk(self) -> None:
        """`tiny` is below MIN_CHUNK_CHARS: navigable, not retrievable."""
        result = extract.extract(_source_file(), FIXTURE.read_text(), Settings())
        defs = {s.name for s in result.symbols if s.tag == "def"}
        chunked = {c.name for c in result.chunks}
        self.assertIn("tiny", defs)
        self.assertNotIn("tiny", chunked)

    def test_substantial_definitions_are_chunked(self) -> None:
        result = extract.extract(_source_file(), FIXTURE.read_text(), Settings())
        chunked = {c.name for c in result.chunks}
        self.assertIn("outer", chunked)
        self.assertIn("Thing", chunked)


class MarkdownSectionSpanTests(unittest.TestCase):
    """A heading owns its own prose, not its subsections'.

    The grammar nests `section` inside `section`, so a span taken whole
    puts every descendant's text in the ancestor's chunk and indexes a
    deep paragraph once per level above it.
    """

    DOC = ("# Top\n\nParent prose long enough to clear the minimum chunk"
           " size comfortably.\n\n"
           "## Sub\n\nChild prose mentioning ONLYINCHILD and also long"
           " enough to be chunked.\n\n"
           "### Deep\n\nDeeper prose mentioning ONLYINDEEP at a length"
           " that clears the minimum.\n")

    def chunks(self):
        f = SourceFile(path="doc.md", mtime_ns=1, size=len(self.DOC),
                       lang="markdown")
        return extract.extract(f, self.DOC, Settings()).chunks

    def test_an_ancestor_does_not_hold_its_descendants_prose(self) -> None:
        for c in self.chunks():
            if c.name == "Top":
                self.assertNotIn("ONLYINCHILD", c.text)
                self.assertNotIn("ONLYINDEEP", c.text)
            if c.name == "Sub":
                self.assertNotIn("ONLYINDEEP", c.text)

    def test_every_heading_still_keeps_its_own_prose(self) -> None:
        """Trimming must not cut a section down to its heading line."""
        by_name = {c.name: c.text for c in self.chunks()}
        self.assertIn("Parent prose", by_name["Top"])
        self.assertIn("ONLYINCHILD", by_name["Sub"])
        self.assertIn("ONLYINDEEP", by_name["Deep"])

    def test_nothing_is_indexed_twice(self) -> None:
        total = sum(len(c.text) for c in self.chunks())
        self.assertLessEqual(total, len(self.DOC))

    def test_a_nested_class_keeps_its_whole_span(self) -> None:
        """Only `section` is trimmed. A class containing a method is
        composition -- the method is part of what the class is."""
        src = ("class Outer:\n"
               "    class Inner:\n"
               "        def work(self):\n"
               "            return 'a value long enough to be chunked'\n")
        f = SourceFile(path="a.py", mtime_ns=1, size=len(src), lang="python")
        outer = [c for c in extract.extract(f, src, Settings()).chunks
                 if c.name == "Outer"]
        self.assertTrue(outer)
        self.assertIn("Inner", outer[0].text)


class LexicalEnrichmentTests(unittest.TestCase):
    """semble's FTS input shape, behind `lexical_enrich`."""

    @staticmethod
    def lexical(path: str, body: str) -> str:
        return extract.lexical(
            path=path, body=body, settings=Settings(lexical_enrich=True)
        )

    def test_the_stem_is_repeated(self) -> None:
        """Doubled on purpose: a file named suppression.py should be
        reachable by "suppression" even when the body never says it."""
        out = self.lexical("a/b/suppression.py", "def x(): pass")
        self.assertEqual(2, out.split().count("suppression"))

    def test_the_last_directories_are_included(self) -> None:
        out = self.lexical("src/doorbell/suppression.py", "body here")
        self.assertIn("doorbell", out.split())

    def test_only_the_last_three_directories(self) -> None:
        out = self.lexical("a/b/c/d/e/f.py", "body")
        self.assertNotIn("a", out.split())
        self.assertIn("e", out.split())

    def test_the_setting_selects_the_shape(self) -> None:
        plain = extract.lexical(path="src/pay.py", body="def x(): pass",
                                settings=Settings(lexical_enrich=False))
        rich = extract.lexical(path="src/pay.py", body="def x(): pass",
                               settings=Settings(lexical_enrich=True))
        self.assertEqual(1, rich.split().count("src"))
        self.assertNotEqual(plain, rich)


if __name__ == "__main__":
    unittest.main()
