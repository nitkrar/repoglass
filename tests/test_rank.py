"""Query-shape and reranking behaviour."""

from __future__ import annotations

import unittest

from repoglass.config import Settings
from repoglass.search import fuse, lexical, rank


def cand(sid: int, path: str, text: str = "", score: float = 0.0):
    return rank.Candidate(symbol_id=sid, path=path, text=text, score=score)


class QueryShapeTests(unittest.TestCase):
    """Identifier-shaped queries take a different retrieval path."""

    def test_identifiers_are_symbol_queries(self) -> None:
        for q in ("getUserById", "UserStore", "_private", "Sinatra::Base",
                  "http_server", "Client"):
            self.assertTrue(rank.is_symbol_query(q), q)

    def test_sentences_and_plain_words_are_not(self) -> None:
        for q in ("stop the same notification twice", "session",
                  "how does the vault unlock", "payment"):
            self.assertFalse(rank.is_symbol_query(q), q)

    def test_short_semantic_phrases_stay_between_the_two_tests(self) -> None:
        """Neither classifier claims a short phrase: `looks_like_prose`
        wants four words and `is_symbol_query` wants an identifier."""
        q = "request logging middleware"
        self.assertFalse(lexical.looks_like_prose(q, min_words=4))
        self.assertFalse(rank.is_symbol_query(q))

    def test_adaptive_alpha_uses_the_selected_bucket(self) -> None:
        s = Settings(alpha_symbol=0.3, alpha_prose=0.5)
        self.assertEqual({"exact": 1.0, "vector": 0.3, "lexical": 0.7},
                         fuse.tier_weights(s, is_prose=False))
        self.assertEqual({"exact": 1.0, "vector": 0.5, "lexical": 0.5},
                         fuse.tier_weights(s, is_prose=True))


class IdentifierTests(unittest.TestCase):
    def test_camel_case_keeps_the_compound_and_the_parts(self) -> None:
        self.assertEqual(["handlerstack", "handler", "stack"],
                         rank.split_identifier("HandlerStack"))

    def test_snake_case(self) -> None:
        self.assertEqual(["my_func", "my", "func"],
                         rank.split_identifier("my_func"))

    def test_a_plain_word_is_not_split(self) -> None:
        self.assertEqual(["simple"], rank.split_identifier("simple"))

    def test_acronyms_split_on_the_boundary(self) -> None:
        self.assertEqual(["gethttpresponse", "get", "http", "response"],
                         rank.split_identifier("getHTTPResponse"))


class TokenizerTests(unittest.TestCase):
    def test_lexical_and_rerank_tokenizers_stay_distinct(self) -> None:
        query = "LOCKOUT_TIER3_ATTEMPTS"
        self.assertEqual(
            ["lockout_tier3_attempts", "lockout", "tier", "attempts"],
            lexical.tokenize_query(query),
        )
        self.assertEqual(
            ["lockout_tier3_attempts", "lockout", "tier3", "attempts"],
            rank.tokenize(query),
        )




class DefinitionDetectionTests(unittest.TestCase):
    def test_finds_a_definition(self) -> None:
        self.assertTrue(rank.defines_symbol("def register_thing(a):", "register_thing"))
        self.assertTrue(rank.defines_symbol("class UserStore:", "UserStore"))
        self.assertTrue(rank.defines_symbol("func handlePane() {", "handlePane"))

    def test_a_call_is_not_a_definition(self) -> None:
        self.assertFalse(rank.defines_symbol("register_thing(42)", "register_thing"))
        self.assertFalse(rank.defines_symbol("x = UserStore()", "UserStore"))

    def test_matching_is_case_sensitive_for_keywords(self) -> None:
        """IGNORECASE matches the word 'Module' in prose and 'Class'
        used as a method name."""
        self.assertFalse(rank.defines_symbol("Class Foo:", "Foo"))

    def test_sql_ddl_is_case_insensitive(self) -> None:
        self.assertTrue(rank.defines_symbol("create table sessions (", "sessions"))
        self.assertTrue(rank.defines_symbol("CREATE TABLE sessions (", "sessions"))


class PathPenaltyTests(unittest.TestCase):
    def test_test_files_and_dirs(self) -> None:
        self.assertAlmostEqual(0.3, rank.path_penalty("tests/test_a.py"))
        self.assertAlmostEqual(0.3, rank.path_penalty("a/FooTests.swift"))
        self.assertAlmostEqual(0.3, rank.path_penalty("src/a.spec.ts"))

    def test_penalties_compound(self) -> None:
        """A test-directory __init__.py takes both."""
        self.assertAlmostEqual(0.15, rank.path_penalty("tests/__init__.py"))

    def test_ordinary_source_is_untouched(self) -> None:
        self.assertEqual(1.0, rank.path_penalty("src/core/vault.py"))

    def test_attestation_is_not_a_test_file(self) -> None:
        """The match is segment-anchored, not substring-based."""
        self.assertEqual(1.0, rank.path_penalty("src/attestation.py"))


class LayerOrderTests(unittest.TestCase):
    S = Settings(rerank=True, file_coherence=0.2, stem_boost=1.0,
                 definition_boost=3.0, saturation_decay=0.5)

    def test_the_stem_boost_outweighs_the_coherence_boost(self) -> None:
        """A stem match should beat file coherence alone."""
        cands = [cand(1, "docs/protocol.md", score=0.50),
                 cand(2, "doorbell/suppression.py", score=0.40),
                 cand(3, "doorbell/suppression.py", score=0.39)]
        out = rank.rerank(cands, "notification suppression", self.S, limit=3)
        self.assertEqual(2, out[0].symbol_id)

    def test_a_definition_beats_a_mention(self) -> None:
        cands = [cand(1, "a/caller.py", "result = UserStore()", score=0.50),
                 cand(2, "a/store.py", "class UserStore:\n    pass", score=0.20)]
        out = rank.rerank(cands, "UserStore", self.S, limit=2)
        self.assertEqual(2, out[0].symbol_id)

    def test_saturation_spreads_results_across_files(self) -> None:
        cands = [cand(i, "big.py", score=1.0 - i * 0.01) for i in range(1, 6)]
        cands.append(cand(99, "other.py", score=0.90))
        out = rank.rerank(cands, "some prose query here", self.S, limit=3)
        self.assertIn(99, [c.symbol_id for c in out])

    def test_penalties_can_be_skipped(self) -> None:
        """content="tests" must not down-rank the tests it asked for."""
        cands = [cand(1, "tests/test_a.py", score=0.50),
                 cand(2, "src/a.py", score=0.30)]
        kept = rank.rerank(list(cands), "some prose query", self.S,
                           penalise_paths=False, limit=2)
        self.assertEqual(1, kept[0].symbol_id)
        hit = rank.rerank([cand(1, "tests/test_a.py", score=0.50),
                           cand(2, "src/a.py", score=0.30)],
                          "some prose query", self.S,
                          penalise_paths=True, limit=2)
        self.assertEqual(2, hit[0].symbol_id)

    def test_non_candidates_can_enter_the_results(self) -> None:
        """The only mechanism that surfaces a chunk no tier retrieved."""
        retrieved = [cand(1, "a/other.py", "x = 1", score=0.50)]
        missed = cand(7, "a/userstore.py", "class UserStore:\n    pass")
        out = rank.rerank(retrieved, "UserStore", self.S,
                          load_non_candidates=lambda names: [missed], limit=3)
        self.assertIn(7, [c.symbol_id for c in out])

    def test_an_empty_pool_stays_empty(self) -> None:
        self.assertEqual([], rank.rerank([], "anything", self.S, limit=5))

    def test_the_limit_is_respected(self) -> None:
        cands = [cand(i, f"f{i}.py", score=1.0 / i) for i in range(1, 30)]
        self.assertEqual(5, len(rank.rerank(cands, "prose query here",
                                            self.S, limit=5)))


class RerankReachableTests(unittest.TestCase):
    """Public search paths must reach the reranker without raising."""

    def test_search_with_rerank_on_returns_results(self) -> None:
        import tempfile
        from pathlib import Path

        from repoglass import Index
        from repoglass.config import Paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "vault.py").write_text(
                "def unlock_vault(key):\n"
                "    '''Open the vault with the supplied key.'''\n"
                "    return crypto.open(key)\n"
            )
            idx = Index.open(root, Settings(rerank=True, embed_backend="none"),
                             paths=Paths(root=root, home=Path(tmp) / "home"))
            idx.refresh()
            self.assertTrue(idx.search("unlock the vault", k=5))

    def test_it_also_works_under_a_content_filter(self) -> None:
        """`penalise_paths` is computed from `content`, so a filtered
        search reaches the layer by a different route."""
        import tempfile
        from pathlib import Path

        from repoglass import Index
        from repoglass.config import Paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "vault.py").write_text(
                "def unlock_vault(key):\n"
                "    '''Open the vault with the supplied key.'''\n"
                "    return crypto.open(key)\n"
            )
            idx = Index.open(root, Settings(rerank=True, embed_backend="none"),
                             paths=Paths(root=root, home=Path(tmp) / "home"))
            idx.refresh()
            idx.search("unlock the vault", k=5, content="code")


if __name__ == "__main__":
    unittest.main()
