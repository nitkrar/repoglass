"""Tier merging and candidate depth."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.search import fuse


class MergeTests(unittest.TestCase):
    def test_ranker_none_takes_first_non_empty_tier(self) -> None:
        empty = fuse.ranked("exact", ())
        lex = fuse.ranked("lexical", ((9, -1.0),), lower_is_better=True)
        vec = fuse.ranked("vector", ((4, 0.9),))
        out = fuse.merge([empty, lex, vec], Settings(ranker="none"))
        self.assertEqual([9], [sid for sid, _ in out])

    def test_scores_are_normalised_whatever_the_ranker(self) -> None:
        lex = fuse.ranked("lexical", ((9, -1.0), (8, -3.0)), lower_is_better=True)
        for ranker in ("none", "rrf"):
            out = fuse.merge([lex], Settings(ranker=ranker))
            self.assertTrue(all(0.0 <= s <= 1.0 for _, s in out), ranker)


class RankerNoneTests(unittest.TestCase):
    """With ranker='none' the first tier's own order is the answer."""

    def test_the_best_bm25_match_comes_first(self) -> None:
        from repoglass import Index
        from repoglass.config import Paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            (root / "strong.py").write_text(
                "def settle_ledger():\n"
                "    # ledger ledger ledger: settle the ledger balance\n"
                "    return ledger.settle_ledger_balance()\n")
            (root / "weak.py").write_text(
                "def unrelated_helper():\n"
                "    # formats a report; mentions a ledger only once here\n"
                "    return format_the_report_for_printing()\n")
            idx = Index.open(root, Settings(embed_backend="none", ranker="none",
                                            rerank=False),
                             paths=Paths(root=root, home=Path(tmp) / "h"))
            hits = idx.search("ledger", k=2, content="all")
            idx.close()
        self.assertEqual(["strong.py", "weak.py"], [h.path for h in hits])
        lexical = [dict(h.tiers)["lexical"] for h in hits]
        self.assertLess(lexical[0], lexical[1])

    def test_equal_exact_matches_keep_path_order_after_a_refresh(self) -> None:
        from repoglass import Index
        from repoglass.config import Paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            for path in ("a.py", "b.py"):
                (root / path).write_text(
                    "def shared_name():\n"
                    "    # long enough to be a retrievable chunk\n"
                    f"    return {path!r}\n")
            settings = Settings(embed_backend="none", ranker="none",
                                rerank=False)
            idx = Index.open(root, settings,
                             paths=Paths(root=root, home=Path(tmp) / "h"))
            idx.refresh()
            (root / "a.py").write_text(
                "def shared_name():\n"
                "    # changed and long enough to be a retrievable chunk\n"
                "    return 'changed'\n")
            idx.refresh()
            hits = idx.search("shared_name", k=2, content="all")
            idx.close()
        self.assertEqual(["a.py", "b.py"], [hit.path for hit in hits])


class CandidateDepthTests(unittest.TestCase):
    """The top results must not depend on how many the caller asked for.

    A depth of `k * overfetch` gives each tier a longer list as `k`
    grows, and RRF -- which rewards appearing in several tiers -- then
    lets deep-but-broad chunks overtake a shallow-but-strong one. The
    same query against the same index answers differently depending on
    how many results were asked for.
    """

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.root.mkdir(parents=True)
        for i in range(12):
            (self.root / f"m{i}.py").write_text(
                f"def handle_payment_{i}(amount):\n"
                f"    # settle payment number {i} against the ledger\n"
                f"    return ledger.settle(amount, {i})\n")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_top_three_is_the_same_whatever_k_is(self) -> None:
        from repoglass import Index
        from repoglass.config import Paths
        idx = Index.open(self.root, Settings(embed_backend="none"),
                         paths=Paths(root=self.root, home=Path(self.tmp.name) / "h"))
        idx.refresh()
        first = [(h.path, h.start_line) for h in idx.search("settle payment ledger", k=3)]
        for k in (5, 10, 25):
            wider = [(h.path, h.start_line)
                     for h in idx.search("settle payment ledger", k=k)][:3]
            self.assertEqual(first, wider, f"top-3 changed at k={k}")


if __name__ == "__main__":
    unittest.main()
