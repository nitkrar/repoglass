"""Retrieval tiers and fusion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.search import fuse


class RankedListTests(unittest.TestCase):
    def test_ranks_are_one_based_and_ordered(self) -> None:
        rl = fuse.RankedList(tier="lexical", items=((7, -1.2), (9, -0.4)))
        self.assertEqual({7: 1, 9: 2}, rl.ranks())


class NormaliseTests(unittest.TestCase):
    def test_maps_onto_zero_one_higher_better(self) -> None:
        out = dict(fuse.normalise([(1, -5.0), (2, -1.0), (3, -3.0)]))
        self.assertAlmostEqual(1.0, out[2])   # bm25 least-negative is best
        self.assertAlmostEqual(0.0, out[1])
        self.assertTrue(0.0 < out[3] < 1.0)

    def test_single_item_is_top_scored(self) -> None:
        self.assertEqual([(4, 1.0)], fuse.normalise([(4, -2.0)]))

    def test_identical_scores_do_not_divide_by_zero(self) -> None:
        out = fuse.normalise([(1, 2.0), (2, 2.0)])
        self.assertEqual(2, len(out))
        self.assertTrue(all(0.0 <= s <= 1.0 for _, s in out))

    def test_empty_input(self) -> None:
        self.assertEqual([], fuse.normalise([]))


class RrfTests(unittest.TestCase):
    def test_appearing_in_both_tiers_beats_appearing_in_one(self) -> None:
        """The actual value of fusion: corroboration across retrievers.

        Note RRF does NOT favour a consistent middle rank over a first
        place. With k=60, rank 1 + rank 3 scores 1/61 + 1/63 = 0.032266,
        just above rank 2 twice at 1/62 + 1/62 = 0.032258.
        """
        a = fuse.RankedList("lexical", ((1, 0.0), (2, 0.0)))
        b = fuse.RankedList("vector", ((1, 0.0), (3, 0.0)))
        merged = [sid for sid, _ in fuse.rrf([a, b], k=60)]
        self.assertEqual(1, merged[0])

    def test_a_lone_top_hit_can_outrank_a_consistent_runner_up(self) -> None:
        a = fuse.RankedList("lexical", ((1, 0.0), (2, 0.0), (3, 0.0)))
        b = fuse.RankedList("vector", ((3, 0.0), (2, 0.0), (1, 0.0)))
        scores = dict(fuse.rrf([a, b], k=60))
        self.assertGreater(scores[1], scores[2])
        self.assertAlmostEqual(scores[1], scores[3])

    def test_single_list_preserves_order(self) -> None:
        a = fuse.RankedList("lexical", ((5, 0.0), (6, 0.0)))
        self.assertEqual([5, 6], [sid for sid, _ in fuse.rrf([a], k=60)])


class MergeTests(unittest.TestCase):
    def test_ranker_none_takes_first_non_empty_tier(self) -> None:
        empty = fuse.RankedList("exact", ())
        lex = fuse.RankedList("lexical", ((9, -1.0),))
        vec = fuse.RankedList("vector", ((4, 0.9),))
        out = fuse.merge([empty, lex, vec], Settings(ranker="none"))
        self.assertEqual([9], [sid for sid, _ in out])

    def test_scores_are_normalised_whatever_the_ranker(self) -> None:
        lex = fuse.RankedList("lexical", ((9, -1.0), (8, -3.0)))
        for ranker in ("none", "rrf"):
            out = fuse.merge([lex], Settings(ranker=ranker))
            self.assertTrue(all(0.0 <= s <= 1.0 for _, s in out), ranker)


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
