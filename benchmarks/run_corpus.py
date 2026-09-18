"""Score repoglass on the semble benchmark corpus.

Usage:
    ./.venv/bin/python benchmarks/run_corpus.py
    ./.venv/bin/python benchmarks/run_corpus.py --config path/to/repoglass.toml
    ./.venv/bin/python benchmarks/run_corpus.py --languages python go
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repoglass import Index                      # noqa: E402
from repoglass.config import Paths, Settings     # noqa: E402

from corpus import REFS, repos, tasks            # noqa: E402

TOP_K = 10


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevant_ranks: list[int], n_relevant: int, k: int) -> float:
    if n_relevant == 0:
        return 0.0
    relevances = [0] * k
    for rank in relevant_ranks:
        if 1 <= rank <= k:
            relevances[rank - 1] = 1
    ideal = dcg([1] * min(k, n_relevant))
    return dcg(relevances) / ideal if ideal > 0 else 0.0


def path_matches(file_path: str, target_path: str) -> bool:
    """Suffix either way, because we index a `benchmark_root` while the
    labels are written against the repository root."""
    a = file_path.replace("\\", "/")
    b = target_path.replace("\\", "/")
    return a == b or a.endswith(f"/{b}") or b.endswith(f"/{a}")


def covers(path: str, start: int, end: int, target) -> bool:
    """semble's `target_matches_location`: the path must match, and if
    the label pins a span the result must overlap it."""
    if not path_matches(path, target.path):
        return False
    if target.start_line is None or target.end_line is None:
        return True
    return not (end < target.start_line or start > target.end_line)


def first_rank(hits, target) -> int | None:
    for i, h in enumerate(hits, 1):
        if covers(h.path, h.start_line, h.end_line, target):
            return i
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    # Retrieval settings come from the same config loader as the product.
    ap.add_argument("--config", type=Path,
                    help="TOML overriding the defaults, same schema as"
                         " `as_toml()` output")
    # Selection and output paths only below here.
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--repos", nargs="*")
    ap.add_argument("--home", default=str(REFS / "index"),
                    help="where built indexes go; not /tmp, which gets swept")
    ap.add_argument("--json", dest="out")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    chosen = {
        n: r for n, r in repos().items()
        if r.ready
        and (not args.languages or r.language in args.languages)
        and (not args.repos or n in args.repos)
    }
    if not chosen:
        print("  nothing synced; run benchmarks/corpus.py --sync first")
        return 1

    from repoglass.config import load
    settings = load(repo_config=args.config)
    # Benchmarks build once and query many times; short auto-refresh is noise.
    settings = replace(settings, rescan_after_seconds=3600)

    by_repo = defaultdict(list)
    for t in tasks(chosen):
        by_repo[t.repo].append(t)

    rows: list[dict] = []
    index_s: dict[str, float] = {}
    for name, repo in sorted(chosen.items()):
        t0 = time.perf_counter()
        idx = Index.open(repo.index_root, settings,
                         paths=Paths(root=repo.index_root,
                                     home=Path(args.home)))
        idx.refresh()
        index_s[name] = time.perf_counter() - t0
        for task in by_repo[name]:
            t0 = time.perf_counter()
            hits = idx.search(task.query, k=TOP_K)
            ms = (time.perf_counter() - t0) * 1000
            ranks = [r for tgt in task.relevant
                     if (r := first_rank(hits, tgt)) is not None]
            rows.append({
                "repo": name, "language": task.language,
                "category": task.category, "query": task.query,
                "ndcg10": ndcg_at_k(ranks, len(task.relevant), TOP_K),
                "ndcg5": ndcg_at_k(ranks, len(task.relevant), 5),
                "hit": bool(ranks), "ms": round(ms, 2),
            })

    def report(title: str, key) -> None:
        print(f"\n  {title:<22}{'n':>6}{'NDCG@10':>10}{'NDCG@5':>9}{'any hit':>10}")
        print("  " + "-" * 57)
        groups = defaultdict(list)
        for r in rows:
            groups[key(r)].append(r)
        for g, rs in sorted(groups.items()):
            print(f"  {str(g):<22}{len(rs):>6}"
                  f"{sum(x['ndcg10'] for x in rs)/len(rs):>10.3f}"
                  f"{sum(x['ndcg5'] for x in rs)/len(rs):>9.3f}"
                  f"{sum(x['hit'] for x in rs)/len(rs):>9.0%}")

    n = len(rows)
    ms = sorted(r["ms"] for r in rows)
    print(f"\n  coverage={settings.coverage}  rerank={settings.rerank}"
          f"  saturation={settings.saturation_decay}"
          f"  depth={settings.candidate_depth}"
          f"  embedder={settings.embed_backend}"
          f"  {len(chosen)} repos  {n} queries")
    print(f"  NDCG@10 {sum(r['ndcg10'] for r in rows)/n:.3f}"
          f"   NDCG@5 {sum(r['ndcg5'] for r in rows)/n:.3f}"
          f"   any hit {sum(r['hit'] for r in rows)/n:.0%}")
    print(f"  index {sum(index_s.values()):.1f}s total"
          f"   query p50 {ms[len(ms)//2]:.1f}ms  p95 {ms[int(n*0.95)]:.1f}ms")
    report("by category", lambda r: r["category"])
    report("by language", lambda r: r["language"])

    if args.out:
        Path(args.out).write_text(json.dumps({
            "label": args.label or f"repoglass {settings.coverage}",
            "meta": {"coverage": settings.coverage, "repos": len(chosen),
                     "queries": n,
                     "ndcg10": round(sum(r["ndcg10"] for r in rows)/n, 4),
                     "query_ms_p50": ms[len(ms)//2],
                     "index_s_total": round(sum(index_s.values()), 1)},
            "results": rows}, indent=1))
        print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
