"""Compare repoglass symbol navigation with grep on Python trees.

Usage:
    ./.venv/bin/python benchmarks/navigation.py <tree> [--sample 40]
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus import REFS                          # noqa: E402
from repoglass import Index                      # noqa: E402
from repoglass.config import Paths, Settings     # noqa: E402


def oracle(root: Path) -> tuple[dict[str, set], dict[str, set]]:
    """name -> {(relpath, line)} for definitions and for references."""
    defs: dict[str, set] = defaultdict(set)
    refs: dict[str, set] = defaultdict(set)
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        try:
            tree = ast.parse(path.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                defs[node.name].add((rel, node.lineno))
            elif isinstance(node, ast.Name):
                refs[node.id].add((rel, node.lineno))
            elif isinstance(node, ast.Attribute):
                refs[node.attr].add((rel, node.lineno))
    return defs, refs


def score(found: set, truth: set) -> tuple[float, float]:
    """Precision and recall at file level.

    Files, not lines: the question a caller asks is "which files
    should I open", and a one-line offset between two parsers is not
    a wrong answer to that.
    """
    f = {p for p, _ in found}
    t = {p for p, _ in truth}
    if not f and not t:
        return 1.0, 1.0
    precision = len(f & t) / len(f) if f else 0.0
    recall = len(f & t) / len(t) if t else 1.0
    return precision, recall


def grep(root: Path, pattern: str, extended: bool = False) -> set:
    cmd = ["grep", "-rn", "--include=*.py"]
    cmd.append("-E" if extended else "-w")
    cmd += [pattern, "."]
    out = subprocess.run(cmd, cwd=root, capture_output=True, text=True).stdout
    found = set()
    for line in out.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        found.add((parts[0].removeprefix("./"), int(parts[1])))
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", type=Path)
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--home", default=str(REFS / "index-nav"))
    args = ap.parse_args()
    root = args.tree.resolve()

    true_defs, true_refs = oracle(root)
    # Names worth asking about: defined exactly once in this tree, so
    # "where is it defined" has one right answer, and referenced at
    # least twice, so "who uses it" is a real question.
    candidates = [n for n, d in true_defs.items()
                  if len(d) == 1 and len(true_refs.get(n, ())) >= 2
                  and not n.startswith("__")]
    candidates.sort(key=lambda n: -len(true_refs[n]))
    names = candidates[:args.sample]
    print(f"tree={root}  python files={len(list(root.rglob('*.py')))}"
          f"  symbols tested={len(names)}")

    idx = Index.open(root, Settings(embed_backend="none"),
                     paths=Paths(root=root, home=Path(args.home)))
    t = time.perf_counter()
    idx.refresh()
    print(f"index built in {time.perf_counter() - t:.1f}s\n")

    totals: dict[tuple[str, str], list] = defaultdict(list)
    counts: Counter = Counter()
    timing: Counter = Counter()

    for name in names:
        t = time.perf_counter()
        rg_def = {(s.path, s.start_line) for s in idx.definitions(name)}
        timing["rpg defs"] += time.perf_counter() - t

        t = time.perf_counter()
        gp_def = grep(root, rf"^[[:space:]]*(def|class)[[:space:]]+{re.escape(name)}\b",
                      extended=True)
        timing["grep defs"] += time.perf_counter() - t

        for label, got in (("rpg defs", rg_def), ("grep defs", gp_def)):
            p, r = score(got, true_defs[name])
            totals[(label, "p")].append(p)
            totals[(label, "r")].append(r)
            counts[label] += len({x for x, _ in got})

        t = time.perf_counter()
        rg_ref = {(s.path, s.start_line) for s in idx.references(name)}
        timing["rpg refs"] += time.perf_counter() - t

        t = time.perf_counter()
        gp_ref = grep(root, name)
        timing["grep refs"] += time.perf_counter() - t

        for label, got in (("rpg refs", rg_ref), ("grep refs", gp_ref)):
            p, r = score(got, true_refs[name])
            totals[(label, "p")].append(p)
            totals[(label, "r")].append(r)
            counts[label] += len({x for x, _ in got})

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    print(f"{'':<12}{'precision':>11}{'recall':>9}{'F1':>8}"
          f"{'files/query':>13}{'ms/query':>10}")
    print("-" * 63)
    for label in ("rpg defs", "grep defs", "rpg refs", "grep refs"):
        p, r = mean(totals[(label, "p")]), mean(totals[(label, "r")])
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        print(f"{label:<12}{p:>11.3f}{r:>9.3f}{f1:>8.3f}"
              f"{counts[label] / len(names):>13.1f}"
              f"{timing[label] / len(names) * 1000:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
