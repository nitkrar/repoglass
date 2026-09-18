"""Manage the semble benchmark corpus.

Usage:
    ./.venv/bin/python benchmarks/corpus.py --sync --languages python go
    ./.venv/bin/python benchmarks/corpus.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: Cache root for benchmark inputs and checkouts. Override with REPOGLASS_REFS.
REFS = Path(os.environ.get("REPOGLASS_REFS", "~/.cache/repoglass/refs")).expanduser()
#: Upstream benchmark checkout containing `repos.json` and `annotations/`.
SEMBLE = REFS / "semble" / "benchmarks"
#: Local checkouts of the benchmark repositories.
CORPUS = REFS / "corpus"


@dataclass(frozen=True)
class Repo:
    name: str
    language: str
    url: str
    revision: str
    benchmark_root: str | None = None

    @property
    def checkout(self) -> Path:
        return CORPUS / self.name

    @property
    def index_root(self) -> Path:
        """Subdirectory the labels target, when the benchmark pins one."""
        return (self.checkout if self.benchmark_root is None
                else self.checkout / self.benchmark_root)

    @property
    def ready(self) -> bool:
        return self.index_root.is_dir() and self._head() == self.revision

    def _head(self) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""


@dataclass(frozen=True)
class Target:
    """A labelled answer. Most are a bare path; some pin a span, and
    then a result only counts if it overlaps those lines."""

    path: str
    start_line: int | None = None
    end_line: int | None = None


@dataclass(frozen=True)
class Task:
    repo: str
    language: str
    query: str
    category: str
    #: Primary and secondary labels are pooled, which is what semble's
    #: `all_relevant` does. Keeping them apart would change the
    #: denominator and make our NDCG incomparable to theirs.
    relevant: tuple[Target, ...]


def _target(raw) -> Target:
    if isinstance(raw, str):
        return Target(path=raw)
    return Target(path=raw["path"], start_line=raw.get("start_line"),
                  end_line=raw.get("end_line"))


def repos() -> dict[str, Repo]:
    raw = json.loads((SEMBLE / "repos.json").read_text())
    return {r["name"]: Repo(**r) for r in raw}


def tasks(only: dict[str, Repo] | None = None) -> list[Task]:
    known = only if only is not None else repos()
    out: list[Task] = []
    for f in sorted((SEMBLE / "annotations").glob("*.json")):
        name = f.stem
        if name not in known:
            continue
        for item in json.loads(f.read_text()):
            repo = item.get("repo", name)
            if repo not in known:
                continue
            out.append(Task(
                repo=repo,
                language=known[repo].language,
                query=item["query"],
                category=item.get("category", "semantic"),
                relevant=tuple(_target(t) for t in item.get("relevant", []))
                         + tuple(_target(t) for t in item.get("secondary", [])),
            ))
    return out


def sync(repo: Repo) -> str:
    """Fetch just the pinned revision.

    A depth-1 fetch of one commit rather than a clone: nothing in the
    measurement reads history, and the corpus is large.
    """
    if repo.ready:
        return "ok"
    repo.checkout.mkdir(parents=True, exist_ok=True)
    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(args, cwd=repo.checkout,
                              capture_output=True, text=True)
    if not (repo.checkout / ".git").is_dir():
        run("git", "init", "-q")
        run("git", "remote", "add", "origin", repo.url)
    r = run("git", "fetch", "-q", "--depth", "1", "origin", repo.revision)
    if r.returncode:
        return f"fetch failed: {r.stderr.strip().splitlines()[-1:]}"
    r = run("git", "checkout", "-q", "--detach", repo.revision)
    if r.returncode:
        return f"checkout failed: {r.stderr.strip()}"
    if not repo.index_root.is_dir():
        return f"benchmark_root missing: {repo.benchmark_root}"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--languages", nargs="*")
    ap.add_argument("--repos", nargs="*")
    args = ap.parse_args()

    chosen = {
        n: r for n, r in repos().items()
        if (not args.languages or r.language in args.languages)
        and (not args.repos or n in args.repos)
    }
    if args.list:
        ready = sum(r.ready for r in chosen.values())
        print(f"  {len(chosen)} repos selected, {ready} synced, "
              f"{len(tasks(chosen))} queries")
        for n, r in sorted(chosen.items(), key=lambda kv: kv[1].language):
            q = len(tasks({n: r}))
            print(f"    {'ok ' if r.ready else '-- '}{r.language:<12}{n:<20}{q:>4} queries")
        return 0

    if args.sync:
        CORPUS.mkdir(parents=True, exist_ok=True)
        failed = 0
        for n, r in sorted(chosen.items()):
            status = sync(r)
            failed += status != "ok"
            print(f"  {n:<22}{r.language:<12}{status}")
        total = sum(f.stat().st_size for f in CORPUS.rglob("*") if f.is_file())
        print(f"\n  {len(chosen) - failed}/{len(chosen)} ready, "
              f"{total / 1e9:.2f} GB on disk")
        return 1 if failed else 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
