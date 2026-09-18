"""Path penalties and greedy saturation for the reranking layer."""

from __future__ import annotations

import functools
import re
from pathlib import Path

from .boosting import Candidate

_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:"
    r"test_[^/]*\.py|[^/]*_test\.py|[^/]*_test\.go|[^/]*Tests?\.java"
    r"|[^/]*Test\.php|[^/]*_spec\.rb|[^/]*_test\.rb"
    r"|[^/]*\.test\.[jt]sx?|[^/]*\.spec\.[jt]sx?"
    r"|[^/]*Tests?\.kt|[^/]*Spec\.kt|[^/]*Tests?\.swift|[^/]*Spec\.swift"
    r"|[^/]*Tests?\.cs|test_[^/]*\.cpp|[^/]*_test\.cpp|test_[^/]*\.c"
    r"|[^/]*_test\.c|[^/]*Spec\.scala|[^/]*Suite\.scala|[^/]*Test\.scala"
    r"|[^/]*_test\.dart|test_[^/]*\.dart|[^/]*_spec\.lua|[^/]*_test\.lua"
    r"|test_[^/]*\.lua|test_helpers?[^/]*\.\w+"
    r")$"
)
_TEST_DIR_RE = re.compile(r"(?:^|/)(?:tests?|__tests__|spec|testing)(?:/|$)")
_COMPAT_DIR_RE = re.compile(r"(?:^|/)(?:compat|_compat|legacy)(?:/|$)")
_EXAMPLES_DIR_RE = re.compile(r"(?:^|/)(?:_?examples?|docs?_src)(?:/|$)")
_TYPE_DEFS_RE = re.compile(r"\.d\.ts$")
_REEXPORT = frozenset({"__init__.py", "package-info.java"})

_STRONG, _MODERATE, _MILD = 0.3, 0.5, 0.7


@functools.lru_cache(maxsize=4096)
def path_penalty(path: str) -> float:
    """A multiplier in (0, 1] for where a chunk lives, not what it says.

    Categories compound: `compat/__init__.py` takes both the compat and
    the re-export penalty.
    """
    norm = path.replace("\\", "/")
    p = 1.0
    if _TEST_FILE_RE.search(norm) or _TEST_DIR_RE.search(norm):
        p *= _STRONG
    if Path(path).name in _REEXPORT:
        p *= _MODERATE
    if _COMPAT_DIR_RE.search(norm):
        p *= _STRONG
    if _EXAMPLES_DIR_RE.search(norm):
        p *= _STRONG
    if _TYPE_DEFS_RE.search(norm):
        p *= _MILD
    return p


def _select_top(cands: list[Candidate], settings, *, limit: int,
                penalise_paths: bool) -> list[Candidate]:
    """Penalise, then take the top `limit` with saturation decay.

    Decay is applied greedily during selection rather than up front,
    because whether a chunk is the second from its file depends on what
    was selected before it.

    Two axes share the multiplier: one file filling every slot, and the
    same bytes appearing under several paths -- which any tree carrying
    vendored copies or dated snapshots produces. Repeats are demoted
    rather than dropped, since identical spans in different files are one
    answer to what the code does and several to where it lives.
    """
    scored = [
        (c.score * (path_penalty(c.path) if penalise_paths else 1.0), c)
        for c in cands
    ]
    scored.sort(key=lambda t: (-t[0], t[1].symbol_id))

    per_file: dict[str, int] = {}
    per_text: dict[str, int] = {}
    chosen: list[tuple[float, Candidate]] = []
    floor = float("inf")
    decay = settings.saturation_decay
    for base, c in scored:
        if len(chosen) >= limit and base <= floor:
            break
        n = per_file.get(c.path, 0)
        # Only a body that exists can be a repeat of another. A
        # candidate carrying no text is one retrieval could not load,
        # not one that says the same thing as its neighbour.
        track = bool(c.text)
        repeats = per_text.get(c.text, 0) if track else 0
        eff = base * (decay ** (n + repeats)) if n or repeats else base
        chosen.append((eff, c))
        per_file[c.path] = n + 1
        if track:
            per_text[c.text] = repeats + 1
        if len(chosen) >= limit:
            floor = min(s for s, _ in chosen)

    chosen.sort(key=lambda t: (-t[0], t[1].symbol_id))
    out: list[Candidate] = []
    for score, c in chosen[:limit]:
        c.score = score
        out.append(c)
    return out
