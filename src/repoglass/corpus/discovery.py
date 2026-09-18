"""Directory walk, exclusion, classification.

No dependency on git: untracked files are indexed and the root need not be a
repository. `.gitignore` is read as a file, not queried through the binary.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..config import Paths, Settings
from ..config import IGNORE_FILE_NAME, NEVER_INDEX_LANGS
from ..models import SourceFile
from . import languages
from .classify import classify

log = logging.getLogger(__name__)


def walk(paths: Paths, settings: Settings) -> Iterator[SourceFile]:
    """Yield every indexable file under `paths.root`.

    Ignore files are honoured per directory and inherit downward, as git
    does -- see `_load_ignore_for_dir`.

    Directories: `is_dir(follow_symlinks=False)`. Symlinked directories
    are never descended -- this is what bounds the walk, since
    `hard_exclude` matches names and cannot break a cycle. File symlinks
    are followed and deduplicated by `os.path.realpath`, so a link and
    its target index once.

    `OSError` from any entry (broken symlink, permission denied) is
    logged and skipped. One unreadable entry must not abort a walk, and
    must never surface from a read method.
    """
    root = paths.root
    pruned = set(settings.hard_exclude)

    stack: list[tuple[Path, tuple]] = [(root, ())]
    seen_real: set[str] = set()
    while stack:
        directory, inherited = stack.pop()
        spec = _load_ignore_for_dir(directory, settings)
        if spec is not None:
            inherited = (*inherited, _IgnoreSpec(base=directory, spec=spec))
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            log.debug("skipping unreadable directory %s: %s", directory, exc)
            continue
        for entry in entries:
            try:
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in pruned and not _is_ignored(
                        root, path, inherited, is_dir=True
                    ):
                        stack.append((path, inherited))
                    continue
                rel = os.path.relpath(entry.path, root).replace(os.sep, "/")
                if not is_indexable(rel):
                    continue
                if _is_ignored(root, path, inherited, is_dir=False):
                    continue
                real = os.path.realpath(entry.path)
                if real in seen_real:
                    continue
                seen_real.add(real)
                st = entry.stat(follow_symlinks=True)
            except OSError as exc:
                log.debug("skipping unreadable entry %s: %s", entry.path, exc)
                continue
            if st.st_size > settings.max_file_bytes:
                log.debug("skipping %s: %d bytes", rel, st.st_size)
                continue
            # Last, because it is the only filter that opens the file.
            if _is_generated(path, st.st_size):
                log.debug("skipping %s: lines too long to be written", rel)
                continue
            lang = languages.detect(rel)
            if lang is None:
                continue
            content_type = classify(lang, rel, settings)
            if content_type in settings.index_excluded:
                continue
            yield SourceFile(
                path=rel,
                mtime_ns=st.st_mtime_ns,
                size=st.st_size,
                lang=lang,
                content_type=content_type,
            )


#: A file whose typical line runs this long was generated, not
#: written. Hand-written code and prose wrap; a minified bundle, an
#: encoded blob or a base64 payload does not, and sits two orders of
#: magnitude above ordinary source. Set well clear of the longest
#: real files -- long changelogs and API signature dumps -- so the
#: test never has to adjudicate a close call.
MAX_MEDIAN_LINE = 1_000
#: Files smaller than this are left alone. A small file cannot
#: contribute the thousands of mangled symbols this guards against,
#: and skipping the check keeps the walk from opening almost
#: everything it sees.
_PROBE_FLOOR = 16_384
#: A generated file is generated from its first byte, so a prefix
#: answers as well as the whole and bounds the cost.
_PROBE_BYTES = 65_536


def _is_generated(path: Path, size: int) -> bool:
    """Whether the file's shape says no one typed it.

    Extension cannot separate a bundle from source -- both are
    `.js` -- and neither can size, since a few hundred kilobytes is
    an ordinary file. Line length can: the distinguishing property
    of generated output is that it does not wrap.
    """
    if size < _PROBE_FLOOR:
        return False
    try:
        with open(path, "rb") as fh:
            probe = fh.read(_PROBE_BYTES)
    except OSError:
        return False
    # Drop the final line: a truncated read almost always cuts one,
    # and a partial line is shorter than it really is.
    lines = probe.split(b"\n")[:-1]
    if len(lines) < 3:
        # Fewer than three newlines in 64 KB is itself the signature.
        return True
    lines.sort(key=len)
    return len(lines[len(lines) // 2]) > MAX_MEDIAN_LINE


def is_indexable(path: str) -> bool:
    """Whether the extension maps to a language at all.

    A tags query buys symbols and references. Retrieval does not need
    one: `window_chunks` groups lines to `window_chars` when there is no
    tags query, so a language with no `<lang>-tags.scm` is searchable as
    unnamed windows and merely not navigable.

    Files with no grammar -- .env, keys, archives, images, video,
    binaries -- are excluded by construction. `NEVER_INDEX_LANGS` covers
    what a grammar exists for and must still never be indexed.
    """
    lang = languages.detect(path)
    return lang is not None and lang not in NEVER_INDEX_LANGS


@dataclass(frozen=True)
class _IgnoreSpec:
    """One ignore file's patterns, and the directory they are relative to."""

    base: Path
    spec: object


def _load_ignore_for_dir(directory: Path, settings: Settings):
    """Compile `.gitignore` + `.repoglassignore` for one directory.

    Ignore files are per-directory in git, not per-repository: a
    monorepo with `packages/foo/.gitignore` expects those patterns to
    apply inside that package and nowhere else.

    `.gitignore` lines come first so `.repoglassignore` settles any
    disagreement within the same directory -- last match wins.
    """
    from pathspec import GitIgnoreSpec

    lines: list[str] = []
    if settings.gitignore:
        lines += _read_lines(directory / ".gitignore")
    lines += _read_lines(directory / IGNORE_FILE_NAME)
    return GitIgnoreSpec.from_lines(lines) if lines else None


def _is_ignored(rel_to: Path, path: Path, specs, is_dir: bool) -> bool:
    """Whether any inherited spec excludes `path`.

    Walk every pattern of every spec in order and keep the last
    verdict, rather than asking each spec independently. That is what
    lets a `!` pattern in a nearer ignore file re-admit a file an outer
    `.gitignore` excluded.
    """
    ignored = False
    for entry in specs:
        try:
            relative = path.relative_to(entry.base)
        except ValueError:
            continue
        text = relative.as_posix() + ("/" if is_dir else "")
        for pattern in entry.spec.patterns:
            if pattern.include is None:
                continue
            if pattern.match_file(text) is not None:
                ignored = pattern.include
    return ignored


def _read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
