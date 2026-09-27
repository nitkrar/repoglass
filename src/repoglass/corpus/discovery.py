"""Directory walk, exclusion, classification.

No dependency on git: untracked files are indexed and the root need not be a
repository. `.gitignore` is read as a file, not queried through the binary.
"""

from __future__ import annotations

import logging
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

    semsift's `discover` walks the tree and applies `.gitignore` (when
    `settings.gitignore`) and `.repoglassignore` per directory, the
    latter settling disagreements; `hard_exclude` prunes directories by
    name. repoglass then drops what it cannot or should not index: files
    over `max_file_bytes`, generated files, languages it does not detect,
    and `index_excluded` content types.
    """
    from semsift.files import discover

    root = paths.root
    pruned = set(settings.hard_exclude)
    ignore_files = ((".gitignore",) if settings.gitignore else ()) + (IGNORE_FILE_NAME,)

    def include(rel: str, is_dir: bool) -> bool:
        if is_dir:
            return rel.rsplit("/", 1)[-1] not in pruned
        return is_indexable(rel)

    for found in discover(root, ignore_files=ignore_files, include=include):
        if found.size > settings.max_file_bytes:
            log.debug("skipping %s: %d bytes", found.path, found.size)
            continue
        # Last, because it is the only filter that opens the file.
        if _is_generated(root / found.path, found.size):
            log.debug("skipping %s: lines too long to be written", found.path)
            continue
        lang = languages.detect(found.path)
        if lang is None:
            continue
        content_type = classify(lang, found.path, settings)
        if content_type in settings.index_excluded:
            continue
        yield SourceFile(path=found.path, mtime_ns=found.mtime_ns, size=found.size,
                         lang=lang, content_type=content_type)


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
