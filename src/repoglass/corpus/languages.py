"""Language detection, tag-query resolution, capability.

`manifest.json` records which source set each query came from. The sets
target different grammar versions and are not interchangeable. Some
languages also append local `<lang>-refs.scm` coverage because the
vendored tags query alone does not supply the reference captures the
extractor needs.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from ..models import LanguageCapability

QUERY_DIR = Path(__file__).parent / "queries"


def detect(path: str) -> str | None:
    from grep_ast import filename_to_lang

    return filename_to_lang(path)


def tag_query(lang: str) -> str | None:
    """Combined definition and reference query source for a language.

    Concatenates the vendored tags query with the supplemental
    `<lang>-refs.scm` where one exists.
    """
    tags = QUERY_DIR / f"{lang}-tags.scm"
    if not tags.is_file():
        return None
    parts = [tags.read_text()]
    refs = QUERY_DIR / f"{lang}-refs.scm"
    if refs.is_file():
        parts.append(refs.read_text())
    return "\n".join(parts)


#: `@name.reference.call` -> "call". Kinds are read off the query
#: rather than listed here, so a supplemental file adding one needs no
#: second edit to be reported.
_REFERENCE_KIND = re.compile(r"@name\.reference\.([a-z_]+)")


def capability(lang: str) -> LanguageCapability:
    """What this language's queries actually capture.

    Callers use this to distinguish "no references in this language" from
    "no references found". Which kinds, not whether any: a query
    capturing only type positions answers the second question yes and
    still cannot say who calls a function.
    """
    src = tag_query(lang)
    if src is None:
        return LanguageCapability(lang=lang, definitions=False)
    return LanguageCapability(
        lang=lang,
        definitions="name.definition." in src,
        reference_kinds=frozenset(_REFERENCE_KIND.findall(src)),
    )


def extractor_rev() -> str:
    """Hash over the resolved .scm set.

    Part of index identity: changing a query shifts spans, so a mismatch
    forces a reindex.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for path in sorted(QUERY_DIR.glob("*.scm")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=None)
def compiled_query(lang: str):
    """The compiled tree-sitter Query for a language, or None.

    Cached because compilation is expensive and the result is
    immutable, while `get_parser` and `get_language` are near-free and
    already cached by grep_ast.

    Safe to share: `Query` holds only the compiled pattern. The per-run
    state lives in `QueryCursor`, which callers construct fresh.
    """
    src = tag_query(lang)
    if src is None:
        return None
    from grep_ast.tsl import get_language
    from tree_sitter import Query

    return Query(get_language(lang), src)
