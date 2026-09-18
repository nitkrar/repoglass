"""Which category a file belongs to.

Resolved once at index time and stored on the file row, so a filtered
query is a plain test on a column rather than generated SQL calling a
registered Python function -- leaving a database any `sqlite3`
connection can read.

Categories are mutually exclusive and tried in a fixed order, so every
file belongs to exactly one:

    docs -> config -> data -> tests -> code

Language decides the first three, path decides `tests`, and `code` is
the remainder. Language before path is deliberate: a markdown file
under `tests/` is documentation, and a CSV fixture is data.
"""

from __future__ import annotations

import re
from functools import lru_cache

from ..config import Settings
from ..models import ContentType


def classify(lang: str, path: str, settings: Settings) -> ContentType:
    lowered = (lang or "").lower()
    if lowered in _lower(settings.doc_languages):
        return "docs"
    if lowered in _lower(settings.config_languages):
        return "config"
    if lowered in _lower(settings.data_languages):
        return "data"
    if is_test(path, settings.test_markers):
        return "tests"
    return "code"


def is_test(path: str, markers: tuple[str, ...]) -> bool:
    """Whether a marker appears as a path segment or a delimited name part.

    A bare substring makes `inspection.py` a test, because it contains
    "spec". `attestation.py` and `latest.py` are the same family for
    the "test" marker.
    """
    if not path or not markers:
        return False
    loose, camel = _patterns(markers)
    return bool(loose.search(path) or camel.search(path))


@lru_cache(maxsize=32)
def _lower(langs: tuple[str, ...]) -> frozenset[str]:
    return frozenset(str(x).lower() for x in langs)


@lru_cache(maxsize=32)
def _patterns(markers: tuple[str, ...]) -> tuple[re.Pattern, re.Pattern]:
    """The loose and camelCase marker patterns for a set of markers.

    Ecosystems mark tests in three forms, and missing one silently
    reclassifies a whole language:

        directory   tests/ spec/ testing/ __tests__/   (note the plural)
        delimited   test_a.py  a_test.go  a.spec.ts
        camelCase   FooTests.swift  CaptureControllerSpec.swift

    Two patterns rather than one because the camelCase form must be
    case-SENSITIVE: under IGNORECASE the rule "a lowercase letter
    followed by Test" also matches `latest` and `protest`.
    """
    alt = "|".join(re.escape(m) for m in markers)
    loose = re.compile(
        rf"(?:^|/)__?(?:{alt})s?__?(?:/|$)"
        rf"|(?:^|/)(?:{alt})(?:s|ing)?(?:/|$)"
        rf"|(?:^|/|[_.\-])(?:{alt})s?[_.\-]"
        rf"|[_.\-](?:{alt})s?\.[A-Za-z0-9]+$",
        re.IGNORECASE,
    )
    camel = re.compile(
        rf"[a-z0-9](?:{'|'.join(m.title() for m in markers)})s?\.[A-Za-z0-9]+$"
    )
    return loose, camel
