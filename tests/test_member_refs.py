"""References written through a member: `Type.member`, `x.field`.

Each `fixtures/refs/<lang>/` holds a `sample.*` and `expected.txt`:
every reference a reader would navigate from it, `<line> <name>` per
line, written by hand. Exact, so a duplicate or an extra capture fails
as surely as a missing one.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

FIXTURES = Path(__file__).parent / "fixtures" / "refs"


class MemberReferenceTests(unittest.TestCase):
    def test_references_are_exactly_the_written_uses(self) -> None:
        dirs = sorted(d for d in FIXTURES.iterdir() if d.is_dir())
        self.assertTrue(dirs)
        for d in dirs:
            with self.subTest(lang=d.name):
                sample = next(p for p in d.iterdir() if p.stem == "sample")
                text = sample.read_text()
                f = SourceFile(path=sample.name, mtime_ns=0, size=len(text), lang=d.name)
                actual = sorted(
                    (s.start_line, s.name)
                    for s in extract.extract(f, text, Settings()).symbols
                    if s.tag == "ref"
                )
                expected = sorted(
                    (int(line), name)
                    for line, name in (
                        row.split(" ", 1)
                        for row in (d / "expected.txt").read_text().splitlines()
                    )
                )
                self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
