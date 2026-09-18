"""Definition chunks, plus windows over what no definition owns.

`definition` leaves module-level code in no chunk at all, so a PHP
script with no class and no function is unreachable by any query.
Windows-only covers everything but drops the symbol link the exact tier
uses. Hybrid takes the coverage without paying that, which is why it is
the only other `coverage` value.
"""

from __future__ import annotations

import unittest

from repoglass.config import Settings
from repoglass.corpus import extract
from repoglass.models import SourceFile

MODULE_LEVEL = '''"""What this module is for, in a docstring nothing owns."""

TIMEOUT_SECONDS = 30
RETRY_BUDGET = 5

HANDLERS = {
    "ping": "respond immediately to a liveness probe",
    "drain": "stop accepting work and finish what is in flight",
}


def reticulate(splines):
    """A definition, which the tag query does capture."""
    return [s for s in splines if s.length > TIMEOUT_SECONDS]
'''

NO_DEFINITIONS = '''# A script. Top level from first line to last, like a PHP endpoint.
config = load_config()
if config.mode == "serve":
    start_server(config.port)
else:
    run_batch(config.input_path)
'''


def chunks(body: str, mode: str, path: str = "a.py"):
    f = SourceFile(path=path, mtime_ns=1, size=len(body), lang="python")
    return extract.extract(f, body, Settings(coverage=mode)).chunks


def covered(body: str, mode: str) -> set[int]:
    out: set[int] = set()
    for c in chunks(body, mode):
        out.update(range(c.start_line, c.end_line + 1))
    return out & set(range(1, len(body.splitlines()) + 1))


class HybridTests(unittest.TestCase):
    def test_definition_mode_leaves_module_level_uncovered(self) -> None:
        self.assertNotIn(1, covered(MODULE_LEVEL, "definition"))
        self.assertNotIn(3, covered(MODULE_LEVEL, "definition"))

    def test_hybrid_covers_the_module_level(self) -> None:
        got = covered(MODULE_LEVEL, "hybrid")
        self.assertIn(1, got, "module docstring")
        self.assertIn(3, got, "a top-level constant")
        self.assertIn(6, got, "a top-level dict")

    def test_hybrid_keeps_the_definition_chunk_and_its_name(self) -> None:
        """What window mode loses: the symbol link the exact tier uses."""
        names = {c.name for c in chunks(MODULE_LEVEL, "hybrid")}
        self.assertIn("reticulate", names)

    def test_a_file_with_no_definitions_is_still_chunked(self) -> None:
        """The PHP script of the module docstring, in miniature."""
        self.assertEqual([], list(chunks(NO_DEFINITIONS, "definition")))
        self.assertTrue(chunks(NO_DEFINITIONS, "hybrid"))

    def test_hybrid_does_not_duplicate_the_definition_body(self) -> None:
        """A window mostly overlapping a definition adds nothing but
        another candidate competing for the same slot."""
        body_lines = set()
        for c in chunks(MODULE_LEVEL, "hybrid"):
            span = frozenset(range(c.start_line, c.end_line + 1))
            self.assertNotIn(span, body_lines, "identical span twice")
            body_lines.add(span)
        defn = next(c for c in chunks(MODULE_LEVEL, "hybrid")
                    if c.name == "reticulate")
        overlap = [c for c in chunks(MODULE_LEVEL, "hybrid")
                   if c.name == ""
                   and len(set(range(c.start_line, c.end_line + 1))
                           & set(range(defn.start_line, defn.end_line + 1)))
                   * 2 > (c.end_line - c.start_line + 1)]
        self.assertEqual([], overlap, "a window mostly inside a definition")


if __name__ == "__main__":
    unittest.main()
