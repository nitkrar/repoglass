"""Top-level modules may import only inward through the package layers."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "repoglass"

#: module -> layers it may import from
ALLOWED: dict[str, set[str]] = {
    "models": set(),
    "text": set(),              # string normalisation; both corpus and store need it
    "config": {"models"},       # type aliases are domain vocabulary
    "corpus": {"models", "config", "text"},
    "embeddings": {"models", "config"},
    "store": {"models", "config", "text"},
    "search": {"models", "config", "store", "embeddings"},
    "index": {"models", "config", "corpus", "embeddings", "store", "search",
              "text"},
    # Above index: a consumer of the public surface, imported by nothing.
    "cli": {"models", "config", "index"},
}


def _layer_of(path: Path) -> str:
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else rel.stem


def _internal_imports(path: Path) -> set[str]:
    """Layers this file imports from, resolving relative imports."""
    tree = ast.parse(path.read_text())
    own = _layer_of(path)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("repoglass."):
                    found.add(node.module.split(".")[1])
                continue
            # relative: level 1 inside a package is a sibling of that package
            if node.module:
                head = node.module.split(".")[0]
                found.add(head if node.level >= 2 else (own if path.parent.name == own else head))
            else:
                for alias in node.names:
                    found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("repoglass."):
                    found.add(alias.name.split(".")[1])
    return {f for f in found if f in ALLOWED} - {own}


class ImportDirectionTests(unittest.TestCase):
    def test_no_upward_imports(self) -> None:
        violations: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            if path.name == "__init__.py" and path.parent == SRC:
                continue  # the public re-export module sees everything
            layer = _layer_of(path)
            permitted = ALLOWED.get(layer)
            if permitted is None:
                continue
            for imported in _internal_imports(path):
                if imported not in permitted:
                    violations.append(
                        f"{path.relative_to(SRC)}: {layer} -> {imported} (not allowed)"
                    )
        self.assertEqual([], violations, "\n".join(violations))

    def test_every_layer_is_covered(self) -> None:
        """A new top-level module must be added to ALLOWED deliberately."""
        present = {
            _layer_of(p)
            for p in SRC.rglob("*.py")
            if not (p.name == "__init__.py" and p.parent == SRC)
        }
        self.assertEqual(set(), present - set(ALLOWED), "unclassified modules")


if __name__ == "__main__":
    unittest.main()
