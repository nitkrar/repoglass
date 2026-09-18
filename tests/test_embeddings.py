"""Embedder protocol, vector packing, backend selection.

No test may reach the network: model2vec downloads on first use, so the
fake is the only backend tests construct.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repoglass import embeddings
from repoglass.config import Paths, Settings


class FakeEmbedderTests(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        a = embeddings.FakeEmbedder(dims=8)
        b = embeddings.FakeEmbedder(dims=8)
        self.assertEqual(a.encode(["hello"]), b.encode(["hello"]))

    def test_different_text_gives_different_vectors(self) -> None:
        e = embeddings.FakeEmbedder(dims=8)
        self.assertNotEqual(e.encode(["a"])[0], e.encode(["b"])[0])

    def test_respects_requested_dims(self) -> None:
        e = embeddings.FakeEmbedder(dims=16)
        self.assertEqual(16, len(e.encode(["x"])[0]))
        self.assertEqual(16, e.dims)

    def test_encodes_a_batch(self) -> None:
        e = embeddings.FakeEmbedder(dims=4)
        self.assertEqual(3, len(e.encode(["a", "b", "c"])))


class BuildTests(unittest.TestCase):
    def test_none_backend_returns_none(self) -> None:
        paths = Paths(root=Path("/r"), home=Path("/h"))
        self.assertIsNone(embeddings.build(Settings(embed_backend="none"), paths))

    def test_http_backend_requires_an_endpoint(self) -> None:
        paths = Paths(root=Path("/r"), home=Path("/h"))
        with self.assertRaises(ValueError):
            embeddings.build(Settings(embed_backend="http"), paths)


if __name__ == "__main__":
    unittest.main()
