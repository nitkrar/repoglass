"""repoglass settings to an embedder.

No test may reach the network: model2vec downloads on first use.
"""

from __future__ import annotations

import unittest
from unittest import mock

from repoglass import embeddings
from repoglass.config import Settings


class BuildTests(unittest.TestCase):
    def test_none_backend_returns_none(self) -> None:
        self.assertIsNone(embeddings.build(Settings(embed_backend="none")))

    def test_http_backend_requires_an_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            embeddings.build(Settings(embed_backend="http"))

    def test_an_unknown_backend_is_refused(self) -> None:
        """Typos in embed_backend must not silently fall through to the
        static path, which is what an else-branch default would do."""
        with self.assertRaises(ValueError):
            embeddings.build(Settings(embed_backend="bogus"))  # type: ignore[arg-type]

    def test_onnx_dependency_error_names_the_repoglass_extra(self) -> None:
        cause = ImportError("No module named 'onnxruntime'",
                            name="onnxruntime")
        error = ImportError("OnnxEncoder needs the onnx extra: pip install 'semsift[onnx]'")
        error.__cause__ = cause
        with mock.patch.object(embeddings, "OnnxEncoder", side_effect=error):
            with self.assertRaisesRegex(ValueError, r"repoglass\[onnx\]"):
                embeddings.build(Settings(embed_backend="onnx"))


class SpaceTests(unittest.TestCase):
    def test_the_onnx_graph_file_is_part_of_the_space(self) -> None:
        a = Settings(embed_backend="onnx", embed_model="BAAI/bge-small-en-v1.5")
        b = Settings(embed_backend="onnx", embed_model="BAAI/bge-small-en-v1.5",
                     embed_onnx_file="onnx/model_quantized.onnx")
        self.assertNotEqual(embeddings.space(a, 384), embeddings.space(b, 384))

    def test_the_http_endpoint_is_part_of_the_space(self) -> None:
        a = Settings(embed_backend="http", embed_endpoint="http://a")
        b = Settings(embed_backend="http", embed_endpoint="http://b")
        self.assertNotEqual(embeddings.space(a, 8), embeddings.space(b, 8))

    def test_settings_a_backend_ignores_leave_the_space_alone(self) -> None:
        a = Settings(embed_backend="static")
        b = Settings(embed_backend="static", embed_endpoint="http://b",
                     embed_onnx_file="other.onnx")
        self.assertEqual(embeddings.space(a, 8), embeddings.space(b, 8))


class PrefixTests(unittest.TestCase):
    QWEN = "onnx-community/Qwen3-Embedding-0.6B-ONNX"

    def test_code_models_get_a_code_instruction(self) -> None:
        for model in (self.QWEN, "nomic-ai/CodeRankEmbed"):
            got = embeddings.query_prefix(Settings(embed_model=model))
            self.assertIn("code", got, model)

    def test_other_models_get_the_model_default(self) -> None:
        s = Settings(embed_model="BAAI/bge-small-en-v1.5")
        self.assertTrue(embeddings.query_prefix(s))
        self.assertEqual("", embeddings.query_prefix(
            Settings(embed_model="minishlab/potion-base-8M")))

    def test_an_explicit_setting_overrides_the_recommendation(self) -> None:
        s = Settings(embed_model=self.QWEN, embed_query_prefix="mine: ")
        self.assertEqual("mine: ", embeddings.query_prefix(s))

    def test_empty_string_is_an_override_not_an_absence(self) -> None:
        s = Settings(embed_model=self.QWEN, embed_query_prefix="")
        self.assertEqual("", embeddings.query_prefix(s))

    def test_document_prefix_follows_the_same_rules(self) -> None:
        e5 = "intfloat/e5-small-v2"
        self.assertEqual("passage: ", embeddings.doc_prefix(Settings(embed_model=e5)))
        self.assertEqual("", embeddings.doc_prefix(
            Settings(embed_model=e5, embed_doc_prefix="")))


if __name__ == "__main__":
    unittest.main()
