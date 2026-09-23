"""Embedding backends and the index contracts around them.

Queries can need a different encoding shape from documents, and a
backend change moves vectors into a different space, so both behaviors
must be pinned.
"""

from __future__ import annotations

import unittest
import unittest.mock

from repoglass import embeddings
from repoglass.embeddings import backends
from repoglass.config import Paths, Settings


class QueryAsymmetryTests(unittest.TestCase):
    """bge-family models want a prefix on the query side and nothing on
    the document side. Encoding both the same way silently costs recall,
    and nothing would fail loudly, so it is pinned."""

    def test_default_embedder_treats_query_and_document_alike(self) -> None:
        emb = embeddings.FakeEmbedder(dims=8)
        self.assertEqual(emb.encode(["abc"]), emb.encode_query(["abc"]))

    def test_a_prefix_changes_only_the_query_side(self) -> None:
        emb = embeddings.FakeEmbedder(dims=8)
        emb.query_prefix = "PREFIX: "
        self.assertEqual(emb.encode(["abc"]), emb.encode(["abc"]))
        self.assertNotEqual(emb.encode(["abc"]), emb.encode_query(["abc"]))
        # The prefix is literally prepended, not hashed in some other way.
        self.assertEqual(emb.encode(["PREFIX: abc"]), emb.encode_query(["abc"]))

    def test_known_models_carry_a_recommended_prefix(self) -> None:
        self.assertTrue(embeddings.default_query_prefix("BAAI/bge-small-en-v1.5"))
        self.assertEqual("", embeddings.default_query_prefix("minishlab/potion-base-8M"))

    def test_an_explicit_setting_overrides_the_recommendation(self) -> None:
        s = Settings(embed_backend="none", embed_model="BAAI/bge-small-en-v1.5",
                     embed_query_prefix="mine: ")
        self.assertEqual("mine: ", embeddings.resolve_query_prefix(s))

    def test_empty_string_is_an_override_not_an_absence(self) -> None:
        """None means 'use the recommendation'; '' means 'no prefix'."""
        s = Settings(embed_backend="none", embed_model="BAAI/bge-small-en-v1.5",
                     embed_query_prefix="")
        self.assertEqual("", embeddings.resolve_query_prefix(s))


class PoolingTests(unittest.TestCase):
    """Pooling is per-model and getting it wrong fails silently.

    bge uses CLS, e5 and MiniLM use mean, Qwen3-Embedding uses the last
    non-pad token. All three produce a plausible unit vector, so a wrong
    choice shows up only as bad ranking.
    """

    def setUp(self) -> None:
        import numpy as np

        self.np = np
        # batch of 2, 4 tokens, 3 dims. Second sequence is 2 tokens + pad.
        self.h = np.array([
            [[1., 0, 0], [2., 0, 0], [3., 0, 0], [4., 0, 0]],
            [[10., 0, 0], [20., 0, 0], [99., 0, 0], [99., 0, 0]],
        ], dtype="float32")
        self.mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype="int64")

    def test_cls_takes_the_first_token(self) -> None:
        got = embeddings._pool(self.h, self.mask, "cls")
        self.assertEqual([1.0, 10.0], [got[0][0], got[1][0]])

    def test_mean_ignores_padding(self) -> None:
        got = embeddings._pool(self.h, self.mask, "mean")
        self.assertAlmostEqual(2.5, float(got[0][0]))    # (1+2+3+4)/4
        self.assertAlmostEqual(15.0, float(got[1][0]))   # (10+20)/2, not /4

    def test_last_takes_the_last_real_token_not_the_last_row(self) -> None:
        """With right padding, hidden[:, -1] is a pad vector for every
        sequence shorter than the batch maximum."""
        got = embeddings._pool(self.h, self.mask, "last")
        self.assertAlmostEqual(4.0, float(got[0][0]))
        self.assertAlmostEqual(20.0, float(got[1][0]))   # not 99
        self.assertNotAlmostEqual(99.0, float(got[1][0]))

    def test_known_models_map_to_their_trained_pooling(self) -> None:
        self.assertEqual("cls", embeddings.default_pooling("BAAI/bge-small-en-v1.5"))
        self.assertEqual("last", embeddings.default_pooling(
            "onnx-community/Qwen3-Embedding-0.6B-ONNX"))
        self.assertEqual("mean", embeddings.default_pooling("nomic-ai/CodeRankEmbed"))


class BackendSelectionTests(unittest.TestCase):
    def test_none_backend_builds_nothing(self) -> None:
        s = Settings(embed_backend="none")
        self.assertIsNone(embeddings.build(s, Paths.for_root(__file__ and ".")))

    def test_http_backend_requires_an_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            embeddings.build(Settings(embed_backend="http"),
                             Paths.for_root("."))

    def test_onnx_backend_is_a_recognised_value(self) -> None:
        """Typos in embed_backend must not silently fall through to the
        static path, which is what an else-branch default would do."""
        with self.assertRaises(ValueError):
            embeddings.build(Settings(embed_backend="bogus"),  # type: ignore[arg-type]
                             Paths.for_root("."))


class OnnxEmbedderTests(unittest.TestCase):
    """Guarded: needs onnxruntime and a cached model. Skips, never fails,
    so the suite stays offline-safe."""

    MODEL = "BAAI/bge-small-en-v1.5"
    #: Not the configured default, which is webgpu and needs a device no
    #: CI runner has. What these four assert is pooling and prefix, which
    #: a provider does not change; which provider attaches is asserted
    #: against a fake onnxruntime below.
    PROVIDERS = "cpu"

    def setUp(self) -> None:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime not installed")
        try:
            self.emb = embeddings.OnnxEmbedder(
                self.MODEL, local_only=True, providers=self.PROVIDERS
            )
        except Exception as exc:                     # not cached, no network
            self.skipTest(f"model unavailable: {type(exc).__name__}")

    def test_dims_are_learned_not_declared(self) -> None:
        self.assertEqual(384, self.emb.dims)

    def test_vectors_are_unit_length(self) -> None:
        import math

        v = self.emb.encode(["a chunk of text about payment handling"])[0]
        self.assertAlmostEqual(1.0, math.sqrt(sum(x * x for x in v)), places=4)

    def test_it_separates_two_unrelated_texts(self) -> None:
        """The reason this backend exists, and not a tautology: a wrong
        pooling or prefix choice still yields a plausible unit vector,
        and shows up only as the near text scoring below the far
        one."""
        docs = self.emb.encode([
            "records a delivery so the same notification is not sent twice",
            "locate an executable on the PATH and return its absolute path",
        ])
        q = self.emb.encode_query(["stop duplicate notifications to an agent"])[0]
        near = sum(a * b for a, b in zip(docs[0], q))
        far = sum(a * b for a, b in zip(docs[1], q))
        self.assertGreater(near, far)

    def test_the_default_prefix_is_applied(self) -> None:
        self.assertTrue(self.emb.query_prefix)
        self.assertNotEqual(self.emb.encode(["x"]), self.emb.encode_query(["x"]))


class _FakeSession:
    def __init__(self, path, arg=None, providers=None):
        self.path, self.arg, self.providers = path, arg, providers


class _FakeOptions:
    def __init__(self) -> None:
        self.devices = None

    def add_provider_for_devices(self, devices, options) -> None:
        self.devices = devices


class _FakeDevice:
    def __init__(self, ep_name: str) -> None:
        self.ep_name = ep_name


class _FakeOrt:
    """Enough onnxruntime to see which attachment path was taken."""

    InferenceSession = _FakeSession
    SessionOptions = _FakeOptions

    def __init__(self, devices=()) -> None:
        self._devices = devices

    def get_available_providers(self):
        return ["CPUExecutionProvider"]

    def get_ep_devices(self):
        return list(self._devices)


class ProviderAttachmentTests(unittest.TestCase):
    """A plugin provider named in `providers=` is dropped rather than
    refused: the session builds, reports success, and runs on CPU. So
    the two attachment paths are pinned by which one each choice takes,
    not by the session merely constructing."""

    def test_a_named_provider_goes_through_the_providers_argument(self) -> None:
        sess = backends._session(_FakeOrt(), "m.onnx", "cpu")
        self.assertEqual(["CPUExecutionProvider"], sess.providers)
        self.assertIsNone(sess.arg)

    def test_webgpu_attaches_by_device_and_never_by_name(self) -> None:
        name = "WebGpuExecutionProvider"
        ort = _FakeOrt(devices=[_FakeDevice("CPUExecutionProvider"),
                                _FakeDevice(name)])
        with unittest.mock.patch.object(backends, "_webgpu",
                                        return_value=name):
            sess = backends._session(ort, "m.onnx", "webgpu")
        # `providers=` unused: a plugin name there would silently run on CPU.
        self.assertIsNone(sess.providers)
        self.assertEqual([name], [d.ep_name for d in sess.arg.devices])

    def test_webgpu_without_a_device_fails_rather_than_falling_back(self) -> None:
        with unittest.mock.patch.object(backends, "_webgpu",
                                        return_value="WebGpuExecutionProvider"):
            with self.assertRaises(ValueError):
                backends._session(_FakeOrt(), "m.onnx", "webgpu")


if __name__ == "__main__":
    unittest.main()
