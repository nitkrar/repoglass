"""Packed vectors use float16."""

from __future__ import annotations

import unittest

from repoglass.embeddings import VECTOR_DTYPE, pack


def unpack(blob: bytes):
    """Read a packed vector back. Local to the test: nothing in the
    library reads a vector one at a time -- the search tier reads the
    whole matrix at once."""
    import numpy as np

    return np.frombuffer(blob, dtype=VECTOR_DTYPE).astype("float32")


class PackingTests(unittest.TestCase):
    def test_two_bytes_per_dimension(self) -> None:
        self.assertEqual(512, len(pack([0.1] * 256)))

    def test_packing_normalises(self) -> None:
        got = unpack(pack([3.0, 4.0]))
        self.assertAlmostEqual(0.6, got[0], places=3)
        self.assertAlmostEqual(0.8, got[1], places=3)

    def test_a_zero_vector_does_not_divide_by_zero(self) -> None:
        self.assertEqual([0.0, 0.0], list(unpack(pack([0.0, 0.0]))))

    def test_half_precision_holds_the_range_we_use(self) -> None:
        """Components of a unit vector are in [-1, 1], where float16
        resolution is ~0.0005 -- far below the score gaps we rank on."""
        import numpy as np

        v = list(np.linspace(-1.0, 1.0, 256))
        back = unpack(pack(v))
        norm = float(np.linalg.norm(v))
        self.assertTrue(
            np.allclose([x / norm for x in v], back, atol=1e-3))

    def test_the_dtype_is_declared_once(self) -> None:
        """store and search both derive their width from this."""
        self.assertEqual("float16", VECTOR_DTYPE)


if __name__ == "__main__":
    unittest.main()
