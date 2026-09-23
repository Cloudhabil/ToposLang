"""Unit tests for Formal Chain Group C_k(K; Z)."""

import unittest
from topos.core.cell import GlobularCell
from topos.core.chain import Chain


class TestChain(unittest.TestCase):
    def setUp(self) -> None:
        self.v0 = GlobularCell("v0", dim=0)
        self.v1 = GlobularCell("v1", dim=0)
        self.v2 = GlobularCell("v2", dim=0)
        self.e0 = GlobularCell("e0", dim=1, source=self.v0, target=self.v1)
        self.e1 = GlobularCell("e1", dim=1, source=self.v1, target=self.v2)

    def test_canonical_zero(self) -> None:
        z = Chain.zero(0)
        self.assertTrue(z.is_zero())
        self.assertEqual(len(z), 0)
        self.assertEqual(z, 0)
        self.assertEqual(repr(z), "0_0")

    def test_elementary_chain(self) -> None:
        c = Chain.from_cell(self.v0, coeff=2)
        self.assertEqual(c[self.v0], 2)
        self.assertEqual(len(c), 1)
        self.assertFalse(c.is_zero())

    def test_addition_and_commutativity(self) -> None:
        c1 = Chain.from_cell(self.v0, 1)
        c2 = Chain.from_cell(self.v1, 3)
        
        sum1 = c1 + c2
        sum2 = c2 + c1
        self.assertEqual(sum1, sum2)
        self.assertEqual(sum1[self.v0], 1)
        self.assertEqual(sum1[self.v1], 3)

    def test_zero_pruning(self) -> None:
        c1 = Chain.from_cell(self.v0, 2)
        c2 = Chain.from_cell(self.v0, -2)
        c_sum = c1 + c2
        self.assertTrue(c_sum.is_zero())
        self.assertEqual(len(c_sum), 0)
        self.assertNotIn(self.v0, c_sum)

    def test_subtraction_and_inverses(self) -> None:
        c1 = Chain.from_cell(self.v0, 5)
        c2 = Chain.from_cell(self.v0, 5)
        diff = c1 - c2
        self.assertTrue(diff.is_zero())
        self.assertEqual(c1 + (-c1), Chain.zero(0))

    def test_scalar_multiplication(self) -> None:
        c = Chain.from_cell(self.v0, 2) + Chain.from_cell(self.v1, -3)
        scaled = 3 * c
        self.assertEqual(scaled[self.v0], 6)
        self.assertEqual(scaled[self.v1], -9)

        zero_scaled = 0 * c
        self.assertTrue(zero_scaled.is_zero())

    def test_dimension_mismatch_error(self) -> None:
        c_0 = Chain.from_cell(self.v0)
        c_1 = Chain.from_cell(self.e0)
        with self.assertRaises(ValueError):
            _ = c_0 + c_1

        with self.assertRaises(ValueError):
            _ = c_0 - c_1


if __name__ == "__main__":
    unittest.main()
