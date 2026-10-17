"""Unit tests for Four-Dimensional Graded DACCs and Four-Tier Boundary Nilpotency.

Tests:
- 4-hypercube [0,1]^4 construction and cell counting (16, 32, 24, 8, 1)
- 4-torus T^4 construction
- Exact four-tier nilpotency verification (B_1 B_2 = 0, B_2 B_3 = 0, B_3 B_4 = 0)
- Combinatorial Hodge Laplacians L_0, L_1, L_2, L_3, L_4 over Q
- Exact discrete Hodge decomposition over Q
"""

from fractions import Fraction
import unittest

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.four_tier_dacc import (
    FourTierDACC,
    create_4_hypercube,
    create_4_torus,
)


class TestFourTierDACC(unittest.TestCase):
    def test_hypercube_cell_counts(self):
        hc = create_4_hypercube("Hypercube4D")
        self.assertEqual(hc.num_cells(0), 16)
        self.assertEqual(hc.num_cells(1), 32)
        self.assertEqual(hc.num_cells(2), 24)
        self.assertEqual(hc.num_cells(3), 8)
        self.assertEqual(hc.num_cells(4), 1)

    def test_hypercube_four_tier_nilpotency(self):
        hc = create_4_hypercube("Hypercube4D")
        report = hc.verify_four_tier_nilpotency(raise_on_error=True)
        self.assertTrue(report.tier1_valid)
        self.assertTrue(report.tier2_valid)
        self.assertTrue(report.tier3_valid)
        self.assertTrue(report.all_valid)

    def test_torus_four_tier_nilpotency(self):
        torus = create_4_torus("Torus4D")
        report = torus.verify_four_tier_nilpotency(raise_on_error=True)
        self.assertTrue(report.all_valid)
        self.assertEqual(torus.num_cells(0), 1)
        self.assertEqual(torus.num_cells(1), 4)
        self.assertEqual(torus.num_cells(2), 6)
        self.assertEqual(torus.num_cells(3), 4)
        self.assertEqual(torus.num_cells(4), 1)

    def test_hypercube_hodge_laplacians(self):
        hc = create_4_hypercube("Hypercube4D")
        for k in range(5):
            Lk = hc.compute_hodge_laplacian(k)
            num_k = hc.num_cells(k)
            self.assertEqual(Lk.rows, num_k)
            self.assertEqual(Lk.cols, num_k)
            # Verify symmetry L_k^T = L_k
            for r in range(Lk.rows):
                for c in range(Lk.cols):
                    self.assertEqual(Lk.get(r, c), Lk.get(c, r))

    def test_hodge_decomposition(self):
        hc = create_4_hypercube("Hypercube4D")
        decomp = hc.discrete_hodge_decomposition(dim=1)
        self.assertEqual(decomp["dim_Ck"], 32)
        self.assertEqual(decomp["dim_im_Bk_T"], 15)
        self.assertEqual(decomp["dim_im_Bk1"], 17)
        self.assertEqual(decomp["dim_ker_Lk"], 0)


if __name__ == "__main__":
    unittest.main()
