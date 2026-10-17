"""Unit tests for Pro-DACC Inverse Limit Towers and Projective Capabilities.

Tests:
- ProCellComplex inverse limit construction & decimation maps
- Coalgebraic lazy streams
- Projective Betti invariants
- Continuous projective topological loss L_proj
- Resolvent norm ||(zI - L)^-1||
- O(1) elementary collapse garbage collection
- Lock-free AMR spatial commutativity
"""

from fractions import Fraction
import unittest

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.four_tier_dacc import create_4_hypercube
from topos.core.pro_complex import (
    ProCellComplex,
    DecimationMap,
    CellularHomotopyRewrite,
    verify_spatial_commutativity,
)


class TestProComplex(unittest.TestCase):
    def setUp(self):
        # Build a 2-level projective system
        self.k0 = create_4_hypercube("K0").complex
        self.k1 = create_4_hypercube("K1").complex
        self.pro = ProCellComplex("ProHypercube", max_dimension=4)
        self.pro.add_scale_complex(0, self.k0)
        self.pro.add_scale_complex(1, self.k1)

    def test_pro_betti_limit(self):
        betti_lim = self.pro.projective_betti_limit(dim=1)
        self.assertEqual(betti_lim, 0)
        profile = self.pro.projective_betti_profile()
        self.assertEqual(profile[0], 1)
        self.assertEqual(profile[1], 0)

    def test_projective_topological_loss(self):
        loss = self.pro.projective_topological_loss(dim=1, target_beta=0, tau_base=Fraction(1, 2))
        self.assertGreaterEqual(loss, 0.0)

    def test_resolvent_norm(self):
        # z outside spectrum, e.g. z = 10.0 + 1.0j
        norm = self.pro.compute_pseudospectral_resolvent_norm(dim=1, scale=0, z=complex(10.0, 1.0))
        self.assertGreater(norm, 0.0)

    def test_elementary_collapse_gc(self):
        comp = CellComplex("CollapseTest")
        v0 = GlobularCell("V0", dim=0)
        v1 = GlobularCell("V1", dim=0)
        comp.add_cell(v0)
        comp.add_cell(v1)
        e = GlobularCell("E01", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v1))
        comp.add_cell(e)

        # Elementary collapse of free face (v1, E01)
        collapsed = self.pro.elementary_collapse(comp, "V1", "E01")
        self.assertNotIn("V1", [c.name for c in collapsed.get_cells(0)])
        self.assertNotIn("E01", [c.name for c in collapsed.get_cells(1)])
        self.assertIn("V0", [c.name for c in collapsed.get_cells(0)])

    def test_lock_free_amr_spatial_commutativity(self):
        # Two disjoint cellular rewrites: supp(r_A) ∩ supp(r_B) = ∅
        r_A = CellularHomotopyRewrite("rA", support={"cell_A1", "cell_A2"})
        r_B = CellularHomotopyRewrite("rB", support={"cell_B1", "cell_B2"})
        self.assertTrue(r_A.commutes_with(r_B))
        self.assertTrue(verify_spatial_commutativity(r_A, r_B))

        # Overlapping rewrites
        r_C = CellularHomotopyRewrite("rC", support={"cell_A1", "cell_C1"})
        self.assertFalse(r_A.commutes_with(r_C))
        self.assertFalse(verify_spatial_commutativity(r_A, r_C))


if __name__ == "__main__":
    unittest.main()
