"""Unit tests for Dual-Tier Spatial Wavefront Scheduler and Theorem 2.2."""

import unittest
from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.clifford import Multivector, PGA301
from topos.core.complex import CellComplex
from topos.runtime.wavefront import RewriteRule, DualTierWavefrontScheduler


class TestWavefrontCommutativity(unittest.TestCase):
    def setUp(self):
        # Build 1D/2D grid complex
        self.complex = CellComplex("GridTest")
        v0 = GlobularCell("V0", dim=0)
        v1 = GlobularCell("V1", dim=0)
        v2 = GlobularCell("V2", dim=0)
        v3 = GlobularCell("V3", dim=0)
        for v in (v0, v1, v2, v3):
            self.complex.add_cell(v)

        e01 = GlobularCell("E01", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v1))
        e12 = GlobularCell("E12", dim=1, source=Chain.from_cell(v1), target=Chain.from_cell(v2))
        e23 = GlobularCell("E23", dim=1, source=Chain.from_cell(v2), target=Chain.from_cell(v3))
        for e in (e01, e12, e23):
            self.complex.add_cell(e)

        self.scheduler = DualTierWavefrontScheduler(self.complex)

    def test_tier1_disjoint_support_commutativity(self):
        # r1 operates on E01 (support {E01, V0, V1})
        # r2 operates on E23 (support {E23, V2, V3})
        # Supports are completely disjoint
        r1 = RewriteRule("Rule1", lhs_cells=("E01",), rhs_cells=())
        r2 = RewriteRule("Rule2", lhs_cells=("E23",), rhs_cells=())

        self.assertFalse(self.scheduler.check_interference(r1, r2))
        self.assertTrue(self.scheduler.are_commuting(r1, r2))

    def test_overlapping_support_interference(self):
        # r1 operates on E01, r2 operates on E12 (shared boundary vertex V1)
        # Without non-overlapping geometric blades, they interfere
        r1 = RewriteRule("Rule1", lhs_cells=("E01",), rhs_cells=())
        r2 = RewriteRule("Rule2", lhs_cells=("E12",), rhs_cells=())

        self.assertTrue(self.scheduler.check_interference(r1, r2))
        self.assertFalse(self.scheduler.are_commuting(r1, r2))

    def test_tier2_continuous_blade_wedge_nullification(self):
        # Both rules touch V1, but have parallel/collinear geometric blades whose wedge vanishes
        blade_parallel_1 = Multivector.from_named_blades(PGA301, {"e1": 1.0})
        blade_parallel_2 = Multivector.from_named_blades(PGA301, {"e1": 2.5})
        # e1 ^ (2.5 e1) == 0

        r1 = RewriteRule("Rule1", lhs_cells=("E01",), rhs_cells=(), blade_state=blade_parallel_1)
        r2 = RewriteRule("Rule2", lhs_cells=("E12",), rhs_cells=(), blade_state=blade_parallel_2)

        # Because S(r1) ^ S(r2) == 0, Tier 2 holds and interference is False
        self.assertFalse(self.scheduler.check_interference(r1, r2))
        self.assertTrue(self.scheduler.are_commuting(r1, r2))

    def test_wavefront_partitioning(self):
        r1 = RewriteRule("Rule1", lhs_cells=("E01",), rhs_cells=())
        r2 = RewriteRule("Rule2", lhs_cells=("E12",), rhs_cells=())
        r3 = RewriteRule("Rule3", lhs_cells=("E23",), rhs_cells=())

        # r1 and r3 are disjoint and can be in Wavefront 0.
        # r2 conflicts with both r1 and r3 (shares V1 with r1, shares V2 with r3).
        # So r2 must go into Wavefront 1.
        wavefronts = self.scheduler.partition_into_wavefronts([r1, r2, r3])

        self.assertEqual(len(wavefronts), 2)
        self.assertIn(r1, wavefronts[0])
        self.assertIn(r3, wavefronts[0])
        self.assertEqual(wavefronts[1], [r2])


if __name__ == "__main__":
    unittest.main()
