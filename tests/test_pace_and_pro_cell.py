"""Unit tests for ProCellComplex, PaceOperator, and CenterGradedSheaf extensions.

Verifies:
  • ScaleComplex & ProCellComplex multiscale hierarchy
  • DecimationMap pushforward chain cochain scaling
  • Projective topological loss and pseudospectral resolvent bounds
  • Spatial refinement commutativity & elementary collapse
  • PaceOperator Clebsch-Gordan cone forward-invariance R(C) ⊆ C
  • Theorem 16.28 one-pace lower floor contraction
  • Emergent continuum mass gap calculation Delta > 0
  • CenterGradedSheaf flux sectors on T^4 (e.g. 64 for SU(2), 729 for SU(3))
  • Bochner moment positivity eta(k) >= 0
"""

import unittest
from topos.core.pro_cell_complex import (
    DecimationMap,
    ScaleComplex,
    ProCellComplex,
)
from topos.physics.pace_operator import PaceOperator
from topos.ir.gauge_sheaf import CenterGradedSheaf


class TestProCellComplex(unittest.TestCase):
    def test_decimation_map(self):
        d_same = DecimationMap(source_scale=1, target_scale=1)
        self.assertEqual(d_same.pushforward_chain([1.0, 2.0], source_dim=2), [1.0, 2.0])

        d_diff = DecimationMap(source_scale=2, target_scale=1)
        # scale_ratio = 2^(2-1) = 2. For source_dim=2, factor = 1 / (2^2) = 0.25
        result = d_diff.pushforward_chain([4.0, 8.0], source_dim=2)
        self.assertEqual(result, [1.0, 2.0])

    def test_scale_complex_counts(self):
        sc0 = ScaleComplex(scale=0)
        self.assertEqual(len(sc0.cells[0]), 16)
        self.assertEqual(len(sc0.cells[1]), 32)
        self.assertEqual(len(sc0.cells[2]), 24)
        self.assertEqual(len(sc0.cells[3]), 8)
        self.assertEqual(len(sc0.cells[4]), 1)
        self.assertEqual(sc0.n_total, 81)

        sc1 = ScaleComplex(scale=1)
        self.assertEqual(len(sc1.cells[0]), 16 * 16)
        self.assertEqual(sc1.n_total, 81 * 16)

    def test_pro_cell_complex(self):
        pro = ProCellComplex(base_dim=4)
        sc0 = pro.get_scale_complex(0)
        self.assertIsNotNone(sc0)
        self.assertEqual(sc0.scale, 0)

        # Projective loss
        loss = pro.compute_projective_topological_loss(m=2)
        self.assertGreaterEqual(loss, 0.0)

        # Resolvent bound
        bound = pro.compute_pseudospectral_resolvent_bound(m=1, z=0.5 + 0.0j)
        self.assertGreater(bound, 0.0)

        # AMR spatial commutativity
        self.assertTrue(pro.verify_spatial_refinement_commutativity())

        # Elementary collapse
        self.assertTrue(pro.elementary_collapse(1, ("v0", "e0")))


class TestPaceOperator(unittest.TestCase):
    def test_pace_operator_cone_invariance(self):
        sheaf = CenterGradedSheaf(gauge_n=2)
        pace = PaceOperator(sheaf)
        self.assertTrue(pace.verify_cone_forward_invariance(n_iterations=5))

    def test_theorem_16_28_floor(self):
        sheaf = CenterGradedSheaf(gauge_n=2)
        pace = PaceOperator(sheaf)
        floor_val = pace.compute_theorem_16_28_one_pace_lower_floor()
        self.assertAlmostEqual(floor_val, 0.187500)
        self.assertGreater(floor_val, 0.0)

    def test_continuum_mass_gap(self):
        sheaf = CenterGradedSheaf(gauge_n=2)
        pace = PaceOperator(sheaf)
        gap = pace.compute_continuum_mass_gap(bare_coupling=0.8)
        self.assertGreater(gap, 0.0)


class TestCenterGradedSheafExtensions(unittest.TestCase):
    def test_default_initialization(self):
        sheaf = CenterGradedSheaf()
        self.assertEqual(sheaf.gauge_n, 2)
        self.assertEqual(sheaf.dim, 4)
        self.assertEqual(sheaf.count_flux_sectors_4torus(), 64)
        self.assertTrue(sheaf.verify_bochner_moment_positivity())

    def test_su3_flux_sectors(self):
        sheaf_su3 = CenterGradedSheaf(gauge_n=3)
        self.assertEqual(sheaf_su3.gauge_n, 3)
        # 3^6 = 729 sectors
        self.assertEqual(sheaf_su3.count_flux_sectors_4torus(), 729)
        self.assertTrue(sheaf_su3.verify_bochner_moment_positivity())

    def test_dynamical_screening_fallback(self):
        sheaf = CenterGradedSheaf(gauge_n=2)
        op = sheaf.construct_dynamical_screening_operator(complex_obj=None)
        self.assertEqual(op, sheaf)


if __name__ == "__main__":
    unittest.main()
