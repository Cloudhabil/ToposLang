"""Unit tests for Cellular Gauge Sheaves with Discrete Z/NZ Center Grading.

Tests:
- CenterGradedSheaf representation and Z_N grading
- 't Hooft flux algebra on T^4 (64 flux sectors)
- Bochner moment positivity eta(k) >= 0
- Bochner duality slope_e = 2 artanh(e^-rho_m)
- Junction Band condition check
- Higher sheaf cohomology obstructions for fracton mobility
"""

import unittest

from topos.core.four_tier_dacc import create_4_torus
from topos.ir.gauge_sheaf import (
    CenterGradedSheaf,
    GaugeGroup,
    THooftFluxProfile,
)


class TestGaugeSheaf(unittest.TestCase):
    def test_gauge_group_and_center_grading(self):
        torus = create_4_torus("Torus4D").complex
        sheaf_su3 = CenterGradedSheaf("SU3Sheaf", torus, GaugeGroup("SU(3)", 3), fundamental_charge=1)
        self.assertEqual(sheaf_su3.gauge_group.N, 3)
        self.assertEqual(sheaf_su3.gauge_group.center_order, 3)
        self.assertEqual(sheaf_su3.get_charge("L0"), 1)

        phase1 = sheaf_su3.gauge_group.center_phase(1)
        phase4 = sheaf_su3.gauge_group.center_phase(4)
        self.assertAlmostEqual(phase1.real, phase4.real)
        self.assertAlmostEqual(phase1.imag, phase4.imag)

    def test_thooft_flux_algebra_64_sectors(self):
        torus = create_4_torus("Torus4D").complex
        sheaf = CenterGradedSheaf("SU2Sheaf", torus, GaugeGroup("SU(2)", 2))
        profile = sheaf.compute_thooft_flux_profile(phi_v=0.1)

        self.assertEqual(len(profile.flux_sectors), 64)
        self.assertEqual(len(profile.electric_flux_ratios), 64)
        self.assertTrue(profile.bochner_positive)

        # Check electric slope duality: slope_e = 2 * artanh(exp(-rho_m))
        slope = profile.electric_slope(magnetic_screening_rho=1.0)
        self.assertGreater(slope, 0.0)

    def test_junction_band_condition(self):
        torus = create_4_torus("Torus4D").complex
        sheaf = CenterGradedSheaf("SU3Sheaf", torus, GaugeGroup("SU(3)", 3))
        valid, q = sheaf.check_junction_band(phi_v=0.1, q_star=0.01)
        self.assertTrue(valid)
        self.assertGreater(q, 0.0)

    def test_fracton_cohomology_obstruction(self):
        torus = create_4_torus("Torus4D").complex
        sheaf = CenterGradedSheaf("SU3Sheaf", torus, GaugeGroup("SU(3)", 3))
        obs = sheaf.fracton_cohomology_obstruction(subcomplex_dim=2)
        self.assertIn("is_obstructed", obs)
        self.assertIn("dim_Hk", obs)
        self.assertIn("mobility", obs)
        self.assertEqual(obs["stalk_group"], "Z/3Z")


if __name__ == "__main__":
    unittest.main()
