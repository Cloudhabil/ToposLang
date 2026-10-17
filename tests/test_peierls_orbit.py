"""Unit tests for Constructive Yang-Mills Modules in ToposLang Physics.

Tests:
1. OrbitSpaceManifold & Wilson loop basis generation on 4-hypercube and 4-torus.
2. Dirichlet form energy evaluation and strictly positive mass gap in orbit space.
3. PeierlsVortexAnalyzer: entropy bound mu_4 <= 14, Peierls inequality, Balaban handover.
4. BakryEmeryCurvatureEvaluator: Carré du champ Gamma and Gamma_2, Ricci curvature lower bound,
   Poincaré spectral gap.
"""

from fractions import Fraction
import math
import unittest

from topos.core.four_tier_dacc import FourTierDACC, create_4_hypercube, create_4_torus
from topos.physics.orbit_space import OrbitSpaceManifold, OrbitState, WilsonLoopObservable
from topos.physics.peierls_vortex import PeierlsVortexAnalyzer
from topos.physics.bakry_emery import BakryEmeryCurvatureEvaluator


class TestConstructiveYangMills(unittest.TestCase):
    def setUp(self):
        self.hypercube = create_4_hypercube()
        self.torus = create_4_torus()

    def test_orbit_space_manifold_loop_basis(self):
        # 4-hypercube has 24 plaquettes (2-cells)
        orbit_hyper = OrbitSpaceManifold(self.hypercube, gauge_group="SU(2)")
        self.assertEqual(orbit_hyper.loop_dimension, 24)

        # 4-torus has 6 plaquettes + 4 fundamental non-bounding 1-cycles
        orbit_torus = OrbitSpaceManifold(self.torus, gauge_group="SU(2)")
        self.assertEqual(orbit_torus.loop_dimension, 10)

        # Check trivial trace
        for loop in orbit_hyper.fundamental_loops:
            self.assertEqual(loop.evaluate_trivial(), 2.0)

    def test_orbit_space_dirichlet_and_spectral_gap(self):
        orbit_torus = OrbitSpaceManifold(self.torus, gauge_group="SU(2)")
        
        # Test Dirichlet form energy on an excited orbit state
        state = OrbitState(
            state_id="glueball_0",
            loop_amplitudes={"W_P01": 0.5 + 0.5j, "Polyakov_L0": 1.0 + 0.0j},
        )
        energy = orbit_torus.dirichlet_form_energy(state)
        # W_P01 (area 1): 1 * 1 * (0.5^2 + 0.5^2) = 0.5
        # Polyakov_L0 (area 0, min 1): 1 * 1 * 1.0 = 1.0
        # Total energy = 1.5
        self.assertAlmostEqual(energy, 1.5, places=5)

        # Spectral gap at coupling beta = 2.0
        gap = orbit_torus.spectral_mass_gap(beta=2.0)
        self.assertGreater(gap, 0.0)

        # Spectrum ordered: lambda_0 == 0.0, lambda_1 > 0
        spectrum = orbit_torus.compute_orbit_laplacian_spectrum(beta=2.0)
        self.assertEqual(spectrum[0], 0.0)
        self.assertGreater(spectrum[1], 0.0)

    def test_peierls_vortex_analyzer(self):
        analyzer = PeierlsVortexAnalyzer(self.hypercube, gauge_group_n=2, lattice_dim=4)

        # Surface entropy constant: mu_4 <= 14 in 4D (Aizenman-Froehlich bound)
        self.assertLessEqual(analyzer.surface_entropy_mu, 14.0)
        self.assertAlmostEqual(analyzer.surface_entropy_mu, 14.0, places=5)

        # Critical strong-coupling threshold u_c = 1/14 for SU(2)
        self.assertAlmostEqual(analyzer.u_critical, 1.0 / 14.0, places=10)

        # PHYSICAL SIGN CONVENTION: confinement is in SMALL beta (strong coupling).
        # beta_c_strong for SU(2) ~ 0.286 (where u(beta)*mu4 = 1).
        # At beta=0.05 (deep strong coupling), u*mu4 ~ 0.175 < 1 -> confined.
        beta_strong = 0.05
        u_strong = analyzer.character_weight(beta_strong)
        self.assertLess(u_strong * analyzer.surface_entropy_mu, 1.0)  # Peierls active
        sigma_lat = analyzer.string_tension_lattice(beta_strong)
        self.assertGreater(sigma_lat, 0.0)

        # At large beta (weak coupling / continuum), u -> 1, Peierls bound is NOT active.
        beta_weak = 100.0
        u_weak = analyzer.character_weight(beta_weak)
        self.assertGreater(u_weak, 0.9)  # u close to 1 at weak coupling
        product_weak = u_weak * analyzer.surface_entropy_mu
        self.assertGreater(product_weak, 1.0)  # NO naive Peierls confinement at weak coupling
        sigma_weak = analyzer.string_tension_lattice(beta_weak)
        self.assertEqual(sigma_weak, 0.0)

        # Peierls area law suppression: at beta=0.05, (u*mu4)^10 ~ 0.175^10 ~ very small
        p_suppression = analyzer.verify_peierls_area_law(beta=0.05, minimal_area=10)
        self.assertLess(p_suppression, 1e-7)

        # Balaban UV-to-IR Peierls handover:
        # Start at UV with bare coupling g0=0.8 (a=0.05 fm) and flow to IR (a=0.5 fm).
        cert = analyzer.execute_balaban_peierls_handover(
            bare_coupling_g0=0.8, lattice_spacing_fm=0.05, target_scale_fm=0.5
        )
        # IR effective coupling must be larger (stronger) than UV bare coupling
        self.assertGreater(cert.effective_coupling_g_sq_ir, cert.bare_coupling_g0_sq)
        # IR beta must be smaller than UV beta = 2N/g0^2 = 4/0.64 = 6.25
        beta_uv = 2.0 * 2 / cert.bare_coupling_g0_sq
        self.assertLess(cert.lattice_coupling_beta_ir, beta_uv)
        # Lambda_YM scale must be non-zero (purely from RGE, no circular assumption)
        self.assertGreater(cert.lambda_ym_mev, 0.0)
        # Mass gap must be non-zero (from dimensional transmutation)
        self.assertGreater(cert.mass_gap_phys_gev, 0.0)

    def test_bakry_emery_curvature_and_poincare_gap(self):
        orbit_hyper = OrbitSpaceManifold(self.hypercube, gauge_group="SU(2)")
        evaluator = BakryEmeryCurvatureEvaluator(orbit_hyper)

        cert = evaluator.estimate_curvature_bound(beta=2.0)
        self.assertTrue(cert.is_strictly_positive)
        self.assertGreater(cert.ricci_lower_bound_kappa, 0.0)
        self.assertGreater(cert.spectral_mass_gap, 0.0)
        self.assertAlmostEqual(cert.spectral_mass_gap, cert.ricci_lower_bound_kappa)

        # Test Gamma operator
        f = {loop.loop_id: 2.0 for loop in orbit_hyper.fundamental_loops}
        g = {loop.loop_id: 3.0 for loop in orbit_hyper.fundamental_loops}
        gamma = evaluator.compute_gamma(f, g)
        for val in gamma.values():
            self.assertEqual(val, 6.0)


if __name__ == "__main__":
    unittest.main()
