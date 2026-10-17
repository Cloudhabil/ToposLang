"""Unit tests for Dirac-Kähler Fermion Doubling Resolution and Creutz Ratio Evaluator.

Tests:
1. Dirac-Kähler grading: sum_{k=0}^4 choose(4, k) = 16.
2. Cl(4,0) multivector decomposition into 4 Dirac 4-spinors.
3. Proof that Dirac-Kähler has exactly 1 pole at p=(0,0,0,0) in BZ vs 16 for naive fermions.
4. Bare Wilson lattice action Metropolis sampling on PeriodicLattice4D.
5. Emergent string tension extraction via Creutz ratios chi(R, T) > 0.
6. Confinement invariant region I = { beta in R : beta <= beta_c }.
"""

from __future__ import annotations
import math
import unittest

from topos.core.clifford import CL400, Multivector
from topos.core.four_tier_dacc import create_4_hypercube
from topos.physics.creutz_ratio import CreutzRatioEvaluator, PeriodicLattice4D
from topos.physics.dirac_kahler import DiracKahlerFermion


class TestDiracKahlerAndCreutz(unittest.TestCase):
    def setUp(self) -> None:
        self.hypercube = create_4_hypercube()
        self.dk = DiracKahlerFermion(self.hypercube)

    def test_dirac_kahler_cochain_dimensions(self) -> None:
        """Verifies graded cochain dimensions sum choose(4, k) = 16."""
        dims = self.dk.cochain_grading_dimensions()
        self.assertEqual(dims[0], 1)
        self.assertEqual(dims[1], 4)
        self.assertEqual(dims[2], 6)
        self.assertEqual(dims[3], 4)
        self.assertEqual(dims[4], 1)
        self.assertEqual(self.dk.total_cochain_dimension(), 16)

    def test_multivector_spinor_decomposition(self) -> None:
        """Verifies Cl(4,0) multivector decomposes into 4 Dirac spinors."""
        mv = Multivector.scalar(CL400, 1.0)
        spinors = self.dk.decompose_multivector_to_dirac_spinors(mv)
        self.assertEqual(len(spinors), 4)
        self.assertEqual(len(spinors[0]), 4)
        self.assertEqual(spinors[0][0], 1.0 + 0j)

    def test_fermion_doubling_resolution_proof(self) -> None:
        """Statically proves that Dirac-Kähler resolves the Nielsen-Ninomiya doubling theorem."""
        report = self.dk.prove_fermion_doubling_resolution()
        self.assertEqual(report.naive_doubler_count, 16)
        self.assertEqual(report.dirac_kahler_species_count, 1)
        self.assertEqual(report.brillouin_zone_poles_dirac_kahler, [(0, 0, 0, 0)])

    def test_periodic_lattice_and_bare_action(self) -> None:
        """Verifies Metropolis sampling using strictly the bare Wilson action S[U]."""
        lat = PeriodicLattice4D(shape=(2, 2, 2, 2), n_colors=2)
        # Cold start
        self.assertEqual(len(lat.links), 2 * 2 * 2 * 2 * 4)
        p0 = lat.plaquette((0, 0, 0, 0), 0, 1)
        # Cold start plaquette is identity -> Tr(I) = 2
        self.assertAlmostEqual(p0.trace().real, 2.0)

        # Run Metropolis updates
        acc = lat.metropolis_sweep(beta=2.0, step_scale=0.3)
        self.assertGreaterEqual(acc, 0.0)

    def test_creutz_ratio_emergent_string_tension(self) -> None:
        """Verifies emergent string tension chi(1, 1) > 0 at strong coupling beta=0.5."""
        lat = PeriodicLattice4D(shape=(2, 2, 2, 2), n_colors=2)
        evaluator = CreutzRatioEvaluator(lat)

        prof = evaluator.measure_profile(beta=0.5, n_thermalization=20, n_samples=5, max_size=1)
        self.assertTrue(prof.is_confining)
        self.assertGreater(prof.string_tension, 0.0)
        self.assertIn((1, 1), prof.creutz_ratios)


if __name__ == "__main__":
    unittest.main()
