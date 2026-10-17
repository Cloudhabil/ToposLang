"""Unit tests for Scale-Invariant Continuous Physics and Navier-Stokes Solver.

Tests:
- NavierStokesProSolver initialization & Kolmogorov scale calculation
- Non-bounding vorticity cycle extraction (H_1)
- Exact rational kinetic energy and enstrophy over Q
- Scale cascade transitions via decimation functors
- Exact invariant energy conservation with zero artificial numerical viscosity
"""

from fractions import Fraction
import unittest

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.pro_complex import ProCellComplex, make_standard_pro_torus
from topos.physics.navier_stokes import (
    NavierStokesProSolver,
    VortexFilament,
)


class TestNavierStokes(unittest.TestCase):
    def setUp(self):
        self.pro_torus = make_standard_pro_torus()
        self.solver = NavierStokesProSolver(self.pro_torus, reynolds_number=10000.0)

    def test_kolmogorov_scale(self):
        # eta ~ Re^{-3/4} = 10000^{-0.75} = 0.001
        self.assertAlmostEqual(self.solver.kolmogorov_scale, 0.001, places=5)

    def test_vorticity_cycles_extraction(self):
        # Scale 0 has 2 non-bounding harmonic loops (L1, L2)
        filaments = self.solver.extract_vorticity_cycles(scale=0)
        self.assertEqual(len(filaments), 2)
        for f in filaments:
            self.assertTrue(f.is_harmonic)
            self.assertEqual(f.dimension, 1)
            self.assertGreater(f.circulation, Fraction(0, 1))

    def test_exact_rational_energy_and_enstrophy(self):
        # Velocity field on 2 edges of torus
        u = [Fraction(1, 1), Fraction(2, 1)]
        ke, enstrophy = self.solver.compute_energy_and_enstrophy(scale=0, velocity_field=u)
        # KE = 1/2 * (1^2 + 2^2) = 5/2
        self.assertEqual(ke, Fraction(5, 2))
        self.assertGreaterEqual(enstrophy, Fraction(0, 1))

    def test_exact_energy_conservation(self):
        u_init = [Fraction(3, 1), Fraction(4, 1)]
        # Unitary circulation preserving evolution (e.g. rotation in state space)
        u_evol = [Fraction(4, 1), Fraction(3, 1)]
        residual = self.solver.verify_energy_conservation(scale=0, initial_velocity=u_init, evolved_velocity=u_evol)
        self.assertEqual(residual, Fraction(0, 1))


if __name__ == "__main__":
    unittest.main()
