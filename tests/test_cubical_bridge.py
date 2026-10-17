"""Unit tests for the ToposLang-Cubical Homotopy Bridge and Realization Functor R."""

import unittest
from fractions import Fraction
from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.clifford import Multivector, PGA301
from topos.core.complex import CellComplex
from topos.cubical.bridge import RealizationFunctor
from topos.topology.homology import HomologyEngine


class TestCubicalBridge(unittest.TestCase):
    """Verifies Realization Functor, cycle detection, Kan box synthesis, and DACC re-integration."""

    def test_boundary_column_to_face_mask(self) -> None:
        functor = RealizationFunctor(PGA301)
        # Column with +1 and -1 entries (e.g. edge from v0 to v1)
        col = [Fraction(-1, 1), Fraction(1, 1)]
        fmask = functor.boundary_column_to_face_mask(col, dimension=2)
        # First non-zero (-1) -> axis 0, value 0 -> bit 0 -> mask 1
        # Second non-zero (+1) -> axis 1, value 1 -> bit 2*1+1 = 3 -> mask 8
        self.assertTrue(fmask.has_face(0, 0))
        self.assertTrue(fmask.has_face(1, 1))

    def test_circle_loop_resolution_via_cubical_bridge(self) -> None:
        # Build Circle S1: base 0-cell, loop 1-cell (beta_0 = 1, beta_1 = 1)
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)

        s1 = CellComplex("S1_Bridge")
        s1.add_cell(base)
        s1.add_cell(loop)

        homology_before = HomologyEngine(s1).betti_profile()
        self.assertEqual(homology_before[1], 1)

        # Initialize Realization Functor with PGA(3,0,1)
        functor = RealizationFunctor(signature=PGA301, max_homotopy_dim=2)

        # Attach initial multivector section to loop
        u_init = Multivector.from_named_blades(PGA301, {"1": 1.0, "e1": 2.5})
        functor.attach_section("loop", u_init)

        # Resolve harmonic cycle via Kan composition bridge
        # Rotor bivector e1e2 for 90-degree continuous rotation
        e1e2 = Multivector.from_named_blades(PGA301, {"e1e2": 1.0})
        res = functor.resolve_cycle(
            complex_obj=s1,
            dim=1,
            generator_bivector=e1e2,
            step_index=0,
            rule_name="loop_closure",
        )

        self.assertIsNotNone(res)
        self.assertEqual(res.betti_before[1], 1)
        self.assertEqual(res.betti_after[1], 0)  # Loop closed! beta_1 dropped to 0!
        self.assertEqual(res.dimension, 2)
        self.assertIn("homotopy_loop_closure_1_0", s1.graded_cells[2][0].name)

        # Verify terminal multivector state
        terminal_mv = res.terminal_multivector
        self.assertAlmostEqual(terminal_mv.components[0], 1.0)
        self.assertIsNotNone(functor.get_section(res.homotopy_cell_name))


if __name__ == "__main__":
    unittest.main()
