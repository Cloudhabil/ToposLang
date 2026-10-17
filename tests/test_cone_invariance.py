"""Unit tests for Reflection-Positive Cone Invariance and Continuum Mass Gap.

Tests:
- CharacterExpansion and reflection-positive cone membership (c_j >= 0)
- Clebsch-Gordan fusion algebra for SU(2)
- Pace Operator R = D o B forward invariance (R(C) subset C)
- Unconditional one-pace lower floor (Theorem 16.28)
- Dimensional transmutation & continuum non-analytic mass gap Delta > 0
"""

import math
import unittest

from topos.topology.cone_invariance import (
    CharacterExpansion,
    PaceOperator,
    MassGapCalculator,
)


class TestConeInvariance(unittest.TestCase):
    def test_cone_membership(self):
        w_in_cone = CharacterExpansion({"0": 1.0, "1/2": 0.2, "1": 0.05})
        self.assertTrue(w_in_cone.is_in_reflection_positive_cone())

        w_outside_cone = CharacterExpansion({"0": 1.0, "1/2": -0.1, "1": 0.05})
        self.assertFalse(w_outside_cone.is_in_reflection_positive_cone())

    def test_clebsch_gordan_fusion(self):
        # 1/2 x 1/2 = 0 + 1
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(0.5, 0.5, 0.0), 1)
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(0.5, 0.5, 1.0), 1)
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(0.5, 0.5, 0.5), 0)

        # 1 x 1 = 0 + 1 + 2
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(1.0, 1.0, 0.0), 1)
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(1.0, 1.0, 1.0), 1)
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(1.0, 1.0, 2.0), 1)
        self.assertEqual(CharacterExpansion.clebsch_gordan_mult(1.0, 1.0, 3.0), 0)

    def test_forward_cone_invariance(self):
        # Initial weight in cone
        w = CharacterExpansion({"0": 1.0, "1/2": 0.3, "1": 0.1})
        pace_op = PaceOperator(w)

        # Forward invariance: R(C) subset C
        self.assertTrue(pace_op.verify_forward_cone_invariance())

        w_bond, w_pace = pace_op.execute_pace()
        self.assertTrue(w_bond.is_in_reflection_positive_cone())
        self.assertTrue(w_pace.is_in_reflection_positive_cone())

    def test_theorem_16_28_one_pace_floor(self):
        w = CharacterExpansion({"0": 1.0, "1/2": 0.2, "1": 0.05})
        pace_op = PaceOperator(w)
        floor_val = pace_op.unconditional_one_pace_floor()
        self.assertGreater(floor_val, 0.0)

    def test_mass_gap_positivity(self):
        scale = MassGapCalculator.dimensional_transmutation_scale(beta=6.0)
        self.assertGreater(scale, 0.0)

        delta = MassGapCalculator.compute_mass_gap(beta=6.0, q_odd=0.15)
        self.assertGreater(delta, 0.0)


if __name__ == "__main__":
    unittest.main()
