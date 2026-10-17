"""Unit tests for Roofline Model and Static Compiler Invariant Engine."""

import unittest

from topos.compiler.roofline import (
    RooflineCompilerVerifier,
    RooflineContractReport,
    verify_roofline_machine_balance,
)
from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex


class TestRoofline(unittest.TestCase):
    def test_direct_machine_balance_invariant(self):
        # L=4 prime lanes, d=6 -> L*d = 24 >= 19
        rep = verify_roofline_machine_balance(prime_lanes_L=4, column_density_d=6.0)
        self.assertTrue(rep.invariant_satisfied)
        self.assertTrue(rep.is_compute_bound)
        self.assertGreater(rep.arithmetic_intensity, 16.5)

        # L=2, d=4 -> L*d = 8 < 19
        rep_fail = verify_roofline_machine_balance(prime_lanes_L=2, column_density_d=4.0)
        self.assertFalse(rep_fail.invariant_satisfied)
        self.assertFalse(rep_fail.is_compute_bound)

    def test_roofline_compiler_verifier_complex(self):
        # Build 2-cell triangle
        comp = CellComplex("RooflineTriangle")
        v0 = GlobularCell("V0", dim=0)
        v1 = GlobularCell("V1", dim=0)
        v2 = GlobularCell("V2", dim=0)
        for v in (v0, v1, v2):
            comp.add_cell(v)

        e01 = GlobularCell("E01", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v1))
        e12 = GlobularCell("E12", dim=1, source=Chain.from_cell(v1), target=Chain.from_cell(v2))
        e02 = GlobularCell("E02", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v2))
        for e in (e01, e12, e02):
            comp.add_cell(e)

        verifier = RooflineCompilerVerifier()
        report = verifier.verify_complex(comp, dim=1)

        self.assertGreater(report.num_cells_N, 0)
        self.assertGreater(report.max_tile_size_T, 0)
        self.assertGreater(report.hadamard_bit_bound_L_max, 0.0)
        self.assertGreaterEqual(report.required_prime_lanes_L, 4)


if __name__ == "__main__":
    unittest.main()
