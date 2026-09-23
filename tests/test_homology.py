"""Unit tests for HomologyEngine, boundary matrices, and Betti numbers."""

import unittest
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.topology.homology import HomologyEngine, ExactMatrix
from topos.topology.invariant import (
    betti_numbers,
    euler_characteristic,
    verify_euler_poincare,
    is_connected,
    is_simply_connected,
    has_uncontracted_loops,
)


class TestHomology(unittest.TestCase):
    def test_exact_matrix_rank(self) -> None:
        # 3x3 matrix of rank 2
        # [1, 2, 3]
        # [4, 5, 6]
        # [7, 8, 9] (row 3 = 2*row2 - row1)
        m = ExactMatrix.from_integers(3, 3, [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        self.assertEqual(m.rank(), 2)
        self.assertEqual(m.nullity(), 1)

    def test_point_homology(self) -> None:
        k = CellComplex("Point")
        k.add_cell(Simplex(("P",)))
        engine = HomologyEngine(k)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 0)
        self.assertTrue(is_connected(k))
        self.assertTrue(is_simply_connected(k))

    def test_circle_hit_s1(self) -> None:
        """Circle S1: beta_0 = 1, beta_1 = 1."""
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)

        s1 = CellComplex("S1")
        s1.add_cell(base)
        s1.add_cell(loop)

        engine = HomologyEngine(s1)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 1)
        self.assertTrue(has_uncontracted_loops(s1))
        self.assertFalse(is_simply_connected(s1))
        self.assertTrue(verify_euler_poincare(s1))

    def test_hollow_triangle_vs_filled_triangle(self) -> None:
        """Hollow triangle (1-sphere S1) has beta_1 = 1; filled triangle has beta_1 = 0."""
        vA, vB, vC = Simplex(("A",)), Simplex(("B",)), Simplex(("C",))
        eAB, eBC, eAC = Simplex(("A", "B")), Simplex(("B", "C")), Simplex(("A", "C"))

        hollow = CellComplex("HollowTriangle")
        for v in (vA, vB, vC):
            hollow.add_cell(v)
        for e in (eAB, eBC, eAC):
            hollow.add_cell(e)

        hollow_engine = HomologyEngine(hollow)
        self.assertEqual(hollow_engine.betti_number(0), 1)
        self.assertEqual(hollow_engine.betti_number(1), 1)
        self.assertTrue(has_uncontracted_loops(hollow))

        # Fill the triangle with 2-simplex
        filled = CellComplex.from_simplices([Simplex(("A", "B", "C"))])
        filled_engine = HomologyEngine(filled)
        self.assertEqual(filled_engine.betti_number(0), 1)
        self.assertEqual(filled_engine.betti_number(1), 0)  # loop contracted!
        self.assertEqual(filled_engine.betti_number(2), 0)
        self.assertTrue(is_simply_connected(filled))
        self.assertTrue(verify_euler_poincare(filled))

    def test_sphere_s2_homology(self) -> None:
        """Hollow tetrahedron (topological 2-sphere S2): beta_0=1, beta_1=0, beta_2=1."""
        # 4 faces of tetrahedron
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))

        s2 = CellComplex("S2")
        for f in (f1, f2, f3, f4):
            s2.add_cell(f, auto_add_boundaries=True)

        engine = HomologyEngine(s2)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 0)
        self.assertEqual(engine.betti_number(2), 1)
        self.assertEqual(euler_characteristic(s2), 2)  # chi(S2) = 2
        self.assertTrue(verify_euler_poincare(s2))

    def test_torus_hit_t2(self) -> None:
        """Torus T2: beta_0 = 1, beta_1 = 2, beta_2 = 1."""
        v = GlobularCell("v", dim=0)
        a = GlobularCell("a", dim=1, source=v, target=v)
        b = GlobularCell("b", dim=1, source=v, target=v)

        source_chain = Chain.from_cell(a) + Chain.from_cell(b)
        target_chain = Chain.from_cell(b) + Chain.from_cell(a)
        face = GlobularCell("face", dim=2, source=source_chain, target=target_chain)

        torus = CellComplex("Torus")
        torus.add_cell(v)
        torus.add_cell(a)
        torus.add_cell(b)
        torus.add_cell(face)

        engine = HomologyEngine(torus)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 2)
        self.assertEqual(engine.betti_number(2), 1)
        self.assertEqual(euler_characteristic(torus), 0)  # chi(T2) = 0
        self.assertTrue(verify_euler_poincare(torus))


    def test_exact_matrix_rank_zero_columns(self) -> None:
        """Verifies exact rank computation when leading columns or intermediate columns have zero pivots."""
        from topos.topology.homology import ExactMatrix
        m = ExactMatrix.from_integers(3, 3, [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ])
        self.assertEqual(m.rank(), 2)
        self.assertEqual(m.nullity(), 1)


if __name__ == "__main__":
    unittest.main()
