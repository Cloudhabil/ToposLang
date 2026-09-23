"""Unit tests for Simplicial boundary formulas and delta complexes."""

import unittest
from topos.core.cell import Simplex
from topos.core.boundary import boundary
from topos.core.complex import CellComplex


class TestSimplex(unittest.TestCase):
    def test_0_simplex(self) -> None:
        v = Simplex(("A",))
        self.assertEqual(v.dim, 0)
        self.assertEqual(v.name, "[A]")
        self.assertTrue(v.boundary().is_zero())

    def test_1_simplex_boundary(self) -> None:
        e = Simplex(("A", "B"))
        self.assertEqual(e.dim, 1)
        b = e.boundary()
        self.assertEqual(b.dim, 0)
        # d([A, B]) = [B] - [A]
        v_A = Simplex(("A",))
        v_B = Simplex(("B",))
        self.assertEqual(b[v_B], 1)
        self.assertEqual(b[v_A], -1)

    def test_2_simplex_boundary(self) -> None:
        t = Simplex(("A", "B", "C"))
        self.assertEqual(t.dim, 2)
        b = t.boundary()
        self.assertEqual(b.dim, 1)
        # d([A, B, C]) = [B, C] - [A, C] + [A, B]
        e_BC = Simplex(("B", "C"))
        e_AC = Simplex(("A", "C"))
        e_AB = Simplex(("A", "B"))
        self.assertEqual(b[e_BC], 1)
        self.assertEqual(b[e_AC], -1)
        self.assertEqual(b[e_AB], 1)

    def test_3_simplex_boundary(self) -> None:
        tet = Simplex(("A", "B", "C", "D"))
        self.assertEqual(tet.dim, 3)
        b = tet.boundary()
        self.assertEqual(b.dim, 2)
        self.assertEqual(len(b), 4)
        # Verify alternating signs
        self.assertEqual(b[Simplex(("B", "C", "D"))], 1)
        self.assertEqual(b[Simplex(("A", "C", "D"))], -1)
        self.assertEqual(b[Simplex(("A", "B", "D"))], 1)
        self.assertEqual(b[Simplex(("A", "B", "C"))], -1)

    def test_simplicial_complex_generation(self) -> None:
        tet = Simplex(("A", "B", "C", "D"))
        k = CellComplex.from_simplices([tet], name="Tetrahedron")
        self.assertEqual(k.dim, 3)
        # 1 tetrahedron (dim 3), 4 triangles (dim 2), 6 edges (dim 1), 4 vertices (dim 0)
        self.assertEqual(len(k.get_cells(3)), 1)
        self.assertEqual(len(k.get_cells(2)), 4)
        self.assertEqual(len(k.get_cells(1)), 6)
        self.assertEqual(len(k.get_cells(0)), 4)
        # Euler characteristic for contractible space (simplex) is 1: 4 - 6 + 4 - 1 = 1
        self.assertEqual(k.euler_characteristic(), 1)


if __name__ == "__main__":
    unittest.main()
