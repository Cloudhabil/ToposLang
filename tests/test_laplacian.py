"""Unit tests for discrete Hodge Laplacians and harmonic forms."""

import unittest
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.topology.homology import BoundaryMatrix
from topos.topology.laplacian import HodgeLaplacian


class TestLaplacian(unittest.TestCase):
    def test_matrix_nilpotence(self) -> None:
        """Matrix version of d^2 = 0: B_{k-1} @ B_k == 0."""
        tet = Simplex(("A", "B", "C", "D"))
        k = CellComplex.from_simplices([tet])

        B1 = BoundaryMatrix.from_complex(k, 1).matrix
        B2 = BoundaryMatrix.from_complex(k, 2).matrix
        B3 = BoundaryMatrix.from_complex(k, 3).matrix

        # B1 @ B2 must be 0
        prod_1_2 = B1.matmul(B2)
        for r in range(prod_1_2.rows):
            for c in range(prod_1_2.cols):
                self.assertEqual(prod_1_2.get(r, c), 0)

        # B2 @ B3 must be 0
        prod_2_3 = B2.matmul(B3)
        for r in range(prod_2_3.rows):
            for c in range(prod_2_3.cols):
                self.assertEqual(prod_2_3.get(r, c), 0)

    def test_circle_hodge_laplacian(self) -> None:
        """Verify dim(ker(L_k)) == beta_k for Circle S1."""
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)

        s1 = CellComplex("S1")
        s1.add_cell(base)
        s1.add_cell(loop)

        L0 = HodgeLaplacian(s1, 0)
        L1 = HodgeLaplacian(s1, 1)

        self.assertEqual(L0.harmonic_dimension(), 1)
        self.assertEqual(L1.harmonic_dimension(), 1)

    def test_sphere_s2_hodge_laplacian(self) -> None:
        """Verify dim(ker(L_k)) == beta_k for 2-sphere S2."""
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))

        s2 = CellComplex("S2")
        for f in (f1, f2, f3, f4):
            s2.add_cell(f, auto_add_boundaries=True)

        L0 = HodgeLaplacian(s2, 0)
        L1 = HodgeLaplacian(s2, 1)
        L2 = HodgeLaplacian(s2, 2)

        self.assertEqual(L0.harmonic_dimension(), 1)  # beta_0 = 1
        self.assertEqual(L1.harmonic_dimension(), 0)  # beta_1 = 0
        self.assertEqual(L2.harmonic_dimension(), 1)  # beta_2 = 1

    def test_torus_t2_hodge_laplacian(self) -> None:
        """Verify dim(ker(L_k)) == beta_k for Torus T2."""
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

        L0 = HodgeLaplacian(torus, 0)
        L1 = HodgeLaplacian(torus, 1)
        L2 = HodgeLaplacian(torus, 2)

        self.assertEqual(L0.harmonic_dimension(), 1)  # beta_0 = 1
        self.assertEqual(L1.harmonic_dimension(), 2)  # beta_1 = 2
        self.assertEqual(L2.harmonic_dimension(), 1)  # beta_2 = 1


if __name__ == "__main__":
    unittest.main()
