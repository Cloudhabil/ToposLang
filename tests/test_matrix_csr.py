"""Unit tests for CSROperator, Finite Fields (F_p, F_2, Q), and direct plaquette down-Laplacian."""

import unittest
from fractions import Fraction
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.matrix_csr import (
    PrimeField,
    BinaryField,
    RationalField,
    CSROperator,
    compute_plaquette_down_laplacian,
    CSRHomologyEngine,
)


class TestMatrixCSR(unittest.TestCase):
    def test_prime_field_arithmetic(self) -> None:
        f = PrimeField(7)
        self.assertEqual(f.add(4, 5), 2)
        self.assertEqual(f.sub(2, 5), 4)
        self.assertEqual(f.mul(3, 4), 5)
        self.assertEqual(f.inv(3), 5)  # 3 * 5 = 15 = 1 mod 7
        self.assertEqual(f.mul(3, f.inv(3)), 1)
        with self.assertRaises(ZeroDivisionError):
            f.inv(0)

    def test_binary_field_arithmetic(self) -> None:
        f2 = BinaryField()
        self.assertEqual(f2.add(1, 1), 0)
        self.assertEqual(f2.sub(1, 0), 1)
        self.assertEqual(f2.mul(1, 1), 1)
        self.assertEqual(f2.mul(1, 0), 0)
        self.assertEqual(f2.inv(1), 1)
        with self.assertRaises(ZeroDivisionError):
            f2.inv(0)

    def test_csr_dense_roundtrip(self) -> None:
        grid = [
            [1, 0, 2],
            [0, 0, 3],
            [4, 5, 0],
        ]
        csr = CSROperator.from_dense(grid, field=RationalField())
        self.assertEqual(csr.rows, 3)
        self.assertEqual(csr.cols, 3)
        self.assertEqual(csr.nnz, 5)

        dense = csr.to_dense()
        expected = [
            [Fraction(1), Fraction(0), Fraction(2)],
            [Fraction(0), Fraction(0), Fraction(3)],
            [Fraction(4), Fraction(5), Fraction(0)],
        ]
        self.assertEqual(dense, expected)

    def test_csr_transpose(self) -> None:
        grid = [
            [1, 2, 0],
            [0, 3, 4],
        ]
        csr = CSROperator.from_dense(grid, field=RationalField())
        csr_t = csr.transpose()
        self.assertEqual(csr_t.rows, 3)
        self.assertEqual(csr_t.cols, 2)
        expected_dense = [
            [Fraction(1), Fraction(0)],
            [Fraction(2), Fraction(3)],
            [Fraction(0), Fraction(4)],
        ]
        self.assertEqual(csr_t.to_dense(), expected_dense)

    def test_csr_matmul(self) -> None:
        # A: 2x3, B: 3x2
        # A = [[1, 2, 0], [0, 3, 4]]
        # B = [[1, 0], [0, 2], [3, 1]]
        # A @ B = [[1, 4], [12, 10]]
        A = CSROperator.from_dense([[1, 2, 0], [0, 3, 4]], field=RationalField())
        B = CSROperator.from_dense([[1, 0], [0, 2], [3, 1]], field=RationalField())
        C = A.matmul(B)
        self.assertEqual(C.rows, 2)
        self.assertEqual(C.cols, 2)
        expected = [
            [Fraction(1), Fraction(4)],
            [Fraction(12), Fraction(10)],
        ]
        self.assertEqual(C.to_dense(), expected)

    def test_csr_gaussian_elimination_and_rank_over_fp(self) -> None:
        # Rank-2 matrix in F_7:
        # [1, 2, 3]
        # [2, 4, 6] (2 * row 1)
        # [0, 1, 4]
        f7 = PrimeField(7)
        mat = CSROperator.from_dense([
            [1, 2, 3],
            [2, 4, 6],
            [0, 1, 4],
        ], field=f7)
        self.assertEqual(mat.rank(), 2)
        self.assertEqual(mat.nullity(), 1)

    def test_direct_plaquette_down_laplacian(self) -> None:
        """Verifies L_2^down = B_2^T @ B_2 directly couples adjacent plaquettes sharing an edge."""
        # 2 adjacent square plaquettes sharing edge e_mid:
        # v0 --- e_top0 --- v1 --- e_top1 --- v2
        #  |                 |                 |
        # e_left0          e_mid             e_right1
        #  |                 |                 |
        # v3 --- e_bot0 --- v4 --- e_bot1 --- v5
        complex_obj = CellComplex("TwoPlaquettes")
        v = [GlobularCell(f"v{i}", dim=0) for i in range(6)]
        for vertex in v:
            complex_obj.add_cell(vertex)

        e_top0 = GlobularCell("e_top0", dim=1, source=v[0], target=v[1])
        e_top1 = GlobularCell("e_top1", dim=1, source=v[1], target=v[2])
        e_left0 = GlobularCell("e_left0", dim=1, source=v[0], target=v[3])
        e_mid = GlobularCell("e_mid", dim=1, source=v[1], target=v[4])
        e_right1 = GlobularCell("e_right1", dim=1, source=v[2], target=v[5])
        e_bot0 = GlobularCell("e_bot0", dim=1, source=v[3], target=v[4])
        e_bot1 = GlobularCell("e_bot1", dim=1, source=v[4], target=v[5])

        for e in [e_top0, e_top1, e_left0, e_mid, e_right1, e_bot0, e_bot1]:
            complex_obj.add_cell(e)

        # Plaquette 0: boundary = e_top0 + e_mid - e_bot0 - e_left0
        # target_chain = e_top0 + e_mid, source_chain = e_left0 + e_bot0 => boundary = target - source
        p0 = GlobularCell("p0", dim=2,
                          source=Chain.from_cell(e_left0) + Chain.from_cell(e_bot0),
                          target=Chain.from_cell(e_top0) + Chain.from_cell(e_mid))
        # Plaquette 1: boundary = e_top1 + e_right1 - e_bot1 - e_mid
        # (notice e_mid is traversed downwards for p0 (+1) and upwards/opposing for p1 (-1))
        p1 = GlobularCell("p1", dim=2,
                          source=Chain.from_cell(e_mid) + Chain.from_cell(e_bot1),
                          target=Chain.from_cell(e_top1) + Chain.from_cell(e_right1))

        complex_obj.add_cell(p0)
        complex_obj.add_cell(p1)

        f_q = RationalField()
        L2_down = compute_plaquette_down_laplacian(complex_obj, field=f_q)

        self.assertEqual(L2_down.rows, 2)
        self.assertEqual(L2_down.cols, 2)
        # Plaquette 0 has 4 edges: self-degree = 4
        # Plaquette 1 has 4 edges: self-degree = 4
        # Shared edge is e_mid with coeff +1 in p0 and -1 in p1: coupling = -1
        dense_L2 = L2_down.to_dense()
        self.assertEqual(dense_L2[0][0], Fraction(4, 1))
        self.assertEqual(dense_L2[1][1], Fraction(4, 1))
        self.assertEqual(dense_L2[0][1], Fraction(-1, 1))
        self.assertEqual(dense_L2[1][0], Fraction(-1, 1))


if __name__ == "__main__":
    unittest.main()
