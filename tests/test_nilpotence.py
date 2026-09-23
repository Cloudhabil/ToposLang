"""Unit and integration tests for Nilpotency Verifier (d^2 = 0)."""

import unittest
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import (
    boundary,
    NilpotencyVerifier,
    NilpotencyViolationError,
    CellNotFoundError,
)


class TestNilpotence(unittest.TestCase):
    def test_simplicial_nilpotence(self) -> None:
        """Verify d_{k-1} o d_k = 0 across 1, 2, and 3-simplices."""
        # 1-simplex
        e = Simplex(("A", "B"))
        self.assertTrue(NilpotencyVerifier.verify_cell(e))

        # 2-simplex
        tri = Simplex(("A", "B", "C"))
        self.assertTrue(NilpotencyVerifier.verify_cell(tri))
        # Direct check on chain
        d1 = boundary(tri)
        d2 = boundary(d1)
        self.assertTrue(d2.is_zero())

        # 3-simplex
        tet = Simplex(("A", "B", "C", "D"))
        self.assertTrue(NilpotencyVerifier.verify_cell(tet))
        d_tet_1 = boundary(tet)
        d_tet_2 = boundary(d_tet_1)
        self.assertTrue(d_tet_2.is_zero())

    def test_circle_hit_s1(self) -> None:
        """Verify S1 circle Higher Inductive Type: point base, path loop : base -> base."""
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)

        complex_s1 = CellComplex(name="S1")
        complex_s1.add_cell(base)
        complex_s1.add_cell(loop)

        # Boundary of loop: target - source = base - base = 0
        self.assertTrue(loop.boundary().is_zero())
        self.assertTrue(complex_s1.validate(enforce_nilpotence=True))
        # Euler characteristic of S1 is chi = 1 - 1 = 0
        self.assertEqual(complex_s1.euler_characteristic(), 0)

    def test_torus_hit_t2(self) -> None:
        """Verify T2 torus Higher Inductive Type: 1 point, 2 loops, 1 surface."""
        v = GlobularCell("v", dim=0)
        a = GlobularCell("a", dim=1, source=v, target=v)
        b = GlobularCell("b", dim=1, source=v, target=v)

        # Commutator boundary: (a * b) => (b * a)
        source_chain = Chain.from_cell(a) + Chain.from_cell(b)
        target_chain = Chain.from_cell(b) + Chain.from_cell(a)
        torus_2cell = GlobularCell(
            "face", dim=2, source=source_chain, target=target_chain
        )

        torus = CellComplex(name="Torus")
        torus.add_cell(v)
        torus.add_cell(a)
        torus.add_cell(b)
        torus.add_cell(torus_2cell)

        self.assertTrue(NilpotencyVerifier.verify_cell(torus_2cell))
        self.assertTrue(torus.validate(enforce_nilpotence=True))
        # Euler characteristic of Torus is 1 - 2 + 1 = 0
        self.assertEqual(torus.euler_characteristic(), 0)

    def test_negative_nilpotence_rejection(self) -> None:
        """A corrupted 2-cell whose boundary is not closed must raise NilpotencyViolationError."""
        vA = GlobularCell("A", dim=0)
        vB = GlobularCell("B", dim=0)
        vC = GlobularCell("C", dim=0)

        # e1: A -> B, e2: B -> C
        e1 = GlobularCell("e1", dim=1, source=vA, target=vB)
        e2 = GlobularCell("e2", dim=1, source=vB, target=vC)

        # Deliberately invalid 2-cell where source is e1 (A->B) and target is e2 (B->C).
        # d(e2) - d(e1) = (C - B) - (B - A) = C - 2B + A != 0.
        corrupted_cell = GlobularCell("corrupted_2cell", dim=2, source=e1, target=e2)

        with self.assertRaises(NilpotencyViolationError) as ctx:
            NilpotencyVerifier.verify_cell(corrupted_cell)

        err = ctx.exception
        self.assertEqual(err.cell, corrupted_cell)
        self.assertEqual(err.dim, 2)
        # Verify residual contains non-zero terms
        self.assertFalse(err.residual.is_zero())
        self.assertEqual(err.residual[vC], 1)
        self.assertEqual(err.residual[vB], -2)
        self.assertEqual(err.residual[vA], 1)

    def test_missing_boundary_closure_rejection(self) -> None:
        """Adding a 2-cell without registering its 1-cells should fail closure check."""
        vA = Simplex(("A",))
        vB = Simplex(("B",))
        e = Simplex(("A", "B"))

        complex_obj = CellComplex(name="OpenEdge")
        complex_obj.add_cell(vA)
        # vB is missing!
        with self.assertRaises(CellNotFoundError):
            complex_obj.add_cell(e, validate_closure=True)


if __name__ == "__main__":
    unittest.main()
