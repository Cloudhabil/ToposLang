"""Differential verification harness comparing legacy ExactMatrix against CSROperator over Q and F_p."""

import unittest
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.matrix_csr import PrimeField, RationalField, CSRHomologyEngine
from topos.topology.homology import HomologyEngine


class TestDifferentialEngine(unittest.TestCase):
    def _assert_engines_match(self, complex_obj: CellComplex) -> None:
        """Verifies that legacy HomologyEngine matches CSRHomologyEngine over Q and F_p (p=65537)."""
        legacy_engine = HomologyEngine(complex_obj)
        csr_q_engine = CSRHomologyEngine(complex_obj, field=RationalField())
        csr_fp_engine = CSRHomologyEngine(complex_obj, field=PrimeField(65537))

        max_dim = complex_obj.dim
        for d in range(max_dim + 1):
            legacy_betti = legacy_engine.betti_number(d)
            csr_q_betti = csr_q_engine.betti_number(d)
            csr_fp_betti = csr_fp_engine.betti_number(d)

            # Invariant I: Rational Homological Equivalence
            self.assertEqual(
                legacy_betti,
                csr_q_betti,
                f"Mismatch in dim {d} for {complex_obj.name}: legacy={legacy_betti} != csr_q={csr_q_betti}"
            )
            # Invariant II: Modular Isomorphism on Torsion-Free Topologies
            self.assertEqual(
                csr_q_betti,
                csr_fp_betti,
                f"Modular collapse in dim {d} for {complex_obj.name}: csr_q={csr_q_betti} != csr_fp={csr_fp_betti}"
            )

    def test_periodic_ring_circle_s1(self) -> None:
        """Periodic 1D ring (Circle S1): beta_0 = 1, beta_1 = 1."""
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)

        s1 = CellComplex("S1_Ring")
        s1.add_cell(base)
        s1.add_cell(loop)

        self._assert_engines_match(s1)

    def test_discretized_periodic_ring_n_cycle(self) -> None:
        """Periodic ring graph with N=8 vertices and 8 directed edges: beta_0 = 1, beta_1 = 1."""
        N = 8
        ring = CellComplex("PeriodicRing8")
        vertices = [GlobularCell(f"v_{i}", dim=0) for i in range(N)]
        for v in vertices:
            ring.add_cell(v)

        for i in range(N):
            edge = GlobularCell(f"e_{i}", dim=1, source=vertices[i], target=vertices[(i + 1) % N])
            ring.add_cell(edge)

        self._assert_engines_match(ring)

    def test_torus_t2(self) -> None:
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

        self._assert_engines_match(torus)

    def test_sphere_s2(self) -> None:
        """Hollow tetrahedron (topological 2-sphere S2): beta_0=1, beta_1=0, beta_2=1."""
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))

        s2 = CellComplex("S2")
        for f in (f1, f2, f3, f4):
            s2.add_cell(f, auto_add_boundaries=True)

        self._assert_engines_match(s2)

    def test_hollow_and_filled_triangle(self) -> None:
        vA, vB, vC = Simplex(("A",)), Simplex(("B",)), Simplex(("C",))
        eAB, eBC, eAC = Simplex(("A", "B")), Simplex(("B", "C")), Simplex(("A", "C"))

        hollow = CellComplex("HollowTriangle")
        for v in (vA, vB, vC):
            hollow.add_cell(v)
        for e in (eAB, eBC, eAC):
            hollow.add_cell(e)

        self._assert_engines_match(hollow)

        filled = CellComplex.from_simplices([Simplex(("A", "B", "C"))])
        self._assert_engines_match(filled)


if __name__ == "__main__":
    unittest.main()
