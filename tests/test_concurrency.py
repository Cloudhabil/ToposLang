"""Unit and integration tests for T-IR DACC, Contractions, and Spatial Concurrency."""

import unittest
from topos.core.cell import GlobularCell
from topos.core.complex import CellComplex
from topos.ir.tir import DACC
from topos.ir.contract import HomotopyContractionPass
from topos.runtime.engine import RewriteRule
from topos.runtime.scheduler import SpatialPartitioner, SpatialScheduler
from topos.topology.homology import HomologyEngine


class TestConcurrency(unittest.TestCase):
    def test_dacc_causal_ordering(self) -> None:
        """DACC sorts cells causally: 0-cells precede 1-cells, 1-cells precede 2-cells."""
        vA = GlobularCell("A", dim=0)
        vB = GlobularCell("B", dim=0)
        e = GlobularCell("e", dim=1, source=vA, target=vB)

        k = CellComplex("Edge")
        k.add_cell(vA)
        k.add_cell(vB)
        k.add_cell(e)

        dacc = DACC(k)
        self.assertTrue(dacc.is_acyclic())
        order = dacc.causal_order()
        self.assertEqual(len(order), 3)

        # 0-cells must appear before the 1-cell that depends on them
        e_idx = order.index(e)
        vA_idx = order.index(vA)
        vB_idx = order.index(vB)
        self.assertLess(vA_idx, e_idx)
        self.assertLess(vB_idx, e_idx)

    def test_homotopy_contraction_pass(self) -> None:
        """Collapsing a contractible pendant edge preserves global homology exactly."""
        # Hollow triangle: A, B, C with e1, e2, e3 (beta_1 = 1)
        vA = GlobularCell("A", dim=0)
        vB = GlobularCell("B", dim=0)
        vC = GlobularCell("C", dim=0)
        e1 = GlobularCell("e1", dim=1, source=vA, target=vB)
        e2 = GlobularCell("e2", dim=1, source=vB, target=vC)
        e3 = GlobularCell("e3", dim=1, source=vA, target=vC)

        # Pendant vertex and contractible edge: P with e_pendant : P -> A
        vP = GlobularCell("P", dim=0)
        e_pendant = GlobularCell("e_pendant", dim=1, source=vP, target=vA)

        k = CellComplex("TriangleWithTail")
        for v in (vA, vB, vC, vP):
            k.add_cell(v)
        for e in (e1, e2, e3, e_pendant):
            k.add_cell(e)

        # Before contraction: beta_0 = 1, beta_1 = 1
        homology_before = HomologyEngine(k)
        self.assertEqual(homology_before.betti_number(0), 1)
        self.assertEqual(homology_before.betti_number(1), 1)

        # Contract the pendant edge
        contracted = HomotopyContractionPass.contract_edge(k, "e_pendant")

        # After contraction: beta_0 = 1, beta_1 = 1 (identical!), P and e_pendant are gone
        homology_after = HomologyEngine(contracted)
        self.assertEqual(homology_after.betti_number(0), 1)
        self.assertEqual(homology_after.betti_number(1), 1)
        self.assertEqual(len(contracted.get_cells(0)), 3)
        self.assertEqual(len(contracted.get_cells(1)), 3)
        self.assertTrue(contracted.validate(enforce_nilpotence=True))

    def test_spatial_partitioner_support_and_interference(self) -> None:
        """Verifies that shared boundary vertices produce interference, while disjoint cells do not."""
        vA = GlobularCell("A", dim=0)
        vB = GlobularCell("B", dim=0)
        vC = GlobularCell("C", dim=0)
        vD = GlobularCell("D", dim=0)

        # e1: A -> B, e2: B -> C (shared vertex B!)
        # e3: C -> D, e4: C -> D
        e1 = GlobularCell("e1", dim=1, source=vA, target=vB)
        e2 = GlobularCell("e2", dim=1, source=vB, target=vC)
        e3 = GlobularCell("e3", dim=1, source=vC, target=vD)
        e4 = GlobularCell("e4", dim=1, source=vC, target=vD)

        k = CellComplex("Linear")
        for v in (vA, vB, vC, vD):
            k.add_cell(v)
        for e in (e1, e2, e3, e4):
            k.add_cell(e)

        rule_1 = RewriteRule("r1", lhs=("e1",), rhs=("e1",))
        rule_2 = RewriteRule("r2", lhs=("e2",), rhs=("e2",))

        # r1 support includes {e1, A, B}
        supp1 = SpatialPartitioner.compute_support(k, rule_1)
        self.assertIn("A", supp1)
        self.assertIn("B", supp1)

        # r2 support includes {e2, B, C}
        supp2 = SpatialPartitioner.compute_support(k, rule_2)
        self.assertIn("B", supp2)
        self.assertIn("C", supp2)

        # Shared vertex B causes interference
        self.assertFalse(supp1.isdisjoint(supp2))

        # Partitioning will put r1 and r2 into separate wavefronts
        wfs = SpatialPartitioner.partition_wavefronts(k, [rule_1, rule_2])
        self.assertEqual(len(wfs), 2)

    def test_parallel_disjoint_rewrites(self) -> None:
        """Simultaneous parallel execution of 2 disjoint loop contractions on worker threads."""
        # Cluster 1: Triangle 1 (A, B, C with e1, e2, e3)
        vA = GlobularCell("A", dim=0)
        vB = GlobularCell("B", dim=0)
        vC = GlobularCell("C", dim=0)
        e1 = GlobularCell("e1", dim=1, source=vA, target=vB)
        e2 = GlobularCell("e2", dim=1, source=vB, target=vC)
        e3 = GlobularCell("e3", dim=1, source=vA, target=vC)

        # Cluster 2: Triangle 2 (D, E, F with e4, e5, e6)
        vD = GlobularCell("D", dim=0)
        vE = GlobularCell("E", dim=0)
        vF = GlobularCell("F", dim=0)
        e4 = GlobularCell("e4", dim=1, source=vD, target=vE)
        e5 = GlobularCell("e5", dim=1, source=vE, target=vF)
        e6 = GlobularCell("e6", dim=1, source=vD, target=vF)

        complex_obj = CellComplex("DualDisjointClusters")
        for v in (vA, vB, vC, vD, vE, vF):
            complex_obj.add_cell(v)
        for e in (e1, e2, e3, e4, e5, e6):
            complex_obj.add_cell(e)

        # Initially 2 disjoint components, 2 holes: beta_0 = 2, beta_1 = 2
        engine = HomologyEngine(complex_obj)
        self.assertEqual(engine.betti_number(0), 2)
        self.assertEqual(engine.betti_number(1), 2)

        rule1 = RewriteRule("collapse_cluster1", lhs=("e1", "e2"), rhs=("e3",))
        rule2 = RewriteRule("collapse_cluster2", lhs=("e4", "e5"), rhs=("e6",))

        # Partitioner recognizes they are topologically disjoint and bundles them into 1 wavefront
        wavefronts = SpatialPartitioner.partition_wavefronts(complex_obj, [rule1, rule2])
        self.assertEqual(len(wavefronts), 1)
        self.assertEqual(len(wavefronts[0]), 2)

        # Execute parallel wavefront across worker threads with 0 locks
        scheduler = SpatialScheduler(max_workers=2)
        trace_steps = scheduler.execute_wavefront(complex_obj, wavefronts[0])

        self.assertEqual(len(trace_steps), 2)
        # Verify that beta_1 dropped simultaneously from 2 to 0
        final_homology = HomologyEngine(complex_obj)
        self.assertEqual(final_homology.betti_number(0), 2)
        self.assertEqual(final_homology.betti_number(1), 0)

        # Verify full complex nilpotency d^2 = 0 after parallel merge
        self.assertTrue(complex_obj.validate(enforce_nilpotence=True))


if __name__ == "__main__":
    unittest.main()
