"""Unit tests for Distributed Consensus with Infinite Horizons and Dynamic Churn.

Tests:
- DynamicChurnNetwork channels and dynamic scale complexes
- Deadlock detection via Hodge 1-Laplacian harmonic cycles in ker(L_1)
- Deadlock resolution via homotopy 2-cell capping
- Invariant evaluation: until betti(lim Km, dim=1) == 0
"""

import unittest

from topos.consensus.infinite_consensus import (
    DynamicChurnNetwork,
    ConsensusReport,
)


class TestInfiniteConsensus(unittest.TestCase):
    def setUp(self):
        self.net = DynamicChurnNetwork("P2PConsensusNet")
        # Set up a circular dependency: A -> B -> C -> A
        self.net.add_channel("ch_AB", "A", "B")
        self.net.add_channel("ch_BC", "B", "C")
        self.net.add_channel("ch_CA", "C", "A")

    def test_deadlock_detection(self):
        comp = self.net.build_scale_complex(scale=0, active_nodes=["A", "B", "C"])
        self.net.pro_complex.set_scale(0, comp)

        deadlocks = self.net.detect_deadlocks(comp)
        self.assertEqual(len(deadlocks), 1)
        cycle_channels = deadlocks[0]
        self.assertIn("ch_AB", cycle_channels)
        self.assertIn("ch_BC", cycle_channels)
        self.assertIn("ch_CA", cycle_channels)

        report = self.net.evaluate_projective_consensus(max_scale=0)
        self.assertFalse(report.is_consensus_achieved)
        self.assertEqual(report.open_deadlocks, 1)

    def test_deadlock_resolution_via_cycle_contraction(self):
        comp = self.net.build_scale_complex(scale=0, active_nodes=["A", "B", "C"])
        self.net.pro_complex.set_scale(0, comp)

        deadlocks = self.net.detect_deadlocks(comp)
        self.assertEqual(len(deadlocks), 1)

        # Resolve circular deadlock by attaching capping homotopy 2-cell
        self.net.resolve_deadlock(comp, deadlocks[0], resolution_id="res_abc")

        # After resolution, harmonic cycle is contracted
        remaining_deadlocks = self.net.detect_deadlocks(comp)
        self.assertEqual(len(remaining_deadlocks), 0)

        report = self.net.evaluate_projective_consensus(max_scale=0)
        self.assertTrue(report.is_consensus_achieved)
        self.assertEqual(report.open_deadlocks, 0)


if __name__ == "__main__":
    unittest.main()
