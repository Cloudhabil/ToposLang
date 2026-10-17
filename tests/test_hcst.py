"""Unit tests for the Hodge-Clifford Spatial Transformer (HCST) Engine."""

import math
import unittest

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.clifford import Multivector, PGA301
from topos.core.complex import CellComplex
from topos.core.four_tier_dacc import FourTierDACC
from topos.physics.hcst import (
    DualStreamCliffordAttention,
    HCSTTransformerLayer,
    HodgeDiracMessagePassing,
    QKVProjectionParams,
    TopologicalPoolingEngine,
)


class TestHCST(unittest.TestCase):
    def setUp(self):
        # Build 2-cell triangle complex
        self.complex = CellComplex("TriangleDACC")
        v0 = GlobularCell("V0", dim=0)
        v1 = GlobularCell("V1", dim=0)
        v2 = GlobularCell("V2", dim=0)
        for v in (v0, v1, v2):
            self.complex.add_cell(v)

        e01 = GlobularCell("E01", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v1))
        e12 = GlobularCell("E12", dim=1, source=Chain.from_cell(v1), target=Chain.from_cell(v2))
        e02 = GlobularCell("E02", dim=1, source=Chain.from_cell(v0), target=Chain.from_cell(v2))
        for e in (e01, e12, e02):
            self.complex.add_cell(e)

        # Triangle face F0
        f0 = GlobularCell("F0", dim=2, source=Chain.from_cell(e01) + Chain.from_cell(e12), target=Chain.from_cell(e02))
        self.complex.add_cell(f0)

        self.dacc = FourTierDACC(self.complex)

    def test_dual_stream_clifford_attention(self):
        attn = DualStreamCliffordAttention(PGA301, wedge_weight_lambda=0.5)

        # Vector along x (e1) and vector along y (e2)
        q = Multivector.from_named_blades(PGA301, {"e1": 1.0})
        k = Multivector.from_named_blades(PGA301, {"e2": 1.0})

        # Metric alignment: <e1 * ~e2>_0 = <e1 * e2>_0 = <e12>_0 = 0
        # Exterior wedge: e1 ^ e2 = e12, norm^2 = 1.0
        # Expected unnormalized score = (0 - 0.5 * 1.0) / sqrt(16) = -0.5 / 4 = -0.125
        score = attn.compute_attention_score(q, k)
        self.assertAlmostEqual(score, -0.125)

        # Collinear vectors: e1 and e1
        # Metric alignment: <e1 * ~e1>_0 = <e1 * e1>_0 = 1.0
        # Exterior wedge: e1 ^ e1 = 0
        # Score = (1.0 - 0) / 4 = 0.25
        score_collinear = attn.compute_attention_score(q, q)
        self.assertAlmostEqual(score_collinear, 0.25)
        self.assertGreater(score_collinear, score)

    def test_qkv_projection(self):
        params = QKVProjectionParams.identity(PGA301)
        mv = Multivector.from_named_blades(PGA301, {"e1": 2.0, "e1e2": 1.5})
        projected = params.project_multivector(mv)
        self.assertAlmostEqual(projected.get_blade("e1"), 2.0)
        self.assertAlmostEqual(projected.get_blade("e1e2"), 1.5)

    def test_hcst_forward_layer_norm_preservation(self):
        layer = HCSTTransformerLayer(self.dacc)

        # Initialize input multivector cochain on 1-cells
        v_e01 = Multivector.from_named_blades(PGA301, {"e1": 1.0, "e2": 0.5})
        v_e12 = Multivector.from_named_blades(PGA301, {"e2": 1.0, "e3": 0.5})
        v_e02 = Multivector.from_named_blades(PGA301, {"e1": 0.5, "e3": 1.0})

        input_cochain = {"E01": v_e01, "E12": v_e12, "E02": v_e02}
        init_norm = v_e01.norm()

        out_cochain = layer.forward(dim=1, cochain_k=input_cochain)

        self.assertIn("E01", out_cochain)
        self.assertIn("E12", out_cochain)
        self.assertIn("E02", out_cochain)

        # Norm preservation under rotor sandwich: ||R * A * ~R|| == ||A||
        out_norm = out_cochain["E01"].norm()
        self.assertAlmostEqual(init_norm, out_norm, places=6)

    def test_heat_kernel_betti_surrogate(self):
        layer = HCSTTransformerLayer(self.dacc)
        # Heat trace Tr(exp(-tau L_1)) > 0
        surrogate = layer.compute_heat_kernel_betti_surrogate(dim=1, tau_heat=1.0)
        self.assertGreater(surrogate, 0.0)

    def test_homology_preserving_topological_pooling(self):
        # In our triangle complex, face F0 has boundary {E01, E12, E02}.
        # Each edge is shared only by F0 (total coboundary degree = 1).
        # Thus (E01, F0) forms a strict free pair!
        pooling_engine = TopologicalPoolingEngine(self.dacc)
        free_pairs = pooling_engine.find_strict_free_pairs(dim=2)
        self.assertGreaterEqual(len(free_pairs), 1)

        face_cell, parent_cell = free_pairs[0]
        collapse_res = pooling_engine.execute_elementary_collapse(face_cell, parent_cell)

        self.assertTrue(collapse_res.betti_preserved)
        # In the pooled complex, face_cell and parent_cell have been retracted
        pooled_names = [c.name for c in collapse_res.pooled_dacc.get_cells(dim=parent_cell.dim)]
        self.assertNotIn(parent_cell.name, pooled_names)


if __name__ == "__main__":
    unittest.main()
