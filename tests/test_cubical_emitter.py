"""Tests for the topos_cubical MLIR dialect emitter and --engine tri-state flag."""

import unittest

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.compiler.emitter import MLIREmitter
from topos.runtime.engine import RewriteEngine, VALID_ENGINE_MODES


def _make_annulus_complex() -> CellComplex:
    """Build a minimal S¹-like complex: 1 vertex, 1 loop edge — β₁=1."""
    K = CellComplex("annulus")
    base = GlobularCell("base", dim=0)
    loop = GlobularCell("loop", dim=1, source=base, target=base)
    K.add_cell(base)
    K.add_cell(loop)
    return K


class TestCubicalDialectEmitter(unittest.TestCase):

    def setUp(self):
        self.K = _make_annulus_complex()

    def test_emit_cubical_returns_string(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        self.assertIsInstance(ir, str)
        self.assertGreater(len(ir), 50)

    def test_emit_cubical_contains_module(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        self.assertIn("module attributes", ir)

    def test_emit_cubical_contains_bridge_lower(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        self.assertIn("topos_cubical.bridge_lower", ir)

    def test_emit_cubical_contains_hcomp(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        self.assertIn("topos_cubical.hcomp", ir)

    def test_emit_cubical_contains_truncate(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        self.assertIn("topos_cubical.truncate", ir)

    def test_emit_cubical_algebra_annotation(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K, algebra="CGA(4,1,0)")
        self.assertIn("CGA(4,1,0)", ir)

    def test_emit_cubical_attaches_all_cells(self):
        emitter = MLIREmitter(target="cubical")
        ir = emitter.emit(self.K)
        for name in ["base", "loop"]:
            self.assertIn(name, ir)

    def test_emit_topos_still_works(self):
        emitter = MLIREmitter(target="topos")
        ir = emitter.emit(self.K)
        self.assertIn("topos.create_complex", ir)

    def test_emit_standard_still_works(self):
        emitter = MLIREmitter(target="standard")
        ir = emitter.emit(self.K)
        self.assertIn("func.func", ir)


class TestEngineModeFlag(unittest.TestCase):

    def test_default_engine_mode_is_auto(self):
        engine = RewriteEngine()
        self.assertEqual(engine.engine_mode, "auto")

    def test_dacc_only_mode(self):
        engine = RewriteEngine(engine_mode="dacc-only")
        self.assertEqual(engine.engine_mode, "dacc-only")

    def test_cubical_forced_mode(self):
        engine = RewriteEngine(engine_mode="cubical-forced")
        self.assertEqual(engine.engine_mode, "cubical-forced")

    def test_invalid_engine_mode_raises(self):
        with self.assertRaises(ValueError):
            RewriteEngine(engine_mode="invalid-mode")

    def test_valid_engine_modes_constant(self):
        self.assertIn("auto", VALID_ENGINE_MODES)
        self.assertIn("dacc-only", VALID_ENGINE_MODES)
        self.assertIn("cubical-forced", VALID_ENGINE_MODES)


if __name__ == "__main__":
    unittest.main()
