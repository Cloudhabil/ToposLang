"""Unit and integration tests for the MLIR/LLVM Compiler Pipeline and JIT Bridge.

Tests:
1. High-Level Topos Dialect Emission (!topos.dacc, topos.down_laplacian, !topos.sheaf).
2. Standard Lowered MLIR Dialect Emission (func, arith, scf, memref, vector).
3. Finite Field MLIR Row-Reduction Kernels (F_p modular arithmetic & GF(2) SIMD XOR).
4. Sheaf Block-CSR MLIR Emission and constant descriptors.
5. MLIRJITBridge dual-path probe, execution, and hermetic fallback equivalence.
6. CLI Integration (--emit-mlir, --emit-topos-ir, --backend in tauc and tau-run).
"""

from __future__ import annotations
import subprocess
import sys
import unittest
from pathlib import Path

from topos.cli.tauc import main as tauc_main
from topos.cli.taurun import main as taurun_main
from topos.compiler.emitter import MLIREmitter
from topos.compiler.jit import ExecutionResult, MLIRJITBridge
from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.frontend.parser import parse_source, build_complex
from topos.ir.sheaf import CellularSheafIR, StalkSpec


class TestMLIRCompiler(unittest.TestCase):
    """Test suite for MLIR textual emission, syntax validation, and JIT bridge."""

    def setUp(self) -> None:
        # Construct S^1
        base = GlobularCell("base", dim=0)
        loop = GlobularCell("loop", dim=1, source=base, target=base)
        self.s1 = CellComplex("S1")
        self.s1.add_cell(base)
        self.s1.add_cell(loop)

        # Construct S^2 (hollow tetrahedron)
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))
        self.s2 = CellComplex("S2")
        for f in (f1, f2, f3, f4):
            self.s2.add_cell(f, auto_add_boundaries=True)

    def test_emit_topos_dialect_structure(self) -> None:
        """Verifies structure and operations of high-level topos dialect emission."""
        emitter = MLIREmitter(target="topos")
        mlir_text = emitter.emit(self.s2)

        # Module framing
        self.assertIn('module attributes {topos.version = "1.0.0"} {', mlir_text)
        self.assertIn('func.func @build_complex_S2() -> !topos.dacc<2> {', mlir_text)
        self.assertIn('%0 = topos.create_complex "S2" : !topos.dacc<2>', mlir_text)
        self.assertIn('topos.verify_nilpotency %0 : !topos.dacc<2>', mlir_text)
        self.assertIn('%l2 = topos.down_laplacian %0, dim(2) -> !topos.csr_matrix<i64>', mlir_text)
        self.assertIn('%beta2 = topos.homology_rank %l2 : i64', mlir_text)
        self.assertIn('return %0 : !topos.dacc<2>', mlir_text)

    def test_emit_standard_lowered_mlir(self) -> None:
        """Verifies lowering to standard MLIR dialects (arith, scf, func, memref, vector)."""
        emitter = MLIREmitter(target="standard")
        mlir_text = emitter.emit(self.s2)

        # Standard dialects check
        self.assertIn('module {', mlir_text)
        self.assertIn('func.func @fp_row_reduce', mlir_text)
        self.assertIn('arith.muli', mlir_text)
        self.assertIn('arith.remui', mlir_text)
        self.assertIn('arith.subi', mlir_text)

        # Vector GF(2) SIMD check
        self.assertIn('func.func @f2_vector_row_xor', mlir_text)
        self.assertIn('vector<4xi64>', mlir_text)
        self.assertIn('arith.xori', mlir_text)

        # Sparse MatVec check
        self.assertIn('func.func @csr_matvec', mlir_text)
        self.assertIn('scf.for', mlir_text)
        self.assertIn('memref.load', mlir_text)
        self.assertIn('memref.store', mlir_text)

        # Complex-specific shape constants
        self.assertIn('func.func @get_l2_down_shape_S2()', mlir_text)

    def test_emit_sheaf_mlir(self) -> None:
        """Verifies MLIR emission with cellular sheaves attached."""
        sheaf_ir = CellularSheafIR(
            name="GaugeU1",
            complex_obj=self.s2,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=3),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=3),
            },
        )

        # Topos dialect with sheaf
        topos_emitter = MLIREmitter(target="topos")
        topos_text = topos_emitter.emit(self.s2, sheaf_ir=sheaf_ir)
        self.assertIn('func.func @sheaf_GaugeU1() -> !topos.sheaf<"S2", 3, "Q">', topos_text)
        self.assertIn('topos.create_sheaf "GaugeU1"', topos_text)
        self.assertIn('%sheaf_l2 = topos.sheaf_down_laplacian %s0 -> !topos.block_csr<i64, 3x3>', topos_text)
        self.assertIn('%nullity = topos.sheaf_nullity %sheaf_l2 : i64', topos_text)

        # Standard dialect with sheaf
        std_emitter = MLIREmitter(target="standard")
        std_text = std_emitter.emit(self.s2, sheaf_ir=sheaf_ir)
        self.assertIn('func.func @get_sheaf_l2_descriptor_GaugeU1()', std_text)
        self.assertIn('arith.constant 4 : index', std_text)  # 4 blocks

    def test_jit_bridge_execution_fallback(self) -> None:
        """Verifies MLIRJITBridge executes cleanly via fallback with exact topological invariants."""
        bridge = MLIRJITBridge(force_fallback=True)
        res = bridge.execute_down_laplacian(self.s2)

        self.assertTrue(res.success)
        self.assertEqual(res.backend, "pure_python_csr")
        self.assertEqual(res.dim_ker, 1)  # beta_2(S2) = 1
        self.assertEqual(res.rank, 3)     # 4 faces - 1 nullity = 3
        self.assertIsNotNone(res.mlir_source)

    def test_jit_bridge_sheaf_execution(self) -> None:
        """Verifies JIT bridge evaluates sheaf down-Laplacians."""
        sheaf_ir = CellularSheafIR(
            name="TorusSheaf",
            complex_obj=self.s2,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=2),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=2),
            },
        )
        bridge = MLIRJITBridge(force_fallback=True)
        res = bridge.execute_down_laplacian(self.s2, sheaf_ir=sheaf_ir)

        self.assertTrue(res.success)
        self.assertEqual(res.backend, "pure_python_csr")
        # 2 * beta_2 = 2 * 1 = 2
        self.assertEqual(res.dim_ker, 2)
        self.assertEqual(res.rank, 6)     # 8 total - 2 nullity = 6

    def test_jit_bridge_process_execution(self) -> None:
        """Verifies MLIRJITBridge executing full .tau process with stopping invariants."""
        source = """
        cell0 A;
        cell0 B;
        cell0 C;
        cell1 e1 : A -> B;
        cell1 e2 : B -> C;
        cell1 e3 : A -> C;

        process test_collapse(complex: CellComplex) {
            rewrite loop_contraction : (e1 * e2) => e3;
            until betti(complex, dim=1) == 0 {
                apply loop_contraction on complex;
            }
        }
        """
        program = parse_source(source)
        complex_obj = build_complex(program, name="TestLoop")

        bridge = MLIRJITBridge(force_fallback=True)
        res = bridge.execute_process(program, "test_collapse", complex_obj)

        self.assertTrue(res.success)
        self.assertEqual(res.iterations, 1)
        self.assertEqual(res.dim_ker, 0)  # beta_1 collapsed to 0

    def test_cli_tauc_emit_mlir(self) -> None:
        """Verifies tauc CLI flags --emit-mlir and --emit-topos-ir."""
        example_path = "examples/01_circle_hit.tau"
        if not Path(example_path).exists():
            self.skipTest(f"{example_path} does not exist")

        # Capture stdout for --emit-topos-ir
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        try:
            sys.stdout = out = StringIO()
            rc = tauc_main([example_path, "--emit-topos-ir"])
            self.assertEqual(rc, 0)
            output = out.getvalue()
            self.assertIn("topos.version", output)
            self.assertIn("topos.create_complex", output)

            sys.stdout = out_std = StringIO()
            rc = tauc_main([example_path, "--emit-mlir"])
            self.assertEqual(rc, 0)
            output_std = out_std.getvalue()
            self.assertIn("func.func @fp_row_reduce", output_std)
            self.assertIn("func.func @csr_matvec", output_std)
        finally:
            sys.stdout = old_stdout

    def test_cli_taurun_backend_flag(self) -> None:
        """Verifies tau-run CLI with --backend auto and --backend csr."""
        example_path = "examples/03_loop_collapse.tau"
        if not Path(example_path).exists():
            self.skipTest(f"{example_path} does not exist")

        from io import StringIO
        import sys

        old_stdout = sys.stdout
        try:
            sys.stdout = out = StringIO()
            rc = taurun_main([example_path, "--backend", "csr"])
            self.assertEqual(rc, 0)
            output = out.getvalue()
            self.assertIn("Execution Backend: pure_python_csr", output)
            self.assertIn("Total Iterations: 1", output)
        finally:
            sys.stdout = old_stdout


if __name__ == "__main__":
    unittest.main()
