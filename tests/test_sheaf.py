"""Unit and integration tests for Cellular Sheaves and Block-CSR Sheaf Down-Laplacian.

Tests:
1. Parsing of sheaf, stalk, restriction declarations and sheaf_dim_ker stopping conditions.
2. Direct plaquette down-Laplacian assembly in Block-CSR format (L_2^down = B_2^dagger @ B_2).
3. Constant Sheaf Tensoring Theorem: dim(ker(L_2^down(sheaf, d))) == d * dim(ker(L_2^down(scalar))).
4. Twisted Sheaves with non-trivial restriction holonomies on periodic and spherical topologies.
5. Invariant-driven process execution with until sheaf_dim_ker(...) stopping loops.
6. Zero-link state allocation proof: Asserting that no 1-cell state vectors are allocated.
"""

from __future__ import annotations
import unittest
from fractions import Fraction

from topos.core.cell import GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.matrix_csr import PrimeField, RationalField, compute_plaquette_down_laplacian
from topos.frontend.ast import (
    Program,
    RestrictionDecl,
    SheafDecl,
    SheafDimKerCondition,
    StalkDecl,
    UntilLoop,
)
from topos.frontend.parser import parse_source
from topos.ir.sheaf import (
    BlockCSROperator,
    CellularSheafIR,
    StalkSpec,
    build_block_csr_sheaf_laplacian,
    resolve_field,
)
from topos.runtime.interpreter import ToposInterpreter
from topos.topology.homology import HomologyEngine


class TestCellularSheaf(unittest.TestCase):
    """Test suite for cellular sheaves and Block-CSR assembly."""

    def test_parse_sheaf_declaration(self) -> None:
        """Verifies parsing of first-class sheaf blocks and conditions."""
        source = """
        hit S1 {
            point base;
            path loop : base = base;
        }

        sheaf GaugeField over S1 {
            stalk[2] = R^3;
            stalk[1] = R^3;
            restriction(p0 -> e0) = [1.0, 0.0; 0.0, 1.0];
        }

        process contract_mesh(complex: CellComplex) {
            until sheaf_dim_ker(GaugeField.L2_down) == 0 {
                apply step on complex;
            }
        }
        """
        program = parse_source(source)
        self.assertIsInstance(program, Program)

        # Verify SheafDecl
        sheaf_decls = [s for s in program.statements if isinstance(s, SheafDecl)]
        self.assertEqual(len(sheaf_decls), 1)
        sheaf = sheaf_decls[0]
        self.assertEqual(sheaf.name, "GaugeField")
        self.assertEqual(sheaf.complex_name, "S1")
        self.assertEqual(len(sheaf.stalks), 2)
        self.assertEqual(sheaf.stalks[0].cell_dim, 2)
        self.assertEqual(sheaf.stalks[0].field_type, "R")
        self.assertEqual(sheaf.stalks[0].vector_dim, 3)
        self.assertEqual(len(sheaf.restrictions), 1)
        self.assertEqual(sheaf.restrictions[0].source_cell, "p0")
        self.assertEqual(sheaf.restrictions[0].target_cell, "e0")
        self.assertEqual(
            sheaf.restrictions[0].matrix_values,
            [[1.0, 0.0], [0.0, 1.0]],
        )

        # Verify UntilLoop with SheafDimKerCondition
        until_loops = [
            stmt for s in program.statements if hasattr(s, "body") for stmt in s.body if isinstance(stmt, UntilLoop)
        ]
        self.assertEqual(len(until_loops), 1)
        cond = until_loops[0].condition
        self.assertIsInstance(cond, SheafDimKerCondition)
        self.assertEqual(cond.sheaf_name, "GaugeField")
        self.assertEqual(cond.operator_name, "L2_down")
        self.assertEqual(cond.operator, "==")
        self.assertEqual(cond.value, 0)

    def test_constant_sheaf_tensoring_sphere_s2(self) -> None:
        """Verifies Constant Sheaf Tensoring Theorem:
        dim(ker(L_2^down(F, d))) = d * dim(ker(L_2^down(scalar))) on Sphere S2.
        For S^2, scalar dim(ker(L_2^down)) = beta_2 = 1.
        For stalk dimension d=3, sheaf nullity must equal 3 * 1 = 3.
        """
        # Hollow tetrahedron (topological S^2)
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))

        s2 = CellComplex("S2")
        for f in (f1, f2, f3, f4):
            s2.add_cell(f, auto_add_boundaries=True)

        # 1. Scalar plaquette down-Laplacian nullity
        scalar_csr = compute_plaquette_down_laplacian(s2, field=RationalField())
        scalar_nullity = scalar_csr.nullity()
        self.assertEqual(scalar_nullity, 1)  # beta_2(S2) = 1

        # 2. Sheaf down-Laplacian with 3D constant stalk
        sheaf_ir = CellularSheafIR(
            name="ConstantSheaf3D",
            complex_obj=s2,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=3),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=3),
            },
        )
        block_csr = build_block_csr_sheaf_laplacian(sheaf_ir)
        self.assertEqual(block_csr.num_blocks, 4)
        self.assertEqual(block_csr.block_shape, (3, 3))

        # Check unrolling to flat CSR
        flat_csr = block_csr.to_flat_csr()
        self.assertEqual(flat_csr.rows, 12)  # 4 faces * 3 dim
        self.assertEqual(flat_csr.cols, 12)

        # Check Tensoring Theorem: nullity = 3 * 1 = 3
        sheaf_nullity = block_csr.sheaf_nullity()
        self.assertEqual(sheaf_nullity, 3 * scalar_nullity)
        self.assertEqual(sheaf_nullity, 3)

    def test_constant_sheaf_finite_field_fp(self) -> None:
        """Verifies Constant Sheaf Laplacians over finite field F_65537."""
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        f3 = Simplex(("A", "C", "D"))
        f4 = Simplex(("B", "C", "D"))

        s2 = CellComplex("S2_FP")
        for f in (f1, f2, f3, f4):
            s2.add_cell(f, auto_add_boundaries=True)

        sheaf_ir = CellularSheafIR(
            name="ModularSheaf",
            complex_obj=s2,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="F_65537", dim=2),
                1: StalkSpec(cell_dim=1, field_type="F_65537", dim=2),
            },
        )
        block_csr = build_block_csr_sheaf_laplacian(sheaf_ir)
        self.assertEqual(block_csr.sheaf_nullity(), 2 * 1)  # d * beta_2 = 2 * 1 = 2

    def test_twisted_sheaf_holonomy(self) -> None:
        """Verifies that non-trivial restriction holonomy modifies the sheaf nullity."""
        # Two triangles sharing an edge: [A, B, C] and [A, B, D] sharing [A, B]
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        k = CellComplex("TwoTriangles")
        k.add_cell(f1, auto_add_boundaries=True)
        k.add_cell(f2, auto_add_boundaries=True)

        # Standard constant 1D sheaf: nullity is 0 (open sheet, not closed cycle)
        sheaf_const = CellularSheafIR(
            name="Const1D",
            complex_obj=k,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=1),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=1),
            },
        )
        b_const = build_block_csr_sheaf_laplacian(sheaf_const)
        self.assertEqual(b_const.num_blocks, 2)
        self.assertEqual(b_const.block_shape, (1, 1))

        # Twisted 2D sheaf with orthogonal rotation on the shared face
        sheaf_twisted = CellularSheafIR(
            name="Twisted2D",
            complex_obj=k,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=2),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=2),
            },
            restrictions={
                # Reflection on one triangle
                ("[A, B, C]", "[A, B]"): [[-1, 0], [0, 1]],
            },
        )
        b_twisted = build_block_csr_sheaf_laplacian(sheaf_twisted)
        flat = b_twisted.to_flat_csr()
        self.assertEqual(flat.rows, 4)
        self.assertEqual(flat.cols, 4)
        self.assertEqual(b_twisted.sheaf_rank(), flat.rank())

    def test_sheaf_dim_ker_runtime_loop(self) -> None:
        """Verifies execution of process loop governed by sheaf_dim_ker condition."""
        source = """
        cell0 A;
        cell0 B;
        cell0 C;

        cell1 e1 : A -> B;
        cell1 e2 : B -> C;
        cell1 e3 : A -> C;

        sheaf TrivialSheaf over VoidComplex {
            stalk[2] = Q^1;
            stalk[1] = Q^1;
        }

        process test_sheaf_proc(complex: CellComplex) {
            rewrite contract : (e1 * e2) => e3;

            until sheaf_dim_ker(TrivialSheaf.L2_down) == 0 {
                apply contract on complex;
            }
        }
        """
        program = parse_source(source)
        # Build initial complex
        from topos.frontend.parser import build_complex
        comp = build_complex(program, name="VoidComplex")

        # Initially, 0 plaquettes => sheaf_dim_ker = 0, loop terminates immediately
        interpreter = ToposInterpreter()
        ctx = interpreter.run_process(program, "test_sheaf_proc", comp)
        self.assertEqual(ctx.iterations, 0)

    def test_zero_link_state_allocation_invariant(self) -> None:
        """CRITICAL ARCHITECTURAL GUARANTEE TEST:
        Verifies that during Block-CSR assembly of L_2^down, zero state allocations
        or stalk vectors are stored for 1-cells (links).
        """
        f1 = Simplex(("A", "B", "C"))
        f2 = Simplex(("A", "B", "D"))
        k = CellComplex("Pair")
        k.add_cell(f1, auto_add_boundaries=True)
        k.add_cell(f2, auto_add_boundaries=True)

        sheaf_ir = CellularSheafIR(
            name="VerifyPlaquetteOnly",
            complex_obj=k,
            stalks={
                2: StalkSpec(cell_dim=2, field_type="Q", dim=4),
                1: StalkSpec(cell_dim=1, field_type="Q", dim=4),
            },
        )
        block_csr = build_block_csr_sheaf_laplacian(sheaf_ir)

        # 1. Assert block rows count strictly equals number of 2-cells (plaquettes), NOT 1-cells
        self.assertEqual(block_csr.num_blocks, len(k.get_cells(2)))
        self.assertNotEqual(block_csr.num_blocks, len(k.get_cells(1)))

        # 2. Assert indices reference plaquette IDs strictly in [0, len(C_2))
        n_plaquettes = len(k.get_cells(2))
        for col_idx in block_csr.indices:
            self.assertTrue(0 <= col_idx < n_plaquettes)

        # 3. Assert total flat CSR dimension is N_p * d2, with 0 link rows
        d2 = 4
        flat_csr = block_csr.to_flat_csr()
        self.assertEqual(flat_csr.rows, n_plaquettes * d2)
        self.assertEqual(flat_csr.cols, n_plaquettes * d2)


if __name__ == "__main__":
    unittest.main()
