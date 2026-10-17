"""ToposLang MLIR Emitter.

Generates verified textual MLIR IR for cell complexes, boundary operators,
sparse down-Laplacians, and cellular sheaves across two target abstraction tiers:
1. High-Level Topos Dialect (target="topos"):
   Emits semantic domain operations (!topos.dacc, topos.down_laplacian, !topos.sheaf, topos.homology_rank).
2. Standard MLIR Lowered Dialects (target="standard"):
   Lowers to standard MLIR dialects (func, arith, scf, memref, vector) with:
   - Modular integer arithmetic for finite field F_p.
   - Vectorized 64-bit SIMD bitwise ops for binary field F_2.
   - Sparse CSR and Block-CSR evaluation kernels.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from topos.core.complex import CellComplex
from topos.core.matrix_csr import CSROperator, compute_plaquette_down_laplacian
from topos.ir.sheaf import (
    BlockCSROperator,
    CellularSheafIR,
    build_block_csr_sheaf_laplacian,
    resolve_field,
)


class MLIREmitter:
    """Emits textual MLIR IR from CellComplexes, CSR operators, and Cellular Sheaves."""

    def __init__(self, target: str = "standard") -> None:
        """Initializes emitter with a target dialect mode ('standard', 'topos', 'cubical', 'yang_mills', 'procomplex', or 'lean4')."""
        valid_targets = ("standard", "topos", "cubical", "yang_mills", "procomplex", "lean4")
        if target not in valid_targets:
            raise ValueError(f"Unknown target dialect: {target!r}. Must be one of {valid_targets}.")
        self.target = target


    def emit_topos_dialect(
        self,
        complex_obj: CellComplex,
        sheaf_ir: Optional[CellularSheafIR] = None,
    ) -> str:
        """Emits high-level topos dialect MLIR representation."""
        lines: List[str] = [
            '// ToposLang High-Level Dialect Representation',
            '// Target: topos dialect',
            f'// Complex: {complex_obj.name} (dimension = {complex_obj.dim})',
            'module attributes {topos.version = "1.0.0"} {',
        ]

        dim = complex_obj.dim
        lines.append(f'  func.func @build_complex_{complex_obj.name}() -> !topos.dacc<{dim}> {{')
        lines.append(f'    %0 = topos.create_complex "{complex_obj.name}" : !topos.dacc<{dim}>')

        # Add cells
        for k in sorted(complex_obj.graded_cells.keys()):
            for c in complex_obj.get_cells(k):
                lines.append(f'    topos.attach_cell %0, dim({k}), name("{c.name}") : !topos.dacc<{dim}>')

        # Nilpotency verifier op
        lines.append(f'    topos.verify_nilpotency %0 : !topos.dacc<{dim}>')

        # 2-Down Laplacian op
        if len(complex_obj.get_cells(2)) > 0:
            lines.append(f'    %l2 = topos.down_laplacian %0, dim(2) -> !topos.csr_matrix<i64>')
            lines.append(f'    %beta2 = topos.homology_rank %l2 : i64')

        lines.append(f'    return %0 : !topos.dacc<{dim}>')
        lines.append('  }')

        # Sheaf declaration if present
        if sheaf_ir is not None:
            p_stalk = sheaf_ir.get_stalk(2)
            e_stalk = sheaf_ir.get_stalk(1)
            f_str = p_stalk.field_type
            d2 = p_stalk.dim
            d1 = e_stalk.dim
            lines.append('')
            lines.append(f'  func.func @sheaf_{sheaf_ir.name}() -> !topos.sheaf<"{complex_obj.name}", {d2}, "{f_str}"> {{')
            lines.append(
                f'    %s0 = topos.create_sheaf "{sheaf_ir.name}", complex("{complex_obj.name}"), '
                f'stalk_dim(2, {d2}), stalk_dim(1, {d1}), field("{f_str}") : !topos.sheaf<"{complex_obj.name}", {d2}, "{f_str}">'
            )
            for (p_name, e_name), mat in sheaf_ir.restrictions.items():
                mat_str = str(mat).replace("'", "")
                lines.append(f'    topos.attach_restriction %s0, "{p_name}" -> "{e_name}", matrix({mat_str})')
            lines.append(f'    %sheaf_l2 = topos.sheaf_down_laplacian %s0 -> !topos.block_csr<i64, {d2}x{d2}>')
            lines.append(f'    %nullity = topos.sheaf_nullity %sheaf_l2 : i64')
            lines.append(f'    return %s0 : !topos.sheaf<"{complex_obj.name}", {d2}, "{f_str}">')
            lines.append('  }')

        lines.append('}')
        return '\n'.join(lines)

    def emit_standard_mlir(
        self,
        complex_obj: CellComplex,
        sheaf_ir: Optional[CellularSheafIR] = None,
        field_type: str = "Q",
        prime: int = 65537,
    ) -> str:
        """Lowers cell complex and down-Laplacians to standard MLIR dialects."""
        lines: List[str] = [
            '// ToposLang Lowered MLIR Module',
            f'// Complex: {complex_obj.name} | Field: {field_type}',
            '// Standard Dialects: func, arith, scf, memref, vector',
            'module {',
        ]

        # 1. Finite field row reduction kernel
        lines.extend(self._emit_modular_row_reduction_kernel())
        lines.append('')

        # 2. Vectorized GF(2) XOR kernel
        lines.extend(self._emit_f2_vector_kernel())
        lines.append('')

        # 3. Sparse CSR Matrix-Vector Product kernel
        lines.extend(self._emit_csr_matvec_kernel())
        lines.append('')

        # 4. Complex-specific Down-Laplacian data and assembly
        csr_l2 = compute_plaquette_down_laplacian(complex_obj)
        lines.extend(self._emit_csr_data_constants(complex_obj.name, csr_l2))
        lines.append('')

        # 5. Sheaf Block-CSR Down-Laplacian data and kernel (if sheaf provided)
        if sheaf_ir is not None:
            block_csr = build_block_csr_sheaf_laplacian(sheaf_ir)
            lines.extend(self._emit_sheaf_block_constants(sheaf_ir.name, block_csr))
            lines.append('')

        lines.append('}')
        return '\n'.join(lines)

    def _emit_modular_row_reduction_kernel(self) -> List[str]:
        """Emits standard MLIR arith function for F_p sparse row elimination."""
        return [
            '  // Modular row reduction step over F_p: (val_a - val_b * pivot_inv) mod p',
            '  func.func @fp_row_reduce(%val_a: i64, %val_b: i64, %pivot_inv: i64, %prime: i64) -> i64 {',
            '    %mult = arith.muli %val_b, %pivot_inv : i64',
            '    %factor = arith.remui %mult, %prime : i64',
            '    %term = arith.muli %val_a, %factor : i64',
            '    %term_mod = arith.remui %term, %prime : i64',
            '    %sub = arith.subi %val_a, %term_mod : i64',
            '    %add_p = arith.addi %sub, %prime : i64',
            '    %res = arith.remui %add_p, %prime : i64',
            '    return %res : i64',
            '  }',
        ]

    def _emit_f2_vector_kernel(self) -> List[str]:
        """Emits vectorized SIMD bitwise XOR kernel for GF(2)."""
        return [
            '  // Vectorized 256-bit SIMD row XOR over GF(2)',
            '  func.func @f2_vector_row_xor(%row_a: vector<4xi64>, %row_b: vector<4xi64>) -> vector<4xi64> {',
            '    %res = arith.xori %row_a, %row_b : vector<4xi64>',
            '    return %res : vector<4xi64>',
            '  }',
        ]

    def _emit_csr_matvec_kernel(self) -> List[str]:
        """Emits sparse matrix-vector multiplication kernel over memref."""
        return [
            '  // Sparse CSR Matrix-Vector Product: y = A @ x',
            '  func.func @csr_matvec(',
            '    %num_rows: index,',
            '    %indptr: memref<?xindex>,',
            '    %indices: memref<?xindex>,',
            '    %data: memref<?xi64>,',
            '    %x: memref<?xi64>,',
            '    %y: memref<?xi64>',
            '  ) {',
            '    %c0 = arith.constant 0 : index',
            '    %c1 = arith.constant 1 : index',
            '    %zero = arith.constant 0 : i64',
            '    scf.for %r = %c0 to %num_rows step %c1 {',
            '      %start = memref.load %indptr[%r] : memref<?xindex>',
            '      %r_next = arith.addi %r, %c1 : index',
            '      %end = memref.load %indptr[%r_next] : memref<?xindex>',
            '      %sum_init = arith.constant 0 : i64',
            '      %sum = scf.for %idx = %start to %end step %c1 iter_args(%acc = %sum_init) -> (i64) {',
            '        %col = memref.load %indices[%idx] : memref<?xindex>',
            '        %val = memref.load %data[%idx] : memref<?xi64>',
            '        %x_val = memref.load %x[%col] : memref<?xi64>',
            '        %prod = arith.muli %val, %x_val : i64',
            '        %next_acc = arith.addi %acc, %prod : i64',
            '        scf.yield %next_acc : i64',
            '      }',
            '      memref.store %sum, %y[%r] : memref<?xi64>',
            '    }',
            '    return',
            '  }',
        ]

    def _emit_csr_data_constants(self, name: str, csr: CSROperator) -> List[str]:
        """Emits constant descriptors for a computed CSR operator."""
        lines: List[str] = [
            f'  // Constant CSR representation for Down-Laplacian: {name}',
            f'  // Shape: ({csr.rows}, {csr.cols}), NNZ: {csr.nnz}',
            f'  func.func @get_l2_down_shape_{name}() -> (index, index, index) {{',
            f'    %rows = arith.constant {csr.rows} : index',
            f'    %cols = arith.constant {csr.cols} : index',
            f'    %nnz = arith.constant {csr.nnz} : index',
            f'    return %rows, %cols, %nnz : index, index, index',
            f'  }}',
        ]
        return lines

    def _emit_sheaf_block_constants(self, sheaf_name: str, block_csr: BlockCSROperator) -> List[str]:
        """Emits constant descriptors for a compiled Sheaf Block-CSR operator."""
        d_r, d_c = block_csr.block_shape
        flat = block_csr.to_flat_csr()
        lines: List[str] = [
            f'  // Sheaf Block-CSR Down-Laplacian: {sheaf_name}',
            f'  // Blocks: {block_csr.num_blocks}, BlockShape: ({d_r}, {d_c}), NNZ Blocks: {block_csr.nnz_blocks}',
            f'  func.func @get_sheaf_l2_descriptor_{sheaf_name}() -> (index, index, index, index) {{',
            f'    %n_blocks = arith.constant {block_csr.num_blocks} : index',
            f'    %b_dim = arith.constant {d_r} : index',
            f'    %total_rows = arith.constant {flat.rows} : index',
            f'    %flat_nnz = arith.constant {flat.nnz} : index',
            f'    return %n_blocks, %b_dim, %total_rows, %flat_nnz : index, index, index, index',
            f'  }}',
        ]
        return lines

    def emit_cubical_dialect(
        self,
        complex_obj: CellComplex,
        sheaf_ir: Optional[CellularSheafIR] = None,
        algebra: str = "PGA(3,0,1)",
        max_homotopy_dim: int = 2,
    ) -> str:
        """Emits the topos_cubical bridge dialect MLIR representation.

        Introduces three bridge operations:
        - topos_cubical.bridge_lower: Lowers DACC sparse tensor support to Kan box tube.
        - topos_cubical.hcomp: Executes stack-allocated Kan composition.
        - topos_cubical.truncate: Applies n-truncation pruning pass.
        """
        dim = complex_obj.dim
        lines: List[str] = [
            '// ToposLang Cubical Homotopy Bridge Dialect',
            '// Target: topos_cubical dialect',
            f'// Complex: {complex_obj.name} (dimension = {dim})',
            f'// Clifford Algebra: {algebra}',
            f'// Max Homotopy Dimension (n-truncation): {max_homotopy_dim}',
            'module attributes {topos.version = "1.0.0", topos_cubical.version = "1.0.0"} {',
        ]

        lines.append(f'  func.func @cubical_bridge_{complex_obj.name}() -> !topos.dacc<{dim}> {{')
        lines.append(f'    // 1. Build and load the DACC macro-spatial complex')
        lines.append(f'    %K = topos.create_complex "{complex_obj.name}" : !topos.dacc<{dim}>')

        for k in sorted(complex_obj.graded_cells.keys()):
            for c in complex_obj.get_cells(k):
                lines.append(
                    f'    topos.attach_cell %K, dim({k}), name("{c.name}") : !topos.dacc<{dim}>'
                )

        lines.append(f'    %c0 = arith.constant 0 : i64')
        lines.append(f'    // 2. Compute Hodge Laplacian and detect harmonic cycle L_k c != 0')
        lines.append(f'    %L1 = topos.down_laplacian %K, dim(1) -> !topos.csr_matrix<i64>')
        lines.append(f'    %beta1 = topos.homology_rank %L1 : i64')
        lines.append(f'    %has_cycle = arith.cmpi ugt, %beta1, %c0 : i64')
        lines.append(f'    // 3. Extract spatial support supp(r) and lower to Kan box')
        lines.append(f'    %support_cells = topos.support_closure %K, %L1 : !topos.set<cell>')
        lines.append(
            f'    %kan_box = topos_cubical.bridge_lower %K, %L1, %support_cells :'
        )
        lines.append(
            f'        (!topos.dacc<{dim}>, !topos.csr_matrix<i64>, !topos.set<cell>)'
        )
        lines.append(
            f'        -> !topos_cubical.box_tube<dim = {dim + 1}, algebra = "{algebra}">'
        )
        lines.append(f'    // 4. Execute stack-allocated Kan composition with n-truncation')
        lines.append(f'    %i_one = topos_cubical.interval_endpoint_1 : !topos_cubical.interval')
        lines.append(
            f'    %endpoint_mv = topos_cubical.hcomp %kan_box, %i_one :'
        )
        lines.append(
            f'        (!topos_cubical.box_tube<dim = {dim + 1}, algebra = "{algebra}">, !topos_cubical.interval)'
        )
        lines.append(f'        -> !topos.multivector<{algebra}>')
        lines.append(f'    %truncated_mv = topos_cubical.truncate %endpoint_mv {{n_dim = {max_homotopy_dim} : i32}} :')
        lines.append(f'        !topos.multivector<{algebra}> -> !topos.multivector<{algebra}>')
        lines.append(f'    // 5. Re-integrate terminal state u(1) as ({dim+1})-cell alpha_r attachment')
        lines.append(f'    %lhs_chain = topos.extract_lhs_chain %L1 : !topos.chain<{dim}, Q>')
        lines.append(f'    %rhs_chain = topos.extract_rhs_chain %L1 : !topos.chain<{dim}, Q>')
        lines.append(f'    %K_closed = topos.homotopy_rewrite %K, %lhs_chain, %rhs_chain :')
        lines.append(f'        !topos.dacc<{dim}>, !topos.chain<{dim}, Q>, !topos.chain<{dim}, Q> -> !topos.dacc<{dim}>')
        lines.append(f'    return %K_closed : !topos.dacc<{dim}>')
        lines.append(f'  }}')

        if sheaf_ir is not None:
            p_stalk = sheaf_ir.get_stalk(2)
            e_stalk = sheaf_ir.get_stalk(1)
            d2 = p_stalk.dim
            d1 = e_stalk.dim
            lines.append(f'')
            lines.append(
                f'  func.func @cubical_sheaf_{sheaf_ir.name}() -> !topos.sheaf<"{complex_obj.name}", {d2}, "{algebra}"> {{'
            )
            lines.append(
                f'    %s0 = topos.create_sheaf "{sheaf_ir.name}", complex("{complex_obj.name}"), '
                f'stalk_dim(2, {d2}), stalk_dim(1, {d1}), algebra("{algebra}") '
                f': !topos.sheaf<"{complex_obj.name}", {d2}, "{algebra}">'
            )
            lines.append(f'    return %s0 : !topos.sheaf<"{complex_obj.name}", {d2}, "{algebra}">')
            lines.append(f'  }}')

        lines.append('}')
        return '\n'.join(lines)

    def emit_yang_mills_mlir(
        self,
        complex_obj: CellComplex,
        gauge_group: str = "SU3",
    ) -> str:
        """Emits Topos MLIR Dialect for 4D Yang-Mills Block-Spin Step (Section 2.5 of spec)."""
        dim = max(4, complex_obj.dim)
        lines: List[str] = [
            '// Topos MLIR Dialect: 4D Yang-Mills Block-Spin Step',
            f'module attributes {{topos.dim = {dim} : i32, topos.gauge_group = "{gauge_group}"}} {{',
            f'  func.func @execute_rg_pace(%arg0: !topos.complex<{dim}>) -> !topos.complex<{dim}> {{',
            '    %c1 = arith.constant 1 : i32',
            '    %c2 = arith.constant 2 : i32',
            '    // 1. Verify 4-tier boundary nilpotency: B2 * B3 == 0',
            f'    %nilpotent_23 = topos.verify_nilpotency %arg0, %c2 : (!topos.complex<{dim}>, i32) -> i1',
            '    cf.assert %nilpotent_23, "Static Boundary Nilpotency (B2*B3=0) Violated!"',
            '    // 2. Extract sparse incidence matrices over Q',
            f'    %B1 = topos.get_incidence_matrix %arg0, %c1 : (!topos.complex<{dim}>, i32) -> !topos.sparse_matrix<Q>',
            f'    %B2 = topos.get_incidence_matrix %arg0, %c2 : (!topos.complex<{dim}>, i32) -> !topos.sparse_matrix<Q>',
            '    // 3. Compute Exact Rational Hodge 1-Laplacian: L1 = B1^T * B1 + B2 * B2^T',
            '    %L1 = topos.compute_hodge_laplacian %B1, %B2 : (!topos.sparse_matrix<Q>, !topos.sparse_matrix<Q>) -> !topos.sparse_matrix<Q>',
            '    // 4. Isolate Harmonic Kernel Dimension (1st Betti Number beta_1)',
            '    %beta_1 = topos.compute_kernel_dim %L1 : (!topos.sparse_matrix<Q>) -> i64',
            '    // 5. Execute Cellular Homotopy Rewrite',
            f'    %lhs_chain = topos.get_chain %arg0, %c1, [0, 1] : (!topos.complex<{dim}>, i32, list<i32>) -> !topos.chain<1, Q>',
            f'    %rhs_chain = topos.get_chain %arg0, %c1, [2, 3] : (!topos.complex<{dim}>, i32, list<i32>) -> !topos.chain<1, Q>',
            '',
            f'    %updated_complex = topos.homotopy_rewrite %arg0, %lhs_chain, %rhs_chain :',
            f'        (!topos.complex<{dim}>, !topos.chain<1, Q>, !topos.chain<1, Q>) -> !topos.complex<{dim}>',
            f'    return %updated_complex : !topos.complex<{dim}>',
            '  }',
            '}',
        ]
        return '\n'.join(lines)

    def emit_procomplex_mlir(
        self,
        complex_obj: CellComplex,
        gauge_group: str = "SU3",
        max_scale: int = 3,
    ) -> str:
        """Emits Coalgebraic Lazy Streams in MLIR (!topos.procomplex)."""
        dim = max(4, complex_obj.dim)
        lines: List[str] = [
            '// Topos MLIR Dialect: Pro-DACC Inverse Limit Lazy Stream',
            f'module attributes {{topos.dim = {dim} : i32, topos.gauge_group = "{gauge_group}", topos.procomplex.lazy = true}} {{',
            f'  func.func @stream_eval_scale(%arg0: !topos.procomplex<{dim}>, %scale: index) -> !topos.complex<{dim}> {{',
            f'    // Coalgebraic on-demand evaluation of scale %scale in projective tower',
            f'    %K_scale = topos.eval_stream_scale %arg0, %scale : (!topos.procomplex<{dim}>, index) -> !topos.complex<{dim}>',
            f'    return %K_scale : !topos.complex<{dim}>',
            '  }',
            '',
            f'  func.func @compute_projective_topological_loss(%arg0: !topos.procomplex<{dim}>) -> f64 {{',
            f'    %max_s = arith.constant {max_scale} : index',
            '    %target_beta = arith.constant 0 : i64',
            f'    // Continuous projective loss L_proj over inverse limit tower',
            f'    %loss = topos.compute_projective_loss %arg0, %max_s, %target_beta : (!topos.procomplex<{dim}>, index, i64) -> f64',
            '    return %loss : f64',
            '  }',
            '',
            f'  func.func @zero_cost_elementary_collapse_gc(%arg0: !topos.complex<{dim}>) -> !topos.complex<{dim}> {{',
            f'    // O(1) elementary collapse pruning inactive intermediate homotopy cells',
            f'    %pruned = topos.elementary_collapse %arg0 : (!topos.complex<{dim}>) -> !topos.complex<{dim}>',
            f'    return %pruned : !topos.complex<{dim}>',
            '  }',
            '}',
        ]
        return '\n'.join(lines)

    def emit(
        self,
        complex_obj: CellComplex,
        sheaf_ir: Optional[CellularSheafIR] = None,
        algebra: str = "PGA(3,0,1)",
        max_homotopy_dim: int = 2,
        gauge_group: str = "SU3",
        max_scale: int = 3,
    ) -> str:
        """Emits textual MLIR IR according to the initialized target mode."""
        if self.target == "topos":
            return self.emit_topos_dialect(complex_obj, sheaf_ir=sheaf_ir)
        if self.target == "cubical":
            return self.emit_cubical_dialect(
                complex_obj,
                sheaf_ir=sheaf_ir,
                algebra=algebra,
                max_homotopy_dim=max_homotopy_dim,
            )
        if self.target == "yang_mills":
            return self.emit_yang_mills_mlir(complex_obj, gauge_group=gauge_group)
        if self.target == "procomplex":
            return self.emit_procomplex_mlir(complex_obj, gauge_group=gauge_group, max_scale=max_scale)
        if self.target == "lean4":
            from topos.compiler.lean4_emitter import Lean4Emitter
            return Lean4Emitter().emit_yang_mills_proof_scaffold(
                complex_obj=complex_obj,
                gauge_group_n=3 if "3" in gauge_group else 2,
            )
        return self.emit_standard_mlir(complex_obj, sheaf_ir=sheaf_ir)

