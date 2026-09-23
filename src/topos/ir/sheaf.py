"""Cellular Sheaf Intermediate Representation (Sheaf-IR) and Block-CSR Down-Laplacian.

Implements:
1. StalkSpec and RestrictionMap: Local algebraic data over n-cells.
2. CellularSheafIR: Global cellular sheaf structure over a CellComplex.
3. BlockCSROperator: Block Compressed Sparse Row representation of the direct plaquette
   sheaf down-Laplacian:
       (L_2^down)_{p_i, p_j} = sum_{e in boundary(p_i) cap boundary(p_j)} s_i * s_j * (rho_{p_i -> e}^T @ rho_{p_j -> e})
   completely bypassing 1-cell (link) state allocations.
4. Unrolling from Block-CSR to flat CSROperator over any Field K (F_p, F_2, Q).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from topos.core.cell import Cell
from topos.core.complex import CellComplex
from topos.core.matrix_csr import (
    BinaryField,
    CSROperator,
    Field,
    PrimeField,
    RationalField,
)
from topos.frontend.ast import RestrictionDecl, SheafDecl, StalkDecl


def resolve_field(field_type_str: str) -> Field:
    """Resolves field type identifier string into a Field trait instance."""
    s = field_type_str.strip()
    if s == "Q":
        return RationalField()
    elif s == "F_2":
        return BinaryField()
    elif s.startswith("F_"):
        prime = int(s[2:])
        if prime == 2:
            return BinaryField()
        return PrimeField(prime)
    elif s in ("R", "C"):
        # For pure-Python execution, rational arithmetic with Fraction guarantees exact zero-roundoff
        return RationalField()
    else:
        # Default fallback
        return RationalField()


@dataclass
class StalkSpec:
    """Specification of a stalk vector space attached to k-cells."""
    cell_dim: int
    field_type: str = "R"
    dim: int = 1


class CellularSheafIR:
    """Intermediate Representation of a Cellular Sheaf over a CellComplex."""

    def __init__(
        self,
        name: str,
        complex_obj: CellComplex,
        stalks: Optional[Dict[int, StalkSpec]] = None,
        restrictions: Optional[Dict[Tuple[str, str], List[List[Any]]]] = None,
    ) -> None:
        self.name = name
        self.complex = complex_obj
        self.stalks: Dict[int, StalkSpec] = stalks or {}
        # restrictions[(p_name, e_name)] = matrix of shape (fiber_dim, stalk_dim)
        self.restrictions: Dict[Tuple[str, str], List[List[Any]]] = restrictions or {}

    def get_stalk(self, cell_dim: int) -> StalkSpec:
        """Retrieves stalk specification for a given cell dimension."""
        if cell_dim in self.stalks:
            return self.stalks[cell_dim]
        # Default stalk: 1-dimensional
        return StalkSpec(cell_dim=cell_dim, field_type="R", dim=1)

    def get_restriction(self, source_cell: str, target_cell: str) -> List[List[Any]]:
        """Retrieves restriction matrix rho: source -> target.
        
        Defaults to Identity matrix if dimensions match, or canonical projection/embedding.
        """
        pair = (source_cell, target_cell)
        if pair in self.restrictions:
            return self.restrictions[pair]

        # Canonical default: Identity if dimensions match
        src_cell = self.complex.get_cell(source_cell)
        tgt_cell = self.complex.get_cell(target_cell)
        src_dim = self.get_stalk(src_cell.dim).dim
        tgt_dim = self.get_stalk(tgt_cell.dim).dim

        field_obj = resolve_field(self.get_stalk(src_cell.dim).field_type)
        identity = [
            [field_obj.one() if r == c else field_obj.zero() for c in range(src_dim)]
            for r in range(tgt_dim)
        ]
        return identity

    @classmethod
    def from_ast(cls, sheaf_decl: SheafDecl, complex_obj: CellComplex) -> CellularSheafIR:
        """Builds a CellularSheafIR from AST declarations."""
        stalks: Dict[int, StalkSpec] = {}
        for s in sheaf_decl.stalks:
            stalks[s.cell_dim] = StalkSpec(
                cell_dim=s.cell_dim,
                field_type=s.field_type,
                dim=s.vector_dim,
            )

        restrictions: Dict[Tuple[str, str], List[List[Any]]] = {}
        for r in sheaf_decl.restrictions:
            restrictions[(r.source_cell, r.target_cell)] = r.matrix_values

        return cls(
            name=sheaf_decl.name,
            complex_obj=complex_obj,
            stalks=stalks,
            restrictions=restrictions,
        )


class BlockCSROperator:
    """Block Compressed Sparse Row representation of the Sheaf Down-Laplacian L_2^down."""

    def __init__(
        self,
        num_blocks: int,
        block_shape: Tuple[int, int],
        indptr: Sequence[int],
        indices: Sequence[int],
        block_data: Sequence[List[List[Any]]],
        field_obj: Optional[Field] = None,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_shape = block_shape  # (d2, d2)
        self.indptr = list(indptr)
        self.indices = list(indices)
        self.block_data = list(block_data)
        self.field = field_obj if field_obj is not None else RationalField()

    @property
    def nnz_blocks(self) -> int:
        return len(self.block_data)

    def to_flat_csr(self) -> CSROperator:
        """Unrolls Block-CSR into a flat CSROperator of shape (N * d2, N * d2)."""
        d_r, d_c = self.block_shape
        total_rows = self.num_blocks * d_r
        total_cols = self.num_blocks * d_c
        f = self.field

        if self.num_blocks == 0 or self.nnz_blocks == 0:
            return CSROperator.empty(total_rows, total_cols, field=f)

        flat_indptr = [0]
        flat_indices: List[int] = []
        flat_data: List[Any] = []

        for p_i in range(self.num_blocks):
            b_start = self.indptr[p_i]
            b_end = self.indptr[p_i + 1]

            # Process each local row of block p_i
            for local_r in range(d_r):
                row_entries: Dict[int, Any] = {}
                for b_idx in range(b_start, b_end):
                    p_j = self.indices[b_idx]
                    blk = self.block_data[b_idx]
                    col_base = p_j * d_c
                    for local_c in range(d_c):
                        val = blk[local_r][local_c]
                        field_val = f.from_int(val) if isinstance(val, int) and not isinstance(f, RationalField) else (
                            Fraction(str(val)) if isinstance(val, (float, int)) and isinstance(f, RationalField) else val
                        )
                        if not f.is_zero(field_val):
                            global_c = col_base + local_c
                            if global_c in row_entries:
                                row_entries[global_c] = f.add(row_entries[global_c], field_val)
                            else:
                                row_entries[global_c] = field_val

                sorted_row = sorted(row_entries.items(), key=lambda x: x[0])
                for c, v in sorted_row:
                    flat_indices.append(c)
                    flat_data.append(v)
                flat_indptr.append(len(flat_indices))

        return CSROperator(total_rows, total_cols, flat_indptr, flat_indices, flat_data, field=f)

    def sheaf_nullity(self) -> int:
        """Computes dim(ker(L_2^down)) for the sheaf down-Laplacian."""
        return self.to_flat_csr().nullity()

    def sheaf_rank(self) -> int:
        """Computes rank(L_2^down) for the sheaf down-Laplacian."""
        return self.to_flat_csr().rank()


def _matrix_transpose(mat: List[List[Any]]) -> List[List[Any]]:
    """Transposes a 2D matrix list."""
    if not mat or not mat[0]:
        return []
    rows = len(mat)
    cols = len(mat[0])
    return [[mat[r][c] for r in range(rows)] for c in range(cols)]


def _matrix_matmul(a: List[List[Any]], b: List[List[Any]], field_obj: Field) -> List[List[Any]]:
    """Multiplies two 2D matrix lists over a Field trait."""
    rows = len(a)
    cols = len(b[0]) if b else 0
    k_len = len(b)
    f = field_obj

    res = [[f.zero() for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            total = f.zero()
            for k in range(k_len):
                val_a = a[r][k]
                val_b = b[k][c]
                fa = f.from_int(val_a) if isinstance(val_a, int) and not isinstance(f, RationalField) else (
                    Fraction(str(val_a)) if isinstance(val_a, (float, int)) and isinstance(f, RationalField) else val_a
                )
                fb = f.from_int(val_b) if isinstance(val_b, int) and not isinstance(f, RationalField) else (
                    Fraction(str(val_b)) if isinstance(val_b, (float, int)) and isinstance(f, RationalField) else val_b
                )
                total = f.add(total, f.mul(fa, fb))
            res[r][c] = total
    return res


def _matrix_add(a: List[List[Any]], b: List[List[Any]], field_obj: Field) -> List[List[Any]]:
    """Adds two 2D matrix lists of same dimensions over a Field trait."""
    rows = len(a)
    cols = len(a[0]) if a else 0
    f = field_obj
    res = [[f.zero() for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            va = a[r][c]
            vb = b[r][c]
            fa = f.from_int(va) if isinstance(va, int) and not isinstance(f, RationalField) else (
                Fraction(str(va)) if isinstance(va, (float, int)) and isinstance(f, RationalField) else va
            )
            fb = f.from_int(vb) if isinstance(vb, int) and not isinstance(f, RationalField) else (
                Fraction(str(vb)) if isinstance(vb, (float, int)) and isinstance(f, RationalField) else vb
            )
            res[r][c] = f.add(fa, fb)
    return res


def _matrix_scale(mat: List[List[Any]], scalar: int, field_obj: Field) -> List[List[Any]]:
    """Multiplies matrix by an integer scalar (+1 or -1)."""
    rows = len(mat)
    cols = len(mat[0]) if mat else 0
    f = field_obj
    s_val = f.from_int(scalar)
    res = [[f.zero() for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            v = mat[r][c]
            fv = f.from_int(v) if isinstance(v, int) and not isinstance(f, RationalField) else (
                Fraction(str(v)) if isinstance(v, (float, int)) and isinstance(f, RationalField) else v
            )
            res[r][c] = f.mul(s_val, fv)
    return res


def build_block_csr_sheaf_laplacian(sheaf_ir: CellularSheafIR) -> BlockCSROperator:
    """Assembles the Sheaf 2-Down Laplacian L_2^down = B_2^dagger @ B_2 directly into Block-CSR.
    
    CRITICAL ARCHITECTURAL GUARANTEE:
    1-cells (links) serve strictly as incidence indexing bounds during compilation.
    Zero link (1-cell) state variables or runtime buffers are allocated.
    Runtime state and sheaf dynamics operate strictly on 2-cells (plaquettes).
    """
    complex_obj = sheaf_ir.complex
    plaquettes = complex_obj.get_cells(2)
    n_p = len(plaquettes)

    p_stalk = sheaf_ir.get_stalk(2)
    d2 = p_stalk.dim
    f = resolve_field(p_stalk.field_type)

    if n_p == 0:
        return BlockCSROperator(0, (d2, d2), [0], [], [], field_obj=f)

    # 1. Map 1-cells to incidence lookup without allocating state vectors
    edges = complex_obj.get_cells(1)
    edge_index = {e.name: idx for idx, e in enumerate(edges)}

    # 2. Build incidence mapping: e_idx -> list of (plaquette_index, signed_coeff, plaquette_name)
    edge_incidences: Dict[int, List[Tuple[int, int, str]]] = {e_idx: [] for e_idx in range(len(edges))}

    for p_idx, p in enumerate(plaquettes):
        boundary_chain = p.boundary()
        for face_cell, coeff in boundary_chain:
            if face_cell.name in edge_index:
                e_idx = edge_index[face_cell.name]
                edge_incidences[e_idx].append((p_idx, coeff, p.name))

    # 3. Direct Plaquette Coupling Blocks:
    # (L_2^down)_{p_i, p_j} = sum_e (B_2)_{e, p_i} * (B_2)_{e, p_j} * (rho_{p_i -> e}^T @ rho_{p_j -> e})
    block_couplings: Dict[Tuple[int, int], List[List[Any]]] = {}

    for e_idx, incident_list in edge_incidences.items():
        if not incident_list:
            continue
        edge_name = edges[e_idx].name
        for p_i, s_i, p_i_name in incident_list:
            rho_i = sheaf_ir.get_restriction(p_i_name, edge_name)
            rho_i_T = _matrix_transpose(rho_i)

            for p_j, s_j, p_j_name in incident_list:
                rho_j = sheaf_ir.get_restriction(p_j_name, edge_name)
                # Compute block = (rho_i^T @ rho_j)
                coupling_term = _matrix_matmul(rho_i_T, rho_j, f)
                sign = s_i * s_j
                scaled_term = _matrix_scale(coupling_term, sign, f)

                pair = (p_i, p_j)
                if pair in block_couplings:
                    block_couplings[pair] = _matrix_add(block_couplings[pair], scaled_term, f)
                else:
                    block_couplings[pair] = scaled_term

    # 4. Construct Block-CSR arrays
    indptr = [0]
    indices: List[int] = []
    block_data: List[List[List[Any]]] = []

    def _is_block_zero(b: List[List[Any]]) -> bool:
        for row in b:
            for val in row:
                if not f.is_zero(val):
                    return False
        return True

    for p_i in range(n_p):
        row_entries = [
            (p_j, blk) for (pi, p_j), blk in block_couplings.items()
            if pi == p_i and not _is_block_zero(blk)
        ]
        row_entries.sort(key=lambda x: x[0])
        for p_j, blk in row_entries:
            indices.append(p_j)
            block_data.append(blk)
        indptr.append(len(indices))

    return BlockCSROperator(
        num_blocks=n_p,
        block_shape=(d2, d2),
        indptr=indptr,
        indices=indices,
        block_data=block_data,
        field_obj=f,
    )
