"""Compressed Sparse Row (CSR) matrix backend and finite field abstractions.

Implements:
1. Field traits: Finite Field F_p, Binary Field F_2 (GF(2)), and Rational Field Q adapter.
2. CSROperator: Compressed Sparse Row matrix supporting sparse-sparse multiplication,
   transposition, addition, Gaussian elimination, and exact rank/nullity over any field K.
3. Direct Plaquette Down-Laplacian: L_2^down = B_2^T @ B_2 in CSR format without
   allocating 1-cell (link) state vectors.
4. CSRHomologyEngine: Exact topological homology and Betti numbers over F_p, F_2, and Q.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from topos.core.cell import Cell
from topos.core.complex import CellComplex


# ---------------------------------------------------------------------------
# 1. Algebraic Field Abstractions
# ---------------------------------------------------------------------------

class Field(ABC):
    """Abstract algebraic field K."""

    @abstractmethod
    def zero(self) -> Any:
        """Additive identity 0."""
        pass

    @abstractmethod
    def one(self) -> Any:
        """Multiplicative identity 1."""
        pass

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Field addition a + b."""
        pass

    @abstractmethod
    def sub(self, a: Any, b: Any) -> Any:
        """Field subtraction a - b."""
        pass

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        """Field multiplication a * b."""
        pass

    @abstractmethod
    def inv(self, a: Any) -> Any:
        """Multiplicative inverse a^(-1). Raises ZeroDivisionError if a == 0."""
        pass

    @abstractmethod
    def neg(self, a: Any) -> Any:
        """Additive inverse -a."""
        pass

    @abstractmethod
    def is_zero(self, a: Any) -> bool:
        """Check if element is additive identity."""
        pass

    @abstractmethod
    def from_int(self, n: int) -> Any:
        """Coerce integer to field element."""
        pass


class PrimeField(Field):
    """Finite Field F_p for prime p."""

    def __init__(self, p: int) -> None:
        if p < 2:
            raise ValueError(f"Prime p must be >= 2, got {p}")
        self.p = p

    def zero(self) -> int:
        return 0

    def one(self) -> int:
        return 1

    def from_int(self, n: int) -> int:
        return ((n % self.p) + self.p) % self.p

    def add(self, a: int, b: int) -> int:
        return (a + b) % self.p

    def sub(self, a: int, b: int) -> int:
        return (a - b + self.p) % self.p

    def mul(self, a: int, b: int) -> int:
        return (a * b) % self.p

    def neg(self, a: int) -> int:
        return (self.p - (a % self.p)) % self.p

    def inv(self, a: int) -> int:
        a_mod = a % self.p
        if a_mod == 0:
            raise ZeroDivisionError(f"Division by zero in F_{self.p}")
        # Extended Euclidean Algorithm
        g, x, _ = self._extended_gcd(a_mod, self.p)
        if g != 1:
            raise ValueError(f"{a} has no modular inverse in F_{self.p} (p is not prime to a)")
        return (x % self.p + self.p) % self.p

    def is_zero(self, a: int) -> bool:
        return (a % self.p) == 0

    def _extended_gcd(self, a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = self._extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y

    def __repr__(self) -> str:
        return f"F_{self.p}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PrimeField) and self.p == other.p


class BinaryField(PrimeField):
    """Optimized Galois Field GF(2) = F_2."""

    def __init__(self) -> None:
        super().__init__(2)

    def add(self, a: int, b: int) -> int:
        return (a ^ b) & 1

    def sub(self, a: int, b: int) -> int:
        return (a ^ b) & 1

    def mul(self, a: int, b: int) -> int:
        return (a & b) & 1

    def neg(self, a: int) -> int:
        return a & 1

    def inv(self, a: int) -> int:
        if (a & 1) == 0:
            raise ZeroDivisionError("Division by zero in F_2")
        return 1

    def from_int(self, n: int) -> int:
        return n & 1

    def is_zero(self, a: int) -> bool:
        return (a & 1) == 0

    def __repr__(self) -> str:
        return "F_2"


class RationalField(Field):
    """Exact Rational Field Q using fractions.Fraction."""

    def zero(self) -> Fraction:
        return Fraction(0, 1)

    def one(self) -> Fraction:
        return Fraction(1, 1)

    def from_int(self, n: int) -> Fraction:
        return Fraction(n, 1)

    def add(self, a: Fraction, b: Fraction) -> Fraction:
        return a + b

    def sub(self, a: Fraction, b: Fraction) -> Fraction:
        return a - b

    def mul(self, a: Fraction, b: Fraction) -> Fraction:
        return a * b

    def neg(self, a: Fraction) -> Fraction:
        return -a

    def inv(self, a: Fraction) -> Fraction:
        if a == 0:
            raise ZeroDivisionError("Division by zero in Q")
        return Fraction(1, 1) / a

    def is_zero(self, a: Fraction) -> bool:
        return a == 0

    def __repr__(self) -> str:
        return "Q"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RationalField)


# ---------------------------------------------------------------------------
# 2. Compressed Sparse Row (CSR) Operator
# ---------------------------------------------------------------------------

class CSROperator:
    """Compressed Sparse Row (CSR) matrix operator over an arbitrary field K."""

    def __init__(
        self,
        rows: int,
        cols: int,
        indptr: Sequence[int],
        indices: Sequence[int],
        data: Sequence[Any],
        field: Optional[Field] = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.field = field if field is not None else RationalField()
        self.indptr = list(indptr)
        self.indices = list(indices)
        self.data = [self.field.from_int(v) if isinstance(v, int) and not isinstance(self.field, RationalField) else v for v in data]

        if len(self.indptr) != self.rows + 1:
            raise ValueError(f"indptr length {len(self.indptr)} must be rows + 1 ({self.rows + 1})")
        if len(self.indices) != len(self.data):
            raise ValueError(f"indices length {len(self.indices)} must match data length {len(self.data)}")

    @property
    def nnz(self) -> int:
        """Total number of stored non-zero entries."""
        return len(self.data)

    @classmethod
    def empty(cls, rows: int, cols: int, field: Optional[Field] = None) -> CSROperator:
        """Creates an all-zero CSR matrix of shape (rows, cols)."""
        return cls(rows, cols, [0] * (rows + 1), [], [], field=field)

    @classmethod
    def from_dense(cls, grid: List[List[Any]], field: Optional[Field] = None) -> CSROperator:
        """Constructs a CSR operator from a 2D dense list."""
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0
        f = field if field is not None else RationalField()

        indptr = [0]
        indices: List[int] = []
        data: List[Any] = []

        for r in range(rows):
            for c in range(cols):
                val = grid[r][c]
                field_val = f.from_int(val) if isinstance(val, int) else val
                if not f.is_zero(field_val):
                    indices.append(c)
                    data.append(field_val)
            indptr.append(len(indices))

        return cls(rows, cols, indptr, indices, data, field=f)

    @classmethod
    def from_coo(
        cls,
        rows: int,
        cols: int,
        row_indices: Sequence[int],
        col_indices: Sequence[int],
        values: Sequence[Any],
        field: Optional[Field] = None,
    ) -> CSROperator:
        """Constructs a CSR operator from Coordinate (COO) format, summing duplicate entries."""
        f = field if field is not None else RationalField()
        
        # Group and accumulate by (r, c)
        coo_dict: Dict[Tuple[int, int], Any] = {}
        for r, c, val in zip(row_indices, col_indices, values):
            field_val = f.from_int(val) if isinstance(val, int) else val
            if (r, c) in coo_dict:
                coo_dict[(r, c)] = f.add(coo_dict[(r, c)], field_val)
            else:
                coo_dict[(r, c)] = field_val

        indptr = [0]
        indices: List[int] = []
        data: List[Any] = []

        for r in range(rows):
            # Sort non-zero columns for row r
            row_entries = [(c, val) for (row, c), val in coo_dict.items() if row == r and not f.is_zero(val)]
            row_entries.sort(key=lambda x: x[0])
            for c, val in row_entries:
                indices.append(c)
                data.append(val)
            indptr.append(len(indices))

        return cls(rows, cols, indptr, indices, data, field=f)

    def to_dense(self) -> List[List[Any]]:
        """Converts CSR matrix to dense 2D list."""
        dense = [[self.field.zero() for _ in range(self.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for idx in range(start, end):
                c = self.indices[idx]
                dense[r][c] = self.data[idx]
        return dense

    def get(self, r: int, c: int) -> Any:
        """Retrieves element at (r, c)."""
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise IndexError(f"Index ({r}, {c}) out of bounds for ({self.rows}, {self.cols})")
        start = self.indptr[r]
        end = self.indptr[r + 1]
        for idx in range(start, end):
            if self.indices[idx] == c:
                return self.data[idx]
        return self.field.zero()

    def transpose(self) -> CSROperator:
        """Computes the transpose operator M^T in CSR format."""
        if self.nnz == 0:
            return CSROperator.empty(self.cols, self.rows, field=self.field)

        # Count entries per column
        col_counts = [0] * self.cols
        for c in self.indices:
            col_counts[c] += 1

        # Transpose indptr
        t_indptr = [0] * (self.cols + 1)
        for c in range(self.cols):
            t_indptr[c + 1] = t_indptr[c] + col_counts[c]

        t_indices = [0] * self.nnz
        t_data = [self.field.zero()] * self.nnz
        curr_offset = list(t_indptr)

        for r in range(self.rows):
            start = self.indptr[r]
            end = self.indptr[r + 1]
            for idx in range(start, end):
                c = self.indices[idx]
                val = self.data[idx]
                dest = curr_offset[c]
                t_indices[dest] = r
                t_data[dest] = val
                curr_offset[c] += 1

        return CSROperator(self.cols, self.rows, t_indptr, t_indices, t_data, field=self.field)

    def matmul(self, other: CSROperator) -> CSROperator:
        """Sparse-sparse matrix multiplication: self @ other."""
        if self.cols != other.rows:
            raise ValueError(f"Incompatible dimensions for matmul: {self.cols} != {other.rows}")
        if self.field != other.field:
            raise ValueError(f"Field mismatch in matmul: {self.field} != {other.field}")

        f = self.field
        res_indptr = [0]
        res_indices: List[int] = []
        res_data: List[Any] = []

        # Sparse row-based accumulator
        for r in range(self.rows):
            row_acc: Dict[int, Any] = {}
            start = self.indptr[r]
            end = self.indptr[r + 1]

            for idx in range(start, end):
                k = self.indices[idx]
                a_val = self.data[idx]

                # Scan row k of other
                other_start = other.indptr[k]
                other_end = other.indptr[k + 1]
                for o_idx in range(other_start, other_end):
                    c = other.indices[o_idx]
                    b_val = other.data[o_idx]
                    prod = f.mul(a_val, b_val)
                    if c in row_acc:
                        row_acc[c] = f.add(row_acc[c], prod)
                    else:
                        row_acc[c] = prod

            # Filter out zeros and sort by column
            sorted_entries = [(c, val) for c, val in sorted(row_acc.items()) if not f.is_zero(val)]
            for c, val in sorted_entries:
                res_indices.append(c)
                res_data.append(val)
            res_indptr.append(len(res_indices))

        return CSROperator(self.rows, other.cols, res_indptr, res_indices, res_data, field=f)

    def add(self, other: CSROperator) -> CSROperator:
        """Matrix addition: self + other."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Dimension mismatch in add: ({self.rows}, {self.cols}) != ({other.rows}, {other.cols})")
        if self.field != other.field:
            raise ValueError(f"Field mismatch in add: {self.field} != {other.field}")

        f = self.field
        res_indptr = [0]
        res_indices: List[int] = []
        res_data: List[Any] = []

        for r in range(self.rows):
            row_acc: Dict[int, Any] = {}
            # Load self row
            for idx in range(self.indptr[r], self.indptr[r + 1]):
                row_acc[self.indices[idx]] = self.data[idx]
            # Add other row
            for idx in range(other.indptr[r], other.indptr[r + 1]):
                c = other.indices[idx]
                val = other.data[idx]
                if c in row_acc:
                    row_acc[c] = f.add(row_acc[c], val)
                else:
                    row_acc[c] = val

            sorted_entries = [(c, val) for c, val in sorted(row_acc.items()) if not f.is_zero(val)]
            for c, val in sorted_entries:
                res_indices.append(c)
                res_data.append(val)
            res_indptr.append(len(res_indices))

        return CSROperator(self.rows, self.cols, res_indptr, res_indices, res_data, field=f)

    def row_echelon_form(self) -> Tuple[CSROperator, int]:
        """Gaussian elimination over field K with coordinated row/column pointers.
        
        Returns:
            (ref_operator, rank): Reduced row echelon operator and exact rank.
        """
        if self.rows == 0 or self.cols == 0 or self.nnz == 0:
            return (CSROperator.empty(self.rows, self.cols, field=self.field), 0)

        f = self.field
        # Convert to mutable sparse rows: List[Dict[int, Any]]
        sparse_rows: List[Dict[int, Any]] = []
        for r in range(self.rows):
            row_dict = {}
            for idx in range(self.indptr[r], self.indptr[r + 1]):
                row_dict[self.indices[idx]] = self.data[idx]
            sparse_rows.append(row_dict)

        rank = 0
        lead = 0
        r = 0

        while r < self.rows and lead < self.cols:
            # 1. Locate pivot row
            pivot_row = r
            while pivot_row < self.rows and lead not in sparse_rows[pivot_row]:
                pivot_row += 1

            if pivot_row == self.rows:
                # No pivot in column lead, advance column but stay on row r
                lead += 1
                continue

            # 2. Swap pivot row to position r
            sparse_rows[r], sparse_rows[pivot_row] = sparse_rows[pivot_row], sparse_rows[r]

            # 3. Normalize pivot row so pivot element = 1
            pivot_val = sparse_rows[r][lead]
            pivot_inv = f.inv(pivot_val)
            normalized_row = {c: f.mul(v, pivot_inv) for c, v in sparse_rows[r].items()}
            sparse_rows[r] = normalized_row

            # 4. Eliminate lower rows
            for lower_row in range(r + 1, self.rows):
                if lead in sparse_rows[lower_row]:
                    factor = sparse_rows[lower_row][lead]
                    new_row: Dict[int, Any] = {}
                    # Union of columns
                    all_cols = set(sparse_rows[lower_row].keys()) | set(sparse_rows[r].keys())
                    for c in all_cols:
                        curr_val = sparse_rows[lower_row].get(c, f.zero())
                        sub_term = f.mul(factor, sparse_rows[r].get(c, f.zero()))
                        res_val = f.sub(curr_val, sub_term)
                        if not f.is_zero(res_val):
                            new_row[c] = res_val
                    sparse_rows[lower_row] = new_row

            rank += 1
            r += 1
            lead += 1

        # Rebuild CSR
        ref_indptr = [0]
        ref_indices: List[int] = []
        ref_data: List[Any] = []

        for r_dict in sparse_rows:
            sorted_entries = sorted(r_dict.items())
            for c, val in sorted_entries:
                ref_indices.append(c)
                ref_data.append(val)
            ref_indptr.append(len(ref_indices))

        return (CSROperator(self.rows, self.cols, ref_indptr, ref_indices, ref_data, field=f), rank)

    def rank(self) -> int:
        """Calculates exact rank over field K."""
        _, r = self.row_echelon_form()
        return r

    def nullity(self) -> int:
        """Computes dimension of null space (kernel): dim(ker) = cols - rank."""
        return self.cols - self.rank()

    def __repr__(self) -> str:
        return f"CSROperator(shape=({self.rows}, {self.cols}), nnz={self.nnz}, field={self.field})"


# ---------------------------------------------------------------------------
# 3. Direct Plaquette Down-Laplacian Assembler (No Link Allocations)
# ---------------------------------------------------------------------------

def compute_plaquette_down_laplacian(
    complex_obj: CellComplex,
    field: Optional[Field] = None,
) -> CSROperator:
    """Assembles the combinatorial 2-down Laplacian L_2^down = B_2^T @ B_2 directly in CSR format.
    
    CRITICAL ARCHITECTURAL GUARANTEE:
    1-cells (links) serve strictly as incidence indexing bounds during compilation.
    State allocations and runtime coupling operate strictly on 2-cells (plaquettes).
    """
    f = field if field is not None else PrimeField(65537)
    plaquettes = complex_obj.get_cells(2)
    n_p = len(plaquettes)

    if n_p == 0:
        return CSROperator.empty(0, 0, field=f)

    # 1. Map 1-cells to integer indices for incidence lookup
    edges = complex_obj.get_cells(1)
    edge_index = {e.name: idx for idx, e in enumerate(edges)}

    # 2. Build boundary incidence in COO format: (edge_idx, plaquette_idx) -> coeff
    # Note: Link state vectors are never allocated!
    b2_coo: Dict[int, List[Tuple[int, int]]] = {e_idx: [] for e_idx in range(len(edges))}

    for p_idx, p in enumerate(plaquettes):
        boundary_chain = p.boundary()
        for face_cell, coeff in boundary_chain:
            if face_cell.name in edge_index:
                e_idx = edge_index[face_cell.name]
                b2_coo[e_idx].append((p_idx, coeff))

    # 3. Directly couple plaquettes sharing an edge: (L_2^down)_{p_i, p_j} = sum_e (B_2)_{e, p_i} * (B_2)_{e, p_j}
    plaquette_couplings: Dict[Tuple[int, int], int] = {}

    for e_idx, incident_plaquettes in b2_coo.items():
        for p_i, s_i in incident_plaquettes:
            for p_j, s_j in incident_plaquettes:
                prod = s_i * s_j
                key = (p_i, p_j)
                plaquette_couplings[key] = plaquette_couplings.get(key, 0) + prod

    # 4. Assemble CSR representation
    indptr = [0]
    indices: List[int] = []
    data: List[Any] = []

    for p_i in range(n_p):
        row_entries = [
            (p_j, val) for (pi, p_j), val in plaquette_couplings.items()
            if pi == p_i and not f.is_zero(f.from_int(val))
        ]
        row_entries.sort(key=lambda x: x[0])
        for p_j, val in row_entries:
            indices.append(p_j)
            data.append(f.from_int(val))
        indptr.append(len(indices))

    return CSROperator(n_p, n_p, indptr, indices, data, field=f)


# ---------------------------------------------------------------------------
# 4. CSR Homology Engine
# ---------------------------------------------------------------------------

class CSRHomologyEngine:
    """Computes exact boundary matrices and Betti numbers over field K using CSR representation."""

    def __init__(self, complex_obj: CellComplex, field: Optional[Field] = None) -> None:
        self.complex = complex_obj
        self.field = field if field is not None else RationalField()

    def boundary_matrix(self, dim: int) -> CSROperator:
        """Constructs boundary matrix B_dim : C_dim -> C_{dim-1} as a CSROperator."""
        k_cells = self.complex.get_cells(dim)
        lower_cells = self.complex.get_cells(dim - 1) if dim > 0 else ()

        n_rows = len(lower_cells)
        n_cols = len(k_cells)

        if n_rows == 0 or n_cols == 0:
            return CSROperator.empty(n_rows, n_cols, field=self.field)

        lower_index = {cell.name: idx for idx, cell in enumerate(lower_cells)}
        row_indices: List[int] = []
        col_indices: List[int] = []
        values: List[Any] = []

        for col_idx, k_cell in enumerate(k_cells):
            b_chain = k_cell.boundary()
            for face_cell, coeff in b_chain:
                if face_cell.name in lower_index:
                    row_idx = lower_index[face_cell.name]
                    row_indices.append(row_idx)
                    col_indices.append(col_idx)
                    values.append(coeff)

        return CSROperator.from_coo(n_rows, n_cols, row_indices, col_indices, values, field=self.field)

    def betti_number(self, dim: int) -> int:
        """Computes the dim-th Betti number: beta_k = dim(ker(B_k)) - rank(B_{k+1})."""
        k_cells = self.complex.get_cells(dim)
        n_k = len(k_cells)
        if n_k == 0:
            return 0

        # Compute dim(ker(B_k))
        if dim == 0:
            dim_ker_Bk = n_k
        else:
            Bk = self.boundary_matrix(dim)
            dim_ker_Bk = Bk.nullity()

        # Compute rank(B_{k+1})
        next_cells = self.complex.get_cells(dim + 1)
        if not next_cells:
            rank_Bk_plus_1 = 0
        else:
            Bk_plus_1 = self.boundary_matrix(dim + 1)
            rank_Bk_plus_1 = Bk_plus_1.rank()

        beta_k = dim_ker_Bk - rank_Bk_plus_1
        return max(0, beta_k)

    def betti_profile(self) -> Dict[int, int]:
        """Computes all Betti numbers {k: beta_k} up to the complex dimension."""
        max_dim = self.complex.dim
        if max_dim < 0:
            return {}
        return {k: self.betti_number(k) for k in range(max_dim + 1)}
