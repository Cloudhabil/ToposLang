"""Boundary incidence matrices and homology computation over cell complexes.

Computes boundary matrices B_k, exact rank and kernel dimensions over Q (using Fraction),
and calculates Betti numbers beta_k = dim(ker(B_k)) - rank(B_{k+1}).
"""

from __future__ import annotations
from fractions import Fraction
from typing import Dict, List, Sequence, Tuple, Union

from topos.core.cell import Cell
from topos.core.complex import CellComplex


class ExactMatrix:
    """Exact rational matrix over Q using fractions.Fraction to guarantee 0 roundoff error."""

    def __init__(self, rows: int, cols: int, data: List[List[Fraction]] | None = None) -> None:
        self.rows = rows
        self.cols = cols
        if data is not None:
            self._data = data
        else:
            self._data = [[Fraction(0, 1) for _ in range(cols)] for _ in range(rows)]

    @classmethod
    def from_integers(cls, rows: int, cols: int, grid: List[List[int]]) -> ExactMatrix:
        data = [[Fraction(val, 1) for val in row] for row in grid]
        return cls(rows, cols, data)

    def get(self, r: int, c: int) -> Fraction:
        return self._data[r][c]

    def set(self, r: int, c: int, val: Union[Fraction, int]) -> None:
        self._data[r][c] = Fraction(val)

    def transpose(self) -> ExactMatrix:
        """Returns the transpose matrix M^T."""
        transposed_data = [
            [self._data[r][c] for r in range(self.rows)]
            for c in range(self.cols)
        ]
        return ExactMatrix(self.cols, self.rows, transposed_data)

    def matmul(self, other: ExactMatrix) -> ExactMatrix:
        """Matrix multiplication self @ other."""
        if self.cols != other.rows:
            raise ValueError(f"Incompatible dimensions for matmul: {self.cols} != {other.rows}")
        result = ExactMatrix(self.rows, other.cols)
        for r in range(self.rows):
            for c in range(other.cols):
                total = Fraction(0, 1)
                for k in range(self.cols):
                    total += self._data[r][k] * other._data[k][c]
                result.set(r, c, total)
        return result

    def add(self, other: ExactMatrix) -> ExactMatrix:
        """Matrix addition self + other."""
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Dimension mismatch in matrix addition.")
        data = [
            [self._data[r][c] + other._data[r][c] for c in range(self.cols)]
            for r in range(self.rows)
        ]
        return ExactMatrix(self.rows, self.cols, data)

    def row_echelon_form(self) -> Tuple[ExactMatrix, int]:
        """Performs exact Gaussian elimination to transform matrix to Row Echelon Form.
        
        Returns:
            (ref_matrix, rank): The matrix in row echelon form and its exact rank.
        """
        if self.rows == 0 or self.cols == 0:
            return (ExactMatrix(self.rows, self.cols), 0)

        # Clone data
        mat = [[self._data[r][c] for c in range(self.cols)] for r in range(self.rows)]
        rank = 0
        lead = 0

        r = 0
        lead = 0
        while r < self.rows and lead < self.cols:
            # Find pivot in column 'lead' starting from row 'r'
            pivot_row = r
            while pivot_row < self.rows and mat[pivot_row][lead] == 0:
                pivot_row += 1

            if pivot_row == self.rows:
                # No pivot in this column, move to next column but stay on row r
                lead += 1
                continue

            # Swap rows
            mat[r], mat[pivot_row] = mat[pivot_row], mat[r]

            # Normalize pivot row
            pivot_val = mat[r][lead]
            mat[r] = [val / pivot_val for val in mat[r]]

            # Eliminate subsequent rows
            for lower_row in range(r + 1, self.rows):
                factor = mat[lower_row][lead]
                if factor != 0:
                    mat[lower_row] = [
                        mat[lower_row][c] - factor * mat[r][c]
                        for c in range(self.cols)
                    ]

            rank += 1
            r += 1
            lead += 1

        return (ExactMatrix(self.rows, self.cols, mat), rank)

    def rank(self) -> int:
        """Computes the exact rank of the matrix over Q."""
        _, r = self.row_echelon_form()
        return r

    def nullity(self) -> int:
        """Computes the dimension of the null space (kernel): dim(ker) = cols - rank."""
        return self.cols - self.rank()

    def __repr__(self) -> str:
        rows_str = ["[" + ", ".join(str(val) for val in row) + "]" for row in self._data]
        return f"ExactMatrix({self.rows}x{self.cols}):\n" + "\n".join(rows_str)


class BoundaryMatrix:
    """Represents the boundary linear map B_k : C_k(K) -> C_{k-1}(K).
    
    Rows correspond to (k-1)-cells and columns correspond to k-cells.
    """

    def __init__(
        self,
        dim: int,
        k_cells: Sequence[Cell],
        lower_cells: Sequence[Cell],
        matrix: ExactMatrix,
    ) -> None:
        self.dim = dim
        self.k_cells = tuple(k_cells)
        self.lower_cells = tuple(lower_cells)
        self.matrix = matrix

    @classmethod
    def from_complex(cls, complex_obj: CellComplex, dim: int) -> BoundaryMatrix:
        """Constructs boundary matrix B_dim from a CellComplex."""
        k_cells = complex_obj.get_cells(dim)
        lower_cells = complex_obj.get_cells(dim - 1) if dim > 0 else ()

        n_rows = len(lower_cells)
        n_cols = len(k_cells)

        if n_rows == 0 or n_cols == 0:
            return cls(dim, k_cells, lower_cells, ExactMatrix(n_rows, n_cols))

        lower_index = {cell.name: idx for idx, cell in enumerate(lower_cells)}
        matrix = ExactMatrix(n_rows, n_cols)

        for col_idx, k_cell in enumerate(k_cells):
            b_chain = k_cell.boundary()
            for face_cell, coeff in b_chain:
                if face_cell.name in lower_index:
                    row_idx = lower_index[face_cell.name]
                    matrix.set(row_idx, col_idx, coeff)

        return cls(dim, k_cells, lower_cells, matrix)


class HomologyEngine:
    """Computes homology groups and Betti numbers for cell complexes."""

    def __init__(self, complex_obj: CellComplex) -> None:
        self.complex = complex_obj

    def boundary_matrix(self, dim: int) -> BoundaryMatrix:
        """Retrieves boundary matrix B_dim."""
        return BoundaryMatrix.from_complex(self.complex, dim)

    def betti_number(self, dim: int) -> int:
        """Computes the dim-th Betti number: beta_k = dim(ker(B_k)) - rank(B_{k+1})."""
        k_cells = self.complex.get_cells(dim)
        n_k = len(k_cells)
        if n_k == 0:
            return 0

        # Compute dim(ker(B_k))
        if dim == 0:
            dim_ker_Bk = n_k  # B_0 maps to 0-space, so ker(B_0) is entire C_0
        else:
            Bk = self.boundary_matrix(dim)
            dim_ker_Bk = Bk.matrix.nullity()

        # Compute rank(B_{k+1})
        next_cells = self.complex.get_cells(dim + 1)
        if not next_cells:
            rank_Bk_plus_1 = 0
        else:
            Bk_plus_1 = self.boundary_matrix(dim + 1)
            rank_Bk_plus_1 = Bk_plus_1.matrix.rank()

        beta_k = dim_ker_Bk - rank_Bk_plus_1
        return max(0, beta_k)

    def betti_profile(self) -> Dict[int, int]:
        """Computes all Betti numbers {k: beta_k} up to the complex dimension."""
        max_dim = self.complex.dim
        if max_dim < 0:
            return {}
        return {k: self.betti_number(k) for k in range(max_dim + 1)}
