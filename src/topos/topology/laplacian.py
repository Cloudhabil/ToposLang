"""Discrete Hodge Laplacians on Cell Complexes.

Implements the k-th Hodge Laplacian:
    L_k = B_k^T B_k + B_{k+1} B_{k+1}^T = L_k^{down} + L_k^{up}
and verifies the discrete Hodge theorem: dim(ker(L_k)) = beta_k.
"""

from __future__ import annotations
from fractions import Fraction
from typing import Optional, Tuple

from topos.core.complex import CellComplex
from topos.topology.homology import BoundaryMatrix, ExactMatrix


class HodgeLaplacian:
    """Computes the k-th discrete Hodge Laplacian for a cell complex."""

    def __init__(self, complex_obj: CellComplex, dim: int) -> None:
        self.complex = complex_obj
        self.dim = dim
        self.k_cells = complex_obj.get_cells(dim)
        self.n_k = len(self.k_cells)

        self._L_down: Optional[ExactMatrix] = None
        self._L_up: Optional[ExactMatrix] = None
        self._L_total: Optional[ExactMatrix] = None

    @property
    def down_laplacian(self) -> ExactMatrix:
        """L_k^{down} = B_k^T @ B_k."""
        if self._L_down is None:
            if self.dim == 0 or self.n_k == 0:
                self._L_down = ExactMatrix(self.n_k, self.n_k)
            else:
                B_k = BoundaryMatrix.from_complex(self.complex, self.dim).matrix
                B_k_T = B_k.transpose()
                self._L_down = B_k_T.matmul(B_k)
        return self._L_down

    @property
    def up_laplacian(self) -> ExactMatrix:
        """L_k^{up} = B_{k+1} @ B_{k+1}^T."""
        if self._L_up is None:
            next_cells = self.complex.get_cells(self.dim + 1)
            if not next_cells or self.n_k == 0:
                self._L_up = ExactMatrix(self.n_k, self.n_k)
            else:
                B_next = BoundaryMatrix.from_complex(self.complex, self.dim + 1).matrix
                B_next_T = B_next.transpose()
                self._L_up = B_next.matmul(B_next_T)
        return self._L_up

    @property
    def laplacian(self) -> ExactMatrix:
        """Total Hodge Laplacian L_k = L_k^{down} + L_k^{up}."""
        if self._L_total is None:
            if self.n_k == 0:
                self._L_total = ExactMatrix(0, 0)
            else:
                self._L_total = self.down_laplacian.add(self.up_laplacian)
        return self._L_total

    def harmonic_dimension(self) -> int:
        """Dimension of the kernel of L_k (number of zero eigenvalues / harmonic forms).
        
        By Hodge theorem, dim(ker(L_k)) == beta_k.
        """
        if self.n_k == 0:
            return 0
        return self.laplacian.nullity()
