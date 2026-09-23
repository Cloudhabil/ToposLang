"""Boundary Operator and Nilpotency Verifier.

Implements the fundamental homology boundary operator d_k : C_k -> C_{k-1}
and automated verification that d_{k-1} o d_k = 0 for all cells in a complex.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Union

from topos.core.chain import Chain

if TYPE_CHECKING:
    from topos.core.cell import Cell
    from topos.core.complex import CellComplex


class TopologicalValidationError(Exception):
    """Base exception for topological and geometric consistency errors."""
    pass


class NilpotencyViolationError(TopologicalValidationError):
    """Raised when the fundamental boundary nilpotency condition (d^2 = 0) is violated."""

    def __init__(self, message: str, cell: Any = None, dim: int = 0, residual: Any = None) -> None:
        super().__init__(message)
        self.cell = cell
        self.dim = dim
        self.residual = residual


class CellNotFoundError(TopologicalValidationError):
    """Raised when a boundary face or referenced cell is missing from a complex."""
    pass


def boundary(obj: Union[Cell, Chain]) -> Chain:
    """Computes the oriented topological boundary of a cell or chain via linearity.
    
    For a chain C = sum(a_i * c_i), d(C) = sum(a_i * d(c_i)).
    """
    if isinstance(obj, Chain):
        if obj.dim <= 0 or obj.is_zero():
            return Chain.zero(obj.dim - 1)
        
        result = Chain.zero(obj.dim - 1)
        for cell, coeff in obj:
            cell_boundary = cell.boundary()
            result = result + (cell_boundary * coeff)
        return result

    # Handle Cell object
    if obj.dim <= 0:
        return Chain.zero(obj.dim - 1)
    return obj.boundary()


class NilpotencyVerifier:
    """Automated formal verifier enforcing the nilpotency axiom: d_{k-1} o d_k = 0."""

    @staticmethod
    def verify_cell(cell: Cell) -> bool:
        """Verifies that d(d(c)) == 0 for a single cell.
        
        Raises:
            NilpotencyViolationError: If d(d(c)) != 0.
        """
        if cell.dim < 2:
            # For 0-cells and 1-cells, d^2 is trivially 0 since d(0-cell) = 0
            return True

        b1 = boundary(cell)
        b2 = boundary(b1)

        if not b2.is_zero():
            raise NilpotencyViolationError(
                f"Nilpotency axiom d_{cell.dim-1} o d_{cell.dim} = 0 violated on cell {cell.name!r}. "
                f"Residual non-zero boundary: {b2}",
                cell=cell,
                dim=cell.dim,
                residual=b2,
            )
        return True

    @classmethod
    def verify_complex(cls, complex: CellComplex) -> bool:
        """Verifies that every cell in the complex satisfies d^2 = 0.
        
        Raises:
            NilpotencyViolationError: If any cell violates the nilpotency axiom.
        """
        for dim in sorted(complex.graded_cells.keys()):
            for cell in complex.get_cells(dim):
                cls.verify_cell(cell)
        return True
