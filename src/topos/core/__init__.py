"""Topos core mathematical primitives: Cells, Chains, Complexes, and Boundary Operators."""

from topos.core.cell import Cell, Simplex, GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import (
    boundary,
    NilpotencyVerifier,
    TopologicalValidationError,
    NilpotencyViolationError,
    CellNotFoundError,
)
from topos.core.matrix_csr import (
    Field,
    PrimeField,
    BinaryField,
    RationalField,
    CSROperator,
    compute_plaquette_down_laplacian,
    CSRHomologyEngine,
)

__all__ = [
    "Cell",
    "Simplex",
    "GlobularCell",
    "Chain",
    "CellComplex",
    "boundary",
    "NilpotencyVerifier",
    "TopologicalValidationError",
    "NilpotencyViolationError",
    "CellNotFoundError",
    "Field",
    "PrimeField",
    "BinaryField",
    "RationalField",
    "CSROperator",
    "compute_plaquette_down_laplacian",
    "CSRHomologyEngine",
]

