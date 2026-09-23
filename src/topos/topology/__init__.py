"""Topos algebraic topology module: Homology, Hodge Laplacians, and Topological Invariants."""

from topos.topology.homology import ExactMatrix, BoundaryMatrix, HomologyEngine
from topos.topology.laplacian import HodgeLaplacian
from topos.topology.invariant import (
    betti_numbers,
    euler_characteristic,
    verify_euler_poincare,
    is_connected,
    is_simply_connected,
    has_uncontracted_loops,
    has_uncontracted_cavities,
)
from topos.ir.sheaf import (
    BlockCSROperator,
    CellularSheafIR,
    build_block_csr_sheaf_laplacian,
)

__all__ = [
    "ExactMatrix",
    "BoundaryMatrix",
    "HomologyEngine",
    "HodgeLaplacian",
    "BlockCSROperator",
    "CellularSheafIR",
    "build_block_csr_sheaf_laplacian",
    "betti_numbers",
    "euler_characteristic",
    "verify_euler_poincare",
    "is_connected",
    "is_simply_connected",
    "has_uncontracted_loops",
    "has_uncontracted_cavities",
]
