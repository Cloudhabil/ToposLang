"""Topological Intermediate Representation (T-IR): DACC and Contraction Passes."""

from topos.ir.tir import DACC
from topos.ir.contract import HomotopyContractionPass
from topos.ir.sheaf import (
    BlockCSROperator,
    CellularSheafIR,
    StalkSpec,
    build_block_csr_sheaf_laplacian,
)

__all__ = [
    "DACC",
    "HomotopyContractionPass",
    "BlockCSROperator",
    "CellularSheafIR",
    "StalkSpec",
    "build_block_csr_sheaf_laplacian",
]
