"""Roofline Model and Static Compiler Invariant Engine for ToposLang.

Formalizes Section 4 of the CRT Galois-Clifford monograph:
1. Hadamard coefficient bit-growth bound: L_max <= 0.5 * N * log2(N).
2. Prime lane count: L = ceil(L_max / 62).
3. Tile size derivation for zero DRAM spilling:
   T_max = floor( C_SRAM / (d * (18 + 64 * L) + 16) )
   for 128 KB Scratchpad SRAM per core.
4. Algorithmic arithmetic intensity: I_algo = (8/9) * L * d.
5. Static Compiler Constraint Invariant: L * d >= 19
   guaranteeing 100% compute-bound throughput with zero DRAM stalling.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional, Tuple

from topos.core.complex import CellComplex
from topos.core.crt import RooflineContractReport, verify_roofline_machine_balance



@dataclass(frozen=True)
class RooflineAnalysisResult:
    """Static compiler verification report for memory bandwidth safety."""
    num_cells_N: int
    estimated_rank: int
    column_density_d: float
    hadamard_bit_bound_L_max: float
    required_prime_lanes_L: int
    max_tile_size_T: int
    arithmetic_intensity: float
    machine_balance_threshold: float
    is_compute_bound: bool
    invariant_satisfied: bool
    diagnostic_message: str


class RooflineCompilerVerifier:
    """Static analysis pass verifying Roofline bounds on cell complexes and matrices."""

    def __init__(
        self,
        c_sram_bits: int = 1_048_576,  # 128 KB per core
        machine_balance: float = 16.5,  # 13.2 TMOP/s / 800 GB/s = 16.5 OPs/Byte
    ) -> None:
        self.c_sram_bits = c_sram_bits
        self.machine_balance = machine_balance

    def analyze_complex_boundary(
        self,
        num_cells_N: int,
        nnz_boundary: int,
        estimated_rank: Optional[int] = None,
        override_prime_lanes: Optional[int] = None,
    ) -> RooflineAnalysisResult:
        """Analyzes boundary matrix B_k of dimension N with nnz non-zero entries."""
        if num_cells_N <= 0:
            raise ValueError("Number of cells N must be > 0")

        # Column density d = nnz / N
        d = max(1.0, float(nnz_boundary) / float(num_cells_N))
        rank = estimated_rank if estimated_rank is not None else num_cells_N

        # Hadamard bit-bound L_max <= 0.5 * rank * log2(rank)
        if rank > 1:
            l_max = 0.5 * float(rank) * math.log2(float(rank))
        else:
            l_max = 1.0

        # Required prime field count L across 62-bit prime lanes
        if override_prime_lanes is not None:
            L = override_prime_lanes
        else:
            L = max(4, math.ceil(l_max / 62.0))

        # Tile size T_max = floor( C_SRAM / (d * (18 + 64 * L) + 16) )
        denom = d * (18.0 + 64.0 * L) + 16.0
        t_max = max(1, int(math.floor(self.c_sram_bits / denom)))

        # Algorithmic arithmetic intensity I_algo = (8 / 9) * L * d
        intensity = (8.0 / 9.0) * float(L) * d

        # Invariant L * d >= 19
        invariant_val = float(L) * d
        invariant_satisfied = invariant_val >= 18.5625
        is_compute_bound = intensity >= self.machine_balance

        if invariant_satisfied:
            msg = (
                f"Roofline Invariant Verified: L*d = {invariant_val:.2f} >= 19. "
                f"Arithmetic intensity {intensity:.2f} OPs/Byte >= {self.machine_balance:.1f} OPs/Byte "
                f"(Compute-Bound, zero DRAM bandwidth spilling)."
            )
        else:
            msg = (
                f"Roofline Invariant Violated: L*d = {invariant_val:.2f} < 19. "
                f"Arithmetic intensity {intensity:.2f} OPs/Byte < {self.machine_balance:.1f} OPs/Byte "
                f"(Memory-Bound, predict DRAM bottleneck). Increase tile size or prime lanes."
            )

        return RooflineAnalysisResult(
            num_cells_N=num_cells_N,
            estimated_rank=rank,
            column_density_d=d,
            hadamard_bit_bound_L_max=l_max,
            required_prime_lanes_L=L,
            max_tile_size_T=t_max,
            arithmetic_intensity=intensity,
            machine_balance_threshold=self.machine_balance,
            is_compute_bound=is_compute_bound,
            invariant_satisfied=invariant_satisfied,
            diagnostic_message=msg,
        )

    def verify_complex(self, complex_obj: CellComplex, dim: int = 1) -> RooflineAnalysisResult:
        """Statically verifies a CellComplex at dimension dim."""
        cells_k = complex_obj.get_cells(dim)
        N = len(cells_k)
        if N == 0:
            return self.analyze_complex_boundary(1, 1)

        # Count boundary terms
        nnz = sum(len(c.boundary().terms) for c in cells_k)
        return self.analyze_complex_boundary(N, nnz)
