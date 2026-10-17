"""Dual-Tier Spatial Wavefront Scheduler and Lock-Free Commutativity Arbiter.

Implements Theorem 2.2 from SSRN-7618742:
1. Downward support closure cl(e^k) recursively traversing boundary faces down to 0-cells.
2. Total spatial footprint supp(r) = cl(c_lhs) u cl(c_rhs).
3. Dual-Tier Interference Predicate:
   Interference(r1, r2) = (supp(r1) & supp(r2) != empty) & (S(r1) ^ S(r2) != 0)
   If Interference(r1, r2) == False, r1 and r2 commute identically (r1 o r2 = r2 o r1).
4. Lock-free spatial wavefront partitioning for concurrent parallel execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from topos.core.cell import Cell
from topos.core.clifford import Multivector, PGA301
from topos.core.complex import CellComplex


@dataclass(frozen=True)
class RewriteRule:
    """Cellular homotopy rewrite rule r: c_lhs => c_rhs with geometric multivector blade state."""
    name: str
    lhs_cells: Tuple[str, ...]
    rhs_cells: Tuple[str, ...]
    blade_state: Optional[Multivector] = None

    def discrete_support(self, complex_obj: CellComplex) -> FrozenSet[str]:
        """Computes supp(r) = cl(c_lhs) u cl(c_rhs) down to 0-cells."""
        footprint: Set[str] = set()

        def add_downward_closure(cell_name: str) -> None:
            if cell_name in footprint:
                return
            footprint.add(cell_name)
            cell = complex_obj.get_cell(cell_name)
            if cell is not None and hasattr(cell, "boundary"):
                b = cell.boundary()
                for term in b.terms:
                    add_downward_closure(term.name)

        for c in self.lhs_cells:
            add_downward_closure(c)
        for c in self.rhs_cells:
            add_downward_closure(c)

        return frozenset(footprint)


class DualTierWavefrontScheduler:
    """Arbiter for lock-free parallel execution of cellular rewrite rules."""

    def __init__(self, complex_obj: CellComplex) -> None:
        self.complex = complex_obj

    def check_interference(self, r1: RewriteRule, r2: RewriteRule) -> bool:
        """Evaluates Dual-Tier Interference Predicate:
        
        Interference(r1, r2) = (supp(r1) & supp(r2) != empty) & (S(r1) ^ S(r2) != 0).
        Returns True if rules interfere (must serialize), False if they commute.
        """
        supp1 = r1.discrete_support(self.complex)
        supp2 = r2.discrete_support(self.complex)

        # Tier 1: Check discrete support overlap
        disjoint_support = len(supp1.intersection(supp2)) == 0
        if disjoint_support:
            # Completely disjoint spatial support: zero interference
            return False

        # Tier 2: Overlapping discrete cells; check continuous geometric blade meet
        if r1.blade_state is not None and r2.blade_state is not None:
            # Outer wedge product S(r1) ^ S(r2)
            wedge_prod = r1.blade_state.wedge(r2.blade_state)
            is_wedge_zero = max(abs(c) for c in wedge_prod.components) < 1e-12
            if is_wedge_zero:
                # Geometric spans do not intersect in dual space
                return False

        # Overlapping support and non-zero continuous wedge: rules interfere
        return True

    def are_commuting(self, r1: RewriteRule, r2: RewriteRule) -> bool:
        """Returns True if r1 and r2 commute under Theorem 2.2."""
        return not self.check_interference(r1, r2)

    def partition_into_wavefronts(self, rules: List[RewriteRule]) -> List[List[RewriteRule]]:
        """Greedy graph coloring / dependency scheduling into parallel wavefronts.
        
        All rules within each wavefront W_m are mutually non-interfering.
        """
        wavefronts: List[List[RewriteRule]] = []

        for rule in rules:
            placed = False
            for wf in wavefronts:
                # Can rule be added to this wavefront?
                if all(self.are_commuting(rule, existing) for existing in wf):
                    wf.append(rule)
                    placed = True
                    break
            if not placed:
                wavefronts.append([rule])

        return wavefronts
