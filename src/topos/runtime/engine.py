"""Topological Rewrite Engine and Homotopy Application.

Applies higher-dimensional rewrite rules alpha : lhs => rhs as homotopy (n+1)-cells,
recording state transitions and topological trace steps.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import NilpotencyVerifier
from topos.runtime.matcher import PathMatcher
from topos.topology.homology import HomologyEngine


@dataclass(frozen=True)
class RewriteRule:
    """A directional (n+1)-cell rewrite rule alpha : lhs => rhs."""
    name: str
    lhs: Tuple[str, ...]
    rhs: Tuple[str, ...]


@dataclass
class RewriteTraceStep:
    """Record of a single rewrite application step."""
    step_index: int
    rule_name: str
    homotopy_cell_name: str
    betti_before: Dict[int, int]
    betti_after: Dict[int, int]


class RewriteEngine:
    """Executes higher-dimensional rewrite transformations on a CellComplex."""

    def __init__(self) -> None:
        self.rules: Dict[str, RewriteRule] = {}
        self.trace: List[RewriteTraceStep] = []

    def register_rule(self, rule: RewriteRule) -> None:
        self.rules[rule.name] = rule

    def apply_rule(
        self,
        complex_obj: CellComplex,
        rule_name: str,
        step_index: int = 0,
    ) -> Optional[RewriteTraceStep]:
        """Applies a registered rewrite rule to the complex.
        
        Attaches the homotopy 2-cell capping the cycle formed by lhs and rhs,
        reducing the 1st Betti number.
        """
        if rule_name not in self.rules:
            raise KeyError(f"Rewrite rule {rule_name!r} is not registered.")

        rule = self.rules[rule_name]
        lhs_match = PathMatcher.match_path_by_names(complex_obj, rule.lhs)
        rhs_match = PathMatcher.match_path_by_names(complex_obj, rule.rhs)

        if not lhs_match or not rhs_match:
            return None

        # Verify parallel boundaries: source(lhs) == source(rhs) and target(lhs) == target(rhs)
        if (
            lhs_match.start_vertex != rhs_match.start_vertex
            or lhs_match.end_vertex != rhs_match.end_vertex
        ):
            raise ValueError(
                f"Cannot apply rewrite {rule_name}: source/target boundaries are not parallel."
            )

        # Snapshot Betti numbers before
        betti_before = HomologyEngine(complex_obj).betti_profile()

        # Build chains for lhs and rhs
        lhs_chain = Chain.zero(1)
        for c in lhs_match.matched_cells:
            lhs_chain = lhs_chain + Chain.from_cell(c)

        rhs_chain = Chain.zero(1)
        for c in rhs_match.matched_cells:
            rhs_chain = rhs_chain + Chain.from_cell(c)

        # Attach homotopy 2-cell: alpha : lhs => rhs
        homotopy_name = f"homotopy_{rule.name}_{step_index}"
        homotopy_cell = GlobularCell(
            homotopy_name,
            dim=2,
            source=lhs_chain,
            target=rhs_chain,
        )

        # Verify nilpotency before attaching
        NilpotencyVerifier.verify_cell(homotopy_cell)
        complex_obj.add_cell(homotopy_cell)

        # Snapshot Betti numbers after
        betti_after = HomologyEngine(complex_obj).betti_profile()

        trace_step = RewriteTraceStep(
            step_index=step_index,
            rule_name=rule_name,
            homotopy_cell_name=homotopy_name,
            betti_before=betti_before,
            betti_after=betti_after,
        )
        self.trace.append(trace_step)
        return trace_step
