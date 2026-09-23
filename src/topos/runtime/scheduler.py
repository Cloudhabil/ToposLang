"""Spatial Boundary Partitioner and Lock-Free Concurrent Scheduler.

Partitions candidate rewrite rules into mutually disjoint spatial wavefronts
and dispatches them concurrently across worker threads with zero lock contention.
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Set, Tuple

from topos.core.cell import Cell, GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import NilpotencyVerifier
from topos.runtime.engine import RewriteRule, RewriteTraceStep
from topos.runtime.matcher import PathMatcher
from topos.topology.homology import HomologyEngine


class SpatialPartitioner:
    """Computes topological support sets and partitions rewrites into disjoint wavefronts."""

    @staticmethod
    def compute_support(complex_obj: CellComplex, rule: RewriteRule) -> Set[str]:
        """Computes the full topological closure down to 0-cells for a rewrite rule."""
        support: Set[str] = set()

        def add_cell_closure(cell: Cell) -> None:
            support.add(cell.name)
            if cell.dim > 0:
                b_chain = cell.boundary()
                for face_cell in b_chain.terms.keys():
                    add_cell_closure(face_cell)

        # Include all cells in LHS and RHS
        for name in rule.lhs + rule.rhs:
            try:
                cell = complex_obj.get_cell(name, dim=1)
                add_cell_closure(cell)
            except Exception:
                pass

        return support

    @classmethod
    def partition_wavefronts(
        cls,
        complex_obj: CellComplex,
        rules: List[RewriteRule],
    ) -> List[List[RewriteRule]]:
        """Partitions candidate rewrite rules into sequential wavefronts of mutually disjoint rules."""
        if not rules:
            return []

        rule_supports = [(r, cls.compute_support(complex_obj, r)) for r in rules]
        wavefronts: List[List[RewriteRule]] = []
        wavefront_supports: List[Set[str]] = []

        for rule, supp in rule_supports:
            placed = False
            for idx, wf_supp in enumerate(wavefront_supports):
                if supp.isdisjoint(wf_supp):
                    wavefronts[idx].append(rule)
                    wf_supp.update(supp)
                    placed = True
                    break

            if not placed:
                wavefronts.append([rule])
                wavefront_supports.append(set(supp))

        return wavefronts


class SpatialScheduler:
    """Dispatches mutually disjoint rewrites across worker threads with 0 lock contention."""

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self.trace: List[RewriteTraceStep] = []

    def _worker_construct_homotopy(
        self,
        complex_obj: CellComplex,
        rule: RewriteRule,
        step_index: int,
    ) -> Tuple[GlobularCell, RewriteRule, int]:
        """Worker thread task: safely constructs the homotopy 2-cell in isolated spatial partition."""
        lhs_match = PathMatcher.match_path_by_names(complex_obj, rule.lhs)
        rhs_match = PathMatcher.match_path_by_names(complex_obj, rule.rhs)

        if not lhs_match or not rhs_match:
            raise ValueError(f"Rule {rule.name} could not match paths.")

        lhs_chain = Chain.zero(1)
        for c in lhs_match.matched_cells:
            lhs_chain = lhs_chain + Chain.from_cell(c)

        rhs_chain = Chain.zero(1)
        for c in rhs_match.matched_cells:
            rhs_chain = rhs_chain + Chain.from_cell(c)

        homotopy_name = f"homotopy_parallel_{rule.name}_{step_index}"
        cell = GlobularCell(homotopy_name, dim=2, source=lhs_chain, target=rhs_chain)
        NilpotencyVerifier.verify_cell(cell)
        return (cell, rule, step_index)

    def execute_wavefront(
        self,
        complex_obj: CellComplex,
        wavefront: List[RewriteRule],
        step_offset: int = 0,
    ) -> List[RewriteTraceStep]:
        """Executes a wavefront of mutually disjoint rewrite rules concurrently."""
        if not wavefront:
            return []

        betti_before = HomologyEngine(complex_obj).betti_profile()
        step_results: List[RewriteTraceStep] = []

        # Execute parallel worker tasks
        workers = min(len(wavefront), self.max_workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._worker_construct_homotopy, complex_obj, rule, step_offset + i)
                for i, rule in enumerate(wavefront)
            ]
            homotopy_cells = [f.result() for f in futures]

        # Lock-free spatial merge into global complex
        for cell, rule, s_idx in homotopy_cells:
            complex_obj.add_cell(cell)

        # Validate whole complex nilpotence after merge
        complex_obj.validate(enforce_nilpotence=True)
        betti_after = HomologyEngine(complex_obj).betti_profile()

        for cell, rule, s_idx in homotopy_cells:
            trace_step = RewriteTraceStep(
                step_index=s_idx,
                rule_name=rule.name,
                homotopy_cell_name=cell.name,
                betti_before=betti_before,
                betti_after=betti_after,
            )
            step_results.append(trace_step)
            self.trace.append(trace_step)

        return step_results

    def execute_all_parallel(
        self,
        complex_obj: CellComplex,
        rules: List[RewriteRule],
    ) -> List[RewriteTraceStep]:
        """Partitions rules into disjoint wavefronts and executes each in parallel."""
        wavefronts = SpatialPartitioner.partition_wavefronts(complex_obj, rules)
        all_steps: List[RewriteTraceStep] = []
        offset = 0

        for wf in wavefronts:
            steps = self.execute_wavefront(complex_obj, wf, step_offset=offset)
            all_steps.extend(steps)
            offset += len(wf)

        return all_steps
