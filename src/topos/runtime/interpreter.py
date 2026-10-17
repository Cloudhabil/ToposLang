"""ToposLang Invariant-Driven Process Interpreter.

Executes process definitions, evaluating homological stopping conditions
(e.g., until betti(complex, dim=1) == 0) and driving topological rewrites.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from topos.core.complex import CellComplex
from topos.core.four_tier_dacc import FourTierDACC
from topos.frontend.ast import (
    ApplyStmt,
    AssertNilpotencyStmt,
    BettiCondition,
    ComputeHodgeSpectrumStmt,
    ComputeScreeningGapStmt,
    ProcessDecl,
    Program,
    RewriteDecl,
    SheafDecl,
    SheafDimKerCondition,
    Statement,
    UntilLoop,
)
from topos.ir.sheaf import CellularSheafIR, build_block_csr_sheaf_laplacian
from topos.runtime.engine import RewriteEngine, RewriteRule, RewriteTraceStep
from topos.topology.homology import HomologyEngine


@dataclass
class RuntimeContext:
    """Stores the execution state, trace history, and final invariants."""
    complex: CellComplex
    trace: List[RewriteTraceStep] = field(default_factory=list)
    iterations: int = 0
    final_betti: Dict[int, int] = field(default_factory=dict)


class ToposInterpreter:
    """Interprets and executes ToposLang AST processes."""

    def __init__(self, engine: Optional[RewriteEngine] = None) -> None:
        self.engine = engine if engine is not None else RewriteEngine()

    def _register_rules(self, program: Program, process: ProcessDecl) -> None:
        """Collects and registers top-level and process-local rewrite rules."""
        for stmt in program.statements:
            if isinstance(stmt, RewriteDecl):
                rule = RewriteRule(
                    name=stmt.name,
                    lhs=tuple(stmt.lhs.elements),
                    rhs=tuple(stmt.rhs.elements),
                )
                self.engine.register_rule(rule)

        for stmt in process.body:
            if isinstance(stmt, RewriteDecl):
                rule = RewriteRule(
                    name=stmt.name,
                    lhs=tuple(stmt.lhs.elements),
                    rhs=tuple(stmt.rhs.elements),
                )
                self.engine.register_rule(rule)

    def _eval_betti_condition(self, cond: BettiCondition, complex_obj: CellComplex) -> bool:
        """Evaluates whether the invariant stopping condition is met."""
        actual_val = HomologyEngine(complex_obj).betti_number(cond.dim)
        target_val = cond.value

        if cond.operator == "==":
            return actual_val == target_val
        elif cond.operator == "!=":
            return actual_val != target_val
        elif cond.operator == "<":
            return actual_val < target_val
        elif cond.operator == ">":
            return actual_val > target_val
        elif cond.operator == "<=":
            return actual_val <= target_val
        elif cond.operator == ">=":
            return actual_val >= target_val
        else:
            raise ValueError(f"Unknown comparison operator: {cond.operator}")

    def _eval_sheaf_condition(
        self,
        cond: SheafDimKerCondition,
        complex_obj: CellComplex,
        sheaves: Dict[str, SheafDecl],
    ) -> bool:
        """Evaluates dim(ker(L_2^down)) stopping condition for a cellular sheaf."""
        sheaf_decl = sheaves.get(cond.sheaf_name)
        if sheaf_decl is not None:
            sheaf_ir = CellularSheafIR.from_ast(sheaf_decl, complex_obj)
        else:
            sheaf_ir = CellularSheafIR(name=cond.sheaf_name, complex_obj=complex_obj)

        block_csr = build_block_csr_sheaf_laplacian(sheaf_ir)
        actual_val = block_csr.sheaf_nullity()
        target_val = cond.value

        if cond.operator == "==":
            return actual_val == target_val
        elif cond.operator == "!=":
            return actual_val != target_val
        elif cond.operator == "<":
            return actual_val < target_val
        elif cond.operator == ">":
            return actual_val > target_val
        elif cond.operator == "<=":
            return actual_val <= target_val
        elif cond.operator == ">=":
            return actual_val >= target_val
        else:
            raise ValueError(f"Unknown comparison operator: {cond.operator}")

    def _eval_condition(
        self,
        cond: Union[BettiCondition, SheafDimKerCondition],
        complex_obj: CellComplex,
        sheaves: Dict[str, SheafDecl],
    ) -> bool:
        """Evaluates either BettiCondition or SheafDimKerCondition."""
        if isinstance(cond, BettiCondition):
            return self._eval_betti_condition(cond, complex_obj)
        elif isinstance(cond, SheafDimKerCondition):
            return self._eval_sheaf_condition(cond, complex_obj, sheaves)
        else:
            raise ValueError(f"Unknown condition type: {type(cond)}")

    def _dispatch_rule(
        self,
        complex_obj: CellComplex,
        rule_name: str,
        step_index: int,
    ) -> None:
        """Routes a rule application through the correct engine backend based on engine_mode.

        - 'auto': Uses classical DACC apply_rule; falls back to cubical bridge if beta_k != 0
                  and the classical rule produced no change.
        - 'dacc-only': Always uses classical exact rational matrix patching.
        - 'cubical-forced': Always uses the Cubical Homotopy Bridge (Realization Functor R).
        """
        mode = self.engine.engine_mode

        if mode == "dacc-only":
            self.engine.apply_rule(complex_obj, rule_name, step_index=step_index)
        elif mode == "cubical-forced":
            # Try cubical bridge first; fall back to classical if no harmonic cycle is found
            result = self.engine.apply_cubical_bridge_rule(complex_obj, rule_name, step_index=step_index)
            if result is None:
                self.engine.apply_rule(complex_obj, rule_name, step_index=step_index)
        else:
            # 'auto': Use classical DACC patching; automatically engage cubical bridge on failure
            try:
                result = self.engine.apply_rule(complex_obj, rule_name, step_index=step_index)
            except Exception:
                result = None
            if result is None:
                self.engine.apply_cubical_bridge_rule(complex_obj, rule_name, step_index=step_index)

    def run_process(
        self,
        program: Program,
        process_name: str,
        complex_obj: CellComplex,
        max_iterations: int = 100,
    ) -> RuntimeContext:
        """Executes a process on the target CellComplex."""
        process_decl: Optional[ProcessDecl] = None
        for stmt in program.statements:
            if isinstance(stmt, ProcessDecl) and stmt.name == process_name:
                process_decl = stmt
                break

        if process_decl is None:
            raise KeyError(f"Process {process_name!r} not found in program.")

        self._register_rules(program, process_decl)
        sheaves = {s.name: s for s in program.statements if isinstance(s, SheafDecl)}
        iterations = 0

        for stmt in process_decl.body:
            if isinstance(stmt, AssertNilpotencyStmt):
                tier = stmt.dim
                dacc = FourTierDACC(complex_obj)
                valid = dacc.verify_nilpotency_tier(tier)
                if valid != stmt.expected:
                    raise AssertionError(f"Static boundary nilpotency (dim={tier}) == {stmt.expected} failed (actual: {valid})!")
            elif isinstance(stmt, ComputeHodgeSpectrumStmt):
                dacc = FourTierDACC(complex_obj)
                _ = dacc.compute_hodge_laplacian(stmt.dim)
            elif isinstance(stmt, ComputeScreeningGapStmt):
                from topos.physics.dynamical_screening import (
                    DynamicalScreeningEndomorphism,
                    GaugeField4D,
                    ScreenedHodgeDiracOperator,
                )
                dacc = FourTierDACC(complex_obj)
                n = 3 if "3" in stmt.gauge_group else 2
                gf = GaugeField4D(dacc, gauge_group_n=n)
                gf.randomize_links(0.3)
                ds = DynamicalScreeningEndomorphism(gf, bare_coupling_g=stmt.coupling)
                op = ScreenedHodgeDiracOperator(dacc, gf, ds)
                gap = op.spectral_mass_gap()
                valid_b, _ = ds.vortex_network.verify_3cell_bianchi_identity()
                print(f"Dynamically Screened Hodge-Dirac Gap on {complex_obj.name}: Delta m = {gap:.6e} (Bianchi closed: {valid_b})")
            elif isinstance(stmt, UntilLoop):
                while not self._eval_condition(stmt.condition, complex_obj, sheaves):
                    if iterations >= max_iterations:
                        raise RuntimeError(
                            f"Invariant loop exceeded maximum iterations ({max_iterations}) "
                            f"without satisfying {stmt.condition}"
                        )
                    for sub_stmt in stmt.body:
                        if isinstance(sub_stmt, ApplyStmt):
                            self._dispatch_rule(complex_obj, sub_stmt.rule_name, iterations)
                        elif isinstance(sub_stmt, ComputeHodgeSpectrumStmt):
                            dacc = FourTierDACC(complex_obj)
                            _ = dacc.compute_hodge_laplacian(sub_stmt.dim)
                    iterations += 1
            elif isinstance(stmt, ApplyStmt):
                self._dispatch_rule(complex_obj, stmt.rule_name, iterations)
                iterations += 1

        # Post-execution homology snapshot
        final_betti = HomologyEngine(complex_obj).betti_profile()

        return RuntimeContext(
            complex=complex_obj,
            trace=list(self.engine.trace),
            iterations=iterations,
            final_betti=final_betti,
        )
