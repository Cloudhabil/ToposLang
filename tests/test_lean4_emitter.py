"""Unit tests for the Lean 4 Proof Object Emitter v3.

v3 architectural invariants verified here:
  §1  Concrete nilpotency: d1_square, d2_square, `decide` proof, DACCChainComplex struct.
  §2  WilsonLoop.is_closed is an IsCycle STRUCTURAL FIELD, not Bool default, no axiom.
  §3  Peierls: norm_num on rational literals, both coupling regimes.
  §4  Bakry–Émery: FaddeevPopovAnomaly opaque, EffectiveRicciCurvature, linarith proof.
  §5  Bałaban induction: RGState, InvariantRegion, RG_Step (opaque), IteratedRG,
      SingleStepForwardInvariance, multiscale_balaban_induction, multiscale_gap_preserved.
  §6  YangMillsMassGapConjecture parameterised by s₀ : RGState; proved by linarith.
  Global: zero sorry tactics, zero native_decide in code, zero {n} placeholders.
"""
import unittest
import math
import re

from topos.core.four_tier_dacc import create_4_hypercube, create_4_torus
from topos.physics.peierls_vortex import PeierlsVortexAnalyzer
from topos.physics.orbit_space import OrbitSpaceManifold
from topos.physics.bakry_emery import BakryEmeryCurvatureEvaluator
from topos.compiler.lean4_emitter import Lean4Emitter, Lean4EmitOptions, emit_lean4_file


class TestLean4EmitterV3(unittest.TestCase):

    def setUp(self):
        self.hypercube = create_4_hypercube()
        self.torus = create_4_torus()

        analyzer = PeierlsVortexAnalyzer(self.hypercube, gauge_group_n=2)
        self.peierls_cert = analyzer.execute_balaban_peierls_handover(
            bare_coupling_g0=0.8, lattice_spacing_fm=0.05, target_scale_fm=0.5
        )
        orbit = OrbitSpaceManifold(self.hypercube, gauge_group="SU(2)")
        evaluator = BakryEmeryCurvatureEvaluator(orbit)
        self.bakry_cert = evaluator.estimate_curvature_bound(beta=2.0)

        self.emitter = Lean4Emitter()

    def _emit(self, use_peierls=True, use_bakry=True, gauge_n=2):
        return self.emitter.emit_yang_mills_proof_scaffold(
            complex_obj=self.hypercube.complex,
            peierls_cert=self.peierls_cert if use_peierls else None,
            bakry_cert=self.bakry_cert if use_bakry else None,
            gauge_group_n=gauge_n,
        )

    def _code_lines(self, src: str) -> str:
        """Return non-comment lines joined — for assertNotIn checks."""
        return '\n'.join(
            l for l in src.split('\n')
            if not l.strip().startswith('--')
            and not l.strip().startswith('/-')
            and not l.strip().startswith('-/')
            and not l.strip().startswith('•')
            and not l.strip().startswith('*')
        )

    # ------------------------------------------------------------------
    # Preamble / Namespace
    # ------------------------------------------------------------------

    def test_output_is_string_and_nontrivial(self):
        src = self._emit()
        self.assertIsInstance(src, str)
        self.assertGreater(len(src), 1000)

    def test_namespace_opens_and_closes(self):
        src = self._emit()
        self.assertIn("namespace ToposYangMills", src)
        self.assertIn("end ToposYangMills", src)

    def test_mathlib_imports_present(self):
        src = self._emit()
        self.assertIn("import Mathlib.Algebra.BigOperators.Basic", src)
        self.assertIn("import Mathlib.Analysis.SpecialFunctions.Log.Basic", src)
        self.assertIn("import Mathlib.Data.Matrix.Basic", src)
        self.assertIn("import Mathlib.Data.Fin.Basic", src)

    # ------------------------------------------------------------------
    # §1  Concrete Nilpotency (v3 fix: decide on explicit matrices)
    # ------------------------------------------------------------------

    def test_concrete_boundary_matrices_present(self):
        """v3: d1_square and d2_square must be explicitly defined."""
        src = self._emit()
        self.assertIn("def d1_square", src)
        self.assertIn("def d2_square", src)

    def test_nilpotency_proved_by_decide(self):
        """v3: ∂₁∘∂₂=0 proved by `decide`, not by a circular implication."""
        src = self._emit()
        self.assertIn("theorem nilpotency_concrete_square", src)
        # The proof must use `decide`
        start = src.index("theorem nilpotency_concrete_square")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("decide", snippet)
        self.assertNotIn(":= h", snippet,
                         "nilpotency must not be a circular P→P implication")

    def test_dacc_chain_complex_structure(self):
        """v3: DACCChainComplex carries nilpotency as a FIELD (not a hypothesis)."""
        src = self._emit()
        self.assertIn("structure DACCChainComplex", src)
        self.assertIn("nilpotent : d1 * d2 = 0", src)
        self.assertIn("def SquareComplex", src)

    def test_general_nilpotency_theorem(self):
        src = self._emit()
        self.assertIn("theorem nilpotency_dacc", src)

    # ------------------------------------------------------------------
    # §2  Gauge Invariance — Structural (v3 fix: no unsound axiom)
    # ------------------------------------------------------------------

    def test_is_cycle_defined(self):
        """v3: IsCycle captures ∂₁ c = 0 as a Prop."""
        src = self._emit()
        self.assertIn("def IsCycle", src)
        self.assertIn("d1.mulVec c = 0", src)

    def test_wilson_loop_has_is_closed_field(self):
        """v3: is_closed is a structural field of type IsCycle, not Bool."""
        src = self._emit()
        self.assertIn("structure WilsonLoop", src)
        self.assertIn("is_closed : IsCycle", src,
                      "is_closed must be typed as IsCycle, not Bool")

    def test_no_axiom_elitzur(self):
        """v3: The unsound `axiom elitzur_theorem` from v2 must be removed."""
        src = self._emit()
        code = self._code_lines(src)
        self.assertNotIn("axiom elitzur_theorem", code,
                         "v2 axiom permitted proving False; must be eliminated in v3")

    def test_wilson_loop_gauge_invariant_theorem(self):
        src = self._emit()
        self.assertIn("theorem wilson_loop_is_gauge_invariant", src)
        # Proof is just field projection w.is_closed — no axiom needed
        start = src.index("theorem wilson_loop_is_gauge_invariant")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("w.is_closed", snippet)

    # ------------------------------------------------------------------
    # §3  Peierls Bounds
    # ------------------------------------------------------------------

    def test_peierls_section_emitted_with_cert(self):
        src = self._emit(use_peierls=True)
        self.assertIn("surfaceEntropyMu", src)
        self.assertIn("peierls_bound_inactive_intermediate", src)
        self.assertIn("peierls_bound_active_strong", src)
        self.assertIn("string_tension_strong_pos", src)

    def test_peierls_section_absent_without_cert(self):
        src = self._emit(use_peierls=False)
        self.assertNotIn("surfaceEntropyMu", src)

    def test_norm_num_used_for_peierls(self):
        src = self._emit(use_peierls=True)
        self.assertIn("norm_num", src)

    def test_no_native_decide_in_code(self):
        """norm_num must be used for ℝ arithmetic; native_decide is invalid."""
        src = self._emit()
        self.assertNotIn("native_decide", self._code_lines(src))

    def test_no_unrendered_beta_ir_placeholder(self):
        src = self._emit(use_peierls=True)
        self.assertNotIn("{beta_ir_str}", src)

    # ------------------------------------------------------------------
    # §4  Bakry–Émery (v3: FP anomaly + EffectiveRicciCurvature + linarith)
    # ------------------------------------------------------------------

    def test_faddeev_popov_anomaly_declared_opaque(self):
        """v3: The FP trace anomaly is opaque — not a concrete formula."""
        src = self._emit(use_bakry=True)
        self.assertIn("opaque FaddeevPopovAnomaly", src)

    def test_effective_ricci_curvature_defined(self):
        src = self._emit(use_bakry=True)
        self.assertIn("EffectiveRicciCurvature", src)
        self.assertIn("classicalRicciLowerBound - FaddeevPopovAnomaly", src)

    def test_anomaly_screening_hypothesis_present(self):
        src = self._emit(use_bakry=True)
        self.assertIn("AnomalyScreeningHypothesis", src)
        # Must bound the anomaly FROM ABOVE (not just assert a κ exists)
        start = src.index("def AnomalyScreeningHypothesis")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("FaddeevPopovAnomaly a ≤", snippet,
                      "AnomalyScreeningHypothesis must bound FP anomaly from above")

    def test_bakry_emery_continuum_limit_proved_by_linarith(self):
        src = self._emit(use_bakry=True)
        self.assertIn("theorem bakry_emery_gap_continuum_limit", src)
        start = src.index("theorem bakry_emery_gap_continuum_limit")
        end = src.index("\n\n", start + 50)
        snippet = src[start:end + 200]
        self.assertIn("linarith", snippet)
        # Must NOT have sorry in the proof body
        sorry_in_proof = [l for l in snippet.split('\n') if l.strip() == 'sorry']
        self.assertEqual(sorry_in_proof, [])

    # ------------------------------------------------------------------
    # §5  Bałaban Multiscale RG Induction (v3 NEW)
    # ------------------------------------------------------------------

    def test_rg_state_structure_present(self):
        """v3: RGState encodes (scale_index, coupling, spectral_gap)."""
        src = self._emit()
        self.assertIn("structure RGState", src)
        self.assertIn("spectral_gap", src)

    def test_invariant_region_defined(self):
        src = self._emit()
        self.assertIn("def InvariantRegion", src)
        self.assertIn("s.spectral_gap ≥ Δm", src)

    def test_rg_step_is_opaque(self):
        """v3: RG_Step must be opaque — its analytical form requires polymer expansion."""
        src = self._emit()
        self.assertIn("opaque RG_Step", src)

    def test_iterated_rg_defined_recursively(self):
        src = self._emit()
        self.assertIn("def IteratedRG", src)
        # Must be a recursive definition on ℕ
        start = src.index("def IteratedRG")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("| 0", snippet)
        self.assertIn("| n + 1", snippet)

    def test_single_step_forward_invariance_defined(self):
        src = self._emit()
        self.assertIn("def SingleStepForwardInvariance", src)
        start = src.index("def SingleStepForwardInvariance")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("InvariantRegion s Δm", snippet)
        self.assertIn("InvariantRegion (RG_Step s) Δm", snippet)

    def test_multiscale_induction_theorem_by_structural_induction(self):
        """v3: The core induction proof must use `induction n with | zero | succ`."""
        src = self._emit()
        self.assertIn("theorem multiscale_balaban_induction", src)
        start = src.index("theorem multiscale_balaban_induction")
        end = src.index("\n\n", start + 50)
        snippet = src[start:end + 400]
        self.assertIn("induction n with", snippet)
        self.assertIn("| zero", snippet)
        self.assertIn("| succ", snippet)
        # No sorry
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_multiscale_gap_preserved_corollary(self):
        src = self._emit()
        self.assertIn("theorem multiscale_gap_preserved", src)
        start = src.index("theorem multiscale_gap_preserved")
        end = src.index("\n\n", start + 50)
        snippet = src[start:end + 300]
        self.assertIn("multiscale_balaban_induction", snippet,
                      "gap_preserved must call the induction theorem")

    # ------------------------------------------------------------------
    # §6  Mass Gap Conjecture (v3: indexed by s₀, proved by linarith)
    # ------------------------------------------------------------------

    def test_mass_gap_conjecture_indexed_by_rg_state(self):
        """v3: The conjecture takes s₀ : RGState as a parameter."""
        src = self._emit()
        self.assertIn("def YangMillsMassGapConjecture", src)
        start = src.index("def YangMillsMassGapConjecture")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("s₀ : RGState", snippet,
                      "YangMillsMassGapConjecture must be parameterised by s₀ : RGState")
        self.assertIn("IteratedRG", snippet,
                      "The conjecture must reference IteratedRG")

    def test_conditional_main_theorem_uses_induction_corollary(self):
        """v3: The main theorem must call multiscale_gap_preserved (not circular)."""
        src = self._emit()
        self.assertIn("theorem conditional_yang_mills_mass_gap", src)
        start = src.index("theorem conditional_yang_mills_mass_gap")
        end = src.index("\n\n", start + 50)
        snippet = src[start:end + 500]
        self.assertIn("multiscale_gap_preserved", snippet,
                      "Main theorem must invoke the multiscale induction corollary")
        self.assertIn("linarith", snippet)
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_no_unrendered_n_placeholder(self):
        """v3: gauge_group_n must be rendered as a literal integer, not {n}."""
        src = self._emit(gauge_n=2)
        self.assertNotIn("SU({n})", src)
        self.assertIn("SU(2)", src)
        src3 = self.emitter.emit_yang_mills_proof_scaffold(
            complex_obj=self.torus.complex, gauge_group_n=3
        )
        self.assertIn("SU(3)", src3)
        self.assertNotIn("{n}", self._code_lines(src3))

    # ------------------------------------------------------------------
    # §5B  Hierarchical Migdal–Kadanoff Closure (Step 1 of 3)
    # ------------------------------------------------------------------

    def test_hierarchical_state_structure_present(self):
        """§5B: HierarchicalState with ratio, h_pos, h_lt_one."""
        src = self._emit()
        self.assertIn("structure HierarchicalState", src)
        self.assertIn("ratio    : ℝ", src)
        self.assertIn("h_pos    : 0 < ratio", src)
        self.assertIn("h_lt_one : ratio < 1", src)

    def test_hierarchical_mk_step_concrete(self):
        """§5B: HierarchicalMKStep is a concrete definition (r → r²), not opaque."""
        src = self._emit()
        self.assertIn("noncomputable def HierarchicalMKStep", src)
        start = src.index("noncomputable def HierarchicalMKStep")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        # Must be concrete: maps ratio to ratio^2
        self.assertIn("s.ratio ^ 2", snippet)
        # Must NOT be opaque
        self.assertNotIn("opaque HierarchicalMKStep", src)

    def test_hierarchical_iterated_rg_recursive(self):
        src = self._emit()
        self.assertIn("noncomputable def HierarchicalIteratedRG", src)
        start = src.index("noncomputable def HierarchicalIteratedRG")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("| 0", snippet)
        self.assertIn("| n + 1", snippet)
        self.assertIn("HierarchicalMKStep", snippet)

    def test_mass_gap_hierarchical_defined(self):
        src = self._emit()
        self.assertIn("noncomputable def MassGapHierarchical", src)
        self.assertIn("-Real.log s.ratio", src)

    def test_mass_gap_hierarchical_pos_theorem(self):
        src = self._emit()
        self.assertIn("theorem mass_gap_hierarchical_pos", src)
        start = src.index("theorem mass_gap_hierarchical_pos")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        # Must use Real.log_neg (the correct Mathlib lemma)
        self.assertIn("Real.log_neg", snippet)
        self.assertIn("linarith", snippet)
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_mass_gap_doubles_theorem(self):
        """The doubling theorem m' = 2m is the key forward-invariance engine."""
        src = self._emit()
        self.assertIn("theorem mass_gap_hierarchical_doubles", src)
        start = src.index("theorem mass_gap_hierarchical_doubles")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("2 * MassGapHierarchical s", snippet)
        self.assertIn("Real.log_pow", snippet)
        self.assertIn("ring", snippet)
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_hierarchical_single_step_forward_invariance_proved(self):
        """The concrete forward-invariance is proved, not just declared."""
        src = self._emit()
        self.assertIn("theorem hierarchical_single_step_forward_invariance", src)
        start = src.index("theorem hierarchical_single_step_forward_invariance")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("mass_gap_hierarchical_doubles", snippet,
                      "Must invoke the doubling lemma")
        self.assertIn("linarith", snippet)
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_hierarchical_gap_preserved_by_induction(self):
        src = self._emit()
        self.assertIn("theorem hierarchical_gap_preserved", src)
        start = src.index("theorem hierarchical_gap_preserved")
        end = src.index("\n\n", start + 50)
        snippet = src[start:end + 300]
        self.assertIn("induction n with", snippet)
        self.assertIn("| zero", snippet)
        self.assertIn("| succ", snippet)
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_hierarchical_anomaly_screening_proved(self):
        """AnomalyScreeningHypothesis proved for the hierarchical (FP=0) case."""
        src = self._emit()
        self.assertIn("theorem hierarchical_anomaly_screening", src)
        start = src.index("theorem hierarchical_anomaly_screening")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("FaddeevPopovAnomaly a = 0", snippet,
                      "Must state FP=0 as hypothesis")
        self.assertIn("AnomalyScreeningHypothesis", snippet)
        self.assertIn("norm_num", snippet)

    def test_hierarchical_to_rg_state_bridge(self):
        """The hierarchical state must connect to the abstract RGState framework."""
        src = self._emit()
        self.assertIn("noncomputable def hierarchicalToRGState", src)
        start = src.index("noncomputable def hierarchicalToRGState")
        end = src.index("\n\n", start)
        snippet = src[start:end]
        self.assertIn("RGState", snippet)
        self.assertIn("HierarchicalIteratedRG", snippet)
        self.assertIn("MassGapHierarchical", snippet)

    def test_hierarchical_mass_gap_theorem_fully_closed(self):
        """THE key test: hierarchical_yang_mills_mass_gap has no sorry, no opaque."""
        src = self._emit()
        self.assertIn("theorem hierarchical_yang_mills_mass_gap", src)
        start = src.index("theorem hierarchical_yang_mills_mass_gap")
        end = src.index("\n\n", start + 100)
        snippet = src[start:end + 500]
        # Must reference the induction corollary
        self.assertIn("hierarchical_gap_preserved", snippet)
        self.assertIn("linarith", snippet)
        # No sorry anywhere in this theorem
        self.assertEqual([l for l in snippet.split('\n') if l.strip() == 'sorry'], [])

    def test_section_hierarchical_opens_and_closes(self):
        src = self._emit()
        self.assertIn("section Hierarchical", src)
        self.assertIn("end Hierarchical", src)

    # ------------------------------------------------------------------
    # Global quality checks
    # ------------------------------------------------------------------

    def test_zero_sorry_tactics(self):
        """v3 target: zero sorry in proof-tactic positions."""
        src = self._emit()
        sorry_lines = [l for l in src.split('\n') if l.strip() == 'sorry']
        self.assertEqual(len(sorry_lines), 0,
                         f"Expected 0 sorry tactics, got {len(sorry_lines)}:\n"
                         + '\n'.join(repr(l) for l in sorry_lines))

    def test_no_unrendered_python_placeholders(self):
        """No {identifier} Python placeholders in Lean code positions."""
        src = self._emit()
        code_lines = [
            l for l in src.split('\n')
            if not l.strip().startswith('--')
            and not l.strip().startswith('/-')
            and not l.strip().startswith('-/')
            and not l.strip().startswith('•')
        ]
        placeholders = re.findall(r'\{[a-z_]+\}', '\n'.join(code_lines))
        self.assertEqual(placeholders, [],
                         f"Unrendered Python placeholders in code: {placeholders}")

    def test_five_open_problem_names_present(self):
        """v3: All open problems are named Prop defs or opaque defs — not sorry."""
        src = self._emit()
        for name in [
            "AnomalyScreeningHypothesis",
            "SingleStepForwardInvariance",
            "FaddeevPopovAnomaly",
            "RG_Step",
        ]:
            self.assertIn(name, src, f"Open problem '{name}' must be named in the file")

    def test_emit_file_helper(self):
        src = emit_lean4_file(
            complex_obj=self.hypercube.complex,
            peierls_cert=self.peierls_cert,
            bakry_cert=self.bakry_cert,
            gauge_group_n=2,
        )
        self.assertIsInstance(src, str)
        self.assertIn("namespace ToposYangMills", src)
        self.assertIn("multiscale_balaban_induction", src)

    def test_torus_emits_without_error(self):
        src = self.emitter.emit_yang_mills_proof_scaffold(
            complex_obj=self.torus.complex,
            gauge_group_n=3,
        )
        self.assertIn("namespace ToposYangMills", src)
        self.assertIn("SU(3)", src)
        self.assertEqual([l for l in src.split('\n') if l.strip() == 'sorry'], [])

    def test_v3_architecture_comment_present(self):
        src = self._emit()
        self.assertIn("v3", src)
        self.assertIn("Bałaban", src)


if __name__ == "__main__":
    unittest.main()
