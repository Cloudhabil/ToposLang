"""Unit and integration tests for Topological Rewriting Runtime."""

import unittest
from pathlib import Path

from topos.core.cell import GlobularCell
from topos.core.complex import CellComplex
from topos.frontend.parser import parse_source, build_complex
from topos.runtime.matcher import PathMatcher
from topos.runtime.engine import RewriteEngine, RewriteRule
from topos.runtime.interpreter import ToposInterpreter
from topos.topology.homology import HomologyEngine


class TestRuntimeRewriting(unittest.TestCase):
    def setUp(self) -> None:
        # Create a hollow triangle complex:
        # e1: A -> B, e2: B -> C, e3: A -> C
        self.vA = GlobularCell("A", dim=0)
        self.vB = GlobularCell("B", dim=0)
        self.vC = GlobularCell("C", dim=0)

        self.e1 = GlobularCell("e1", dim=1, source=self.vA, target=self.vB)
        self.e2 = GlobularCell("e2", dim=1, source=self.vB, target=self.vC)
        self.e3 = GlobularCell("e3", dim=1, source=self.vA, target=self.vC)

        self.triangle = CellComplex("HollowTriangle")
        for v in (self.vA, self.vB, self.vC):
            self.triangle.add_cell(v)
        for e in (self.e1, self.e2, self.e3):
            self.triangle.add_cell(e)

    def test_path_matcher(self) -> None:
        # Valid continuous path (e1 * e2)
        match = PathMatcher.match_path_by_names(self.triangle, ("e1", "e2"))
        self.assertIsNotNone(match)
        self.assertEqual(match.start_vertex, self.vA)
        self.assertEqual(match.end_vertex, self.vC)
        self.assertEqual(len(match.matched_cells), 2)

        # Disconnected path: e1 (A->B) and e3 (A->C)
        invalid_match = PathMatcher.match_path_by_names(self.triangle, ("e1", "e3"))
        self.assertIsNone(invalid_match)

    def test_rewrite_engine_manual_application(self) -> None:
        engine = RewriteEngine()
        rule = RewriteRule(name="contract_loop", lhs=("e1", "e2"), rhs=("e3",))
        engine.register_rule(rule)

        # Before rewrite: beta_1 = 1
        homology_before = HomologyEngine(self.triangle)
        self.assertEqual(homology_before.betti_number(1), 1)

        trace_step = engine.apply_rule(self.triangle, "contract_loop")
        self.assertIsNotNone(trace_step)
        self.assertEqual(trace_step.betti_before[1], 1)
        self.assertEqual(trace_step.betti_after[1], 0)

        # After rewrite: beta_1 = 0
        homology_after = HomologyEngine(self.triangle)
        self.assertEqual(homology_after.betti_number(1), 0)
        self.assertTrue(self.triangle.validate(enforce_nilpotence=True))

    def test_end_to_end_loop_collapse_process(self) -> None:
        """Run the parsed 03_loop_collapse.tau file through ToposInterpreter."""
        file_path = Path(__file__).parent.parent / "examples" / "03_loop_collapse.tau"
        source = file_path.read_text()
        program = parse_source(source)

        # Build initial complex
        complex_obj = CellComplex("ProcessComplex")
        for v in (self.vA, self.vB, self.vC):
            complex_obj.add_cell(v)
        for e in (self.e1, self.e2, self.e3):
            complex_obj.add_cell(e)

        # Initial invariant check: beta_1 == 1
        self.assertEqual(HomologyEngine(complex_obj).betti_number(1), 1)

        interpreter = ToposInterpreter()
        ctx = interpreter.run_process(program, "contract_vortices", complex_obj)

        # Verified loop termination:
        # Loop ran until betti(complex, dim=1) == 0, taking exactly 1 iteration
        self.assertEqual(ctx.iterations, 1)
        self.assertEqual(ctx.final_betti[1], 0)
        self.assertEqual(len(ctx.trace), 1)
        self.assertEqual(ctx.trace[0].rule_name, "loop_contraction")

        # Confirm the final complex has closed topology and satisfies d^2 = 0
        self.assertTrue(complex_obj.validate(enforce_nilpotence=True))

    def test_infinite_loop_protection(self) -> None:
        """Verify that an unsatisfiable condition raises RuntimeError on exceeding max_iterations."""
        code = """
        process infinite_loop(k: CellComplex) {
            until betti(k, dim=2) == 5 {
                // does nothing to create 2-cells
            }
        }
        """
        program = parse_source(code)
        interpreter = ToposInterpreter()
        with self.assertRaises(RuntimeError) as ctx:
            interpreter.run_process(program, "infinite_loop", self.triangle, max_iterations=5)
        self.assertIn("exceeded maximum iterations", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
