"""Unit and integration tests for Visualizer and CLI tooling (tauc & tau-run)."""

import json
import tempfile
import unittest
from pathlib import Path

from topos.core.cell import GlobularCell
from topos.core.complex import CellComplex
from topos.runtime.visualizer import Visualizer
from topos.runtime.engine import RewriteTraceStep
from topos.cli.tauc import main as tauc_main
from topos.cli.taurun import main as taurun_main


class TestVisualizer(unittest.TestCase):
    def setUp(self) -> None:
        self.v = GlobularCell("base", dim=0)
        self.loop = GlobularCell("loop", dim=1, source=self.v, target=self.v)
        self.s1 = CellComplex("S1")
        self.s1.add_cell(self.v)
        self.s1.add_cell(self.loop)

    def test_to_ascii(self) -> None:
        ascii_out = Visualizer.to_ascii(self.s1)
        self.assertIn("CellComplex: S1", ascii_out)
        self.assertIn("0-Cells", ascii_out)
        self.assertIn("1-Cells", ascii_out)
        self.assertIn("β_0=1, β_1=1", ascii_out)

    def test_to_dot(self) -> None:
        dot_out = Visualizer.to_dot(self.s1)
        self.assertTrue(dot_out.startswith("digraph \"S1\" {"))
        self.assertTrue(dot_out.strip().endswith("}"))
        self.assertIn("\"base\" -> \"base\"", dot_out)

    def test_to_svg(self) -> None:
        svg_out = Visualizer.to_svg(self.s1)
        self.assertTrue(svg_out.startswith("<svg"))
        self.assertTrue(svg_out.strip().endswith("</svg>"))
        self.assertIn("d² ≡ 0", svg_out)
        self.assertIn("base", svg_out)
        self.assertIn("loop", svg_out)

    def test_to_trace_svg(self) -> None:
        step = RewriteTraceStep(
            step_index=0,
            rule_name="collapse",
            homotopy_cell_name="alpha_0",
            betti_before={0: 1, 1: 1},
            betti_after={0: 1, 1: 0},
        )
        trace_svg = Visualizer.to_trace_svg(self.s1, self.s1, [step])
        self.assertTrue(trace_svg.startswith("<svg"))
        self.assertTrue(trace_svg.strip().endswith("</svg>"))
        self.assertIn("Topological Execution Trace", trace_svg)
        self.assertIn("Initial State", trace_svg)
        self.assertIn("Final State", trace_svg)


class TestCLITools(unittest.TestCase):
    def setUp(self) -> None:
        self.examples_dir = Path(__file__).parent.parent / "examples"
        self.circle_tau = str(self.examples_dir / "01_circle_hit.tau")
        self.torus_tau = str(self.examples_dir / "02_torus_hit.tau")
        self.collapse_tau = str(self.examples_dir / "03_loop_collapse.tau")

    def test_tauc_basic_and_betti(self) -> None:
        exit_code = tauc_main([self.circle_tau, "--betti"])
        self.assertEqual(exit_code, 0)

    def test_tauc_svg_and_dot_export(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = str(Path(tmpdir) / "circle.svg")
            dot_path = str(Path(tmpdir) / "circle.dot")
            exit_code = tauc_main([self.circle_tau, "--svg", svg_path, "--dot", dot_path])
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(svg_path).exists())
            self.assertTrue(Path(dot_path).exists())

    def test_tauc_json_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            exit_code = tauc_main([self.torus_tau, "--json"])
        self.assertEqual(exit_code, 0)
        data = json.loads(f.getvalue())
        self.assertTrue(data["valid"])
        self.assertEqual(data["betti"]["1"], 2)
        self.assertEqual(data["betti"]["2"], 1)

    def test_tauc_missing_file_error(self) -> None:
        exit_code = tauc_main(["non_existent_file.tau"])
        self.assertEqual(exit_code, 1)

    def test_taurun_execution_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            svg_path = str(Path(tmpdir) / "trace.svg")
            exit_code = taurun_main([
                self.collapse_tau,
                "--process", "contract_vortices",
                "--trace",
                "--svg", svg_path,
            ])
            self.assertEqual(exit_code, 0)
            self.assertTrue(Path(svg_path).exists())

    def test_taurun_json_output(self) -> None:
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            exit_code = taurun_main([self.collapse_tau, "--json"])
        self.assertEqual(exit_code, 0)
        data = json.loads(f.getvalue())
        self.assertTrue(data["success"])
        self.assertEqual(data["iterations"], 1)
        self.assertEqual(data["initial_betti"]["1"], 1)
        self.assertEqual(data["final_betti"]["1"], 0)


if __name__ == "__main__":
    unittest.main()
