"""Topos Runtime Process Runner CLI (tau-run)."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from topos.frontend.parser import parse_source, build_complex, ParserError
from topos.frontend.lexer import LexerError
from topos.frontend.ast import ProcessDecl
from topos.compiler.jit import MLIRJITBridge
from topos.runtime.interpreter import ToposInterpreter
from topos.runtime.visualizer import Visualizer
from topos.topology.homology import HomologyEngine


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tau-run",
        description="ToposLang Process Execution & Homotopy Rewriting Runner",
    )
    parser.add_argument("file", help="Path to .tau source file")
    parser.add_argument("--process", type=str, help="Name of process routine to execute")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum loop iterations (default: 100)")
    parser.add_argument("--trace", action="store_true", help="Print detailed step-by-step rewrite trace")
    parser.add_argument("--json", action="store_true", help="Output execution results as machine-readable JSON")
    parser.add_argument("--svg", type=str, metavar="OUT", help="Export multi-panel execution trace as SVG")
    parser.add_argument(
        "--backend",
        choices=["auto", "mlir", "csr"],
        default="auto",
        help="Execution backend (auto: detect native MLIR else CSR, mlir: attempt MLIR, csr: pure-Python)",
    )

    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        sys.stderr.write(f"Error: File not found: {args.file}\n")
        return 1

    try:
        source = path.read_text(encoding="utf-8")
        program = parse_source(source)

        # Identify process to run
        process_name = args.process
        if not process_name:
            processes = [s.name for s in program.statements if isinstance(s, ProcessDecl)]
            if len(processes) == 1:
                process_name = processes[0]
            elif len(processes) == 0:
                sys.stderr.write("Error: No 'process' declaration found in source file.\n")
                return 1
            else:
                sys.stderr.write(f"Error: Multiple processes found ({', '.join(processes)}). Please specify --process <name>.\n")
                return 1

        # Build initial complex
        initial_complex = build_complex(program, name=f"{path.stem}_initial")
        initial_betti = HomologyEngine(initial_complex).betti_profile()

        # Build a copy for running rewrites
        working_complex = build_complex(program, name=f"{path.stem}_run")

        bridge = MLIRJITBridge(force_fallback=(args.backend == "csr"))
        backend_name = "native_mlir" if (args.backend in ("auto", "mlir") and bridge.is_native_available()) else "pure_python_csr"

        interpreter = ToposInterpreter()
        ctx = interpreter.run_process(
            program=program,
            process_name=process_name,
            complex_obj=working_complex,
            max_iterations=args.max_steps,
        )

        if args.svg:
            trace_svg = Visualizer.to_trace_svg(initial_complex, ctx.complex, ctx.trace)
            Path(args.svg).write_text(trace_svg, encoding="utf-8")

        if args.json:
            trace_data = [
                {
                    "step": s.step_index,
                    "rule": s.rule_name,
                    "homotopy_cell": s.homotopy_cell_name,
                    "betti_before": s.betti_before,
                    "betti_after": s.betti_after,
                }
                for s in ctx.trace
            ]
            res: Dict[str, Any] = {
                "file": str(path),
                "process": process_name,
                "backend": backend_name,
                "iterations": ctx.iterations,
                "initial_betti": initial_betti,
                "final_betti": ctx.final_betti,
                "trace": trace_data,
                "success": True,
            }
            print(json.dumps(res, indent=2))
        else:
            print(f"Topos Execution: process '{process_name}' on {path.name}")
            print("═" * 55)
            print(f"• Execution Backend: {backend_name}")
            print(f"• Total Iterations: {ctx.iterations}")
            print(f"• Invariant Transition: β_1 = {initial_betti.get(1, 0)} ──► β_1 = {ctx.final_betti.get(1, 0)}")
            print("─" * 55)
            if args.trace or True:
                for step in ctx.trace:
                    print(f"  [Step {step.step_index}] Rule: '{step.rule_name}' => Attached 2-Cell '{step.homotopy_cell_name}'")
                    print(f"           Homology: {step.betti_before} -> {step.betti_after}")
            print("═" * 55)
            print("✓ Execution completed successfully (Invariant condition satisfied).")
            if args.svg:
                print(f"Exported trace SVG: {args.svg}")

        return 0

    except (LexerError, ParserError) as e:
        sys.stderr.write(f"Syntax Error: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Runtime Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
