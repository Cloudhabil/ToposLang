"""Topos Compiler and Topological Type Checker CLI (tauc)."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from topos.frontend.parser import parse_source, build_complex, ParserError
from topos.frontend.lexer import LexerError
from topos.frontend.ast import SheafDecl
from topos.compiler.emitter import MLIREmitter
from topos.ir.sheaf import CellularSheafIR
from topos.core.boundary import TopologicalValidationError
from topos.runtime.visualizer import Visualizer
from topos.topology.homology import HomologyEngine
from topos.topology.invariant import euler_characteristic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tauc",
        description="ToposLang Compiler & Topological Boundary Verifier",
    )
    parser.add_argument("file", help="Path to .tau source file")
    parser.add_argument("--ast", action="store_true", help="Print Abstract Syntax Tree")
    parser.add_argument("--betti", action="store_true", help="Display Betti invariant profile")
    parser.add_argument("--check", action="store_true", help="Perform strict topological check only")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--svg", type=str, metavar="OUT", help="Export cell complex as standalone SVG")
    parser.add_argument("--dot", type=str, metavar="OUT", help="Export cell complex in Graphviz DOT format")
    parser.add_argument("--emit-mlir", action="store_true", help="Emit lowered standard MLIR textual IR")
    parser.add_argument("--emit-topos-ir", action="store_true", help="Emit high-level topos dialect MLIR IR")

    args = parser.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        sys.stderr.write(f"Error: File not found: {args.file}\n")
        return 1

    try:
        source = path.read_text(encoding="utf-8")
        program = parse_source(source)

        if args.ast and not args.json:
            print("Topos AST:")
            for stmt in program.statements:
                print(f"  {stmt}")

        complex_obj = build_complex(program, name=path.stem)

        sheaf_ir = None
        sheaf_decls = [s for s in program.statements if isinstance(s, SheafDecl)]
        if sheaf_decls:
            sheaf_ir = CellularSheafIR.from_ast(sheaf_decls[0], complex_obj)

        if args.emit_topos_ir:
            emitter = MLIREmitter(target="topos")
            print(emitter.emit_topos_dialect(complex_obj, sheaf_ir=sheaf_ir))
            return 0

        if args.emit_mlir:
            emitter = MLIREmitter(target="standard")
            print(emitter.emit_standard_mlir(complex_obj, sheaf_ir=sheaf_ir))
            return 0

        engine = HomologyEngine(complex_obj)
        betti = engine.betti_profile()
        chi = euler_characteristic(complex_obj)

        if args.svg:
            svg_content = Visualizer.to_svg(complex_obj)
            Path(args.svg).write_text(svg_content, encoding="utf-8")

        if args.dot:
            dot_content = Visualizer.to_dot(complex_obj)
            Path(args.dot).write_text(dot_content, encoding="utf-8")

        if args.json:
            res: Dict[str, Any] = {
                "file": str(path),
                "valid": True,
                "complex_name": complex_obj.name,
                "dimension": complex_obj.dim,
                "cells": {str(k): len(v) for k, v in complex_obj.graded_cells.items()},
                "betti": betti,
                "euler_characteristic": chi,
                "nilpotency_verified": True,
            }
            print(json.dumps(res, indent=2))
        elif not args.check:
            print(Visualizer.to_ascii(complex_obj))
            if args.svg:
                print(f"Exported SVG: {args.svg}")
            if args.dot:
                print(f"Exported DOT: {args.dot}")
        else:
            print(f"✓ Topological verification passed for {path.name} (d² ≡ 0)")

        return 0

    except (LexerError, ParserError) as e:
        sys.stderr.write(f"Syntax Error: {e}\n")
        return 1
    except TopologicalValidationError as e:
        sys.stderr.write(f"Topological Nilpotency Violation: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
