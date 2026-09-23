"""Topological Visualizer: ASCII, Graphviz DOT, and standalone SVG export."""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

from topos.core.cell import GlobularCell, Simplex
from topos.core.complex import CellComplex
from topos.runtime.engine import RewriteTraceStep
from topos.topology.homology import HomologyEngine
from topos.topology.invariant import euler_characteristic


class Visualizer:
    """Generates ASCII summaries, Graphviz DOT representations, and SVG vector graphics."""

    @staticmethod
    def to_ascii(complex_obj: CellComplex) -> str:
        """Produces a clean ASCII structural summary of the complex."""
        engine = HomologyEngine(complex_obj)
        betti = engine.betti_profile()
        chi = euler_characteristic(complex_obj)

        lines = [
            f"CellComplex: {complex_obj.name} (Max Dim: {complex_obj.dim})",
            "═" * 50,
        ]

        for dim in sorted(complex_obj.graded_cells.keys()):
            cells = complex_obj.get_cells(dim)
            lines.append(f"• {dim}-Cells ({len(cells)}):")
            for c in cells:
                if dim == 0:
                    lines.append(f"    - {c.name}")
                elif dim == 1 and isinstance(c, GlobularCell):
                    src = tuple(c.source.terms.keys())[0].name if c.source.terms else "?"
                    tgt = tuple(c.target.terms.keys())[0].name if c.target.terms else "?"
                    lines.append(f"    - {c.name} : {src} -> {tgt}")
                else:
                    lines.append(f"    - {c.name} (boundary: {c.boundary()})")

        lines.append("─" * 50)
        betti_str = ", ".join(f"β_{k}={val}" for k, val in sorted(betti.items()))
        lines.append(f"Invariants: χ = {chi} | {betti_str}")
        lines.append("═" * 50)
        return "\n".join(lines)

    @staticmethod
    def to_dot(complex_obj: CellComplex) -> str:
        """Generates Graphviz DOT representation for directed 1-cells and 2-cells."""
        lines = [
            f"digraph \"{complex_obj.name}\" {{",
            "    rankdir=LR;",
            "    node [shape=circle, style=filled, fillcolor=\"#E3F2FD\", fontname=\"Helvetica\"];",
            "    edge [fontname=\"Helvetica\", fontsize=10];",
        ]

        # Add 0-cells
        for v in complex_obj.get_cells(0):
            lines.append(f"    \"{v.name}\" [label=\"{v.name}\"];")

        # Add 1-cells
        for e in complex_obj.get_cells(1):
            if isinstance(e, GlobularCell):
                src = tuple(e.source.terms.keys())[0].name if e.source.terms else ""
                tgt = tuple(e.target.terms.keys())[0].name if e.target.terms else ""
                if src and tgt:
                    lines.append(f"    \"{src}\" -> \"{tgt}\" [label=\"{e.name}\", color=\"#1E88E5\"];")
            elif isinstance(e, Simplex) and len(e.vertices) == 2:
                lines.append(f"    \"{e.vertices[0]}\" -> \"{e.vertices[1]}\" [label=\"{e.name}\", color=\"#1E88E5\"];")

        # Comment 2-cells
        for f in complex_obj.get_cells(2):
            lines.append(f"    // 2-cell: {f.name} with boundary: {f.boundary()}")

        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def to_svg(
        cls,
        complex_obj: CellComplex,
        width: int = 700,
        height: int = 500,
    ) -> str:
        """Generates self-contained, standalone vector graphic (SVG) XML."""
        engine = HomologyEngine(complex_obj)
        betti = engine.betti_profile()
        chi = euler_characteristic(complex_obj)

        vertices = complex_obj.get_cells(0)
        edges = complex_obj.get_cells(1)
        two_cells = complex_obj.get_cells(2)

        # Compute circular positions for vertices
        cx, cy = width / 2, height / 2 + 20
        radius = min(width, height) * 0.32
        n_v = len(vertices)
        pos: Dict[str, Tuple[float, float]] = {}

        if n_v == 1:
            pos[vertices[0].name] = (cx, cy)
        else:
            for idx, v in enumerate(vertices):
                angle = (2 * math.pi * idx / n_v) - (math.pi / 2)
                pos[v.name] = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))

        # Build SVG
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '  <defs>',
            '    <marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            '      <path d="M 0 1 L 10 5 L 0 9 z" fill="#1976D2" />',
            '    </marker>',
            '  </defs>',
            '  <rect width="100%" height="100%" fill="#FAFAFA" rx="8" />',
            f'  <!-- Header HUD -->',
            f'  <text x="24" y="32" font-family="Helvetica, sans-serif" font-size="18" font-weight="bold" fill="#212121">{complex_obj.name}</text>',
            f'  <text x="24" y="52" font-family="Helvetica, sans-serif" font-size="12" fill="#757575">Max Dim: {complex_obj.dim} | Cells: 0-cells={len(vertices)}, 1-cells={len(edges)}, 2-cells={len(two_cells)}</text>',
            f'  <rect x="{width - 240}" y="16" width="220" height="48" rx="6" fill="#E8F5E9" stroke="#81C784" />',
            f'  <text x="{width - 230}" y="36" font-family="monospace" font-size="13" font-weight="bold" fill="#2E7D32">χ = {chi} | ' + ", ".join(f"β_{k}={v}" for k, v in sorted(betti.items())) + '</text>',
            f'  <text x="{width - 230}" y="52" font-family="sans-serif" font-size="10" fill="#388E3C">Boundary Nilpotent: d² ≡ 0</text>',
        ]

        # Draw 2-cells as filled translucent polygons if coordinates are present
        for f in two_cells:
            # Try to get vertex coordinates for polygon
            poly_points: List[Tuple[float, float]] = []
            if isinstance(f, GlobularCell):
                for edge_cell in f.source.terms.keys():
                    if isinstance(edge_cell, GlobularCell):
                        src = tuple(edge_cell.source.terms.keys())[0].name
                        tgt = tuple(edge_cell.target.terms.keys())[0].name
                        if src in pos and pos[src] not in poly_points:
                            poly_points.append(pos[src])
                        if tgt in pos and pos[tgt] not in poly_points:
                            poly_points.append(pos[tgt])
            if len(poly_points) >= 3:
                pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly_points)
                svg.append(f'  <polygon points="{pts_str}" fill="#FFF59D" fill-opacity="0.45" stroke="#FBC02D" stroke-dasharray="4,4" stroke-width="1.5" />')

        # Draw 1-cells (edges)
        for e in edges:
            if isinstance(e, GlobularCell):
                src = tuple(e.source.terms.keys())[0].name if e.source.terms else ""
                tgt = tuple(e.target.terms.keys())[0].name if e.target.terms else ""
                if src in pos and tgt in pos:
                    x1, y1 = pos[src]
                    x2, y2 = pos[tgt]
                    if src == tgt:
                        # Self-loop arc
                        svg.append(f'  <path d="M {x1} {y1 - 10} C {x1 - 40} {y1 - 70}, {x1 + 40} {y1 - 70}, {x1 + 8} {y1 - 8}" fill="none" stroke="#1976D2" stroke-width="2" marker-end="url(#arrow)" />')
                        svg.append(f'  <text x="{x1}" y="{y1 - 75}" font-family="sans-serif" font-size="11" fill="#0D47A1" text-anchor="middle">{e.name}</text>')
                    else:
                        svg.append(f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#1976D2" stroke-width="2" marker-end="url(#arrow)" />')
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2 - 8
                        svg.append(f'  <text x="{mid_x:.1f}" y="{mid_y:.1f}" font-family="sans-serif" font-size="11" fill="#0D47A1" text-anchor="middle">{e.name}</text>')

        # Draw 0-cells (vertices)
        for v in vertices:
            if v.name in pos:
                x, y = pos[v.name]
                svg.append(f'  <circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="#42A5F5" stroke="#1565C0" stroke-width="2" />')
                svg.append(f'  <text x="{x:.1f}" y="{y + 4:.1f}" font-family="Helvetica, sans-serif" font-size="11" font-weight="bold" fill="#FFFFFF" text-anchor="middle">{v.name}</text>')

        svg.append('</svg>')
        return "\n".join(svg)

    @classmethod
    def to_trace_svg(
        cls,
        initial_complex: CellComplex,
        final_complex: CellComplex,
        trace_steps: List[RewriteTraceStep],
        width: int = 1000,
        height: int = 500,
    ) -> str:
        """Generates a side-by-side comparison SVG showing Initial vs Final state with trace timeline."""
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
            '  <rect width="100%" height="100%" fill="#F5F5F5" rx="8" />',
            f'  <text x="30" y="36" font-family="Helvetica, sans-serif" font-size="20" font-weight="bold" fill="#212121">Topological Execution Trace</text>',
            f'  <text x="30" y="58" font-family="sans-serif" font-size="13" fill="#616161">Total Steps: {len(trace_steps)} | Process Invariant Reduction</text>',
        ]

        # Render Left: Initial State
        sub_svg_1 = cls.to_svg(initial_complex, width=460, height=400)
        # Strip header/footer svg tags and place in nested group
        inner_1 = "\n".join(sub_svg_1.split("\n")[1:-1])
        svg.append(f'  <g transform="translate(20, 80)">')
        svg.append(f'    <text x="24" y="-8" font-family="sans-serif" font-size="14" font-weight="bold" fill="#D32F2F">Initial State (Before Rewrites)</text>')
        svg.append(inner_1)
        svg.append('  </g>')

        # Render Right: Final State
        sub_svg_2 = cls.to_svg(final_complex, width=460, height=400)
        inner_2 = "\n".join(sub_svg_2.split("\n")[1:-1])
        svg.append(f'  <g transform="translate(520, 80)">')
        svg.append(f'    <text x="24" y="-8" font-family="sans-serif" font-size="14" font-weight="bold" fill="#2E7D32">Final State (Homotopies Applied)</text>')
        svg.append(inner_2)
        svg.append('  </g>')

        svg.append('</svg>')
        return "\n".join(svg)
