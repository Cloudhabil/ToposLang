"""Directed Acyclic Cell Complex (DACC) - Topological Intermediate Representation (T-IR).

Imposes a causal poset ordering on cells to guarantee polynomial-time subdiagram
matching and structured topological compilation passes.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple

from topos.core.cell import Cell, GlobularCell
from topos.core.complex import CellComplex


class DACC:
    """Directed Acyclic Cell Complex representation of a CellComplex.
    
    Organizes cells into a topologically sorted dependency DAG where each cell
    depends on its boundary faces.
    """

    def __init__(self, complex_obj: CellComplex) -> None:
        self.complex = complex_obj
        # Adjacency list: cell_name -> set of boundary cell names it depends on
        self.dependencies: Dict[str, Set[str]] = {}
        # Reverse adjacency: cell_name -> set of cells that include it in their boundary
        self.dependents: Dict[str, Set[str]] = {}
        self.topological_order: List[Cell] = []

        self._build_dag()

    def _build_dag(self) -> None:
        # Initialize all cell entries
        all_cells: List[Cell] = []
        for dim in sorted(self.complex.graded_cells.keys()):
            for cell in self.complex.get_cells(dim):
                all_cells.append(cell)
                self.dependencies[cell.name] = set()
                self.dependents[cell.name] = set()

        # Build dependency edges: k-cell depends on its (k-1)-faces
        for cell in all_cells:
            if cell.dim > 0:
                b_chain = cell.boundary()
                for face_cell in b_chain.terms.keys():
                    self.dependencies[cell.name].add(face_cell.name)
                    if face_cell.name in self.dependents:
                        self.dependents[face_cell.name].add(cell.name)

        # Compute topological sorting (Kahn's algorithm)
        in_degree = {name: len(deps) for name, deps in self.dependencies.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_names: List[str] = []

        while queue:
            curr = queue.pop(0)
            sorted_names.append(curr)
            for parent in self.dependents.get(curr, set()):
                in_degree[parent] -= 1
                if in_degree[parent] == 0:
                    queue.append(parent)

        name_to_cell = {cell.name: cell for cell in all_cells}
        self.topological_order = [name_to_cell[name] for name in sorted_names if name in name_to_cell]

    def is_acyclic(self) -> bool:
        """Checks if the cell dependency graph contains no cyclic boundary loops."""
        total_cells = sum(len(cells) for cells in self.complex.graded_cells.values())
        return len(self.topological_order) == total_cells

    def causal_order(self) -> Tuple[Cell, ...]:
        """Returns cells in causal topological order (0-cells first, then 1-cells, etc.)."""
        return tuple(self.topological_order)
