"""Cell Complex registry and topological validation.

Maintains graded collections of n-cells, validates cellular closure,
and computes standard topological invariants like Euler characteristic.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple, Union

from topos.core.cell import Cell, Simplex
from topos.core.chain import Chain
from topos.core.boundary import NilpotencyVerifier, CellNotFoundError, boundary


class CellComplex:
    """Represents a graded cell complex K = union_{k >= 0} K_k."""

    def __init__(self, name: str = "Complex") -> None:
        self.name = name
        # Mapping: dimension k -> {cell_name: Cell}
        self._cells_by_dim: Dict[int, Dict[str, Cell]] = {}

    @property
    def graded_cells(self) -> Dict[int, Tuple[Cell, ...]]:
        """Returns a mapping of dimension k to a tuple of registered cells."""
        return {dim: tuple(cells.values()) for dim, cells in self._cells_by_dim.items()}

    @property
    def dim(self) -> int:
        """The maximum dimension of cells present in the complex, or -1 if empty."""
        if not self._cells_by_dim:
            return -1
        non_empty = [d for d, cells in self._cells_by_dim.items() if len(cells) > 0]
        return max(non_empty) if non_empty else -1

    def add_cell(
        self,
        cell: Cell,
        auto_add_boundaries: bool = False,
        validate_closure: bool = False,
    ) -> None:
        """Adds a cell to the complex.
        
        Args:
            cell: The n-cell to add.
            auto_add_boundaries: If True, recursively registers all boundary cells.
            validate_closure: If True, checks that boundary cells already exist.
        """
        k = cell.dim
        if k not in self._cells_by_dim:
            self._cells_by_dim[k] = {}

        if auto_add_boundaries and k > 0:
            b_chain = cell.boundary()
            for b_cell in b_chain.terms.keys():
                if not self.contains(b_cell):
                    self.add_cell(b_cell, auto_add_boundaries=True)

        if validate_closure and k > 0:
            b_chain = cell.boundary()
            lower_cells = set(self.get_cells(k - 1))
            for b_cell in b_chain.terms.keys():
                if b_cell not in lower_cells:
                    raise CellNotFoundError(
                        f"Boundary cell {b_cell} of cell {cell} is not present in complex '{self.name}'"
                    )

        self._cells_by_dim[k][cell.name] = cell

    def get_cells(self, dim: int) -> Tuple[Cell, ...]:
        """Returns all registered cells of dimension dim."""
        if dim not in self._cells_by_dim:
            return ()
        return tuple(self._cells_by_dim[dim].values())

    def get_cell(self, name: str, dim: Optional[int] = None) -> Cell:
        """Retrieves a cell by name (and optionally dimension)."""
        if dim is not None:
            if dim in self._cells_by_dim and name in self._cells_by_dim[dim]:
                return self._cells_by_dim[dim][name]
            raise CellNotFoundError(f"Cell {name!r} with dimension {dim} not found in complex.")
        
        for d, cells in self._cells_by_dim.items():
            if name in cells:
                return cells[name]
        raise CellNotFoundError(f"Cell {name!r} not found in complex.")

    def contains(self, cell: Cell) -> bool:
        """Checks if a given cell is registered in the complex."""
        k = cell.dim
        if k not in self._cells_by_dim:
            return False
        return cell.name in self._cells_by_dim[k]

    def __contains__(self, cell: Cell) -> bool:
        return self.contains(cell)

    def validate(self, enforce_nilpotence: bool = True, enforce_closure: bool = True) -> bool:
        """Validates topological consistency across all cells in the complex."""
        if enforce_closure:
            for k in sorted(self._cells_by_dim.keys()):
                if k == 0:
                    continue
                lower_cells = set(self.get_cells(k - 1))
                for cell in self.get_cells(k):
                    b_chain = cell.boundary()
                    for b_cell in b_chain.terms.keys():
                        if b_cell not in lower_cells:
                            raise CellNotFoundError(
                                f"Boundary cell {b_cell} of cell {cell} is missing from dimension {k - 1}"
                            )

        if enforce_nilpotence:
            NilpotencyVerifier.verify_complex(self)

        return True

    def euler_characteristic(self) -> int:
        """Computes the Euler characteristic chi = sum_{k=0}^n (-1)^k |K_k|."""
        chi = 0
        for k, cells in self._cells_by_dim.items():
            if k >= 0:
                chi += ((-1) ** k) * len(cells)
        return chi

    @classmethod
    def from_simplices(cls, simplices: List[Simplex], name: str = "SimplicialComplex") -> CellComplex:
        """Factory method to build a closed simplicial complex from a list of top-level simplices."""
        complex_obj = cls(name=name)
        for s in simplices:
            complex_obj.add_cell(s, auto_add_boundaries=True)
        return complex_obj

    def __repr__(self) -> str:
        graded_summary = ", ".join(
            f"{k}-cells: {len(cells)}" for k, cells in sorted(self._cells_by_dim.items())
        )
        return f"CellComplex('{self.name}', {graded_summary})"
