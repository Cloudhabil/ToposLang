"""Subdiagram and path pattern matcher for CellComplexes."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Sequence

from topos.core.cell import Cell, GlobularCell
from topos.core.complex import CellComplex


@dataclass
class MatchResult:
    """Contains the matched subdiagram cells and end vertices."""
    matched_cells: Tuple[Cell, ...]
    start_vertex: Cell
    end_vertex: Cell


class PathMatcher:
    """Finds matching 1-cell composite paths in a CellComplex."""

    @staticmethod
    def match_path_by_names(complex_obj: CellComplex, path_names: Sequence[str]) -> Optional[MatchResult]:
        """Matches a specific sequence of 1-cells by name, verifying connectivity.
        
        Args:
            complex_obj: The target CellComplex.
            path_names: Ordered tuple/list of 1-cell names (e.g., ["e1", "e2"]).
            
        Returns:
            MatchResult if cells exist and form a continuous directed path, else None.
        """
        if not path_names:
            return None

        cells: List[Cell] = []
        for name in path_names:
            try:
                c = complex_obj.get_cell(name, dim=1)
                cells.append(c)
            except Exception:
                return None

        # Verify path connectivity
        # For each cell c_i, target(c_i) must equal source(c_{i+1})
        for i in range(len(cells) - 1):
            curr_cell = cells[i]
            next_cell = cells[i + 1]

            if not isinstance(curr_cell, GlobularCell) or not isinstance(next_cell, GlobularCell):
                return None

            curr_tgt = tuple(curr_cell.target.terms.keys())
            next_src = tuple(next_cell.source.terms.keys())

            if not curr_tgt or not next_src or curr_tgt[0] != next_src[0]:
                return None

        if not isinstance(cells[0], GlobularCell) or not isinstance(cells[-1], GlobularCell):
            return None

        start_v = tuple(cells[0].source.terms.keys())[0]
        end_v = tuple(cells[-1].target.terms.keys())[0]

        return MatchResult(
            matched_cells=tuple(cells),
            start_vertex=start_v,
            end_vertex=end_v,
        )
