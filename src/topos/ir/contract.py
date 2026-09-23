"""Topological Cellular Contraction Pass.

Collapses contractible sub-cells and edges while strictly preserving
the global homology and invariant profile of the CellComplex.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple

from topos.core.cell import GlobularCell, Cell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.topology.homology import HomologyEngine


class HomotopyContractionPass:
    """Performs homotopy-preserving cellular contractions."""

    @staticmethod
    def contract_edge(complex_obj: CellComplex, edge_name: str) -> CellComplex:
        """Collapses a contractible 1-cell e : u -> v by identifying v with u.
        
        Guarantees that global Betti numbers remain strictly invariant.
        """
        edge = complex_obj.get_cell(edge_name, dim=1)
        if not isinstance(edge, GlobularCell):
            raise TypeError("Only GlobularCell 1-cells can be contracted directly.")

        u = tuple(edge.source.terms.keys())[0]
        v = tuple(edge.target.terms.keys())[0]

        if u.name == v.name:
            raise ValueError(f"Cannot contract self-loop edge {edge_name!r}.")

        # Snapshot Betti profile before
        betti_before = HomologyEngine(complex_obj).betti_profile()

        contracted = CellComplex(name=f"{complex_obj.name}_contracted")

        # Add all 0-cells except v
        for vertex in complex_obj.get_cells(0):
            if vertex.name != v.name:
                contracted.add_cell(GlobularCell(vertex.name, dim=0))

        u_cell = contracted.get_cell(u.name, dim=0)

        # Helper to remap vertices
        def remap_vertex(vert_name: str) -> GlobularCell:
            if vert_name == v.name:
                return u_cell
            return contracted.get_cell(vert_name, dim=0)

        # Add all 1-cells except edge_name, remapping any incidence on v to u
        for e in complex_obj.get_cells(1):
            if e.name == edge_name:
                continue
            if isinstance(e, GlobularCell):
                src_v = tuple(e.source.terms.keys())[0]
                tgt_v = tuple(e.target.terms.keys())[0]
                new_src = remap_vertex(src_v.name)
                new_tgt = remap_vertex(tgt_v.name)
                contracted.add_cell(GlobularCell(e.name, dim=1, source=new_src, target=new_tgt))

        # Add higher cells
        for dim in sorted(complex_obj.graded_cells.keys()):
            if dim < 2:
                continue
            for cell in complex_obj.get_cells(dim):
                if isinstance(cell, GlobularCell):
                    # Remap 1-cells in source and target chains
                    def remap_chain(ch: Chain) -> Chain:
                        new_terms = {}
                        for c_face, coeff in ch:
                            if c_face.name != edge_name:
                                remapped_c = contracted.get_cell(c_face.name, dim=dim - 1)
                                new_terms[remapped_c] = coeff
                        return Chain(dim - 1, new_terms)

                    new_source = remap_chain(cell.source)
                    new_target = remap_chain(cell.target)
                    contracted.add_cell(
                        GlobularCell(cell.name, dim=dim, source=new_source, target=new_target)
                    )

        # Verify nilpotency
        contracted.validate(enforce_nilpotence=True)

        # Verify invariant preservation (Homotopy Equivalence)
        betti_after = HomologyEngine(contracted).betti_profile()
        for k, beta in betti_before.items():
            if betti_after.get(k, 0) != beta:
                raise ValueError(
                    f"Contraction of edge {edge_name} altered homology: "
                    f"beta_{k} changed from {beta} to {betti_after.get(k, 0)}"
                )

        return contracted
