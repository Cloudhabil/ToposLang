"""ToposLang-Cubical Homotopy Bridge and Realization Functor R.

Implements the Realization Functor:
    R: Chain_{Cl}(U) -> CubSet_{/X}
Bridging ToposLang's discrete Directed Acyclic Cell Complexes (DACC) with
constructive Cubical Type Theory (CTT) and stack-allocated Kan closures.
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from topos.core.cell import Cell, GlobularCell
from topos.core.chain import Chain
from topos.core.clifford import CliffordSignature, Multivector, PGA301
from topos.core.complex import CellComplex
from topos.core.boundary import NilpotencyVerifier
from topos.core.cubical import FaceMask, KanBox, KanComposition, KanStackFrame
from topos.ir.sheaf import CellularSheafIR
from topos.topology.homology import BoundaryMatrix, HomologyEngine
from topos.topology.laplacian import HodgeLaplacian


@dataclass
class CubicalBridgeResult:
    """Outcome of bridging a discrete topological cycle to a Kan composition."""
    rule_name: str
    homotopy_cell_name: str
    dimension: int
    face_mask_phi: int
    initial_multivector: Multivector
    terminal_multivector: Multivector
    betti_before: Dict[int, int]
    betti_after: Dict[int, int]


class RealizationFunctor:
    """Functor R: Chain_{Cl}(U) -> CubSet_{/X} mapping discrete cells to Kan box fibrations."""

    def __init__(
        self,
        signature: CliffordSignature = PGA301,
        max_homotopy_dim: int = 2,
    ) -> None:
        self.signature = signature
        self.max_homotopy_dim = max_homotopy_dim
        # Multivector sections associated with cell names: F(e) in Cl(p, q, r)
        self.sections: Dict[str, Multivector] = {}

    def attach_section(self, cell_name: str, multivector: Multivector) -> None:
        """Attaches a multivector section to a cell in the complex."""
        assert multivector.signature == self.signature
        self.sections[cell_name] = multivector

    def get_section(self, cell_name: str) -> Multivector:
        """Retrieves multivector section for a cell, or default scalar 1.0."""
        if cell_name in self.sections:
            return self.sections[cell_name]
        return Multivector.scalar(self.signature, 1.0)

    @staticmethod
    def extract_support(complex_obj: CellComplex, cell_names: Sequence[str]) -> Set[str]:
        """Computes the downward topological support closure supp(r) = cl(cells) down to 0-cells."""
        support: Set[str] = set()

        def add_cell_closure(cell: Cell) -> None:
            support.add(cell.name)
            if cell.dim > 0:
                b_chain = cell.boundary()
                for face_cell in b_chain.terms.keys():
                    add_cell_closure(face_cell)

        for name in cell_names:
            try:
                cell = complex_obj.get_cell(name)
                add_cell_closure(cell)
            except Exception:
                pass

        return support

    @staticmethod
    def boundary_column_to_face_mask(
        boundary_col: Sequence[Union[Fraction, int]],
        dimension: int = 2,
    ) -> FaceMask:
        """Converts column j of sparse boundary matrix B_k into cubical face constraint mask phi.
        
        Formula from specification:
            phi = ⋁ (x_axis(i) = ε_i) where ε_i = 1 if (B_k)_{ij} == +1, 0 if (B_k)_{ij} == -1.
        """
        fmask = FaceMask(dimension=dimension)
        axis = 0
        for val in boundary_col:
            fval = Fraction(val)
            if fval != 0:
                eps = 1 if fval > 0 else 0
                fmask.add_face(axis=axis % dimension, value=eps)
                axis += 1
        return fmask

    def create_kan_box(
        self,
        base_multivector: Multivector,
        face_mask: FaceMask,
        generator_bivector: Optional[Multivector] = None,
        dimension: int = 2,
    ) -> KanBox:
        """Synthesizes an open Kan box horn with continuous Lie group versor sandwich tube.
        
        u(t) = S_sandwich(exp(t/2 * B), u_0) = M(t) * u_0 * ~M(t).
        """
        sig = self.signature
        if generator_bivector is None:
            # Default bivector generator (e.g. rotation in e1-e2 plane)
            # In PGA, e1e2 is blade at bit 1 | bit 2 = 6
            generator_bivector = Multivector.basis(sig, 6, 1.0)

        gen = generator_bivector

        def tube_evaluator(t: float) -> Multivector:
            # Rotor / Motor M(t) = exp(0.5 * t * B)
            motor = (gen * (0.5 * t)).exp_motor()
            return base_multivector.sandwich(motor)

        return KanBox(
            dimension=dimension,
            face_mask=face_mask,
            base_u0=base_multivector,
            tube=tube_evaluator,
            signature=sig,
        )

    def resolve_cycle(
        self,
        complex_obj: CellComplex,
        dim: int = 1,
        generator_bivector: Optional[Multivector] = None,
        step_index: int = 0,
        rule_name: str = "cubical_bridge_hcomp",
    ) -> Optional[CubicalBridgeResult]:
        """Detects a non-zero harmonic cycle obstruction L_k c != 0, extracts its support supp(r),
        solves the missing face via stack-allocated Kan composition (hcomp),
        and re-integrates the resulting (k+1)-cell into the DACC to clear the obstruction.
        """
        # Step 1: Detect non-zero harmonic cycle via Hodge Laplacian
        laplacian = HodgeLaplacian(complex_obj, dim)
        harmonic_cycles = laplacian.harmonic_cycles()
        if not harmonic_cycles:
            return None

        # Select first harmonic cycle vector c in ker(L_k)
        cycle_vec = harmonic_cycles[0]
        k_cells = complex_obj.get_cells(dim)
        active_cell_names = [
            k_cells[i].name for i in range(len(k_cells)) if cycle_vec[i] != 0
        ]
        if not active_cell_names:
            return None

        # Step 2: Extract spatial support subcomplex supp(r)
        support = self.extract_support(complex_obj, active_cell_names)

        # Snapshot Betti profile before bridge application
        betti_before = HomologyEngine(complex_obj).betti_profile()

        # Step 3: Map boundary non-zeros to FaceMask phi & extract base section
        Bk = BoundaryMatrix.from_complex(complex_obj, dim)
        # Extract column corresponding to first active cell
        active_idx = [i for i in range(len(k_cells)) if cycle_vec[i] != 0][0]
        col_vals = [Bk.matrix.get(r, active_idx) for r in range(Bk.matrix.rows)]
        fmask = self.boundary_column_to_face_mask(col_vals, dimension=dim + 1)

        # Base multivector from active cell section
        u0 = self.get_section(active_cell_names[0])

        # Step 4: Construct open box and execute stack-allocated hcomp with n-truncation
        kbox = self.create_kan_box(
            base_multivector=u0,
            face_mask=fmask,
            generator_bivector=generator_bivector,
            dimension=dim + 1,
        )
        stack_frame = kbox.to_stack_frame()

        terminal_u1 = KanComposition.hcomp(
            kan_box=kbox,
            interval_t=1.0,
            max_homotopy_dim=self.max_homotopy_dim,
        )

        # Step 5: Sheaf re-integration & (k+1)-cell attachment
        # Convert cycle_vec over Q to primitive integer cycle over Z
        import math
        fracs = [Fraction(v) for v in cycle_vec]
        lcm = 1
        for f in fracs:
            if f != 0:
                lcm = (lcm * f.denominator) // math.gcd(lcm, f.denominator)
        int_coeffs = [int(f * lcm) for f in fracs]
        g = 0
        for v in int_coeffs:
            g = math.gcd(g, abs(v))
        if g > 1:
            int_coeffs = [v // g for v in int_coeffs]

        lhs_chain = Chain.zero(dim)
        rhs_chain = Chain.zero(dim)
        for i, coeff in enumerate(int_coeffs):
            if coeff > 0:
                lhs_chain = lhs_chain + (Chain.from_cell(k_cells[i]) * coeff)
            elif coeff < 0:
                rhs_chain = rhs_chain + (Chain.from_cell(k_cells[i]) * (-coeff))

        # If cycle has no negative part, split cycle or cap it
        if rhs_chain.is_zero() and not lhs_chain.is_zero():
            # For a closed cycle like a circle loop: target boundary is 0, source is loop
            # Globular 2-cell capping the loop
            target_chain = Chain.zero(dim)
            source_chain = lhs_chain
        else:
            source_chain = lhs_chain
            target_chain = rhs_chain

        homotopy_cell_name = f"homotopy_{rule_name}_{dim}_{step_index}"
        homotopy_cell = GlobularCell(
            name=homotopy_cell_name,
            dim=dim + 1,
            source=source_chain,
            target=target_chain,
        )

        # Attach and verify nilpotency
        NilpotencyVerifier.verify_cell(homotopy_cell)
        complex_obj.add_cell(homotopy_cell)

        # Store terminal multivector section for new cell
        self.attach_section(homotopy_cell_name, terminal_u1)

        betti_after = HomologyEngine(complex_obj).betti_profile()

        return CubicalBridgeResult(
            rule_name=rule_name,
            homotopy_cell_name=homotopy_cell_name,
            dimension=dim + 1,
            face_mask_phi=stack_frame.face_mask_phi,
            initial_multivector=u0,
            terminal_multivector=terminal_u1,
            betti_before=betti_before,
            betti_after=betti_after,
        )
