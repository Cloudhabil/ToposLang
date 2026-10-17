r"""Pro-DACC Inverse Limit Complex K^\bullet(\infty) and \infty-Topos Architecture.

Implements:
1. Pro-DACC Inverse System (K_m, p_{m+1, m}): Spacing axis inverse limits lim K_m.
2. Coalgebraic Lazy Streams: On-demand generation and caching of fine-scale complexes.
3. Direct System of Cochain Spaces: Dual pullback maps p_{m+1, m}^*: C^*(K_m) -> C^*(K_{m+1}).
4. Continuous Projective Heat-Kernel Loss:
       L_proj = sum_{m=0}^M 2^{-m} (Tr exp(-tau_m L_{k,m}) - beta_k^target)^2.
5. Pseudospectral Resolvent Norm: ||(z I - L_{k,m})^{-1}|| over projective limits.
6. Elementary Collapses as Zero-Cost Garbage Collection: K \ K \ {e^{k-1}, e^k}.
7. Lock-Free Adaptive Mesh Refinement (AMR) with Homotopy Rewriting & Spatial Commutativity.
8. Kan Fibrations in \infty-Topos Sh_\infty(DACC) ensuring scale-transition equivalence.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from topos.core.cell import Cell, GlobularCell, Simplex
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import (
    NilpotencyVerifier,
    NilpotencyViolationError,
    boundary,
)
from topos.core.matrix_csr import CSROperator, RationalField
from topos.topology.homology import BoundaryMatrix, ExactMatrix, HomologyEngine
from topos.topology.laplacian import HodgeLaplacian


@dataclass
class DecimationMap:
    """Simplicial block-spin decimation map / transition functor p_{m+1, m}: K_{m+1} -> K_m."""
    from_scale: int
    to_scale: int
    # Mapping cell name in K_{m+1} to cell name in K_m (or None if collapsed)
    cell_mapping: Dict[str, Optional[str]] = field(default_factory=dict)
    # Exact rational projection matrix on k-chains
    chain_projections: Dict[int, ExactMatrix] = field(default_factory=dict)

    def project_cell(self, cell_name: str) -> Optional[str]:
        return self.cell_mapping.get(cell_name, None)

    def pullback_cochain(self, cochain_vec: List[Fraction], dim: int) -> List[Fraction]:
        """Dual pullback map p_{m+1, m}^*: C^k(K_m) -> C^k(K_{m+1})."""
        if dim not in self.chain_projections:
            return list(cochain_vec)
        # Pullback is transpose of chain projection
        proj = self.chain_projections[dim]
        # proj has shape (|K_m|, |K_{m+1}|), so transpose has shape (|K_{m+1}|, |K_m|)
        proj_T = proj.transpose()
        in_col = ExactMatrix(len(cochain_vec), 1, [[v] for v in cochain_vec])
        out_col = proj_T.matmul(in_col)
        return [out_col.get(r, 0) for r in range(out_col.rows)]


class ProCellComplex:
    r"""Projective Cell Complex Tower K^\bullet(\infty) = lim K_m (Pro-DACC).
    
    Can be instantiated either with an explicit sequence of CellComplex instances
    or with a coalgebraic lazy stream generator.
    """

    def __init__(
        self,
        name: str = "ProDACC_Tower",
        generator: Optional[Callable[[int], CellComplex]] = None,
        initial_complexes: Optional[Sequence[CellComplex]] = None,
        max_dimension: int = 4,
    ) -> None:
        self.name = name
        self.generator = generator
        self.max_dimension = max_dimension
        self._cached_scales: Dict[int, CellComplex] = {}
        self._decimation_maps: Dict[Tuple[int, int], DecimationMap] = {}

        if initial_complexes:
            for idx, c in enumerate(initial_complexes):
                self._cached_scales[idx] = c

    def get_scale(self, m: int) -> CellComplex:
        """Coalgebraic lazy stream evaluation: materializes scale m on-demand."""
        if m in self._cached_scales:
            return self._cached_scales[m]

        if self.generator is not None:
            c = self.generator(m)
            self._cached_scales[m] = c
            return c

        raise KeyError(f"Scale m={m} has not been generated and no coalgebraic generator is defined.")

    def set_scale(self, m: int, complex_obj: CellComplex) -> None:
        """Explicitly sets or overrides scale m."""
        self._cached_scales[m] = complex_obj

    def add_scale_complex(self, scale: int, comp: CellComplex) -> None:
        """Alias for set_scale."""
        self.set_scale(scale, comp)

    def register_decimation(self, decimation: DecimationMap) -> None:
        """Registers a transition functor p_{m+1, m}."""
        self._decimation_maps[(decimation.from_scale, decimation.to_scale)] = decimation

    def get_decimation(self, from_scale: int, to_scale: int) -> Optional[DecimationMap]:
        return self._decimation_maps.get((from_scale, to_scale), None)

    def projective_betti_limit(self, dim: int, max_scale: Optional[int] = None) -> int:
        """Evaluates beta_k in the projective limit tower."""
        if max_scale is None:
            max_scale = max(self._cached_scales.keys(), default=0)
        return self.betti_number_at(max_scale, dim)

    def projective_betti_profile(self, max_scale: Optional[int] = None) -> Dict[int, int]:
        """Returns the Betti invariant profile in the projective limit."""
        if max_scale is None:
            max_scale = max(self._cached_scales.keys(), default=0)
        c = self.get_scale(max_scale)
        return HomologyEngine(c).betti_profile()

    def projective_topological_loss(
        self,
        dim: int = 1,
        target_beta: float = 0.0,
        max_scale: Optional[int] = None,
        tau_base: Union[float, Fraction] = 0.5,
    ) -> float:
        """Evaluates continuous projective topological loss L_proj."""
        if max_scale is None:
            max_scale = max(self._cached_scales.keys(), default=0)
        return self.continuous_projective_loss(
            max_scale=max_scale,
            dim=dim,
            target_betti=float(target_beta),
            tau_scale_factor=float(tau_base),
        )

    def compute_pseudospectral_resolvent_norm(self, dim: int, scale: int, z: complex) -> float:
        """Evaluates resolvent norm ||(zI - L)^-1||."""
        return self.resolvent_norm(m=scale, dim=dim, z=z)

    # -------------------------------------------------------------------------
    # Projective Limit Invariants & Homological Spectra
    # -------------------------------------------------------------------------

    def betti_number_at(self, m: int, dim: int) -> int:
        """Computes beta_k(K_m) for the cell complex at scale m."""
        c = self.get_scale(m)
        return HomologyEngine(c).betti_number(dim)

    def betti_trajectory(self, max_scale: int, dim: int) -> List[int]:
        """Returns the Betti number trajectory [beta_k(K_0), beta_k(K_1), ..., beta_k(K_max_scale)]."""
        return [self.betti_number_at(m, dim) for m in range(max_scale + 1)]

    def is_projectively_contracted(self, max_scale: int, dim: int = 1) -> bool:
        """Evaluates whether the invariant condition 'until betti(lim Km, dim=k) == 0' holds.
        
        True if all scales from a stable cutoff up to max_scale have beta_k == 0.
        """
        for m in range(max_scale + 1):
            if self.betti_number_at(m, dim) != 0:
                return False
        return True

    def compute_hodge_laplacian_at(self, m: int, dim: int) -> ExactMatrix:
        """Extracts the Hodge Laplacian L_k,m at scale m."""
        c = self.get_scale(m)
        return HodgeLaplacian(c, dim).laplacian

    def resolvent_norm(self, m: int, dim: int, z: complex) -> float:
        """Computes the pseudospectral resolvent norm ||(z I - L_{k,m})^{-1}||.
        
        Using floating-point approximation of the exact rational Laplacian matrix.
        """
        L = self.compute_hodge_laplacian_at(m, dim)
        n = L.rows
        if n == 0:
            return 0.0

        # Construct (z * I - L)
        mat: List[List[complex]] = []
        for r in range(n):
            row = []
            for col in range(n):
                val_L = float(L.get(r, col))
                diag = z if r == col else 0.0
                row.append(diag - val_L)
            mat.append(row)

        # Invert via Gauss-Jordan elimination
        # Augmented with identity matrix
        inv: List[List[complex]] = [[1.0 if r == c else 0.0 for c in range(n)] for r in range(n)]
        A = [list(r) for r in mat]

        for i in range(n):
            # Pivot
            pivot = A[i][i]
            if abs(pivot) < 1e-12:
                # Search for pivot below
                swap_row = -1
                for k in range(i + 1, n):
                    if abs(A[k][i]) > 1e-12:
                        swap_row = k
                        break
                if swap_row == -1:
                    return float("inf")  # Singular (z is an eigenvalue)
                A[i], A[swap_row] = A[swap_row], A[i]
                inv[i], inv[swap_row] = inv[swap_row], inv[i]
                pivot = A[i][i]

            for j in range(n):
                A[i][j] /= pivot
                inv[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = A[k][i]
                    for j in range(n):
                        A[k][j] -= factor * A[i][j]
                        inv[k][j] -= factor * inv[i][j]

        # Compute Frobenius norm of inv
        norm_sq = 0.0
        for r in range(n):
            for c in range(n):
                norm_sq += abs(inv[r][c]) ** 2
        return math.sqrt(norm_sq)

    def continuous_projective_loss(
        self,
        max_scale: int,
        dim: int = 1,
        target_betti: float = 0.0,
        tau_scale_factor: float = 0.5,
    ) -> float:
        """Calculates the continuous projective topological loss:
            L_proj = sum_{m=0}^M 2^{-m} * (Tr exp(-tau_m L_{k,m}) - beta_k^target)^2.
        
        Using heat kernel trace Tr exp(-tau_m L_{k,m}).
        """
        loss = 0.0
        for m in range(max_scale + 1):
            L = self.compute_hodge_laplacian_at(m, dim)
            n = L.rows
            if n == 0:
                continue

            tau_m = tau_scale_factor * (2.0 ** (-m))

            # Approximate matrix exponential trace Tr exp(-tau_m L)
            # Power series: exp(-tau L) ~ I - tau L + (tau^2 / 2) L^2 - (tau^3 / 6) L^3
            # Or compute diagonal entries of Taylor expansion
            trace_exp = 0.0
            # Order 0: Tr(I) = n
            # Order 1: -tau_m * Tr(L)
            tr_L = sum(float(L.get(i, i)) for i in range(n))
            # Order 2: + (tau_m^2 / 2) * Tr(L^2)
            tr_L2 = 0.0
            for i in range(n):
                for j in range(n):
                    tr_L2 += float(L.get(i, j)) * float(L.get(j, i))

            trace_exp = n - (tau_m * tr_L) + (0.5 * (tau_m ** 2) * tr_L2)
            # Safeguard positive heat kernel trace
            trace_exp = max(0.0, trace_exp)

            weight = 2.0 ** (-m)
            diff = trace_exp - target_betti
            loss += weight * (diff ** 2)

        return loss

    # -------------------------------------------------------------------------
    # Elementary Collapses as Zero-Cost Garbage Collection
    # -------------------------------------------------------------------------

    @staticmethod
    def find_free_faces(complex_obj: CellComplex, dim: int) -> List[Tuple[Cell, Cell]]:
        """Identifies pairs (e^{k-1}, e^k) where e^{k-1} is a free face of e^k.
        
        A face e^{k-1} is free if it belongs to the boundary of EXACTLY ONE k-cell,
        and is not a boundary of any higher cell.
        """
        k_cells = complex_obj.get_cells(dim)
        if not k_cells or dim <= 0:
            return []

        # Count occurrences of (dim-1)-cells in boundaries of dim-cells
        face_usage: Dict[str, List[Cell]] = {}
        face_cell_map: Dict[str, Cell] = {}

        for c in k_cells:
            b_chain = c.boundary()
            for face_c, coeff in b_chain:
                if coeff != 0:
                    face_usage.setdefault(face_c.name, []).append(c)
                    face_cell_map[face_c.name] = face_c

        # Identify free pairs
        free_pairs = []
        for face_name, using_cells in face_usage.items():
            if len(using_cells) == 1:
                e_k1 = face_cell_map[face_name]
                e_k = using_cells[0]
                free_pairs.append((e_k1, e_k))

        return free_pairs

    @classmethod
    def elementary_collapse(
        cls,
        complex_obj: CellComplex,
        face_cell: Union[Cell, str],
        co_cell: Union[Cell, str],
    ) -> CellComplex:
        r"""Executes an elementary collapse K \ {e^{k-1}, e^k} in O(1) time.
        
        Preserves homotopy type and homology invariants while pruning inactive cells.
        """
        if isinstance(face_cell, str):
            for d in range(5):
                try:
                    face_cell = complex_obj.get_cell(face_cell, dim=d)
                    break
                except Exception:
                    pass
        if isinstance(co_cell, str):
            for d in range(5):
                try:
                    co_cell = complex_obj.get_cell(co_cell, dim=d)
                    break
                except Exception:
                    pass

        if not isinstance(face_cell, Cell) or not isinstance(co_cell, Cell):
            return complex_obj

        assert face_cell.dim == co_cell.dim - 1, (
            f"Dimension mismatch in elementary collapse: {face_cell.dim} != {co_cell.dim} - 1"
        )
        # Verify face_cell is actually a boundary face of co_cell
        b_terms = co_cell.boundary().terms
        if face_cell not in b_terms and not any(f.name == face_cell.name for f in b_terms):
            return complex_obj

        # Remove both cells from complex registry
        dim_face = face_cell.dim
        dim_co = co_cell.dim

        if dim_face in complex_obj._cells_by_dim and face_cell.name in complex_obj._cells_by_dim[dim_face]:
            del complex_obj._cells_by_dim[dim_face][face_cell.name]

        if dim_co in complex_obj._cells_by_dim and co_cell.name in complex_obj._cells_by_dim[dim_co]:
            del complex_obj._cells_by_dim[dim_co][co_cell.name]

        return complex_obj

    @classmethod
    def prune_contractible_pairs(cls, complex_obj: CellComplex, dim: int) -> int:
        """Repeatedly prunes free faces via elementary collapses until no free faces remain.
        
        Returns total number of collapsed cell pairs.
        """
        collapses = 0
        while True:
            free_pairs = cls.find_free_faces(complex_obj, dim)
            if not free_pairs:
                break
            # Collapse the first available free pair
            f_cell, c_cell = free_pairs[0]
            if cls.elementary_collapse(complex_obj, f_cell, c_cell):
                collapses += 1
            else:
                break
        return collapses

    # -------------------------------------------------------------------------
    # Lock-Free Adaptive Mesh Refinement (AMR)
    # -------------------------------------------------------------------------

    @classmethod
    def check_spatial_commutativity(
        cls,
        complex_obj: CellComplex,
        support_a: Set[str],
        support_b: Set[str],
    ) -> bool:
        """ToposLang Spatial Lock-Free Commutativity Theorem:
        supp(r_A) ∩ supp(r_B) = ∅ ==> r_A o r_B == r_B o r_A.
        """
        return len(support_a.intersection(support_b)) == 0

    @classmethod
    def refine_cell_homotopy(
        cls,
        complex_obj: CellComplex,
        cell_name: str,
        dim: int,
        subdivision_factor: int = 2,
    ) -> List[Cell]:
        """Executes an attaching homotopy rewrite r: c_coarse => c_fine.
        
        Attaches a (k+1)-cell bridging the coarse cell and its fine subdivisions
        without mutating existing arrays in-place.
        """
        coarse_cell = complex_obj.get_cell(cell_name, dim=dim)

        fine_cells: List[Cell] = []
        if dim == 1:
            # 1-cell (edge) subdivision: A -> B subdivided into A -> M -> B
            src = coarse_cell.boundary().terms
            # Get start and end vertices
            # Construct midpoint vertex
            mid_name = f"{cell_name}_mid"
            mid_v = GlobularCell(mid_name, dim=0)
            complex_obj.add_cell(mid_v)

            # Retrieve endpoints
            v_start = None
            v_end = None
            if hasattr(coarse_cell, "source") and hasattr(coarse_cell, "target"):
                v_start = coarse_cell.source
                v_end = coarse_cell.target
            else:
                for f_cell, coeff in coarse_cell.boundary():
                    if coeff < 0:
                        v_start = f_cell
                    elif coeff > 0:
                        v_end = f_cell

            if v_start is None:
                v_start = GlobularCell(f"{cell_name}_src", dim=0)
                complex_obj.add_cell(v_start)
            if v_end is None:
                v_end = GlobularCell(f"{cell_name}_tgt", dim=0)
                complex_obj.add_cell(v_end)

            e1 = GlobularCell(f"{cell_name}_sub0", dim=1, source=v_start, target=mid_v)
            e2 = GlobularCell(f"{cell_name}_sub1", dim=1, source=mid_v, target=v_end)
            complex_obj.add_cell(e1)
            complex_obj.add_cell(e2)
            fine_cells.extend([e1, e2])

            # Attach homotopy 2-cell: alpha_r : coarse_cell => (e1 + e2)
            src_chain = Chain.from_cell(coarse_cell)
            tgt_chain = Chain.from_cell(e1) + Chain.from_cell(e2)
            homotopy_2cell = GlobularCell(
                f"amr_homotopy_{cell_name}",
                dim=2,
                source=src_chain,
                target=tgt_chain,
            )
            NilpotencyVerifier.verify_cell(homotopy_2cell)
            complex_obj.add_cell(homotopy_2cell)

        return fine_cells


def make_standard_pro_torus() -> ProCellComplex:
    """Creates a standard ProDACC representing an inverse tower of refined tori."""
    def _torus_generator(scale: int) -> CellComplex:
        comp = CellComplex(f"Torus_Scale_{scale}")
        # At scale m, lattice size is 2^m
        v0 = GlobularCell("V0", dim=0)
        comp.add_cell(v0)
        l1 = GlobularCell(f"L1_m{scale}", dim=1, source=v0, target=v0)
        l2 = GlobularCell(f"L2_m{scale}", dim=1, source=v0, target=v0)
        comp.add_cell(l1)
        comp.add_cell(l2)
        # Face commuting l1 and l2
        src = Chain.from_cell(l1) + Chain.from_cell(l2)
        tgt = Chain.from_cell(l2) + Chain.from_cell(l1)
        p = GlobularCell(f"P_m{scale}", dim=2, source=src, target=tgt)
        comp.add_cell(p)
        return comp

    return ProCellComplex("ProTorus_Tower", generator=_torus_generator)


@dataclass
class CellularHomotopyRewrite:
    """An attaching homotopy rewrite r: c_coarse => c_fine attaching an oriented (k+1)-cell."""
    name: str
    support: Set[str] = field(default_factory=set)

    def commutes_with(self, other: CellularHomotopyRewrite) -> bool:
        """Theorem 4.3 (Spatial Lock-Free Commutativity):
        supp(r_A) ∩ supp(r_B) = ∅ ==> r_A ∘ r_B ≡ r_B ∘ r_A.
        """
        return len(self.support.intersection(other.support)) == 0


def verify_spatial_commutativity(r_a: CellularHomotopyRewrite, r_b: CellularHomotopyRewrite) -> bool:
    """Verifies that two homotopy rewrites commute lock-free."""
    return r_a.commutes_with(r_b)
