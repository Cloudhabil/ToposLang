"""Four-Dimensional Graded DACCs and Four-Tier Boundary Nilpotency.

Implements:
1. Graded 4D Cell Complexes: K = K_0 ⊔ K_1 ⊔ K_2 ⊔ K_3 ⊔ K_4.
2. Four-Tier Boundary Nilpotency: B_1 B_2 = 0, B_2 B_3 = 0, B_3 B_4 = 0 over Q.
3. Exact Rational Combinatorial Hodge Laplacians: L_k = B_k^T B_k + B_{k+1} B_{k+1}^T for k in {0,1,2,3,4}.
4. Discrete Hodge Decomposition: C_k = im(B_k^T) ⊕ im(B_{k+1}) ⊕ ker(L_k), with dim(ker(L_k)) = beta_k.
5. Standard 4D hypercubical lattices including the 4-hypercube and 4D torus T^4.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from topos.core.cell import Cell, GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.boundary import (
    NilpotencyVerifier,
    NilpotencyViolationError,
    boundary,
)
from topos.core.matrix_csr import CSROperator, RationalField
from topos.topology.homology import BoundaryMatrix, ExactMatrix


class FourTierNilpotencyViolation(NilpotencyViolationError):
    """Raised when one of the 4-tier nilpotency conditions (B_k * B_{k+1} = 0) is violated."""
    def __init__(self, tier: int, message: str, residual: Optional[ExactMatrix] = None) -> None:
        super().__init__(message, dim=tier)
        self.tier = tier
        self.residual_matrix = residual


@dataclass
class FourTierNilpotencyReport:
    """Diagnostic report for the 4-tier nilpotency conditions."""
    tier_12_valid: bool
    tier_23_valid: bool
    tier_34_valid: bool
    max_dim: int

    @property
    def tier1_valid(self) -> bool:
        return self.tier_12_valid

    @property
    def tier2_valid(self) -> bool:
        return self.tier_23_valid

    @property
    def tier3_valid(self) -> bool:
        return self.tier_34_valid

    @property
    def all_valid(self) -> bool:
        return self.tier_12_valid and self.tier_23_valid and self.tier_34_valid


class FourTierDACC:
    """Four-Dimensional Directed Acyclic Cell Complex with exact rational linear algebra."""

    def __init__(self, complex_obj: Optional[CellComplex] = None, name: str = "DACC4D") -> None:
        self.complex = complex_obj if complex_obj is not None else CellComplex(name=name)
        self.field = RationalField()

    @property
    def name(self) -> str:
        return self.complex.name

    @property
    def dim(self) -> int:
        return self.complex.dim

    def add_cell(self, cell: Cell, auto_add_boundaries: bool = False, validate_closure: bool = False) -> None:
        """Adds a cell to the underlying complex."""
        if cell.dim > 4:
            raise ValueError(f"FourTierDACC supports dimensions up to 4, got cell with dim={cell.dim}")
        self.complex.add_cell(cell, auto_add_boundaries=auto_add_boundaries, validate_closure=validate_closure)

    def get_cells(self, dim: int) -> Tuple[Cell, ...]:
        return self.complex.get_cells(dim)

    def num_cells(self, dim: int) -> int:
        return len(self.complex.get_cells(dim))

    def get_incidence_matrix(self, dim: int) -> ExactMatrix:
        """Constructs the boundary incidence matrix B_dim : C_dim -> C_{dim-1} over Q.
        
        Returns an ExactMatrix of shape (|K_{dim-1}|, |K_{dim}|).
        If dim <= 0 or dim > 4, returns appropriate zero/empty matrix.
        """
        if dim <= 0 or dim > 4:
            n_curr = len(self.get_cells(dim)) if dim >= 0 else 0
            n_prev = len(self.get_cells(dim - 1)) if dim > 0 else 0
            return ExactMatrix(n_prev, n_curr)
        return BoundaryMatrix.from_complex(self.complex, dim).matrix

    def get_incidence_csr(self, dim: int) -> CSROperator:
        """Returns B_dim as a CSROperator over RationalField."""
        mat = self.get_incidence_matrix(dim)
        grid = [[mat.get(r, c) for c in range(mat.cols)] for r in range(mat.rows)]
        return CSROperator.from_dense(grid, field=self.field)

    def verify_nilpotency_tier(self, tier: int) -> bool:
        """Verifies nilpotency for a specific tier:
        - tier 1: B_1 * B_2 == 0
        - tier 2: B_2 * B_3 == 0
        - tier 3: B_3 * B_4 == 0
        """
        if tier not in (1, 2, 3):
            raise ValueError(f"Tier must be 1, 2, or 3, got {tier}")

        B_k = self.get_incidence_matrix(tier)
        B_k1 = self.get_incidence_matrix(tier + 1)

        if B_k.cols != B_k1.rows:
            # Dimension alignment check: B_k has cols = |K_k|, B_k1 has rows = |K_k|
            return False

        if B_k.rows == 0 or B_k.cols == 0 or B_k1.cols == 0:
            return True

        prod = B_k.matmul(B_k1)
        for r in range(prod.rows):
            for c in range(prod.cols):
                if prod.get(r, c) != Fraction(0, 1):
                    return False
        return True

    def verify_four_tier_nilpotency(self, raise_on_error: bool = True) -> FourTierNilpotencyReport:
        """Statically verifies the complete 4-tier boundary nilpotency:
        B_1 * B_2 = 0, B_2 * B_3 = 0, B_3 * B_4 = 0.
        """
        t12 = self.verify_nilpotency_tier(1)
        if not t12 and raise_on_error:
            B1 = self.get_incidence_matrix(1)
            B2 = self.get_incidence_matrix(2)
            prod = B1.matmul(B2)
            raise FourTierNilpotencyViolation(1, "Four-tier nilpotency violated: B_1 * B_2 != 0", prod)

        t23 = self.verify_nilpotency_tier(2)
        if not t23 and raise_on_error:
            B2 = self.get_incidence_matrix(2)
            B3 = self.get_incidence_matrix(3)
            prod = B2.matmul(B3)
            raise FourTierNilpotencyViolation(2, "Four-tier nilpotency violated: B_2 * B_3 != 0", prod)

        t34 = self.verify_nilpotency_tier(3)
        if not t34 and raise_on_error:
            B3 = self.get_incidence_matrix(3)
            B4 = self.get_incidence_matrix(4)
            prod = B3.matmul(B4)
            raise FourTierNilpotencyViolation(3, "Four-tier nilpotency violated: B_3 * B_4 != 0", prod)

        return FourTierNilpotencyReport(
            tier_12_valid=t12,
            tier_23_valid=t23,
            tier_34_valid=t34,
            max_dim=self.dim,
        )

    def compute_hodge_laplacian(self, dim: int) -> ExactMatrix:
        """Computes the exact rational combinatorial Hodge Laplacian:
        L_k = B_k^T * B_k + B_{k+1} * B_{k+1}^T over Q.
        
        Valid for any k in {0, 1, 2, 3, 4}.
        """
        if dim < 0 or dim > 4:
            raise ValueError(f"Dimension dim must be between 0 and 4, got {dim}")

        n_k = self.num_cells(dim)
        if n_k == 0:
            return ExactMatrix(0, 0)

        # L_down = B_k^T * B_k
        if dim == 0:
            L_down = ExactMatrix(n_k, n_k)
        else:
            B_k = self.get_incidence_matrix(dim)
            B_k_T = B_k.transpose()
            L_down = B_k_T.matmul(B_k)

        # L_up = B_{k+1} * B_{k+1}^T
        if dim >= 4:
            L_up = ExactMatrix(n_k, n_k)
        else:
            B_next = self.get_incidence_matrix(dim + 1)
            B_next_T = B_next.transpose()
            L_up = B_next.matmul(B_next_T)

        return L_down.add(L_up)

    def compute_betti_number(self, dim: int) -> int:
        """Computes beta_k = dim(ker(L_k)) by exact rational Hodge decomposition."""
        n_k = self.num_cells(dim)
        if n_k == 0:
            return 0
        Lk = self.compute_hodge_laplacian(dim)
        return Lk.nullity()

    def betti_profile(self) -> Dict[int, int]:
        """Returns the Betti numbers {0: beta_0, 1: beta_1, 2: beta_2, 3: beta_3, 4: beta_4}."""
        return {k: self.compute_betti_number(k) for k in range(max(1, self.dim + 1))}

    def discrete_hodge_decomposition(self, dim: int) -> Dict[str, int]:
        """Computes dimensions of the orthogonal components over Q:
        C_k = im(B_k^T) ⊕ im(B_{k+1}) ⊕ ker(L_k).
        """
        n_k = self.num_cells(dim)
        if n_k == 0:
            return {"dim_Ck": 0, "dim_im_Bk_T": 0, "dim_im_Bk1": 0, "dim_ker_Lk": 0}

        dim_im_Bk_T = self.get_incidence_matrix(dim).rank() if dim > 0 else 0
        dim_im_Bk1 = self.get_incidence_matrix(dim + 1).rank() if dim < 4 else 0
        dim_ker_Lk = self.compute_hodge_laplacian(dim).nullity()

        assert dim_im_Bk_T + dim_im_Bk1 + dim_ker_Lk == n_k, (
            f"Discrete Hodge decomposition violated at dim={dim}: "
            f"{dim_im_Bk_T} + {dim_im_Bk1} + {dim_ker_Lk} != {n_k}"
        )
        return {
            "dim_Ck": n_k,
            "dim_im_Bk_T": dim_im_Bk_T,
            "dim_im_Bk1": dim_im_Bk1,
            "dim_ker_Lk": dim_ker_Lk,
        }

    @classmethod
    def create_4hypercube(cls, name: str = "Hypercube4D") -> FourTierDACC:
        """Constructs a standard 4-hypercube [0,1]^4 complex with all 0, 1, 2, 3, 4-cells."""
        comp = CellComplex(name=name)

        # 0-cells: 16 vertices indexed by 4-bit tuples (x, y, z, w) in {0, 1}^4
        vertices = {}
        for idx in range(16):
            v_name = f"V_{bin(idx)[2:].zfill(4)}"
            cell = GlobularCell(v_name, dim=0)
            comp.add_cell(cell)
            vertices[idx] = cell

        # 1-cells: 32 edges (x -> x + 2^mu)
        edges = {}
        for idx in range(16):
            for mu in range(4):
                if not (idx & (1 << mu)):
                    nbr = idx | (1 << mu)
                    e_name = f"E_{bin(idx)[2:].zfill(4)}_dir{mu}"
                    cell = GlobularCell(e_name, dim=1, source=vertices[idx], target=vertices[nbr])
                    comp.add_cell(cell)
                    edges[(idx, mu)] = cell

        # 2-cells: 24 square faces (mu < nu)
        faces = {}
        for idx in range(16):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    if not (idx & (1 << mu)) and not (idx & (1 << nu)):
                        # Square boundary: e(idx, mu) + e(idx + 2^mu, nu) - e(idx + 2^nu, mu) - e(idx, nu)
                        e0 = edges[(idx, mu)]
                        e1 = edges[(idx | (1 << mu), nu)]
                        e2 = edges[(idx | (1 << nu), mu)]
                        e3 = edges[(idx, nu)]
                        source_chain = Chain.from_cell(e0) + Chain.from_cell(e1)
                        target_chain = Chain.from_cell(e3) + Chain.from_cell(e2)
                        p_name = f"P_{bin(idx)[2:].zfill(4)}_d{mu}d{nu}"
                        cell = GlobularCell(p_name, dim=2, source=source_chain, target=target_chain)
                        comp.add_cell(cell)
                        faces[(idx, mu, nu)] = cell

        # 3-cells: 8 3-cubes (mu < nu < rho)
        cubes = {}
        for idx in range(16):
            for mu in range(4):
                for nu in range(mu + 1, 4):
                    for rho in range(nu + 1, 4):
                        if not (idx & (1 << mu)) and not (idx & (1 << nu)) and not (idx & (1 << rho)):
                            # Cube boundary has 6 faces
                            f_xy_0 = faces[(idx, mu, nu)]
                            f_xy_1 = faces[(idx | (1 << rho), mu, nu)]
                            f_xz_0 = faces[(idx, mu, rho)]
                            f_xz_1 = faces[(idx | (1 << nu), mu, rho)]
                            f_yz_0 = faces[(idx, nu, rho)]
                            f_yz_1 = faces[(idx | (1 << mu), nu, rho)]

                            source_chain = (
                                Chain.from_cell(f_xy_1)
                                + Chain.from_cell(f_xz_0)
                                + Chain.from_cell(f_yz_1)
                            )
                            target_chain = (
                                Chain.from_cell(f_xy_0)
                                + Chain.from_cell(f_xz_1)
                                + Chain.from_cell(f_yz_0)
                            )
                            c_name = f"C3_{bin(idx)[2:].zfill(4)}_d{mu}d{nu}d{rho}"
                            cell = GlobularCell(c_name, dim=3, source=source_chain, target=target_chain)
                            comp.add_cell(cell)
                            cubes[(idx, mu, nu, rho)] = cell

        # 4-cell: 1 4-hypercube bounded by the 8 3-cubes
        # idx=0, mu=0, nu=1, rho=2, sigma=3
        c_xyz_0 = cubes[(0, 0, 1, 2)]
        c_xyz_1 = cubes[(8, 0, 1, 2)]
        c_xyw_0 = cubes[(0, 0, 1, 3)]
        c_xyw_1 = cubes[(4, 0, 1, 3)]
        c_xzw_0 = cubes[(0, 0, 2, 3)]
        c_xzw_1 = cubes[(2, 0, 2, 3)]
        c_yzw_0 = cubes[(0, 1, 2, 3)]
        c_yzw_1 = cubes[(1, 1, 2, 3)]

        src_4 = (
            Chain.from_cell(c_xyz_1)
            + Chain.from_cell(c_xyw_0)
            + Chain.from_cell(c_xzw_1)
            + Chain.from_cell(c_yzw_0)
        )
        tgt_4 = (
            Chain.from_cell(c_xyz_0)
            + Chain.from_cell(c_xyw_1)
            + Chain.from_cell(c_xzw_0)
            + Chain.from_cell(c_yzw_1)
        )
        hyper_cell = GlobularCell("HYPER_0", dim=4, source=src_4, target=tgt_4)
        comp.add_cell(hyper_cell)

        dacc = cls(comp)
        return dacc

    @classmethod
    def create_4torus_minimal(cls, name: str = "Torus4D") -> FourTierDACC:
        """Constructs a minimal 4D torus T^4 cell complex with 1 vertex and 4 loops."""
        comp = CellComplex(name=name)
        v = GlobularCell("V0", dim=0)
        comp.add_cell(v)

        # 4 fundamental cycle links: L_0, L_1, L_2, L_3
        loops = [GlobularCell(f"L{i}", dim=1, source=v, target=v) for i in range(4)]
        for l in loops:
            comp.add_cell(l)

        # 6 plaquettes (tori faces): P_ij for i < j
        faces = []
        for i in range(4):
            for j in range(i + 1, 4):
                # (Li * Lj) => (Lj * Li), boundary is 0 because Li and Lj commute in homology
                src = Chain.from_cell(loops[i]) + Chain.from_cell(loops[j])
                tgt = Chain.from_cell(loops[j]) + Chain.from_cell(loops[i])
                p = GlobularCell(f"P{i}{j}", dim=2, source=src, target=tgt)
                comp.add_cell(p)
                faces.append(p)

        # 4 3-volumes: V_ijk for i < j < k
        vols = []
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(j + 1, 4):
                    vol = GlobularCell(f"VOL_{i}{j}{k}", dim=3, source=Chain.zero(2), target=Chain.zero(2))
                    comp.add_cell(vol)
                    vols.append(vol)

        # 1 4-hypercube: H_0123
        h = GlobularCell("HYPER_0123", dim=4, source=Chain.zero(3), target=Chain.zero(3))
        comp.add_cell(h)

        return cls(comp)


def create_4_hypercube(name: str = "Hypercube4D") -> FourTierDACC:
    """Convenience function to create a 4-hypercube [0,1]^4."""
    return FourTierDACC.create_4hypercube(name)


def create_4_torus(name: str = "Torus4D") -> FourTierDACC:
    """Convenience function to create a minimal 4D torus T^4."""
    return FourTierDACC.create_4torus_minimal(name)


# Alias for 4D certified cell complexes
CellComplex4D = FourTierDACC

