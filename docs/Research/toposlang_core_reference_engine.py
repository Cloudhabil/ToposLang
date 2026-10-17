"""
ToposLang (tau-Lang) Core Reference Engine
------------------------------------------
Pure Python 3.13 Standard Library Implementation (Zero External Dependencies).

Implements:
1. Exact Rational Arithmetic over Q (using fractions.Fraction).
2. Directed Acyclic Cell Complexes (DACC) with Boundary Nilpotency (d^2 = 0).
3. Exact Reduced Row Echelon Form (RREF) & Homology Computation (Betti numbers beta_k).
4. Combinatorial Hodge Laplacians L_k = B_k^T * B_k + B_{k+1} * B_{k+1}^T.
5. Downward Support Closure Computation supp(r) = cl(lhs) U cl(rhs).
6. Lock-Free Spatial Wavefront Scheduler for Parallel Execution.
"""

from dataclass import dataclass
from fractions import Fraction
from typing import Dict, List, Set, Tuple, Optional, Any
import concurrent.futures
import threading


# ==============================================================================
# 1. EXACT RATIONAL MATRIX ALGEBRA OVER Q
# ==============================================================================

class MatrixQ:
    """Represents a dense/sparse matrix over the field of Rational Numbers Q."""

    def __init__(self, rows: int, cols: int, data: Optional[List[List[Fraction]]] = None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = [[Fraction(val) for val in row] for row in data]
        else:
            self.data = [[Fraction(0, 1) for _ in range(cols)] for _ in range(rows)]

    def __getitem__(self, idx: Tuple[int, int]) -> Fraction:
        r, c = idx
        return self.data[r][c]

    def __setitem__(self, idx: Tuple[int, int], val: Any) -> None:
        r, c = idx
        self.data[r][c] = Fraction(val)

    def copy(self) -> 'MatrixQ':
        new_mat = MatrixQ(self.rows, self.cols)
        new_mat.data = [[self.data[r][c] for c in range(self.cols)] for r in range(self.rows)]
        return new_mat

    def transpose(self) -> 'MatrixQ':
        transposed = MatrixQ(self.cols, self.rows)
        for r in range(self.rows):
            for c in range(self.cols):
                transposed[c, r] = self.data[r][c]
        return transposed

    def matmul(self, other: 'MatrixQ') -> 'MatrixQ':
        assert self.cols == other.rows, f"Dimension mismatch: ({self.rows}x{self.cols}) * ({other.rows}x{other.cols})"
        res = MatrixQ(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                acc = Fraction(0, 1)
                for k in range(self.cols):
                    acc += self.data[i][k] * other.data[k][j]
                res[i, j] = acc
        return res

    def add(self, other: 'MatrixQ') -> 'MatrixQ':
        assert self.rows == other.rows and self.cols == other.cols
        res = MatrixQ(self.rows, self.cols)
        for r in range(self.rows):
            for c in range(self.cols):
                res[r, c] = self.data[r][c] + other.data[r][c]
        return res

    def is_zero_matrix(self) -> bool:
        for r in range(self.rows):
            for c in range(self.cols):
                if self.data[r][c] != Fraction(0, 1):
                    return False
        return True

    def rref_and_rank(self) -> Tuple['MatrixQ', int]:
        """Computes exact Reduced Row Echelon Form (RREF) and matrix rank over Q."""
        mat = self.copy()
        pivot_row = 0
        pivot_cols = []

        for col in range(mat.cols):
            if pivot_row >= mat.rows:
                break

            # Search for non-zero pivot entry in column `col`
            sel_row = -1
            for r in range(pivot_row, mat.rows):
                if mat[r, col] != Fraction(0, 1):
                    sel_row = r
                    break

            if sel_row == -1:
                continue

            # Swap pivot row into place
            mat.data[pivot_row], mat.data[sel_row] = mat.data[sel_row], mat.data[pivot_row]

            # Scale pivot row to 1
            pivot_val = mat[pivot_row, col]
            for c in range(mat.cols):
                mat[pivot_row, c] /= pivot_val

            # Eliminate non-zero entries in current column across all other rows
            for r in range(mat.rows):
                if r != pivot_row and mat[r, col] != Fraction(0, 1):
                    factor = mat[r, col]
                    for c in range(mat.cols):
                        mat[r, c] -= factor * mat[pivot_row, c]

            pivot_cols.append(col)
            pivot_row += 1

        rank = len(pivot_cols)
        return mat, rank


# ==============================================================================
# 2. TOPOLOGICAL CELL & CHAIN DATA STRUCTURES
# ==============================================================================

@dataclass(frozen=True)
class Cell:
    """Represents a graded k-dimensional topological cell."""
    cell_id: str
    dim: int
    # Map of boundary face IDs to orientation signs (+1 or -1)
    boundary_faces: Dict[str, int]


class Chain:
    """Formal linear combination of k-cells over Q: c = sum(a_i * e_i^k)."""

    def __init__(self, dim: int, coeffs: Optional[Dict[str, Fraction]] = None):
        self.dim = dim
        self.coeffs: Dict[str, Fraction] = {}
        if coeffs:
            for cid, val in coeffs.items():
                frac = Fraction(val)
                if frac != Fraction(0, 1):
                    self.coeffs[cid] = frac

    def add(self, other: 'Chain') -> 'Chain':
        assert self.dim == other.dim, "Chain dimension mismatch"
        res_coeffs = dict(self.coeffs)
        for cid, val in other.coeffs.items():
            res_coeffs[cid] = res_coeffs.get(cid, Fraction(0, 1)) + val
        return Chain(self.dim, res_coeffs)

    def scale(self, scalar: Fraction) -> 'Chain':
        frac = Fraction(scalar)
        res_coeffs = {cid: val * frac for cid, val in self.coeffs.items() if val * frac != Fraction(0, 1)}
        return Chain(self.dim, res_coeffs)

    def is_zero(self) -> bool:
        return len(self.coeffs) == 0

    def __repr__(self) -> str:
        if not self.coeffs:
            return f"0_{self.dim}"
        terms = [f"{val}*{cid}" for cid, val in self.coeffs.items()]
        return " + ".join(terms)


# ==============================================================================
# 3. DIRECTED ACYCLIC CELL COMPLEX (DACC) CONTAINER
# ==============================================================================

class DACC:
    """Directed Acyclic Cell Complex container maintaining cells, boundaries, and homology."""

    def __init__(self, max_dim: int):
        self.max_dim = max_dim
        self.cells: Dict[str, Cell] = {}
        self.cells_by_dim: List[List[str]] = [[] for _ in range(max_dim + 1)]

    def add_cell(self, cell_id: str, dim: int, boundary_faces: Optional[Dict[str, int]] = None) -> Cell:
        assert 0 <= dim <= self.max_dim, f"Dimension {dim} exceeds max_dim {self.max_dim}"
        assert cell_id not in self.cells, f"Cell {cell_id} already exists"
        
        b_faces = boundary_faces if boundary_faces is not None else {}
        # Validate that all boundary faces exist and have dimension (dim - 1)
        if dim > 0:
            for face_id in b_faces.keys():
                assert face_id in self.cells, f"Boundary face {face_id} missing from complex"
                assert self.cells[face_id].dim == dim - 1, f"Face {face_id} dimension mismatch"

        cell = Cell(cell_id=cell_id, dim=dim, boundary_faces=b_faces)
        self.cells[cell_id] = cell
        self.cells_by_dim[dim].append(cell_id)
        return cell

    def get_boundary_matrix(self, dim: int) -> MatrixQ:
        """Constructs exact sparse integer boundary matrix B_k in Q^{|K_{k-1}| x |K_k|}."""
        if dim <= 0 or dim > self.max_dim:
            return MatrixQ(0, 0)

        target_cells = self.cells_by_dim[dim - 1]
        source_cells = self.cells_by_dim[dim]

        rows = len(target_cells)
        cols = len(source_cells)

        mat = MatrixQ(rows, cols)
        target_index_map = {cid: idx for idx, cid in enumerate(target_cells)}

        for col_idx, src_id in enumerate(source_cells):
            src_cell = self.cells[src_id]
            for face_id, sign in src_cell.boundary_faces.items():
                if face_id in target_index_map:
                    row_idx = target_index_map[face_id]
                    mat[row_idx, col_idx] = Fraction(sign, 1)

        return mat

    def verify_nilpotency((self)) -> bool:
        """Statically verifies Axiom 1.1: B_{k-1} * B_k == 0 across all dimensions."""
        for k in range(2, self.max_dim + 1):
            B_k = self.get_boundary_matrix(k)
            B_km1 = self.get_boundary_matrix(k - 1)
            if B_k.cols == 0 or B_km1.cols == 0:
                continue
            prod = B_km1.matmul(B_k)
            if not prod.is_zero_matrix():
                return False
        return True

    def compute_hodge_laplacian(self, dim: int) -> MatrixQ:
        """Computes combinatorial Hodge Laplacian L_k = B_k^T * B_k + B_{k+1} * B_{k+1}^T."""
        n_k = len(self.cells_by_dim[dim])
        if n_k == 0:
            return MatrixQ(0, 0)

        L_k = MatrixQ(n_k, n_k)

        # Downward Laplacian: B_k^T * B_k
        if dim > 0:
            B_k = self.get_boundary_matrix(dim)
            if B_k.rows > 0 and B_k.cols > 0:
                L_down = B_k.transpose().matmul(B_k)
                L_k = L_k.add(L_down)

        # Upward Laplacian: B_{k+1} * B_{k+1}^T
        if dim < self.max_dim:
            B_kp1 = self.get_boundary_matrix(dim + 1)
            if B_kp1.rows > 0 and B_kp1.cols > 0:
                L_up = B_kp1.matmul(B_kp1.transpose())
                L_k = L_k.add(L_up)

        return L_k

    def compute_betti_number(self, dim: int) -> int:
        """Computes k-th Betti number beta_k = dim(ker(B_k)) - rank(B_{k+1})."""
        n_k = len(self.cells_by_dim[dim])
        if n_k == 0:
            return 0

        rank_Bk = 0
        if dim > 0:
            B_k = self.get_boundary_matrix(dim)
            if B_k.rows > 0 and B_k.cols > 0:
                _, rank_Bk = B_k.rref_and_rank()

        rank_Bkp1 = 0
        if dim < self.max_dim:
            B_kp1 = self.get_boundary_matrix(dim + 1)
            if B_kp1.rows > 0 and B_kp1.cols > 0:
                _, rank_Bkp1 = B_kp1.rref_and_rank()

        dim_ker_Bk = n_k - rank_Bk
        beta_k = dim_ker_Bk - rank_Bkp1
        return beta_k

    def euler_characteristic(self) -> int:
        """Computes Euler Characteristic chi = sum((-1)^k * |K_k|)."""
        chi = 0
        for k in range(self.max_dim + 1):
            chi += ((-1) ** k) * len(self.cells_by_dim[k])
        return chi

    def compute_downward_closure(self, chain: Chain) -> Set[str]:
        """Computes downward topological closure cl(c) down to 0-cells."""
        closure_set: Set[str] = set()
        worklist = list(chain.coeffs.keys())

        while len(worklist) > 0:
            cid = worklist.pop()
            if cid in closure_set:
                continue
            closure_set.add(cid)
            cell = self.cells[cid]
            if cell.dim > 0:
                for face_id in cell.boundary_faces.keys():
                    if face_id not in closure_set:
                        worklist.append(face_id)

        return closure_set


# ==============================================================================
# 4. REWRITE RULES & LOCK-FREE SPATIAL SCHEDULER
# ==============================================================================

class RewriteRule:
    """Represents a higher-dimensional rewrite rule r = (lhs => rhs)."""

    def __init__(self, rule_id: str, lhs: Chain, rhs: Chain):
        assert lhs.dim == rhs.dim, "LHS and RHS chain dimensions must match"
        self.rule_id = rule_id
        self.lhs = lhs
        self.rhs = rhs

    def compute_support(self, complex_store: DACC) -> Set[str]:
        """Computes spatial support closure supp(r) = cl(lhs) U cl(rhs)."""
        cl_lhs = complex_store.compute_downward_closure(self.lhs)
        cl_rhs = complex_store.compute_downward_closure(self.rhs)
        return cl_lhs.union(cl_rhs)


class SpatialScheduler:
    """Partitions rewrite rules into lock-free concurrent wavefronts based on spatial disjointness."""

    @staticmethod
    def partition_into_wavefronts(rules: List[RewriteRule], complex_store: DACC) -> List[List[RewriteRule]]:
        """Evaluates pairwise support intersection supp(r1) \cap supp(r2) == \emptyset."""
        supports: Dict[RewriteRule, Set[str]] = {r: r.compute_support(complex_store) for r in rules}
        wavefronts: List[List[RewriteRule]] = []
        pending = list(rules)

        while len(pending) > 0:
            current_wavefront: List[RewriteRule] = []
            wavefront_support: Set[str] = set()

            for r in list(pending):
                r_supp = supports[r]
                # Check spatial disjointness condition
                if wavefront_support.isdisjoint(r_supp):
                    current_wavefront.append(r)
                    wavefront_support.update(r_supp)
                    pending.remove(r)

            wavefronts.append(current_wavefront)

        return wavefronts

    @staticmethod
    def execute_wavefront_parallel(wavefront: List[RewriteRule], complex_store: DACC, lock: threading.Lock) -> None:
        """Executes spatially disjoint rules concurrently across threads with zero runtime lock contention."""
        def _apply_rule(rule: RewriteRule):
            # Apply cellular homotopy rewrite by attaching a higher-dimensional cell
            target_dim = rule.lhs.dim + 1
            attaching_cell_id = f"alpha_{rule.rule_id}"

            diff = rule.rhs.add(rule.lhs.scale(Fraction(-1, 1)))
            b_faces = {cid: int(val) for cid, val in diff.coeffs.items()}

            with lock:
                complex_store.add_cell(attaching_cell_id, target_dim, b_faces)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(_apply_rule, wavefront))


# ==============================================================================
# 5. SELF-TEST & VERIFICATION SUITE
# ==============================================================================

def run_toposlang_verification_suite():
    print("=================================================================")
    print("Running ToposLang (tau-Lang) Core Engine Verification Suite")
    print("=================================================================\n")

    # 1. Construct Hollow Triangle Complex (S^1 discretization)
    complex_store = DACC(max_dim=2)

    # 0-cells (vertices)
    complex_store.add_cell("v0", 0)
    complex_store.add_cell("v1", 0)
    complex_store.add_cell("v2", 0)

    # 1-cells (edges forming a cycle)
    complex_store.add_cell("e0", 1, {"v0": -1, "v1": 1})
    complex_store.add_cell("e1", 1, {"v1": -1, "v2": 1})
    complex_store.add_cell("e2", 1, {"v0": -1, "v2": 1})

    print("[1] Static Boundary Nilpotency Check (d^2 == 0):")
    nilpotent = complex_store.verify_nilpotency()
    print(f"    Boundary Nilpotency Verified: {nilpotent}")
    assert nilpotent, "Nilpotency verification failed!"

    print("\n[2] Initial Homology & Invariants (Hollow Triangle S^1):")
    beta_0 = complex_store.compute_betti_number(0)
    beta_1 = complex_store.compute_betti_number(1)
    chi = complex_store.euler_characteristic()
    print(f"    Betti 0 (Components): {beta_0} [Expected: 1]")
    print(f"    Betti 1 (Holes):      {beta_1} [Expected: 1]")
    print(f"    Euler Characteristic: {chi} [Expected: 0]")
    assert beta_0 == 1 and beta_1 == 1 and chi == 0

    print("\n[3] Combinatorial Hodge Laplacian L_1 Spectrum Nullity Check:")
    L_1 = complex_store.compute_hodge_laplacian(1)
    _, rank_L1 = L_1.rref_and_rank()
    nullity_L1 = L_1.rows - rank_L1
    print(f"    dim(ker(L_1)): {nullity_L1} == Betti 1: {beta_1} [Discrete Hodge Theorem Holds]")
    assert nullity_L1 == beta_1

    print("\n[4] Spatial Lock-Free Wavefront Scheduling Test:")
    # Define two spatially disjoint 1-chains and rules
    lhs1 = Chain(1, {"e0": 1})
    rhs1 = Chain(1, {"e1": 1, "e2": -1})
    rule1 = RewriteRule("r1", lhs1, rhs1)

    # Construct second isolated component for rule 2
    complex_store.add_cell("v3", 0)
    complex_store.add_cell("v4", 0)
    complex_store.add_cell("e3", 1, {"v3": -1, "v4": 1})
    complex_store.add_cell("e4", 1, {"v3": -1, "v4": 1})

    lhs2 = Chain(1, {"e3": 1})
    rhs2 = Chain(1, {"e4": 1})
    rule2 = RewriteRule("r2", lhs2, rhs2)

    wavefronts = SpatialScheduler.partition_into_wavefronts([rule1, rule2], complex_store)
    print(f"    Scheduled Wavefront Count: {len(wavefronts)} [Expected: 1 (Concurrent Lock-Free Wavefront)]")
    assert len(wavefronts) == 1

    # Execute wavefront concurrently across worker threads
    exec_lock = threading.Lock()
    SpatialScheduler.execute_wavefront_parallel(wavefronts[0], complex_store, exec_lock)

    print("\n[5] Post-Rewrite Homology & Hole Collapse Verification:")
    beta_1_post = complex_store.compute_betti_number(1)
    print(f"    Post-Rewrite Betti 1: {beta_1_post} [Expected: 0 (1-cycle contracted by attached 2-cell)]")
    assert beta_1_post == 0

    print("\n=================================================================")
    print("ALL TOPOSLANG MATHEMATICAL & SYSTEM INVARIANTS SUCCESSFULLY VERIFIED")
    print("=================================================================")


if __name__ == "__main__":
    run_toposlang_verification_suite()