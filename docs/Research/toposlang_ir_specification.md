<!-- STREAMING_CHUNK:Initializing Document Header and Versioning -->
# ToposLang IR Specification Document
**Version:** 1.0.0-draft
**Status:** First Mover Spec (Approved for Native MLIR Implementation)
**Target Dialect:** `topos`
**Authors:** Principal Geometric Computing Specialist, ToposLang Architecture Team

## 1. Document Header
This specification formalizes the `topos` MLIR dialect. It translates graded cellular homology, exact rational linear algebra, and topological support closures into hardware-aware compiler representations. The dialect bridges high-level $\tau$-Lang syntax down to exact matrix representations ready for spatial wavefront scheduling and parallel tensor execution on TPU/XLA targets.

---

<!-- STREAMING_CHUNK:Defining EBNF Grammar and Syntax -->
## 2. Grammar & Syntax Specification
The `topos` dialect introduces specific topological types and operations. The Extended Backus-Naur Form (EBNF) for the dialect is defined as follows:

```ebnf
// Types
ToposDialect       ::= `!topos.` ( CellType | ChainType | ComplexType )
CellType           ::= `cell<` IntegerLiteral `>`
ChainType          ::= `chain<` IntegerLiteral `,` FieldType `>`
ComplexType        ::= `complex<` IntegerLiteral `>`
FieldType          ::= `Q` | `Z` | `R`

// Operations
ToposBoundaryOp    ::= `%` SSA_Id `=` `topos.boundary` `%` SSA_Id `:` ChainType `->` ChainType
ToposSupportOp     ::= `%` SSA_Id `=` `topos.support_closure` `%` SSA_Id `:` ComplexType `,` ChainType `->` `!topos.set<cell>`
ToposDisjointOp    ::= `%` SSA_Id `=` `topos.spatial_disjoint` `%` SSA_Id `,` `%` SSA_Id `:` `!topos.set<cell>` `->` `i1`
ToposRewriteOp     ::= `%` SSA_Id `=` `topos.homotopy_rewrite` `%` SSA_Id `,` `%` SSA_Id `,` `%` SSA_Id `:` ComplexType `,` ChainType `,` ChainType `->` ComplexType
```

---

<!-- STREAMING_CHUNK:Specifying Formal Type System and Memory Layouts -->
## 3. Formal Type System & Semantics

### 3.1 `!topos.cell<dim: i32>`
*   **Semantics:** Represents a single, oriented $k$-dimensional cell ($e^k \in K_k$).
*   **Memory Layout (C++):**
    ```cpp
    struct ToposCell {
        uint64_t cell_id;
        uint32_t dim;
        // Sparse mapping to boundary faces (cell_id -> orientation sign)
        llvm::SmallDenseMap<uint64_t, int8_t> boundary_faces; 
    };
    ```

### 3.2 `!topos.chain<dim: i32, field: type>`
*   **Semantics:** Represents a formal linear combination $c = \sum a_i e_i^k \in C_k(K; \mathbb{F})$. For exact structural topology, the field is strict $\mathbb{Q}$.
*   **Memory Layout (C++):**
    ```cpp
    struct ToposChain {
        uint32_t dim;
        // Coefficients a_i linked to cell_ids. Uses Exact Rational class.
        llvm::SmallDenseMap<uint64_t, ExactRational> coefficients;
    };
    ```

### 3.3 `!topos.complex<max_dim: i32>`
*   **Semantics:** The Directed Acyclic Cell Complex (DACC) state container $\mathcal{K} = (K, \partial)$.
*   **Memory Layout (C++):**
    ```cpp
    struct ToposComplex {
        uint32_t max_dim;
        // Graded arrays of cells K_0, K_1, ..., K_n
        std::vector<std::vector<ToposCell>> cells_by_dim;
        // Sparse Incidence Matrices B_k
        std::vector<SparseMatrixQ> boundary_matrices; 
    };
    ```

---

<!-- STREAMING_CHUNK:Detailing Operation Semantics and Verification Hooks -->
## 4. Operation Definitions & Verification Hooks

### 4.1 `topos.boundary` ($\partial_k$)
Maps a $k$-chain to its $(k-1)$-chain boundary.
**Signature:** `(chain<k, Q>) -> chain<k-1, Q>`
**Semantics:** Computes $c_{out} = B_k \cdot c_{in}$.

### 4.2 `topos.homotopy_rewrite`
Attaches a higher-dimensional cell $\alpha_r$ to transition state.
**Signature:** `(complex<n>, chain<k, Q>, chain<k, Q>) -> complex<n>`
**Constraint:** Inputs `lhs` and `rhs` must be parallel boundaries ($\partial_k(lhs) == \partial_k(rhs)$).

### 4.3 Static Verification: Boundary Nilpotency ($\partial^2 = 0$)
The compiler *must* statically verify Axiom 1 before lowering.

```cpp
// MLIR C++ Static Verifier Hook for Homotopy Rewrite
LogicalResult HomotopyRewriteOp::verify() {
    ToposChain lhs = this->getLhs();
    ToposChain rhs = this->getRhs();
    
    // 1. Verify topological dimensions match
    if (lhs.dim() != rhs.dim()) {
        return emitOpError("LHS and RHS must possess identical dimensions.");
    }
    
    // 2. Compute algebraic difference representing the boundary of the new cell
    ToposChain target_boundary = rhs - lhs;
    
    // 3. Axiomatic Verification: d(target_boundary) MUST equal 0.
    ToposChain d_squared = computeBoundary(target_boundary);
    
    if (!d_squared.isZero()) {
        return emitOpError() << "Boundary Nilpotency Violation (d^2 != 0).\n"
                             << "The specified rewrite rule creates a non-closed geometric attachment.\n"
                             << "Residual boundary: " << d_squared;
    }
    
    return success();
}
```

---

<!-- STREAMING_CHUNK:Formalizing Spatial Algorithms for Concurrency -->
## 5. Spatial Closure & Scheduler Algorithms

### 5.1 Downward Topological Support Closure ($cl(c)$)
Computes all structural dependencies of a chain down to $0$-cells.

```text
Algorithm: ComputeDownwardClosure
Input: Chain c, Complex K
Output: Set of CellIDs cl_c

1. Initialize cl_c = empty set
2. Initialize Worklist = all cell IDs in c with non-zero coefficients
3. While Worklist is not empty:
4.    Pop cell_id from Worklist
5.    If cell_id is not in cl_c:
6.        Add cell_id to cl_c
7.        cell = K.getCell(cell_id)
8.        If cell.dim > 0:
9.            For each face_id in cell.boundary_faces.keys():
10.               Push face_id to Worklist
11. Return cl_c
```

### 5.2 Lock-Free Disjointness Scheduling
For rewrite rules $R = \{r_1, r_2, \dots, r_m\}$, the spatial support is $supp(r) = cl(lhs) \cup cl(rhs)$.

```text
Algorithm: PartitionSpatialWavefronts
Input: List of RewriteRules R, Complex K
Output: List of independent rule sets (Wavefronts)

1. Compute supp(r) for all r in R
2. Wavefronts = empty list
3. While R is not empty:
4.     CurrentWavefront = empty list
5.     WavefrontSupport = empty set
6.     For each r in R:
7.         If intersection(WavefrontSupport, supp(r)) is empty:
8.             Add r to CurrentWavefront
9.             WavefrontSupport = union(WavefrontSupport, supp(r))
10.            Remove r from R
11.    Append CurrentWavefront to Wavefronts
12. Return Wavefronts
```

**Proof Sketch for Lock-Free Parallelism:**
Let hardware threads $T_1$ and $T_2$ execute rules $r_1$ and $r_2$. By definition, the memory locations accessed by a rule are restricted to the sparse matrix indices and cell objects within $supp(r)$. If $supp(r_1) \cap supp(r_2) = \emptyset$, then $T_1$ and $T_2$ read/modify mutually exclusive matrix rows and columns. Consequently, sparse additions commute exactly, and no hardware race conditions or cache invalidation conflicts can mathematically occur. 

---

<!-- STREAMING_CHUNK:Defining Exact Linear Algebra Specifications -->
## 6. Exact Linear Algebra Specs & Edge Cases

To prevent floating-point catastrophic cancellation when determining Betti numbers ($\beta_k$), all matrix reductions *must* operate over strict Rational Numbers ($\mathbb{Q}$).

### 6.1 Exact Rational Gaussian Elimination (RREF)
*   **Field Requirements:** Fractions $a/b$ simplified using greatest common divisor (GCD).
*   **Edge Case: Zero-Columns:** If a column consists entirely of zeros, it contributes to the nullity. The pivot index must advance to the next column, but the row pointer must *not* advance. 
*   **Degenerate Matrices:** If $B_k \in \mathbb{Z}^{0 \times |K_k|}$ (no boundaries exist), the rank is exactly 0, and the nullity is exactly $|K_k|$.

### 6.2 Combinatorial Hodge Laplacian ($L_k$)
To find holes, we calculate exact rank/nullity on $L_k$:
$$L_k = B_k^T B_k + B_{k+1} B_{k+1}^T$$
$$\beta_k = \dim(\ker(L_k)) = |K_k| - \text{rank}(B_k) - \text{rank}(B_{k+1})$$

---

<!-- STREAMING_CHUNK:Generating Test Cases and Expected Invariants -->
## 7. Test Cases & Boundary Validation Metrics

The MLIR compiler test suite must pass these geometric unit tests validating exact rank and Betti number computation.

### Test Case 1: Hollow Triangle (Topological Circle $S^1$)
*   **Geometry:** 3 Vertices ($v_0, v_1, v_2$), 3 Edges ($e_0, e_1, e_2$) forming a cycle. No 2-cells.
*   **Matrix $B_1$ ($3 \times 3$):**
    $$B_1 = \begin{bmatrix} -1 & 0 & -1 \\ 1 & -1 & 0 \\ 0 & 1 & 1 \end{bmatrix}$$
*   **Rank Analysis:** $\text{rank}(B_1) = 2$.
*   **Expected Invariants:** 
    *   $\beta_0 = 3 - 2 = 1$ (One connected component).
    *   $\beta_1 = 3 - 2 - 0 = 1$ (One central structural hole).
    *   Euler Characteristic $\chi = 3 - 3 = 0$.

### Test Case 2: Solid Triangle (2-Simplex)
*   **Geometry:** Same as Test 1, but with one attached 2-cell ($f_0$) whose boundary is $e_0 + e_1 - e_2$.
*   **Matrix $B_2$ ($3 \times 1$):**
    $$B_2 = \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}$$
*   **Nilpotency Check:** $B_1 \times B_2 = [0, 0, 0]^T$ (Axiom 1 passes).
*   **Expected Invariants:** 
    *   $\beta_0 = 1$.
    *   $\beta_1 = 3 - 2 - 1 = 0$ (Hole has been contracted / filled).
    *   Euler Characteristic $\chi = 3 - 3 + 1 = 1$.

### Test Case 3: Disjoint Edges (Two Separate 1-Simplices)
*   **Geometry:** 4 Vertices ($v_0, v_1, v_2, v_3$), 2 Edges ($e_0 = [v_0, v_1]$, $e_1 = [v_2, v_3]$).
*   **Matrix $B_1$ ($4 \times 2$):**
    $$B_1 = \begin{bmatrix} -1 & 0 \\ 1 & 0 \\ 0 & -1 \\ 0 & 1 \end{bmatrix}$$
*   **Expected Invariants:**
    *   $\text{rank}(B_1) = 2$.
    *   $\beta_0 = 4 - 2 = 2$ (Two independent components).
    *   $\beta_1 = 2 - 2 = 0$.
    *   Euler Characteristic $\chi = 4 - 2 = 2$.