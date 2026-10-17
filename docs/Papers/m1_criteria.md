# Milestone 1: Mathematical, Technical & Verification Criteria

> **Objective**: Establish the core mathematical engine for ToposLang, defining $n$-cells, chain complexes, oriented boundary operators, and an automated formal verifier guaranteeing the topological nilpotency axiom:
> $$\partial_{k-1} \circ \partial_k = 0 \quad (\text{or } d_{k+1} \circ d_k = 0)$$

---

## 1. Mathematical Criteria

The implementation must strictly adhere to algebraic topology and higher category theory axioms:

### 1.1 Graded $n$-Cell Hierarchy
- **Dimension ($k \ge 0$)**: Every cell $c \in K_k$ must have an immutable, non-negative integer dimension $k$.
- **0-Cells (Objects / Vertices)**: Defined without boundaries ($\partial c^0 = 0$).
- **1-Cells (Arrows / Morphisms)**: Defined with an oriented source $\partial^- c^1$ and target $\partial^+ c^1$ (both 0-cells), yielding boundary $\partial c^1 = \partial^+ c^1 - \partial^- c^1$.
- **$k$-Cells ($k \ge 2$)**: Defined with source and target $(k-1)$-chains satisfying globular or simplicial composition laws.

### 1.2 Formal Chain Group ($C_k(K; \mathbb{Z})$)
- **Abelian Group Structure**: A $k$-chain $C = \sum_{i} a_i c_i^k$ is a formal linear combination of $k$-cells with coefficients $a_i \in \mathbb{Z}$.
- **Group Operations**:
  - Addition: $(C_1 + C_2)(c) = C_1(c) + C_2(c)$
  - Scalar Multiplication: $(n \cdot C)(c) = n \cdot C(c)$
  - Negation: $(-C)(c) = -C(c)$
  - Zero Chain: Canonical empty chain representing $0$.
- **Sparsity**: Zero-coefficient terms must automatically normalize and be pruned.

### 1.3 Boundary Operator ($\partial_k : C_k \to C_{k-1}$)
- **Linearity**:
  $$\partial_k\left(\sum_i a_i c_i^k\right) = \sum_i a_i \partial_k(c_i^k)$$
- **Simplicial Boundary Formula**: For any oriented $k$-simplex $[v_0, v_1, \dots, v_k]$:
  $$\partial_k([v_0, \dots, v_k]) = \sum_{j=0}^k (-1)^j [v_0, \dots, \hat{v}_j, \dots, v_k]$$
- **Globular Boundary Formula**: For a 2-cell rewrite $\alpha : f \Rightarrow g$ where $f, g : A \to B$:
  $$\partial^-(\alpha) = f, \quad \partial^+(\alpha) = g, \quad \partial(\alpha) = g - f$$

### 1.4 Nilpotency Axiom ($d^2 = 0$)
- **Formal Guarantee**: For every $k \ge 2$ and every valid $k$-cell $c^k$:
  $$\partial_{k-1}(\partial_k(c^k)) \equiv 0$$
- **Closure Property**: Every face appearing in $\partial c^k$ must be a registered, valid $(k-1)$-cell in the parent complex $K$.

---

## 2. Technical & Architectural Criteria

### 2.1 Dependency Isolation (Zero External Dependencies)
- Must be implemented purely using the **Python 3.13 Standard Library** (`dataclasses`, `typing`, `collections`, `fractions`, `abc`).
- No external binary packages (NumPy, SciPy) strictly required for the core engine, guaranteeing 100% portability in any environment.
- Architecture must provide clean interfaces to plug in NumPy/BLAS acceleration later without rewriting core contracts.

### 2.2 Immutability, Identity & Hashing
- Cells must be **immutable** once constructed (`frozen=True` dataclasses or value objects).
- Stable hashing based on cell dimension and canonical boundary identity to ensure $O(1)$ dictionary lookups and set memberships.

### 2.3 Computational Complexity Targets
- **Chain Addition**: $O(m_1 + m_2)$ where $m_i$ is the number of non-zero terms in chain $i$ (not proportional to total complex size $|K_k|$).
- **Cell Boundary Retrieval**: $O(1)$ amortized lookup.
- **Nilpotency Validation**: $O(E)$ where $E$ is the number of boundary components of the tested cell.

### 2.4 Error Handling & Diagnostics
- Specific, custom domain exceptions:
  - `TopologicalValidationError`: Raised when cell dimensions or boundary attachments do not match.
  - `NilpotencyViolationError`: Raised with explicit diagnostic output showing the residual boundary chain when $\partial^2(c) \neq 0$.
  - `CellNotFoundError`: Raised when referencing a non-existent boundary cell.

---

## 3. Verification & Test Suite Criteria

The Milestone 1 test suite (`tests/`) must pass 100% of the following benchmarks:

### 3.1 Geometric Primitive Tests
1. **0-Simplex (Point)**:
   - Verify $\partial(v_0) = 0$.
2. **1-Simplex (Directed Edge $e: A \to B$)**:
   - $\partial(e) = B - A$.
   - $\partial^2(e) = \partial(B) - \partial(A) = 0 - 0 = 0$.
3. **2-Simplex (Oriented Triangle $[A, B, C]$)**:
   - $\partial([A,B,C]) = [B,C] - [A,C] + [A,B]$.
   - Compute $\partial(\partial([A,B,C]))$ and assert exact zero cancellation.
4. **3-Simplex (Oriented Tetrahedron $[A, B, C, D]$)**:
   - 4 triangular 2-faces.
   - Assert $\partial_1(\partial_2([A,B,C,D])) = 0$.

### 3.2 Topological Space (HIT) Tests
1. **Circle ($S^1$)**:
   - 1 point `base`, 1 edge `loop : base -> base`.
   - $\partial(loop) = base - base = 0$.
2. **Torus ($T^2$)**:
   - Single 2-cell with boundary word $a \cdot b \cdot a^{-1} \cdot b^{-1}$.
   - Verify closed surface property ($\partial^2 = 0$).

### 3.3 Negative Verification Tests
- Construct a corrupted 2-cell whose boundary edges do not form a closed cycle (e.g. disconnected boundary).
- Assert that the boundary verifier rejects it with `NilpotencyViolationError`, pinpointing the dangling boundary chain.

---

## 4. Deliverable Structure for Milestone 1

```
src/topos/core/
├── __init__.py         # Package exports
├── cell.py             # Cell, Simplex, GlobularCell definitions
├── chain.py            # Formal Chain over Z (abelian group operations)
├── complex.py          # CellComplex container and registry
└── boundary.py         # Boundary operator & NilpotencyVerifier engine

tests/
├── test_chain.py       # Chain addition, scalar multiplication, canonical zero
├── test_simplex.py     # Simplicial boundary formulas and delta-complexes
└── test_nilpotence.py  # d ∘ d = 0 verification suite (positive & negative tests)
```
