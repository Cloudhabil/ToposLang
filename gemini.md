# ToposLang (τ-Lang): Higher-Dimensional Topological Programming Language

> **Project Mission**: Design, specify, and build a next-generation programming language where data is represented as $n$-dimensional cell complexes (polygraphs), computation is modeled as $(n+1)$-dimensional continuous topological transformations (homotopies and cobordisms), and program flow is governed by topological invariants.

---

## 1. Executive Summary & Core Philosophy

Traditional programming paradigms model computation as a 1D sequence of instructions mutating discrete, linear memory locations:
- **Imperative**: Turing machines / state-variable mutation over flat address spaces.
- **Functional**: Lambda calculus ($\lambda$-calculus) / 1D term substitution.
- **Object-Oriented**: Encapsulated state machines passing discrete messages.

**ToposLang** introduces a fundamental paradigm shift:
1. **Data as $n$-Cells**: Data is not a flat array or pointer graph; it is a topological space structured as an $n$-dimensional cell complex (0-cells = points/objects, 1-cells = 1D directed paths/morphisms, 2-cells = 2D surfaces/homotopies, $n$-cells = higher equivalences).
2. **Computation as $(n+1)$-Cells**: A step of computation is an $(n+1)$-dimensional continuous rewrite rule (a homotopy or cobordism) transforming an input boundary $\partial^- r$ into an output boundary $\partial^+ r$.
3. **Equality as Paths (HoTT)**: Equality is not a boolean comparison (`==`). In accordance with Homotopy Type Theory, the identity type $x =_A y$ is a continuous path space connecting points $x$ and $y$.
4. **Invariant-Driven Control Flow**: Loops and subroutines do not terminate on arbitrary floating-point loss thresholds (`while loss > 0.001`). Instead, they evaluate discrete topological invariants:
   - Betti numbers $\beta_k = \dim(H_k)$ (e.g., terminate when $\beta_1 = 0$, guaranteeing all loops/holes have been contracted).
   - Euler characteristic $\chi = \sum (-1)^k \beta_k$.
   - Cobordism charges $\Omega_k = 0$.
5. **Intrinsic Concurrency**: Two computations on topologically disjoint sub-complexes that share no boundaries execute simultaneously by geometric definition—unlocking race-condition-free spatial parallelism without mutexes or locks.

### 1.2 State of the Art & Deficit Resolution

Existing paradigms explore aspects of higher topology but suffer from foundational limitations:
- **Dependently Typed Proof Assistants (Lean 4, Cubical Agda)**: Excel at formal boundary verification ($d^2 = 0$) and HoTT, but are designed for interactive theorem proving rather than high-throughput runtime rewriting or GPU tensor lowering.
- **Applied Category Theory (Catlab.jl / AlgebraicJulia)**: Provide expressive string diagrams and functorial structures, but remain embedded DSLs in general-purpose languages lacking invariant-driven control flow and native compiler pipelines.
- **Higher-Dimensional Rewriting (alifib, rewalt, Homotopy.io)**: Treat computation directly as higher-cell homotopies, but are bottlenecked by **superpolynomial subdiagram matching** and lack pathways to physical hardware.
- **Geometric DL & TDA (TopoModelX, Aether-Lang)**: Accelerate matrix calculations on GPUs, but treat complexes as static index arrays without dynamic polygraphic rewriting or formal homotopy type safety.

**ToposLang's Resolution**: A progressive compiler pipeline that combines formal automated verification ($d^2 = 0$) with Directed Acyclic Cell Complexes (DACC) to reduce matching complexity from superpolynomial to polynomial, then lowers cellular complexes into sparse incidence matrices and Hodge Laplacians for hardware execution.

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Cellular Complexes and Polygraphs
A $k$-cell $c^k$ possesses an oriented boundary composed of $(k-1)$-cells:
$$\partial c^k = \partial^+ c^k - \partial^- c^k$$

### 2.2 Boundary Operator Nilpotence
Every cellular structure in ToposLang must strictly satisfy the fundamental homology condition:
$$\partial_{k-1} \circ \partial_k = 0 \quad \text{and} \quad d_{k} \circ d_{k-1} = 0$$
This boundary consistency ensures that "the boundary of a boundary is empty", preventing mathematically invalid topological configurations at compile-time.

### 2.3 Algebraic Topology & Invariant Extraction
- **Boundary Matrices ($B_k$)**: Linear incidence maps from $k$-chains $C_k$ to $(k-1)$-chains $C_{k-1}$.
- **Hodge Laplacians ($L_k$)**:
  $$L_k = B_k^T B_k + B_{k+1} B_{k+1}^T$$
  The kernel $\ker(L_k)$ is isomorphic to the $k$-th real homology group $H_k$, giving exact topological invariants $\beta_k = \dim(\ker(L_k))$.

### 2.4 Cellular Sheaves & Continuous Data Fibers
Beyond discrete combinatorial cells, ToposLang models continuous physics, gauge fields, and distributed data via **Cellular Sheaves** $(\mathcal{F}, \mathcal{E})$ over cell complexes:
- **Stalks / Data Fibers $\mathcal{F}(c)$**: Vector spaces $\mathbb{R}^d$ or algebraic modules associated with each cell $c^k$.
- **Restriction Maps $\mathcal{E}_{u \trianglelefteq v} : \mathcal{F}(u) \to \mathcal{F}(v)$**: Linear maps transporting data between incident faces, enabling sheaf Laplacians and continuous message passing.

---

## 3. The 4-Stage Progressive Compiler Pipeline

To resolve the two fundamental bottlenecks of higher-dimensional computing (**Subdiagram Matching Complexity** and **Silicon Hardware Mismatch**), ToposLang employs a multi-tiered compilation pipeline:

```mermaid
flowchart TD
    subgraph S1 [Stage 1: Front-End]
        Source[".tau Source Code"] --> LexerParser["Lexer & AST Parser"]
        LexerParser --> TypeChecker["Boundary & Homotopy Type Checker\nValidates: d_{k+1} ∘ d_k = 0"]
    end

    subgraph S2 [Stage 2: Topological IR (T-IR)]
        TypeChecker --> DACC["Directed Acyclic Cell Complex (DACC)"]
        DACC --> Contractions["Cellular Contractions\nCollapse Contractible Sub-cells"]
        Contractions --> InvariantSlices["Invariant Boundary Slicing"]
    end

    subgraph S3 [Stage 3: Geometric Lowering IR]
        InvariantSlices --> Cochains["Cochain & Incidence Matrices (B_k)"]
        Cochains --> HodgeEngine["Hodge Laplacians (L_k) & Sheaves"]
        HodgeEngine --> TensorNet["Tensor Networks / Einsum Graph"]
    end

    subgraph S4 [Stage 4: Hardware CodeGen & Execution]
        TensorNet --> SpatialScheduler["Spatial Cell Partitioning\nDisjoint Cells -> Parallel Execution Units"]
        SpatialScheduler --> ExecBackend["Vectorized Sparse CPU/GPU Runtimes"]
    end
```

### Stage Breakdown:
- **Stage 1 (Front-End & Verification)**: Ingests cellular definitions, verifying that source/target boundaries match and the nilpotent boundary axiom holds.
- **Stage 2 (Topological IR - T-IR)**: Solves the matching complexity bottleneck. Enforces directed acyclic orderings (reducing matching from superpolynomial to polynomial) and applies cellular contractions to simplify contractible sub-shapes.
- **Stage 3 (Geometric Lowering IR)**: Solves the hardware mismatch. Maps string diagrams and polygraphs into boundary incidence matrices $B_k$, discrete differential forms, and tensor networks.
- **Stage 4 (Execution & Hardware CodeGen)**: Dispatches independent, disjoint cell rewrites across parallel worker threads/compute blocks without synchronization contention.

---

## 4. Syntax & Language Design (ToposLang DSL)

### 4.1 Declaring Cells & Morphisms
```tau
// 0-Cells (Vertices / Types)
cell0 A;
cell0 B;
cell0 C;

// 1-Cells (Paths / Operations)
cell1 f : A -> B;
cell1 g : B -> C;
cell1 h : A -> C;

// 2-Cell (Homotopy / 2D Surface / Rewrite Rule)
// α transforms path (f ∘ g) into h
cell2 alpha : (f * g) => h;
```

### 4.2 Invariant-Driven Control Flow
```tau
// An optimization process governed by 1-dimensional topological holes
process contract_vortices(complex: CellComplex) {
    rewrite loop_contraction : (e1 * e2) => e3;
    
    // Executes until the 1st Betti number drops to 0 (all loops closed)
    until betti(complex, dim=1) == 0 {
        apply loop_contraction on complex;
    }
}
```

### 4.3 Higher Inductive Type (HIT) Declaration
```tau
// Definition of a 1-Sphere (Circle S¹)
hit S1 {
    point base;
    path loop : base = base;
}

// Definition of a Torus T² = S¹ × S¹
hit Torus {
    point b;
    path p : b = b;
    path q : b = b;
    surface face : (p * q) = (q * p);
}
```

---

## 5. Repository & Architecture Layout

```
topos-lang/
├── gemini.md               # Master project blueprint & roadmap (this document)
├── README.md               # High-level introduction and getting started
├── pyproject.toml          # Project configuration & metadata
├── src/
│   └── topos/
│       ├── __init__.py
│       ├── core/           # Mathematical foundation
│       │   ├── cell.py     # Cell, Chain, and Complex abstractions
│       │   ├── boundary.py # Boundary maps & nilpotence checker (d ∘ d = 0)
│       │   └── polygraph.py# n-polygraph and pasting diagrams
│       ├── topology/       # Algebraic topology engine
│       │   ├── homology.py # Betti numbers, Smith Normal Form / rank
│       │   ├── laplacian.py# Hodge Laplacian L_k calculation
│       │   └── invariant.py# Euler characteristic and cobordism charges
│       ├── frontend/       # Parsing & AST
│       │   ├── ast.py      # Abstract syntax tree for τ-Lang
│       │   ├── lexer.py    # Tokenizer
│       │   └── parser.py   # Recursive-descent parser for .tau files
│       ├── ir/             # Topological Intermediate Representation
│       │   ├── tir.py      # Directed Acyclic Cell Complex IR
│       │   ├── contract.py # Cellular contraction passes
│       │   └── lower.py    # Lowering T-IR to matrix/tensor representations
│       └── runtime/        # Execution & concurrency engine
│           ├── engine.py   # Rewriting interpreter
│           ├── scheduler.py# Spatial disjoint-cell parallel scheduler
│           └── visualizer.py# ASCII / SVG / Graphviz complex visualizer
├── examples/               # Sample Topos programs (.tau)
│   ├── 01_circle_hit.tau
│   ├── 02_loop_collapse.tau
│   └── 03_pasting_diagram.tau
└── tests/                  # Unit and integration test suite
    ├── test_cells.py
    ├── test_boundary_nilpotence.py
    ├── test_homology.py
    ├── test_parser.py
    └── test_runtime_rewriting.py
```

---

## 6. Phased Implementation Roadmap

| Milestone | Deliverables | Verification Criteria | Status |
|---|---|---|---|
| **M1: Mathematical Core** | $n$-Cell, Polygraph, Chain Complex, Nilpotency Verifier ($d^2 = 0$). Zero external dependencies (pure Python standard library). | Validates $d_{k-1}(d_k(c)) = 0$ on simplices, cubes, and string diagrams. | ✅ Completed |
| **M2: Algebraic Topology & Invariants** | Boundary matrices $B_k$, Betti numbers $\beta_k$, Euler characteristic $\chi$, Hodge Laplacians $L_k$. | Correctly computes $\beta_0=1, \beta_1=1$ for $S^1$, and $\beta_0=1, \beta_1=2, \beta_2=1$ for $T^2$. | ✅ Completed |
| **M3: Lexer, Parser & AST** | Tokenizer and recursive descent parser for `.tau` scripts declaring cells, HITs, rewrites, and invariant loops. | Round-trip AST parsing for standard test suites. | ✅ Completed |
| **M4: Topological Rewriting Runtime** | Polygraphic pattern matching and rewrite execution engine with invariant stopping conditions. | Executes loop collapse until $\beta_1 == 0$. | ✅ Completed |
| **M5: Lowering & Disjoint Concurrency** | Spatial boundary partitioner; detects disjoint sub-complexes and executes rewrites concurrently via thread pools. | Benchmark concurrent execution on disjoint sub-complexes with 0 lock contention. | ✅ Completed |
| **M6: Visualizer & Tooling CLI** | CLI `tauc` and `tau-run`, with 2D/3D SVG & Graphviz export of topological execution traces. | Visualizes homotopy deformation traces as diagrams. | ✅ Completed |

---

## 7. Project Status & Future Work

All six foundational milestones of **ToposLang (τ-Lang)** have been implemented with zero external dependencies (pure Python 3.13 standard library) and verified with a 100% passing test suite across 55 automated unit and integration tests.

### Future Native Lowering & Extensions
1. **Cellular Sheaves & Continuous Physics**: Attaching vector stalks $\mathcal{F}(c)$ and restriction maps $\mathcal{E}_{u \trianglelefteq v}$ to cell complexes for gauge fields and continuous neural message passing.
2. **Native MLIR / LLVM Backend (Stage 3 & 4)**: Compiling cellular incidence matrices and tensor network contractions directly into high-throughput GPU/TPU kernels.
3. **Language Server Protocol (LSP)**: Real-time topological invariant diagnostics and diagrammatic previews inside code editors.

