# ToposLang (τ-Lang)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg?style=for-the-badge&logo=creativecommons&logoColor=black" alt="License: CC BY 4.0" />
  <img src="https://img.shields.io/badge/Tests-232%20Passed-success?style=for-the-badge" alt="Tests" />
  <img src="https://img.shields.io/badge/Dependencies-Zero_External-blueviolet?style=for-the-badge" alt="Zero Dependencies" />
  <img src="https://img.shields.io/badge/Architecture-Geometric_Native_Coding-orange?style=for-the-badge" alt="GNC" />
</p>

---

## About

**ToposLang ($\tau$-Lang)** is an open-source, pure-Python programming language and computational engine designed for **Geometric Native Coding (GNC)**, **Hodge-Clifford Spatial Transformers (HCST)**, and **Multi-Prime Galois-Clifford Cellular Sheaves**. 

Created by **Elias Oulad Brahim** ([Cloudhabil](https://github.com/Cloudhabil)), ToposLang addresses the foundational bottlenecks of modern computing identified in the monograph and research papers:
1. **The Concurrency Wall:** Resolves lock contention by treating computation as rewrite operations over oriented, graded Directed Acyclic Cell Complexes (DACC) with automatic **Dual-Tier Spatial Wavefront Scheduling** (Theorem 2.2, zero mutexes).
2. **The Heuristic Termination Flaw:** Replaces arbitrary floating-point convergence thresholds ($\|x_{k+1} - x_k\| < \epsilon$) with **provable homological invariants** (e.g., executing cellular loops until deadlock cycles vanish: `until betti(K, dim=1) == 0`).
3. **Coordinate Fragmentation:** Eliminates Euler angle singularities, quaternions, and matrix conversions by embedding continuous spatial geometry in standardized **$3\text{D}$ Projective Geometric Algebra ($\mathrm{Spin}(3,0,1) \cong \mathrm{SE}(3)$)**.

4. **Non-Archimedean Metric Resolution:** Replaces floating-point SVD degradation with exact **Multi-Prime Chinese Remainder Theorem (CRT)** reduction, while restoring continuous distance comparisons through **Homogeneous Rational Quadratic Forms** and **Garner's Mixed-Radix Re-Lifting**.

ToposLang operates with **100% pure Python and zero external runtime dependencies** (no NumPy, SciPy, or C extensions required).

---

## Core Capabilities

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                               TOPOSLANG ARCHITECTURE                              │
├───────────────────────────────────────────────────────────────────────────────────┤
│  Topos Frontend (.tau)  ──► Lexer & Recursive Descent Parser                      │
│                                      │                                            │
│  Static Homological Type Checker     ▼ Strict Boundary Nilpotency (∂² = 0)        │
│                                      │                                            │
│  Dual-Stream HCST Neural Engine     ──► Metric Alignment & Exterior Wedge Meet   │
│  Multi-Prime CRT Galois Sheaves     ──► 64-bit Prime Lanes, F_{p²}, Garner Lift   │
│  Dual-Tier Spatial Wavefront Arbiter──► Lock-Free Concurrency, 0 Mutexes          │
│  Constructive QFT Modules           ──► Screened Hodge-Dirac, Creutz, Dirac-Kähler│
└───────────────────────────────────────────────────────────────────────────────────┘
```

### 1. Graded Cell Complexes and Boundary Nilpotency
Every spatial domain in ToposLang is represented as a graded cell complex $K = K_0 \sqcup K_1 \sqcup K_2 \sqcup K_3 \sqcup K_4$ governed by the universal axiom of **Boundary Nilpotency**:
$$\partial_{k-1} \circ \partial_k \equiv 0 \iff B_{k-1} B_k = \mathbf{0}$$
Programs with dangling boundary references or torn topological manifolds are rejected statically at compile time.

### 2. Hodge-Clifford Spatial Transformers (HCST)
ToposLang natively implements the full architecture formalized in [SSRN-7618742](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7618742):
* **Multivector Stalks:** Feature channels represented as Clifford multivectors ($Cl(3,0,1)^C$).
* **Equivariant QKV Projections:** $\mathcal{P}_\Theta(A_{k,i}) = \sum_{g=0}^d W_k^{(g)} \langle M_k A_{k,i} \widetilde{M}_k \rangle_g$.
* **Dual-Stream Clifford Attention:** Evaluates metric alignment $\langle Q \widetilde{K} \rangle_0$ against exterior wedge incidence distance $\|Q \wedge K\|_{Cl}^2$.
* **4-Channel Combinatorial Hodge-Dirac Routing:** Aggregates information orthogonally across lower divergence ($N_\downarrow$), upper curl ($N_\uparrow$), boundary faces ($\partial$), and co-faces ($\delta$).
* **Homology-Preserving Topological Pooling:** Reduces mesh complexity via strict free-pair elementary collapses ($\deg_\delta(e_i^{k-1}) = 1$) with certified chain deformation retractions $(\pi_\bullet, \iota_\bullet, h_\bullet)$, preserving Betti numbers and Euler characteristic.
* **Differentiable Backprop across Topology:** Uses functorial adjoint pullbacks and heat-kernel spectral surrogates ($\widetilde{\beta}_k(z; \tau_{\text{heat}}) = \mathrm{Tr}(\exp(-\tau_{\text{heat}} L_k(z)))$).

### 3. Multi-Prime CRT & Non-Archimedean Dual-Lifting
* **Zero Fraction Expansion:** $M$-lane 64-bit residue arithmetic ($p_m \equiv 3 \pmod 4$) in $(2^{62}, 2^{63})$ eliminates arbitrary-precision integer bit-growth during higher-order matrix reductions.
* **Exact Rank Recovery:** $\mathrm{rank}_\mathbb{Q}(B_k) = \max_m \mathrm{rank}_{\mathbb{F}_{p_m}}(B_k \bmod p_m)$.
* **Square-Root-Free Distance Queries:** Compares physical distances $d(P_1, P_2) < a/b$ via exact rational quadratic forms:
  $$b^2 \left[ (X_1 W_2 - X_2 W_1)^2 + (Y_1 W_2 - Y_2 W_1)^2 + (Z_1 W_2 - Z_2 W_1)^2 \right] < a^2 (W_1 W_2)^2$$
* **Garner Mixed-Radix Re-Lifting:** Solves boolean sign and ordering queries ($A > B$) at control loop interfaces in $\mathcal{O}(M^2)$ operations with zero floating-point error.
* **Compiler Roofline Contract:** Enforces static invariant $L \cdot d \ge 19$ for 100% compute-bound execution ($\ge 16.5\text{ OPs/Byte}$) without memory bandwidth stalling.

### 4. Zero-Lock Dual-Tier Wavefront Scheduling
Executes concurrent spatial rewrites across parallel execution cores using the **Dual-Tier Interference Predicate**:
$$\mathrm{Interference}(r_1, r_2) = (\mathrm{supp}(r_1) \cap \mathrm{supp}(r_2) \neq \emptyset) \wedge (S(r_1) \wedge S(r_2) \neq 0)$$
If `False`, rules commute identically ($r_1 \circ r_2 \equiv r_2 \circ r_1$) and execute in parallel with **zero mutex locks and zero cache-line invalidations**.


### 5. Constructive Quantum Field Theory & Gauge Physics
* **Dynamically Screened Hodge-Dirac Operator:** Case B discrete operator $\mathcal{D}_{\text{DACC}}[U]$ with exact oriented discrete Bianchi closure.
* **Emergent String Tension from Bare Wilson Action:** Measures Creutz ratios $\chi(R, T) \to a^2 \sigma > 0$ with no explicit $\Lambda_{\text{YM}}$ in the Lagrangian.
* **Dirac-Kähler Fermion Doubling Resolution:** Maps 16 spinor components onto graded cochains $C^0 \oplus C^1 \oplus C^2 \oplus C^3 \oplus C^4$ with exactly 1 physical pole (resolving Nielsen-Ninomiya).
* **Reflection-Positive Pace Operator:** Proves cone forward-invariance $\mathcal{R}(\mathcal{C}) \subseteq \mathcal{C}$ with certified Theorem 16.28 contraction floor $1 - \phi_{\text{out}} = 0.1875 > 0$.

---

## When to Apply ToposLang

| Domain | Problem in Standard Stacks | ToposLang Solution |
| :--- | :--- | :--- |
| **Robotics & Kinematics** | Gimbal lock; quaternion-matrix conversions; float drift | Standardized $\mathrm{Spin}(3,0,1)$ motors ($M = TR$, $M\widetilde{M} = 1$); coordinate-free sandwiches |

| **Safety & Collision Detection** | Rounding errors classify boundary obstacles incorrectly | Exact rational quadratic distance forms; zero square roots; Garner sign lifting |
| **Topological Deep Learning** | Mesh downsampling destroys cycles/holes; ad-hoc GNNs | HCST dual-stream attention; homology-preserving free-pair pooling; heat-kernel surrogates |
| **Multi-Agent Coordination** | Mutex lock contention; deadlocks undetected | Theorem 2.2 dual-tier lock-free wavefronts; termination loops governed by `betti(K, 1) == 0` |
| **Scientific Computing** | Rational RREF bit-growth explodes memory on large meshes | 64-bit multi-prime CRT modular reduction with static Roofline guarantee ($L \cdot d \ge 19$) |
| **Lattice Gauge Theory** | Nielsen-Ninomiya species doubling; continuum mass gap unproved | DACC Dirac-Kähler 1-pole resolution; reflection-positive Pace operator; bare action Creutz ratios |

---

## Installation & Requirements

ToposLang requires **Python 3.10+** and has **zero external package dependencies**.

```bash
# Clone the repository
git clone https://github.com/Cloudhabil/ToposLang.git
cd ToposLang

# Verify the full test suite (232 tests)
PYTHONPATH=src python3 -m unittest discover -s tests

# Run the complete MVP acceptance suite
./bin/run-mvp-suite
```

---

## Quickstart & Usage

### 1. Running ToposLang Programs (`.tau`)

#### Static Topos Analysis (`tauc`)
Statically verifies boundary nilpotency ($\partial^2 = 0$), type consistency, and topological invariants:
```bash
./bin/tauc examples/06_spatial_agents.tau
```

#### Homotopy Execution Runtime (`tau-run`)
Executes invariant-driven cellular homotopy rewrites until homological termination:
```bash
./bin/tau-run examples/06_spatial_agents.tau
```

---

### 2. Authoring `.tau` Source Code

```tau
// Multi-Agent Spatial Synchronization (examples/06_spatial_agents.tau)

cell0 TaskA_Init;
cell0 TaskA_Running;
cell0 TaskA_Done;

cell1 dispatch_A : TaskA_Init -> TaskA_Running;
cell1 execute_A : TaskA_Running -> TaskA_Done;
cell1 reconcile_A : TaskA_Init -> TaskA_Done;

process synchronize_mesh(complex: CellComplex) {
    rewrite resolve_alpha : (dispatch_A * execute_A) => reconcile_A;

    // Control flow terminates deterministically when 1D deadlock cycles vanish
    until betti(complex, dim=1) == 0 {
        apply resolve_alpha on complex;
    }
}
```

---

### 3. Python API Usage

#### Rigid-Body Spin(3,0,1) Motors (PGA)

```python
import math
from topos.core.clifford import PGA301Motor, Multivector, PGA301

# Create rigid-body motor: 90 deg rotation around z-axis + translation (1, 2, 0)
motor = PGA301Motor.from_axis_angle_translation(
    axis=(0.0, 0.0, 1.0),
    angle=math.pi / 2,
    translation=(1.0, 2.0, 0.0)
)
assert motor.is_unit_motor()  # M * ~M == 1

# Transform a point via sandwich action M * X * ~M
point = Multivector.from_named_blades(PGA301, {"e123": 1.0})
transformed = motor.transform(point)
```

#### Exact Square-Root-Free Distance Queries
```python
from topos.core.projective_metric import ProjectivePoint, ProjectiveMetricEngine

engine = ProjectiveMetricEngine()
p1 = ProjectivePoint(0, 0, z=0, w=1)
p2 = ProjectivePoint(3, 4, z=0, w=1)  # Distance = 5

# Check if d(p1, p2) < 5.5 (represented as 11/2) with zero floating-point error
is_safe = engine.is_within_radius_rational(p1, p2, radius_num=11, radius_den=2)
assert is_safe is True
```

#### Multi-Prime CRT & Exact Rational Rank
```python
from topos.core.crt import CRTMatrixRREF, DEFAULT_CRT_PRIMES

# Boundary matrix of a triangle (rank over Q is 2)
B1 = [
    [-1,  0, -1],
    [ 1, -1,  0],
    [ 0,  1,  1],
]

rref_engine = CRTMatrixRREF(DEFAULT_CRT_PRIMES)
ranks, rational_rank = rref_engine.compute_ranks_across_lanes(B1)
assert rational_rank == 2  # Recovered with 0.0% error and zero bit-growth
```

#### Hodge-Clifford Spatial Transformer (HCST) Layer
```python
from topos.core.four_tier_dacc import create_4_hypercube
from topos.physics.hcst import HCSTTransformerLayer
from topos.core.clifford import Multivector, PGA301

dacc = create_4_hypercube()
layer = HCSTTransformerLayer(dacc)

# Execute forward pass with equivariant attention and isometric rotor updates
cochain = {c.name: Multivector.scalar(PGA301, 1.0) for c in dacc.get_cells(dim=1)}
updated_cochain = layer.forward(dim=1, cochain_k=cochain)
```

---

## Repository Structure

```
topos-lang/
├── bin/
│   ├── tauc                     # Static compiler and topological analyzer
│   ├── tau-run                  # Homotopy execution and rewrite engine
│   └── run-mvp-suite            # Complete acceptance verification test runner
├── src/topos/
│   ├── core/
│   │   ├── clifford.py          # Multivectors, PGA(3,0,1), CGA, standardized motors
│   │   ├── crt.py               # 64-bit multi-prime CRT, F_{p^2}, Garner's algorithm
│   │   ├── projective_metric.py # Exact rational quadratic forms, PGA line meets
│   │   ├── four_tier_dacc.py    # 4D DACC, 4-tier boundary nilpotency (B_k B_{k+1} = 0)
│   │   ├── matrix_csr.py        # Exact rational & finite-field CSR operators, RREF
│   │   └── pro_cell_complex.py  # Dyadic multiscale projective limit complexes
│   ├── physics/
│   │   ├── hcst.py              # Hodge-Clifford Spatial Transformer architecture
│   │   ├── pace_operator.py     # Reflection-positive pace operator & mass gap
│   │   ├── dynamical_screening.py # Case B dynamically screened Hodge-Dirac operator
│   │   ├── creutz_ratio.py      # Bare Wilson action Metropolis & Creutz ratios
│   │   └── dirac_kahler.py      # Nielsen-Ninomiya fermion doubling resolution
│   ├── runtime/
│   │   ├── wavefront.py         # Dual-tier lock-free spatial wavefront scheduler
│   │   ├── engine.py            # Homotopy rewrite execution engine
│   │   └── interpreter.py       # .tau AST interpreter
│   └── compiler/
│       ├── roofline.py          # Static Roofline compiler contract invariant pass
│       └── lean4_emitter.py     # Lean 4 proof scaffold generator
├── examples/                    # Runnable .tau programs (01 through 11)
├── scripts/                     # Monte Carlo ensembles, scaling sweeps, Λ_YM extraction
└── tests/                       # 232 unit tests verifying 100% of mathematical claims
```

---

## Research Publications & References

ToposLang is grounded in the mathematical formalisms introduced in:

1. **Oulad Brahim, Elias**, *Topological and Geometric Native Coding for Higher Dimensional Programming* (September 28, 2026). Available at SSRN: [https://ssrn.com/abstract=7536578](https://ssrn.com/abstract=7536578) or [http://dx.doi.org/10.2139/ssrn.7536578](http://dx.doi.org/10.2139/ssrn.7536578).
2. **Oulad Brahim, Elias**, *ToposLang: An Invariant-driven Topological Programming Language and Lock-free Spatial Concurrency Runtime Over Higher-dimensional Cell Complexes* (September 21, 2026). Available at SSRN: [https://ssrn.com/abstract=7498298](https://ssrn.com/abstract=7498298) or [http://dx.doi.org/10.2139/ssrn.7498298](http://dx.doi.org/10.2139/ssrn.7498298).
3. **Oulad Brahim, Elias**, *Bounded Kan Composition, Clifford Stalk Closures, and Invariant-Driven Cell Rewriting in Topological and Geometric Native Coding for Higher Dimensional Programming Languages* (October 01, 2026). Available at SSRN: [https://ssrn.com/abstract=7548324](https://ssrn.com/abstract=7548324) or [http://dx.doi.org/10.2139/ssrn.7548324](http://dx.doi.org/10.2139/ssrn.7548324).
4. **Oulad Brahim, Elias**, *Hodge-Clifford Spatial Transformers: A Unified Topological and Geometric Native Architecture for Higher-dimensional Programming and Silicon Coprocessors* (October 16, 2026). Available at SSRN: [https://ssrn.com/abstract=7618742](https://ssrn.com/abstract=7618742) or [http://dx.doi.org/10.2139/ssrn.7618742](http://dx.doi.org/10.2139/ssrn.7618742).


---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

[![License: CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

You are free to share and adapt the material for any purpose, including commercially, provided appropriate credit is given to the author (**Elias Oulad Brahim / Cloudhabil**).
