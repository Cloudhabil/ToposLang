# ToposLang (.tau) POC / MVP Test Environment Specification

> **Objective**: Define the criteria, processes, and deployment architecture required to evaluate ToposLang (`.tau`) in an isolated, production-grade test environment as a Proof of Concept (POC) / Minimum Viable Product (MVP).

---

## 1. Scope & MVP Objectives

The primary goal of the MVP is to demonstrate that **computation governed by topological invariants and spatial concurrency** operates reliably, deterministically, and race-free on distributed/agent-based workflows.

### Core MVP Deliverables:
1. **Full Language Pipeline Execution**: End-to-end `.tau` compilation (`tauc`) and execution (`tau-run`).
2. **Topological Invariant Control**: Dynamic termination and branching governed by Betti numbers ($\beta_0, \beta_1$) and Euler characteristic ($\chi$).
3. **Spatial Parallelism Validation**: Automatic detection and lock-free parallel execution of spatially disjoint agent domains (e.g. Cluster A vs Cluster B in `06_spatial_agents.tau`).
4. **Automated Verification Harness**: Continuous boundary nilpotency ($d^2 = 0$) and homology integrity checking throughout program execution.

---

## 2. POC / MVP Evaluation Criteria

### 2.1 Functional Criteria (Acceptance Gates)
| ID | Criterion | Description | Target Metric |
|---|---|---|---|
| **F-01** | **Lexing & Parsing Robustness** | Ingest `.tau` source declaring 0-cells, 1-cells, 2-cells, HITs, and invariant processes. | 100% valid AST generation, descriptive syntax error locations (line/column). |
| **F-02** | **Boundary Nilpotency ($d^2 = 0$)** | Verify that every valid cell and attached homotopy satisfies $\partial_{k-1} \circ \partial_k = 0$. | Zero invalid boundary complexes admitted; 100% rejection of non-closed cycles. |
| **F-03** | **Exact Invariant Computation** | Compute Betti numbers $\beta_k$ via Smith Normal Form / rank without floating-point inaccuracies. | Exact integer invariants for all canonical shapes ($S^1$, $T^2$, agent meshes). |
| **F-04** | **Invariant Termination** | Invariant loops (`until betti(complex, dim=1) == 0`) terminate when topological cycles are contracted. | Deterministic loop termination within configured iteration bounds. |
| **F-05** | **Spatial Disjoint Wavefronts** | Detect mutually disjoint sub-complexes via topological closure and dispatch in parallel. | Zero shared vertices/edges across concurrent wavefront threads. |

### 2.2 Non-Functional & Operational Criteria
| ID | Criterion | Description | Target Metric |
|---|---|---|---|
| **NF-01** | **Zero Lock Contention** | Concurrency achieved via geometric disjointness rather than mutexes/locks. | 0 deadlocks, 0 lock contention overhead across thread workers. |
| **NF-02** | **Portability & Isolation** | Standard Library execution without external binary C/Fortran bindings. | Pure Python 3.13 standard library; packaged in Docker container < 200MB. |
| **NF-03** | **Execution Telemetry** | Rich execution traces with step-by-step invariant changes and exportable visual graphs. | JSON execution logs and SVG/DOT topology export per run. |
| **NF-04** | **Reproducibility** | Repeated execution of the same `.tau` program under identical initial conditions. | 100% identical homology trajectory and final complex structure. |

---

## 3. Reference MVP Use Case: Multi-Agent Spatial Coordination

The canonical benchmark program is [`examples/06_spatial_agents.tau`](file:///home/eliasbrahim/topos-lang/examples/06_spatial_agents.tau):
- **Domain Alpha (Cluster A)**: States `TaskA_Init`, `TaskA_Running`, `TaskA_Done` with 1-cells `dispatch_A`, `execute_A`, `reconcile_A` (open 1-cycle).
- **Domain Beta (Cluster B)**: States `TaskB_Init`, `TaskB_Running`, `TaskB_Done` with 1-cells `dispatch_B`, `execute_B`, `reconcile_B` (open 1-cycle).
- **Initial Invariants**: $\chi = 0$, $\beta_0 = 2$ (2 connected components), $\beta_1 = 2$ (2 independent active task cycles).
- **Execution Target**:
  - `SpatialPartitioner` assigns `resolve_alpha` and `resolve_beta` to Wavefront 0.
  - Workers simultaneously attach 2-cells `homotopy_resolve_alpha_0` and `homotopy_resolve_beta_0`.
  - Final Invariants: $\beta_0 = 2$, $\beta_1 = 0$ (all cycles resolved, zero deadlocks).

---

## 4. Test Environment Architecture & Setup Process

### Step 1: Hermetic Containerization (`Dockerfile`)
Deploy a minimal container environment ensuring reproducible execution across Linux kernels:
- Base: `python:3.13-slim`
- Environment Variables: `PYTHONPATH=/app/src`
- Entrypoints: `/app/bin/tauc`, `/app/bin/tau-run`

### Step 2: Automated Verification Pipeline (CI / Testbed)
A three-tier test pipeline:
1. **Unit & Axiomatic Test Suite**: Run `python3 -m unittest discover tests` (55 baseline tests covering cells, boundary nilpotency, homology, parsing, concurrency).
2. **Stress & Scalability Test**: Scale agent mesh from 2 domains to $N=50$ disjoint clusters to measure wavefront partitioning and throughput.
3. **Negative / Fault Injection Test**: Inject conflicting edges or boundary non-closures to verify error handling and nilpotency failure traps.

### Step 3: Observability & Telemetry Verification
1. Export JSON execution trace (`--trace-json trace.json`).
2. Generate SVG topological graph snapshots before and after execution (`--export-svg`).
3. Verify invariant delta logs: ensure transition $\beta_1: 2 \to 0$ is explicitly auditable.

---

## 5. MVP Sign-Off Checklist

- [ ] Hermetic container builds and executes cleanly (`docker build -t topos-lang:mvp .`).
- [ ] `./bin/tauc examples/06_spatial_agents.tau` compiles and verifies $\beta_0=2, \beta_1=2$.
- [ ] `./bin/tau-run examples/06_spatial_agents.tau` executes and resolves to $\beta_1=0$.
- [ ] 100% test suite pass rate without external dependencies.
- [ ] Concurrency test suite proves parallel execution of disjoint wavefronts without synchronization primitives.
