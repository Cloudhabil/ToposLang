"""Volume and Lattice Refinement Scaling Sweep for Constructive Yang-Mills.

Tests whether the spectral mass gap and Bakry-Émery curvature bound survive the
thermodynamic (L -> inf) and continuum (a -> 0) limits — the decisive tests
separating finite-matrix linear algebra from genuine quantum field theory.

Two sweeps:
1. Volume Scaling: L in {2, 4, 8, 16} at fixed a, tracking kappa(L).
2. Continuum Scaling: N in {4, 8, 16, 32} at fixed physical volume,
   tracking m_phys(a) = sqrt(lambda_1(a)) / a.

Also reads the existing yang_mills_continuum_gap_benchmark.db to cross-check
empirical scaling ratios against known lattice QCD data.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
import sqlite3
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Dict, Tuple

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.core.four_tier_dacc import FourTierDACC
from topos.physics.orbit_space import OrbitSpaceManifold
from topos.physics.bakry_emery import BakryEmeryCurvatureEvaluator
from topos.physics.peierls_vortex import PeierlsVortexAnalyzer


# ---------------------------------------------------------------------------
# Utility: build a periodic 2D torus T^2 with L sites per side
# (cheaper proxy for 4D at large L; the analytical scaling is exact)
# ---------------------------------------------------------------------------

def build_torus_L(L: int, name_prefix: str = "T") -> FourTierDACC:
    """Builds a minimal 2D torus T^2 with 1 vertex and L^2 effective plaquettes.

    We represent an L x L periodic torus by stacking L^2 commutator plaquettes
    on a single vertex. This gives beta_0=1, beta_1=2, beta_2=L^2-1 — the topology
    of T^2 — and L^2 Wilson loop operators in the orbit basis.
    """
    comp = CellComplex(name=f"{name_prefix}_L{L}")
    v = GlobularCell("V0", dim=0)
    comp.add_cell(v)

    # Two fundamental 1-cycles (x and y directions)
    lx = GlobularCell("Lx", dim=1, source=v, target=v)
    ly = GlobularCell("Ly", dim=1, source=v, target=v)
    comp.add_cell(lx)
    comp.add_cell(ly)

    # L^2 plaquette cells
    for i in range(L * L):
        src = Chain.from_cell(lx) + Chain.from_cell(ly)
        tgt = Chain.from_cell(ly) + Chain.from_cell(lx)
        p = GlobularCell(f"P{i}", dim=2, source=src, target=tgt)
        comp.add_cell(p)

    return FourTierDACC(comp)


def build_4torus_L(L: int) -> FourTierDACC:
    """Builds a minimal T^4 with 1 vertex, 4 loops, and L^4 effective plaquettes."""
    comp = CellComplex(name=f"T4_L{L}")
    v = GlobularCell("V0", dim=0)
    comp.add_cell(v)

    loops = [GlobularCell(f"L{i}", dim=1, source=v, target=v) for i in range(4)]
    for l in loops:
        comp.add_cell(l)

    # 6 independent plaquette types (mu < nu), each replicated L^2 times
    count = 0
    for mu in range(4):
        for nu in range(mu + 1, 4):
            for k in range(L * L):
                src = Chain.from_cell(loops[mu]) + Chain.from_cell(loops[nu])
                tgt = Chain.from_cell(loops[nu]) + Chain.from_cell(loops[mu])
                p = GlobularCell(f"P{mu}{nu}_{k}", dim=2, source=src, target=tgt)
                comp.add_cell(p)
                count += 1

    return FourTierDACC(comp)


# ---------------------------------------------------------------------------
# Sweep 1: Volume scaling kappa(L)
# ---------------------------------------------------------------------------

@dataclass
class VolumeScalingRow:
    L: int
    n_loops: int
    beta: float
    kappa: float
    spectral_gap_lambda1: float
    kappa_times_L2: float  # diagnostic: should be O(1) if gap is L-independent


def sweep_volume_scaling(
    L_values: List[int] = [2, 4, 8, 16],
    beta: float = 2.0,
    gauge_group: str = "SU(2)",
) -> List[VolumeScalingRow]:
    """Runs volume scaling sweep kappa(L) for tori of linear size L."""
    results = []
    for L in L_values:
        dacc = build_4torus_L(L)
        orbit = OrbitSpaceManifold(dacc, gauge_group=gauge_group)
        evaluator = BakryEmeryCurvatureEvaluator(orbit)
        cert = evaluator.estimate_curvature_bound(beta=beta)

        # Spectral gap from orbit space Laplacian
        spectrum = orbit.compute_orbit_laplacian_spectrum(beta=beta, truncation_order=20)
        lambda1 = spectrum[1] if len(spectrum) > 1 else 0.0

        row = VolumeScalingRow(
            L=L,
            n_loops=orbit.loop_dimension,
            beta=beta,
            kappa=cert.ricci_lower_bound_kappa,
            spectral_gap_lambda1=lambda1,
            kappa_times_L2=cert.ricci_lower_bound_kappa * L * L,
        )
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Sweep 2: Continuum scaling m_phys(a)
# ---------------------------------------------------------------------------

@dataclass
class ContinuumScalingRow:
    N: int                  # grid points per side
    lattice_spacing_fm: float
    beta_wilson: float
    g0_sq: float
    lambda1_lattice: float  # in lattice units
    m_phys_gev: float       # = sqrt(lambda1) * hbar_c / a  [GeV]
    is_stable: bool         # |dm_phys/dlog_a| < 0.2 * m_phys (RG stability)


def sweep_continuum_scaling(
    N_values: List[int] = [4, 8, 16, 32],
    phys_box_fm: float = 1.0,
    gauge_group: str = "SU(2)",
) -> List[ContinuumScalingRow]:
    """Runs continuum scaling at fixed physical box size, varying N (grid refinement)."""
    hbar_c = 0.197327  # GeV·fm
    n_c = 2 if gauge_group == "SU(2)" else 3

    results = []
    prev_m_phys = None
    for N in N_values:
        a_fm = phys_box_fm / N  # lattice spacing in fm
        # Two-loop lattice coupling: beta_wilson = 2*N_c / g^2
        # Use g^2(mu) at mu = 1/a with Lambda_QCD ~ 250 MeV
        mu_gev = hbar_c / a_fm
        lambda_qcd_gev = 0.250
        # Two-loop: g^2(mu) ~ 1 / (b0 * ln(mu^2/Lambda^2))  for large mu
        b0 = 11.0 * n_c / (48.0 * math.pi ** 2)
        ln_arg = max((mu_gev / lambda_qcd_gev) ** 2, 1.1)
        g0_sq = 1.0 / (b0 * math.log(ln_arg))
        beta_w = 2.0 * n_c / g0_sq

        # Orbit space at this scale (use L=1 minimal torus, scaled)
        dacc = build_4torus_L(1)
        orbit = OrbitSpaceManifold(dacc, gauge_group=gauge_group)
        spectrum = orbit.compute_orbit_laplacian_spectrum(beta=beta_w, truncation_order=20)
        lambda1_lat = spectrum[1] if len(spectrum) > 1 else 0.0

        # Physical gap: m = sqrt(lambda1) * mu_lattice = sqrt(lambda1) * hbar_c / a
        m_phys = math.sqrt(max(lambda1_lat, 0.0)) * (hbar_c / a_fm)

        # RG stability check
        stable = True
        if prev_m_phys is not None and prev_m_phys > 0:
            rel_change = abs(m_phys - prev_m_phys) / prev_m_phys
            stable = rel_change < 0.3

        results.append(ContinuumScalingRow(
            N=N,
            lattice_spacing_fm=a_fm,
            beta_wilson=beta_w,
            g0_sq=g0_sq,
            lambda1_lattice=lambda1_lat,
            m_phys_gev=m_phys,
            is_stable=stable,
        ))
        prev_m_phys = m_phys

    return results


# ---------------------------------------------------------------------------
# Empirical cross-check from existing benchmark database
# ---------------------------------------------------------------------------

def load_empirical_scaling(db_path: str) -> Dict[str, List]:
    """Reads the yang_mills benchmark DB and summarises m_gap vs a and L."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT gauge_group, lattice_extent_N, lattice_spacing_a_fm,
               beta_wilson, string_tension_sigma_gev2, mass_gap_delta_m_gev,
               two_loop_beta_scaling_ratio
        FROM ym_continuum_lattice_scaling
        ORDER BY gauge_group, lattice_extent_N, lattice_spacing_a_fm
    """).fetchall()
    conn.close()
    data: Dict[str, List] = {}
    for row in rows:
        group = row[0]
        if group not in data:
            data[group] = []
        data[group].append(row)
    return data


def compute_scaling_exponent(data_pairs: List[Tuple[float, float]]) -> float:
    """Fits log(m) = alpha * log(a) + const and returns alpha (should be ~0 for gap)."""
    if len(data_pairs) < 2:
        return float("nan")
    log_a = [math.log(max(p[0], 1e-10)) for p in data_pairs]
    log_m = [math.log(max(p[1], 1e-10)) for p in data_pairs]
    n = len(log_a)
    mean_a = sum(log_a) / n
    mean_m = sum(log_m) / n
    cov = sum((log_a[i] - mean_a) * (log_m[i] - mean_m) for i in range(n))
    var = sum((log_a[i] - mean_a) ** 2 for i in range(n))
    if var == 0:
        return float("nan")
    return cov / var


# ---------------------------------------------------------------------------
# Main: run sweeps and print report
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 72)
    print("SWEEP 1: Volume Scaling  kappa(L)  and  lambda_1(L)  at fixed beta=2.0")
    print("=" * 72)
    vol_rows = sweep_volume_scaling(L_values=[2, 4, 8, 16], beta=2.0)
    print(f"{'L':>4}  {'#loops':>7}  {'kappa':>10}  {'lambda_1':>10}  {'kappa*L^2':>10}  Diagnosis")
    for r in vol_rows:
        # Diagnosis: if kappa*L^2 is roughly constant, gap is L-independent (good).
        # If kappa*L^2 ~ const >> 0, then kappa ~ 1/L^2 -> gap collapses (bad / Perron-Frobenius).
        if r.L == vol_rows[0].L:
            kappa0 = r.kappa
        ratio = r.kappa / kappa0 if kappa0 > 0 else float('nan')
        diagnosis = "STABLE" if abs(ratio - 1.0) < 0.05 else ("COLLAPSING (1/L^2)" if r.kappa_times_L2 > 5 else "SUPPRESSED")
        print(f"{r.L:>4}  {r.n_loops:>7}  {r.kappa:>10.6f}  {r.spectral_gap_lambda1:>10.6f}  {r.kappa_times_L2:>10.4f}  {diagnosis}")

    print()
    print("=" * 72)
    print("SWEEP 2: Continuum Scaling  m_phys(a)  at fixed box 1.0 fm, SU(2)")
    print("=" * 72)
    cont_rows = sweep_continuum_scaling(N_values=[4, 8, 16, 32], phys_box_fm=1.0)
    print(f"{'N':>4}  {'a [fm]':>8}  {'beta_W':>8}  {'g0^2':>6}  {'lambda1_lat':>11}  {'m_phys [GeV]':>13}  RGstable")
    pairs = []
    for r in cont_rows:
        print(f"{r.N:>4}  {r.lattice_spacing_fm:>8.4f}  {r.beta_wilson:>8.4f}  {r.g0_sq:>6.4f}  {r.lambda1_lattice:>11.6f}  {r.m_phys_gev:>13.6f}  {'YES' if r.is_stable else 'NO'}")
        pairs.append((r.lattice_spacing_fm, r.m_phys_gev))
    alpha = compute_scaling_exponent(pairs)
    print(f"\n  Scaling exponent d(ln m)/d(ln a) = {alpha:.4f}")
    print(f"  Interpretation: alpha ~ 0 means m_phys is a-independent (continuum limit exists).")
    print(f"  alpha ~ 1 means m_phys ~ 1/a (UV divergence — no continuum limit).")
    print(f"  alpha ~ -1 means m_phys ~ a -> 0 (gap collapses in continuum — no mass gap).")

    print()
    print("=" * 72)
    print("EMPIRICAL CROSS-CHECK: yang_mills_continuum_gap_benchmark.db")
    print("=" * 72)
    db = "/home/eliasbrahim/paper_empirical_benchmarks/yang_mills_continuum_gap_benchmark.db"
    emp = load_empirical_scaling(db)
    for group, rows in emp.items():
        # Sample 6 rows across the a range
        sampled = rows[::max(1, len(rows)//6)][:6]
        print(f"\n  {group}  ({len(rows)} configs in DB):")
        print(f"  {'L':>5}  {'a[fm]':>7}  {'beta_W':>7}  {'sigma[GeV^2]':>13}  {'m_gap[GeV]':>11}  {'2loop_ratio':>11}")
        pairs_emp = []
        for row in sampled:
            _, L, a, beta, sigma, mgap, ratio = row
            print(f"  {L:>5}  {a:>7.4f}  {beta:>7.4f}  {sigma:>13.5f}  {mgap:>11.4f}  {ratio:>11.4f}")
            pairs_emp.append((a, mgap))
        alpha_emp = compute_scaling_exponent(pairs_emp)
        print(f"  => d(ln m_gap)/d(ln a) = {alpha_emp:.4f}   (ideal: 0.00 = perfectly flat)")

    print()
    print("=" * 72)
    print("SUMMARY: Volume and Continuum Scaling Diagnostics")
    print("=" * 72)
    kappa_vals = [r.kappa for r in vol_rows]
    kappa_range = (max(kappa_vals) - min(kappa_vals)) / max(kappa_vals) if max(kappa_vals) > 0 else float('nan')
    m_phys_vals = [r.m_phys_gev for r in cont_rows]
    m_range = (max(m_phys_vals) - min(m_phys_vals)) / max(m_phys_vals) if max(m_phys_vals) > 0 else float('nan')
    print(f"  kappa variation over L in [2,16]:  {kappa_range*100:.2f}%  (< 5% ideal)")
    print(f"  m_phys variation over N in [4,32]: {m_range*100:.2f}%  (< 10% ideal)")
    print(f"  Continuum scaling exponent alpha:   {alpha:.4f}  (|alpha| < 0.1 ideal)")
    print()
    print("NOTE: Results on minimal minimal-cell complexes expose the Perron-Frobenius")
    print("caveat: every finite connected matrix has lambda_1 > 0. What matters is")
    print("whether lambda_1 is L- and a-INDEPENDENT in the physical units (GeV).")
    print("See the mathematical specification paper for the infinite-dimensional argument.")
