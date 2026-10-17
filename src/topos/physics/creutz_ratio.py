"""Emergent String Tension and Creutz Ratio Evaluator on 4D Periodic Lattices.

Implements:
1. Bare Wilson Lattice Action:
   S[U] = beta * sum_{x, mu < nu} (1 - (1/N) Re Tr U_munu(x))
   with NO explicit Lambda_YM or mass parameters inserted by hand.
2. Rectangular Wilson Loops:
   W(R, T) = < (1/N) Re Tr prod_{l in d(R x T)} U_l >
3. Creutz Ratios:
   chi(R, T) = -ln( (W(R, T) * W(R-1, T-1)) / (W(R, T-1) * W(R-1, T)) )
   extracting the non-perturbative emergent string tension a^2 sigma = lim chi(R, T) > 0.
4. Confinement Invariant Region:
   I = { beta in R : beta <= beta_c } where chi(R, T) > 0.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from topos.core.lie_algebra import SUEngine, SUNMatrix


@dataclass
class CreutzRatioProfile:
    """Contains measured Wilson loops, Creutz ratios, and string tension."""
    beta: float
    wilson_loops: Dict[Tuple[int, int], float]
    creutz_ratios: Dict[Tuple[int, int], float]
    string_tension: float
    is_confining: bool


class PeriodicLattice4D:
    """4D Periodic hypercubical lattice for constructive quantum gauge theory."""

    def __init__(self, shape: Tuple[int, int, int, int] = (4, 4, 4, 4), n_colors: int = 2) -> None:
        self.shape = shape
        self.l0, self.l1, self.l2, self.l3 = shape
        self.n_colors = n_colors
        self.engine = SUEngine(n_colors)
        # links[(x, mu)] for mu in {0, 1, 2, 3}
        self.links: Dict[Tuple[int, int, int, int, int], SUNMatrix] = {}
        self.set_cold_start()

    def set_cold_start(self) -> None:
        """Sets all links U_mu(x) = I_N (cold start)."""
        id_n = SUNMatrix.identity(self.n_colors)
        for x0 in range(self.l0):
            for x1 in range(self.l1):
                for x2 in range(self.l2):
                    for x3 in range(self.l3):
                        for mu in range(4):
                            self.links[(x0, x1, x2, x3, mu)] = id_n

    def set_hot_start(self, scale: float = 1.0) -> None:
        """Randomizes all links (hot start)."""
        for k in self.links:
            if self.n_colors == 2:
                axis = (random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
                angle = random.uniform(-scale * math.pi, scale * math.pi)
                self.links[k] = SUNMatrix.from_axis_angle_su2(axis, angle)
            else:
                self.links[k] = SUNMatrix.identity(self.n_colors)

    def _shift(self, x: Tuple[int, int, int, int], mu: int, steps: int = 1) -> Tuple[int, int, int, int]:
        """Periodic shift x + steps * e_mu."""
        x_list = list(x)
        x_list[mu] = (x_list[mu] + steps) % self.shape[mu]
        return (x_list[0], x_list[1], x_list[2], x_list[3])

    def get_link(self, x: Tuple[int, int, int, int], mu: int) -> SUNMatrix:
        return self.links.get((x[0], x[1], x[2], x[3], mu), SUNMatrix.identity(self.n_colors))

    def set_link(self, x: Tuple[int, int, int, int], mu: int, u: SUNMatrix) -> None:
        self.links[(x[0], x[1], x[2], x[3], mu)] = u

    def plaquette(self, x: Tuple[int, int, int, int], mu: int, nu: int) -> SUNMatrix:
        """Computes U_munu(x) = U_mu(x) * U_nu(x+mu) * U_mu(x+nu)^dag * U_nu(x)^dag."""
        u_mu_x = self.get_link(x, mu)
        x_p_mu = self._shift(x, mu, 1)
        u_nu_xpmu = self.get_link(x_p_mu, nu)

        x_p_nu = self._shift(x, nu, 1)
        u_mu_xpnu = self.get_link(x_p_nu, mu)
        u_nu_x = self.get_link(x, nu)

        # Path: x -> x+mu -> x+mu+nu -> x+nu -> x
        step1 = u_mu_x.matmul(u_nu_xpmu)
        step2 = step1.matmul(u_mu_xpnu.dagger())
        p_mat = step2.matmul(u_nu_x.dagger())
        return p_mat

    def local_staple(self, x: Tuple[int, int, int, int], mu: int) -> SUNMatrix:
        """Computes the sum of staples V_mu(x) surrounding link U_mu(x)."""
        staple_sum = SUNMatrix.zero(self.n_colors)
        x_p_mu = self._shift(x, mu, 1)

        for nu in range(4):
            if nu == mu:
                continue
            # Forward staple: U_nu(x+mu) * U_mu(x+nu)^dag * U_nu(x)^dag
            # such that Tr(U_mu(x) * staple) = Tr(P_munu(x))
            x_p_nu = self._shift(x, nu, 1)
            u_nu_xpmu = self.get_link(x_p_mu, nu)
            u_mu_xpnu = self.get_link(x_p_nu, mu)
            u_nu_x = self.get_link(x, nu)
            fwd = u_nu_xpmu.matmul(u_mu_xpnu.dagger()).matmul(u_nu_x.dagger())

            # Backward staple: U_nu(x+mu-nu)^dag * U_mu(x-nu)^dag * U_nu(x-nu)
            x_m_nu = self._shift(x, nu, -1)
            x_p_mu_m_nu = self._shift(x_p_mu, nu, -1)
            u_nu_xpmu_mnu = self.get_link(x_p_mu_m_nu, nu)
            u_mu_xmnu = self.get_link(x_m_nu, mu)
            u_nu_xmnu = self.get_link(x_m_nu, nu)
            bwd = u_nu_xpmu_mnu.dagger().matmul(u_mu_xmnu.dagger()).matmul(u_nu_xmnu)

            staple_sum = staple_sum + fwd + bwd

        return staple_sum

    def metropolis_sweep(self, beta: float, step_scale: float = 0.3) -> float:
        """Executes one full Metropolis sweep over all links using bare Wilson action."""
        accepted = 0
        total_links = self.l0 * self.l1 * self.l2 * self.l3 * 4

        for x0 in range(self.l0):
            for x1 in range(self.l1):
                for x2 in range(self.l2):
                    for x3 in range(self.l3):
                        x = (x0, x1, x2, x3)
                        for mu in range(4):
                            old_u = self.get_link(x, mu)
                            staple = self.local_staple(x, mu)

                            # Local action S = - (beta / N) * Re Tr(U * staple)
                            old_action = -(beta / self.n_colors) * (old_u.matmul(staple)).trace().real

                            # Random rotation near identity
                            axis = (random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))
                            angle = random.uniform(-step_scale, step_scale)
                            r_mat = SUNMatrix.from_axis_angle_su2(axis, angle)
                            new_u = r_mat.matmul(old_u)

                            new_action = -(beta / self.n_colors) * (new_u.matmul(staple)).trace().real
                            delta_s = new_action - old_action

                            if delta_s <= 0.0 or random.random() < math.exp(-min(50.0, delta_s)):
                                self.set_link(x, mu, new_u)
                                accepted += 1

        return accepted / max(1, total_links)

    def evaluate_wilson_loop(self, r: int, t: int, max_samples: int = 50) -> float:
        """Evaluates rectangular Wilson loop W(R, T) = < (1/N) Re Tr prod_{loop} U >."""
        if r <= 0 or t <= 0:
            return 1.0  # Trivial identity loop

        traces: List[float] = []
        # Sample planar loops in (0, 1), (0, 2), (1, 2) planes
        planes = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        sample_count = 0

        for x0 in range(self.l0):
            for x1 in range(self.l1):
                for x2 in range(self.l2):
                    for x3 in range(self.l3):
                        x = (x0, x1, x2, x3)
                        for mu, nu in planes:
                            if sample_count >= max_samples:
                                break
                            sample_count += 1

                            # Traverse R steps along mu
                            curr = x
                            mat = SUNMatrix.identity(self.n_colors)
                            for _ in range(r):
                                mat = mat.matmul(self.get_link(curr, mu))
                                curr = self._shift(curr, mu, 1)

                            # Traverse T steps along nu
                            for _ in range(t):
                                mat = mat.matmul(self.get_link(curr, nu))
                                curr = self._shift(curr, nu, 1)

                            # Traverse R steps backwards along mu
                            for _ in range(r):
                                curr = self._shift(curr, mu, -1)
                                mat = mat.matmul(self.get_link(curr, mu).dagger())

                            # Traverse T steps backwards along nu
                            for _ in range(t):
                                curr = self._shift(curr, nu, -1)
                                mat = mat.matmul(self.get_link(curr, nu).dagger())

                            tr = (mat.trace().real) / self.n_colors
                            traces.append(tr)

                        if sample_count >= max_samples:
                            break

        return sum(traces) / max(1, len(traces))


class CreutzRatioEvaluator:
    """Computes Creutz ratios and extracts emergent non-perturbative string tension."""

    def __init__(self, lattice: PeriodicLattice4D) -> None:
        self.lattice = lattice

    def measure_profile(
        self,
        beta: float,
        n_thermalization: int = 100,
        n_samples: int = 20,
        max_size: int = 2,
    ) -> CreutzRatioProfile:
        """Thermalizes lattice and computes Wilson loops W(R, T) and Creutz ratios chi(R, T)."""
        # Thermalize
        for _ in range(n_thermalization):
            self.lattice.metropolis_sweep(beta=beta)

        w_accum: Dict[Tuple[int, int], float] = {}
        for r in range(max_size + 1):
            for t in range(max_size + 1):
                w_accum[(r, t)] = 0.0

        for _ in range(n_samples):
            self.lattice.metropolis_sweep(beta=beta)
            for r in range(max_size + 1):
                for t in range(max_size + 1):
                    w_accum[(r, t)] += self.lattice.evaluate_wilson_loop(r, t)

        wilson_loops: Dict[Tuple[int, int], float] = {
            k: v / n_samples for k, v in w_accum.items()
        }

        # Compute Creutz ratios chi(R, T) = -ln( (W(R, T) * W(R-1, T-1)) / (W(R, T-1) * W(R-1, T)) )
        creutz: Dict[Tuple[int, int], float] = {}
        for r in range(1, max_size + 1):
            for t in range(1, max_size + 1):
                num = max(1e-15, wilson_loops.get((r, t), 1.0) * wilson_loops.get((r - 1, t - 1), 1.0))
                den = max(1e-15, wilson_loops.get((r, t - 1), 1.0) * wilson_loops.get((r - 1, t), 1.0))
                ratio = num / den
                chi = -math.log(max(1e-15, ratio))
                creutz[(r, t)] = chi

        # Emergent string tension a^2 sigma = chi(max_size, max_size) or chi(1, 1)
        sigma = creutz.get((max_size, max_size), creutz.get((1, 1), 0.0))
        is_confining = sigma > 0.0

        return CreutzRatioProfile(
            beta=beta,
            wilson_loops=wilson_loops,
            creutz_ratios=creutz,
            string_tension=sigma,
            is_confining=is_confining,
        )

    def identify_invariant_confinement_region(
        self,
        beta_candidates: List[float] = [0.5, 1.0, 2.0, 3.0, 4.0],
        n_thermalization: int = 50,
        n_samples: int = 10,
    ) -> Tuple[List[float], float]:
        """Identifies invariant region I = { beta in R : beta <= beta_c } where string tension sigma > 0."""
        invariant_betas: List[float] = []
        beta_c = 0.0

        for beta in beta_candidates:
            prof = self.measure_profile(beta, n_thermalization=n_thermalization, n_samples=n_samples, max_size=1)
            if prof.is_confining and prof.string_tension > 0.01:
                invariant_betas.append(beta)
                beta_c = max(beta_c, beta)

        return invariant_betas, beta_c
