"""Unitary Gauge Groups SU(N) and Lie Algebra su(N) Engine.

Provides coordinate-free matrix representations, center projections,
discrete field strengths, and Lie brackets for constructive gauge theory.
"""

from __future__ import annotations
import cmath
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class SUNMatrix:
    """An N x N complex matrix representing an element of SU(N) or su(N)."""

    def __init__(self, data: Sequence[Sequence[complex]], n: Optional[int] = None) -> None:
        if not data:
            raise ValueError("Matrix data cannot be empty")
        self.n = len(data) if n is None else n
        for row in data:
            if len(row) != self.n:
                raise ValueError(f"Expected square {self.n}x{self.n} matrix, got row length {len(row)}")
        self.data: Tuple[Tuple[complex, ...], ...] = tuple(
            tuple(complex(c) for c in row) for row in data
        )

    @classmethod
    def identity(cls, n: int) -> SUNMatrix:
        """Constructs N x N identity matrix."""
        grid = [[1.0 + 0.0j if r == c else 0.0 + 0.0j for c in range(n)] for r in range(n)]
        return cls(grid, n)

    @classmethod
    def zero(cls, n: int) -> SUNMatrix:
        """Constructs N x N zero matrix."""
        grid = [[0.0 + 0.0j for _ in range(n)] for _ in range(n)]
        return cls(grid, n)

    @classmethod
    def from_axis_angle_su2(cls, axis: Tuple[float, float, float], angle: float) -> SUNMatrix:
        """Constructs SU(2) element from rotation axis and angle:
        U = cos(theta/2) I + i sin(theta/2) (n_x sigma_x + n_y sigma_y + n_z sigma_z).
        """
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm < 1e-15:
            return cls.identity(2)
        nx, ny, nz = ax / norm, ay / norm, az / norm
        half = angle / 2.0
        c = math.cos(half)
        s = math.sin(half)
        # U = [ c + i s nz,   i s nx + s ny ;
        #      i s nx - s ny, c - i s nz ]
        # More standard: (n_x sigma_x + n_y sigma_y + n_z sigma_z)
        # sigma_x = [[0, 1], [1, 0]]
        # sigma_y = [[0, -i], [i, 0]]
        # sigma_z = [[1, 0], [0, -1]]
        u00 = complex(c, s * nz)
        u01 = complex(s * 0.0, s * nx) + complex(s * ny, 0.0) * (-1j) * 1j  # i s (nx sigma_x + ny sigma_y)
        # i (nx sigma_x + ny sigma_y) = [[0, i nx + ny], [i nx - ny, 0]]
        u01 = complex(-s * 0.0 + 0.0, s * nx) + complex(s * ny, 0.0)  # = s*ny + i*s*nx
        # Let's write explicitly:
        # i * n_x * sigma_x = [[0, i nx], [i nx, 0]]
        # i * n_y * sigma_y = [[0, ny], [-ny, 0]]
        # i * n_z * sigma_z = [[i nz, 0], [0, -i nz]]
        # So i * (n . sigma) = [[i nz, ny + i nx], [-ny + i nx, -i nz]]
        # Wait, standard Pauli:
        # sigma_y = [[0, -1j], [1j, 0]]
        # i * sigma_y = [[0, 1], [-1, 0]]
        # So i * (nx sigma_x + ny sigma_y + nz sigma_z) =
        # [[i nz, ny + i nx], [-ny + i nx, -i nz]]
        m00 = c + 1j * s * nz
        m01 = s * ny + 1j * s * nx
        m10 = -s * ny + 1j * s * nx
        m11 = c - 1j * s * nz
        return cls([[m00, m01], [m10, m11]], 2)

    @classmethod
    def random_su2(cls) -> SUNMatrix:
        """Generates random Haar-distributed SU(2) element."""
        # Marsaglia method on 4-sphere S^3
        while True:
            u1 = random.uniform(-1.0, 1.0)
            u2 = random.uniform(-1.0, 1.0)
            s1 = u1 * u1 + u2 * u2
            if s1 < 1.0:
                break
        while True:
            u3 = random.uniform(-1.0, 1.0)
            u4 = random.uniform(-1.0, 1.0)
            s2 = u3 * u3 + u4 * u4
            if s2 < 1.0:
                break
        root = math.sqrt((1.0 - s1) / s2)
        q0 = u1
        q1 = u2
        q2 = u3 * root
        q3 = u4 * root
        # Unit quaternion q = q0 + i q1 + j q2 + k q3 to SU(2) matrix
        # [[q0 + i q3, q2 + i q1], [-q2 + i q1, q0 - i q3]]
        return cls([
            [complex(q0, q3), complex(q2, q1)],
            [complex(-q2, q1), complex(q0, -q3)],
        ], 2)

    @classmethod
    def pauli(cls, index: int) -> SUNMatrix:
        """Returns standard Pauli matrix sigma_1, sigma_2, sigma_3 for SU(2)."""
        if index == 1:
            return cls([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], 2)
        elif index == 2:
            return cls([[0.0 + 0j, -1j], [1j, 0.0 + 0j]], 2)
        elif index == 3:
            return cls([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, -1.0 + 0j]], 2)
        else:
            raise ValueError(f"Pauli index must be 1, 2, or 3, got {index}")

    def __getitem__(self, idx: int) -> Tuple[complex, ...]:
        return self.data[idx]

    def dagger(self) -> SUNMatrix:
        """Conjugate transpose A^dagger."""
        grid = [[self.data[c][r].conjugate() for c in range(self.n)] for r in range(self.n)]
        return SUNMatrix(grid, self.n)

    def trace(self) -> complex:
        """Matrix trace Tr(A)."""
        return sum(self.data[i][i] for i in range(self.n))

    def __add__(self, other: SUNMatrix) -> SUNMatrix:
        assert self.n == other.n, "Matrix dimension mismatch"
        grid = [
            [self.data[r][c] + other.data[r][c] for c in range(self.n)]
            for r in range(self.n)
        ]
        return SUNMatrix(grid, self.n)

    def __sub__(self, other: SUNMatrix) -> SUNMatrix:
        assert self.n == other.n, "Matrix dimension mismatch"
        grid = [
            [self.data[r][c] - other.data[r][c] for c in range(self.n)]
            for r in range(self.n)
        ]
        return SUNMatrix(grid, self.n)

    def __neg__(self) -> SUNMatrix:
        grid = [[-self.data[r][c] for c in range(self.n)] for r in range(self.n)]
        return SUNMatrix(grid, self.n)

    def __mul__(self, scalar: Union[complex, float, int, SUNMatrix]) -> SUNMatrix:
        if isinstance(scalar, SUNMatrix):
            return self.matmul(scalar)
        s = complex(scalar)
        grid = [[self.data[r][c] * s for c in range(self.n)] for r in range(self.n)]
        return SUNMatrix(grid, self.n)

    def __rmul__(self, scalar: Union[complex, float, int]) -> SUNMatrix:
        return self.__mul__(scalar)

    def matmul(self, other: SUNMatrix) -> SUNMatrix:
        assert self.n == other.n, "Matrix dimension mismatch in matmul"
        n = self.n
        grid = [[0.0 + 0.0j for _ in range(n)] for _ in range(n)]
        for r in range(n):
            for k in range(n):
                a_rk = self.data[r][k]
                if a_rk == 0.0:
                    continue
                for c in range(n):
                    grid[r][c] += a_rk * other.data[k][c]
        return SUNMatrix(grid, n)

    def hilbert_schmidt_inner_product(self, other: SUNMatrix) -> complex:
        """Inner product <A, B>_HS = Tr(A^dagger B)."""
        return (self.dagger().matmul(other)).trace()

    def norm_hs(self) -> float:
        """Hilbert-Schmidt Frobenius norm ||A||_HS = sqrt(Re(Tr(A^dagger A)))."""
        prod = self.hilbert_schmidt_inner_product(self)
        return math.sqrt(max(0.0, prod.real))


class SUEngine:
    """Algebraic engine for compact gauge group SU(N) and Lie algebra su(N)."""

    def __init__(self, n: int = 2) -> None:
        if n < 2:
            raise ValueError(f"SU(N) requires N >= 2, got {n}")
        self.n = n

    def center_elements(self) -> List[Tuple[SUNMatrix, complex, int]]:
        """Returns all N center elements z_k = exp(i 2pi k / N) * I_N for k in 0..N-1."""
        res = []
        for k in range(self.n):
            angle = 2.0 * math.pi * k / self.n
            phase = cmath.exp(1j * angle)
            mat = SUNMatrix.identity(self.n) * phase
            res.append((mat, phase, k))
        return res

    def center_flux_projection(self, U_p: SUNMatrix) -> Tuple[SUNMatrix, complex, int]:
        """Projects plaquette holonomy U_p onto center Z(N) by minimizing ||U_p - z I||_HS^2,
        which is equivalent to maximizing Re(z^* Tr(U_p)).
        """
        tr = U_p.trace()
        best_mat = None
        best_phase = None
        best_k = 0
        max_re = -float("inf")

        for mat, phase, k in self.center_elements():
            # Re(phase^* * tr)
            val = (phase.conjugate() * tr).real
            if val > max_re:
                max_re = val
                best_mat = mat
                best_phase = phase
                best_k = k

        assert best_mat is not None and best_phase is not None
        return best_mat, best_phase, best_k

    def topological_defect_density(self, U_p: SUNMatrix) -> float:
        """Defect density nu_p(U) = 1 if z_p != I_N (i.e. k != 0), else 0."""
        _, _, k = self.center_flux_projection(U_p)
        return 1.0 if k != 0 else 0.0

    def plaquette_action_residual(self, U_p: SUNMatrix) -> float:
        """Wilson action residual 1 - (1/N) Re(Tr(U_p)) in [0, 2]."""
        tr = U_p.trace().real
        return max(0.0, 1.0 - (tr / self.n))

    def field_strength_lie_algebra(self, U_p: SUNMatrix) -> SUNMatrix:
        """Discrete field strength Q_p(U) = 1/2 (U_p - U_p^dagger) in su(N)."""
        return (U_p - U_p.dagger()) * 0.5

    def lie_bracket(self, x: SUNMatrix, y: SUNMatrix) -> SUNMatrix:
        """Lie bracket [X, Y] = X Y - Y X."""
        return x.matmul(y) - y.matmul(x)

    def dimensional_transmutation_scale(
        self,
        bare_coupling_g: float,
        lattice_spacing_a: float = 1.0,
    ) -> float:
        """Non-perturbative mass scale Lambda_YM generated by dimensional transmutation:
        Lambda_YM = (1/a) * exp(-1 / (2 b_0 g^2)) * (b_0 g^2)^(-b_1 / (2 b_0^2)).
        """
        g_sq = max(1e-6, bare_coupling_g * bare_coupling_g)
        n = self.n
        # 1-loop and 2-loop beta function coefficients for pure SU(N)
        # b_0 = 11 N / (48 pi^2)
        # b_1 = 34 N^2 / (3 (16 pi^2)^2)
        b0 = (11.0 * n) / (48.0 * (math.pi ** 2))
        b1 = (34.0 * (n ** 2)) / (3.0 * ((16.0 * (math.pi ** 2)) ** 2))

        exponent = -1.0 / (2.0 * b0 * g_sq)
        # Clamped exponent to avoid zero underflow or overflow
        exponent = max(-100.0, min(10.0, exponent))
        exp_factor = math.exp(exponent)

        base = max(1e-12, b0 * g_sq)
        power = -b1 / (2.0 * (b0 ** 2))
        power_factor = math.pow(base, power)

        return (1.0 / max(1e-15, lattice_spacing_a)) * exp_factor * power_factor
