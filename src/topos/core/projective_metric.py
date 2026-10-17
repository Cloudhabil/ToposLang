"""Projective Metric Engine and Non-Archimedean Dual-Lifting Architecture.

Resolves non-Archimedean finite field limitations by separating:
1. Exact topological and algebraic operations in finite CRT rings.
2. Projective homogeneous coordinates (X, Y, Z, W) over integers/rationals.
3. Rational quadratic forms for distance queries:
      b^2 * sum (p_i * q_w - q_i * p_w)^2 < a^2 * (p_w * q_w)^2
   bypassing non-existent square roots and division inside finite fields.
4. Exact decision queries via Garner mixed-radix reconstruction at interface boundaries.
5. Projective line/plane meet semantics in Cl(2,0,1) and Cl(3,0,1) with exact ideal points at infinity.
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Optional, Sequence, Tuple, Union

from topos.core.crt import CRTRing, DEFAULT_CRT_PRIMES


@dataclass(frozen=True)
class ProjectivePoint:
    """Exact homogeneous point in d-dimensional projective space.
    
    Coordinates (x, y, z, w) where w is the homogeneous scaling factor.
    Euclidean position is (x/w, y/w, z/w) when w != 0.
    When w == 0, represents an ideal point at infinity.
    """
    x: int
    y: int
    z: int = 0
    w: int = 1

    def to_euclidean(self) -> Tuple[float, float, float]:
        """Returns inhomogeneous Euclidean coordinates (x/w, y/w, z/w)."""
        if self.w == 0:
            raise ZeroDivisionError("Ideal point at infinity has no finite Euclidean coordinates")
        return (self.x / self.w, self.y / self.w, self.z / self.w)

    def to_crt(self, ring: Optional[CRTRing] = None) -> Tuple[Tuple[int, ...], ...]:
        """Maps homogeneous coordinates into parallel CRT residue lanes."""
        r = ring if ring is not None else CRTRing(DEFAULT_CRT_PRIMES)
        return (
            r.to_crt(self.x),
            r.to_crt(self.y),
            r.to_crt(self.z),
            r.to_crt(self.w),
        )


@dataclass(frozen=True)
class ProjectivePlane:
    """Grade-1 hyperplane in plane-based PGA: Pi = nx e1 + ny e2 + nz e3 + d e0."""
    nx: int
    ny: int
    nz: int = 0
    d: int = 0

    @property
    def is_ideal(self) -> bool:
        """True if plane is the plane at infinity (normal is 0)."""
        return self.nx == 0 and self.ny == 0 and self.nz == 0


class ProjectiveMetricEngine:
    """Evaluates geometric queries, distances, and meets with zero floating-point drift."""

    def __init__(self, crt_ring: Optional[CRTRing] = None) -> None:
        self.ring = crt_ring if crt_ring is not None else CRTRing(DEFAULT_CRT_PRIMES)

    def is_within_radius_rational(
        self,
        p1: ProjectivePoint,
        p2: ProjectivePoint,
        radius_num: int,
        radius_den: int = 1,
    ) -> bool:
        """Evaluates d(p1, p2) < radius_num / radius_den as an exact integer quadratic form.
        
        Formula:
          b^2 * [ (x1*w2 - x2*w1)^2 + (y1*w2 - y2*w1)^2 + (z1*w2 - z2*w1)^2 ] < a^2 * (w1*w2)^2
        where radius = a / b.
        
        Computes polynomial difference in CRT space and reconstructs sign via Garner's algorithm.
        """
        if p1.w == 0 or p2.w == 0:
            return False  # Points at infinity have undefined finite metric distance

        # Numerator cross terms
        dx = p1.x * p2.w - p2.x * p1.w
        dy = p1.y * p2.w - p2.y * p1.w
        dz = p1.z * p2.w - p2.z * p1.w
        dw = p1.w * p2.w

        dist_sq_num = dx * dx + dy * dy + dz * dz
        b_sq = radius_den * radius_den
        a_sq = radius_num * radius_num
        w_sq = dw * dw

        lhs = b_sq * dist_sq_num
        rhs = a_sq * w_sq

        # Evaluate across CRT lanes
        lhs_crt = self.ring.to_crt(lhs)
        rhs_crt = self.ring.to_crt(rhs)

        # Exact boolean test: lhs < rhs <=> ring.sign_compare(rhs, lhs) == 1
        return self.ring.sign_compare(rhs_crt, lhs_crt) == 1

    def squared_euclidean_distance_rational(
        self,
        p1: ProjectivePoint,
        p2: ProjectivePoint,
    ) -> Fraction:
        """Returns exact squared rational distance d^2 in Q."""
        if p1.w == 0 or p2.w == 0:
            raise ValueError("Cannot compute finite distance to point at infinity")

        dx = p1.x * p2.w - p2.x * p1.w
        dy = p1.y * p2.w - p2.y * p1.w
        dz = p1.z * p2.w - p2.z * p1.w
        dw = p1.w * p2.w

        num = dx * dx + dy * dy + dz * dz
        den = dw * dw
        return Fraction(num, den)

    @classmethod
    def meet_lines_2d(
        cls,
        line1: Tuple[int, int, int],  # (a1, b1, c1) for a1*e1 + b1*e2 + c1*e0
        line2: Tuple[int, int, int],  # (a2, b2, c2)
    ) -> Tuple[int, int, int, str]:
        """Evaluates exterior wedge meet L1 ^ L2 in 2D PGA (Cl(2,0,1)).
        
        Lines a1*x + b1*y + c1 = 0 and a2*x + b2*y + c2 = 0.
        Returns homogeneous coordinates (x, y, w) such that (x/w, y/w) is the Euclidean meet:
            w = a1 * b2 - a2 * b1
            x = b1 * c2 - b2 * c1
            y = c1 * a2 - c2 * a1
            classification in {"Point", "Parallel_Ideal", "Coincident"}
        """
        a1, b1, c1 = line1
        a2, b2, c2 = line2

        w = a1 * b2 - a2 * b1
        x = b1 * c2 - b2 * c1
        y = c1 * a2 - c2 * a1

        if w != 0:
            return x, y, w, "Point"
        elif x != 0 or y != 0:
            return x, y, 0, "Parallel_Ideal"
        else:
            return 0, 0, 0, "Coincident"

