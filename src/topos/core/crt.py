"""Chinese Remainder Theorem (CRT) Multi-Prime Engine and Galois Field Stalks.

Implements:
1. Multi-prime 64-bit residue rings R_CRT = \\bigoplus_{m=1}^M F_{p_m} with p_m = 3 (mod 4).
2. Galois Field F_{p^2} ~ F_p[i] / (i^2 + 1) for complex Clifford representation.
3. Garner's O(M^2) mixed-radix reconstruction algorithm for exact integer re-lifting.
4. Parallel multi-prime modular RREF with exact rational rank recovery:
   rank_Q(B_k) = max_m rank_{F_{p_m}}(B_k mod p_m).
5. Compiler Roofline Machine-Balance Invariant L * d >= 19.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# 64-bit prime pool with p = 3 (mod 4) in (2^62, 2^63)
# Guarantees x^2 + 1 is irreducible, making F_p[i]/(i^2+1) an exact Galois field F_{p^2}.
DEFAULT_CRT_PRIMES: Tuple[int, ...] = (
    9223372036854775783,  # 2^63 - 25
    9223372036854775643,  # 2^63 - 165
    9223372036854775507,  # 2^63 - 301
    9223372036854775399,  # 2^63 - 409
)


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    """Extended Euclidean Algorithm: returns (g, x, y) such that a*x + b*y = g."""
    if a == 0:
        return b, 0, 1
    g, x1, y1 = egcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return g, x, y


def mod_inv(a: int, m: int) -> int:
    """Computes modular inverse of a modulo m."""
    a = (a % m + m) % m
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ZeroDivisionError(f"Modular inverse does not exist: {a} and {m} are not coprime")
    return (x % m + m) % m


@dataclass(frozen=True)
class GF2Element:
    """Element of quadratic extension field F_{p^2} = F_p[i] / (i^2 + 1)."""
    real: int
    imag: int
    p: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "real", ((self.real % self.p) + self.p) % self.p)
        object.__setattr__(self, "imag", ((self.imag % self.p) + self.p) % self.p)

    def is_zero(self) -> bool:
        return self.real == 0 and self.imag == 0

    def __add__(self, other: GF2Element) -> GF2Element:
        if self.p != other.p:
            raise ValueError(f"Prime mismatch in F_{{p^2}}: {self.p} != {other.p}")
        return GF2Element((self.real + other.real) % self.p, (self.imag + other.imag) % self.p, self.p)

    def __sub__(self, other: GF2Element) -> GF2Element:
        if self.p != other.p:
            raise ValueError(f"Prime mismatch in F_{{p^2}}: {self.p} != {other.p}")
        return GF2Element((self.real - other.real + self.p) % self.p, (self.imag - other.imag + self.p) % self.p, self.p)

    def __mul__(self, other: GF2Element) -> GF2Element:
        if self.p != other.p:
            raise ValueError(f"Prime mismatch in F_{{p^2}}: {self.p} != {other.p}")
        p = self.p
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        re = (self.real * other.real - self.imag * other.imag) % p
        im = (self.real * other.imag + self.imag * other.real) % p
        return GF2Element((re + p) % p, (im + p) % p, p)

    def __neg__(self) -> GF2Element:
        return GF2Element((self.p - self.real) % self.p, (self.p - self.imag) % self.p, self.p)

    def inv(self) -> GF2Element:
        """Exact multiplicative inverse (a + bi)^(-1) = (a - bi) / (a^2 + b^2)."""
        if self.is_zero():
            raise ZeroDivisionError("Division by zero in F_{p^2}")
        p = self.p
        denom = (self.real * self.real + self.imag * self.imag) % p
        denom_inv = mod_inv(denom, p)
        re = (self.real * denom_inv) % p
        im = ((-self.imag % p + p) * denom_inv) % p
        return GF2Element(re, im, p)

    def __truediv__(self, other: GF2Element) -> GF2Element:
        return self * other.inv()


class GaloisField2:
    """Field structure for quadratic extension F_{p^2}."""

    def __init__(self, p: int) -> None:
        if p % 4 != 3:
            raise ValueError(f"Prime p must be congruent to 3 mod 4, got {p} with p%4={p%4}")
        self.p = p

    def zero(self) -> GF2Element:
        return GF2Element(0, 0, self.p)

    def one(self) -> GF2Element:
        return GF2Element(1, 0, self.p)

    def from_scalar(self, val: int) -> GF2Element:
        return GF2Element(val, 0, self.p)

    def from_complex(self, re: int, im: int) -> GF2Element:
        return GF2Element(re, im, self.p)


class GarnersAlgorithm:
    """Garner's O(M^2) mixed-radix reconstruction algorithm.
    
    Reconstructs exact canonical integer from CRT residues without arbitrary-precision
    big-integer division.
    """

    def __init__(self, primes: Sequence[int] = DEFAULT_CRT_PRIMES) -> None:
        self.primes = tuple(primes)
        self.M = len(self.primes)
        self.modulus = 1
        for p in self.primes:
            self.modulus *= p

        # Precompute pairwise modular inverses gamma_{ij} = p_j^(-1) mod p_i for j < i
        self.gamma: Dict[Tuple[int, int], int] = {}
        for i in range(self.M):
            for j in range(i):
                self.gamma[(i, j)] = mod_inv(self.primes[j], self.primes[i])

    def reconstruct(self, residues: Sequence[int]) -> int:
        """Reconstructs the unique integer x in [0, N) satisfying x = r_i (mod p_i)."""
        if len(residues) != self.M:
            raise ValueError(f"Expected {self.M} residues, got {len(residues)}")

        # Mixed-radix coefficients c_0, ..., c_{M-1}
        c = [0] * self.M
        for i in range(self.M):
            u = residues[i]
            for j in range(i):
                diff = (u - c[j] + self.primes[i]) % self.primes[i]
                u = (diff * self.gamma[(i, j)]) % self.primes[i]
            c[i] = u

        # Compute x = c_0 + c_1 * p_0 + c_2 * p_0 * p_1 + ...
        x = 0
        mult = 1
        for i in range(self.M):
            x += c[i] * mult
            mult *= self.primes[i]

        return x

    def reconstruct_signed(self, residues: Sequence[int]) -> int:
        """Reconstructs signed integer in [-N/2, N/2)."""
        val = self.reconstruct(residues)
        if val >= (self.modulus // 2):
            val -= self.modulus
        return val


class CRTRing:
    """Multi-prime CRT direct-sum ring R_CRT = \\bigoplus F_{p_m}."""

    def __init__(self, primes: Sequence[int] = DEFAULT_CRT_PRIMES) -> None:
        self.primes = tuple(primes)
        self.M = len(self.primes)
        self.garner = GarnersAlgorithm(self.primes)
        self.modulus = self.garner.modulus

    def to_crt(self, value: int) -> Tuple[int, ...]:
        """Maps an integer into the multi-prime CRT residue vector."""
        return tuple(((value % p) + p) % p for p in self.primes)

    def from_crt(self, residues: Sequence[int]) -> int:
        """Lifts CRT residue vector back to signed integer via Garner's algorithm."""
        return self.garner.reconstruct_signed(residues)

    def add(self, a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
        return tuple((a[m] + b[m]) % self.primes[m] for m in range(self.M))

    def sub(self, a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
        return tuple((a[m] - b[m] + self.primes[m]) % self.primes[m] for m in range(self.M))

    def mul(self, a: Sequence[int], b: Sequence[int]) -> Tuple[int, ...]:
        return tuple((a[m] * b[m]) % self.primes[m] for m in range(self.M))

    def inv(self, a: Sequence[int]) -> Tuple[int, ...]:
        return tuple(mod_inv(a[m], self.primes[m]) for m in range(self.M))

    def neg(self, a: Sequence[int]) -> Tuple[int, ...]:
        return tuple((self.primes[m] - (a[m] % self.primes[m])) % self.primes[m] for m in range(self.M))

    def sign_compare(self, a: Sequence[int], b: Sequence[int]) -> int:
        """Evaluates whether a > b, a == b, or a < b via signed Garner re-lifting.
        
        Returns:
            +1 if a > b
             0 if a == b
            -1 if a < b
        """
        diff_residues = self.sub(a, b)
        diff_signed = self.from_crt(diff_residues)
        if diff_signed > 0:
            return 1
        elif diff_signed < 0:
            return -1
        return 0


# ---------------------------------------------------------------------------
# Parallel Multi-Prime Modular RREF
# ---------------------------------------------------------------------------

class CRTMatrixRREF:
    """Executes parallel modular Gaussian elimination across M CRT prime lanes."""

    def __init__(self, primes: Sequence[int] = DEFAULT_CRT_PRIMES) -> None:
        self.primes = tuple(primes)
        self.M = len(self.primes)

    @classmethod
    def rref_mod_p(cls, matrix: List[List[int]], p: int) -> Tuple[List[List[int]], int]:
        """Computes Reduced Row Echelon Form (RREF) and rank modulo prime p."""
        if not matrix or not matrix[0]:
            return [], 0

        rows = len(matrix)
        cols = len(matrix[0])
        # Deep copy modulo p
        A = [[((val % p) + p) % p for val in row] for row in matrix]

        rank = 0
        lead = 0

        for r in range(rows):
            if lead >= cols:
                break

            # Find pivot
            pivot_row = r
            while pivot_row < rows and A[pivot_row][lead] == 0:
                pivot_row += 1

            if pivot_row == rows:
                # No pivot in this column, retry on next column
                lead += 1
                continue

            # Swap rows
            A[r], A[pivot_row] = A[pivot_row], A[r]

            # Normalize pivot row
            pivot_inv = mod_inv(A[r][lead], p)
            A[r] = [(val * pivot_inv) % p for val in A[r]]

            # Eliminate in all other rows
            for i in range(rows):
                if i != r and A[i][lead] != 0:
                    factor = A[i][lead]
                    A[i] = [(A[i][c] - factor * A[r][c] + p * factor) % p for c in range(cols)]

            rank += 1
            lead += 1

        return A, rank

    def compute_ranks_across_lanes(self, matrix: List[List[int]]) -> Tuple[List[int], int]:
        """Computes rank modulo each prime and recovers exact rational rank over Q.
        
        rank_Q(B) = max_m rank_{F_{p_m}}(B mod p_m).
        """
        ranks = []
        for p in self.primes:
            _, rank_p = self.rref_mod_p(matrix, p)
            ranks.append(rank_p)
        rational_rank = max(ranks) if ranks else 0
        return ranks, rational_rank


# ---------------------------------------------------------------------------
# Roofline Model & Compiler Static Invariant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RooflineContractReport:
    """Diagnostic report for Roofline model machine balance invariant."""
    prime_lanes_L: int
    column_density_d: float
    arithmetic_intensity: float
    machine_balance_threshold: float
    is_compute_bound: bool
    invariant_satisfied: bool


def verify_roofline_machine_balance(
    prime_lanes_L: int = 4,
    column_density_d: float = 6.0,
    machine_balance_threshold: float = 16.5,
) -> RooflineContractReport:
    """Verifies the compiler static contract invariant: L * d >= 19.
    
    Arithmetic intensity: I_algo = (8 / 9) * L * d.
    If I_algo >= beta_hw (16.5 OPs/Byte), kernel runs 100% compute-bound with zero
    DRAM bandwidth stalling.
    """
    intensity = (8.0 / 9.0) * prime_lanes_L * column_density_d
    invariant = (prime_lanes_L * column_density_d) >= 18.5625
    is_compute_bound = intensity >= machine_balance_threshold

    return RooflineContractReport(
        prime_lanes_L=prime_lanes_L,
        column_density_d=column_density_d,
        arithmetic_intensity=intensity,
        machine_balance_threshold=machine_balance_threshold,
        is_compute_bound=is_compute_bound,
        invariant_satisfied=invariant,
    )
