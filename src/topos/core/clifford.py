"""Clifford Algebra and Geometric Native Coding Engine.

Provides coordinate-free multivector representations for arbitrary Clifford signatures Cl(p, q, r),
with specialized support for 3D Projective Geometric Algebra PGA(3,0,1).
Supports geometric product, blade decomposition, reversion, motor exponentials, and versor sandwiches.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass(frozen=True)
class CliffordSignature:
    """Metric signature for Clifford algebra Cl(p, q, r).
    
    p: number of basis vectors squaring to +1
    q: number of basis vectors squaring to -1
    r: number of basis vectors squaring to 0 (degenerate/null)
    """
    p: int
    q: int
    r: int
    basis_names: Tuple[str, ...] = ()
    metric: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        total = self.p + self.q + self.r
        if not self.metric:
            # Default metric assignment:
            # r degenerate vectors (0) first, then p positive (+1), then q negative (-1)
            # This matches standard PGA convention where e_0^2 = 0, e_1^2 = e_2^2 = e_3^2 = 1.
            met = (0,) * self.r + (1,) * self.p + (-1,) * self.q
            object.__setattr__(self, "metric", met)
        else:
            if len(self.metric) != total:
                raise ValueError(f"Metric length {len(self.metric)} does not match total generators {total}")

        if not self.basis_names:
            names = []
            for i in range(total):
                names.append(f"e{i}")
            object.__setattr__(self, "basis_names", tuple(names))
        else:
            if len(self.basis_names) != total:
                raise ValueError(f"basis_names length {len(self.basis_names)} does not match total generators {total}")

    @property
    def total_dim(self) -> int:
        return self.p + self.q + self.r

    @property
    def multivector_dim(self) -> int:
        return 1 << self.total_dim


# Predefined standard signatures
PGA301 = CliffordSignature(p=3, q=0, r=1, basis_names=("e0", "e1", "e2", "e3"), metric=(0, 1, 1, 1))
CGA410 = CliffordSignature(p=4, q=1, r=0, basis_names=("e1", "e2", "e3", "e4", "e5"), metric=(1, 1, 1, 1, -1))
STA130 = CliffordSignature(p=1, q=3, r=0, basis_names=("e0", "e1", "e2", "e3"), metric=(1, -1, -1, -1))
EGA300 = CliffordSignature(p=3, q=0, r=0, basis_names=("e1", "e2", "e3"), metric=(1, 1, 1))
CL400 = CliffordSignature(p=4, q=0, r=0, basis_names=("e0", "e1", "e2", "e3"), metric=(1, 1, 1, 1))


class Multivector:
    """A general multivector over CliffordSignature Cl(p, q, r)."""

    # Multiplication tables cache: signature -> (table_sign, table_target)
    _tables_cache: Dict[CliffordSignature, Tuple[List[List[int]], List[List[int]]]] = {}

    def __init__(
        self,
        signature: CliffordSignature,
        components: Optional[Sequence[float]] = None,
    ) -> None:
        self.signature = signature
        dim = signature.multivector_dim
        if components is not None:
            if len(components) != dim:
                raise ValueError(f"Expected {dim} components for {signature}, got {len(components)}")
            self.components = tuple(float(c) for c in components)
        else:
            self.components = (0.0,) * dim

    @classmethod
    def _get_tables(cls, signature: CliffordSignature) -> Tuple[List[List[int]], List[List[int]]]:
        if signature in cls._tables_cache:
            return cls._tables_cache[signature]

        n = signature.total_dim
        dim = signature.multivector_dim
        metric = signature.metric

        signs: List[List[int]] = [[0] * dim for _ in range(dim)]
        targets: List[List[int]] = [[0] * dim for _ in range(dim)]

        for a in range(dim):
            for b in range(dim):
                # Calculate sign and target blade for e_a * e_b
                # a and b are bitmasks of basis vectors
                cur_sign = 1
                cur_metric_val = 1

                # List generators
                gen_a = [i for i in range(n) if (a & (1 << i))]
                gen_b = [i for i in range(n) if (b & (1 << i))]

                # Concatenate generators and bubble sort to standard order
                gens = gen_a + gen_b
                swaps = 0
                i = 0
                while i < len(gens) - 1:
                    if gens[i] > gens[i + 1]:
                        gens[i], gens[i + 1] = gens[i + 1], gens[i]
                        swaps += 1
                        if i > 0:
                            i -= 1
                        else:
                            i += 1
                    elif gens[i] == gens[i + 1]:
                        # e_k * e_k = metric[k]
                        k = gens[i]
                        cur_metric_val *= metric[k]
                        gens.pop(i)
                        gens.pop(i)
                        if i > 0:
                            i -= 1
                    else:
                        i += 1

                if cur_metric_val == 0:
                    signs[a][b] = 0
                    targets[a][b] = 0
                else:
                    if swaps % 2 == 1:
                        cur_sign = -cur_sign
                    final_mask = 0
                    for g in gens:
                        final_mask |= (1 << g)
                    signs[a][b] = cur_sign * cur_metric_val
                    targets[a][b] = final_mask

        cls._tables_cache[signature] = (signs, targets)
        return signs, targets

    @classmethod
    def scalar(cls, signature: CliffordSignature, value: float) -> Multivector:
        comps = [0.0] * signature.multivector_dim
        comps[0] = float(value)
        return cls(signature, comps)

    @classmethod
    def basis(cls, signature: CliffordSignature, blade_index: int, value: float = 1.0) -> Multivector:
        comps = [0.0] * signature.multivector_dim
        comps[blade_index] = float(value)
        return cls(signature, comps)

    @classmethod
    def bivector(cls, signature: CliffordSignature, mu: int, nu: int, value: float = 1.0) -> Multivector:
        """Constructs an oriented bivector blade e_mu ^ e_nu."""
        if mu == nu:
            return cls.scalar(signature, 0.0)
        e_mu = cls.basis(signature, 1 << mu)
        e_nu = cls.basis(signature, 1 << nu)
        prod = e_mu.wedge(e_nu)
        return prod * value

    @classmethod
    def from_named_blades(
        cls,
        signature: CliffordSignature,
        blades: Dict[str, float],
    ) -> Multivector:
        """Constructs multivector from dictionary of blade names to coefficients."""
        comps = [0.0] * signature.multivector_dim
        # Build mapping of blade names to indices
        name_to_idx = cls._get_blade_names_map(signature)
        for name, val in blades.items():
            if name not in name_to_idx:
                raise KeyError(f"Unknown basis blade {name!r} for signature {signature}")
            comps[name_to_idx[name]] = float(val)
        return cls(signature, comps)

    @classmethod
    def _get_blade_names_map(cls, signature: CliffordSignature) -> Dict[str, int]:
        mapping = {"1": 0, "s": 0}
        n = signature.total_dim
        for mask in range(1, 1 << n):
            gens = [signature.basis_names[i] for i in range(n) if (mask & (1 << i))]
            name = "".join(gens)
            mapping[name] = mask
            # Also support e012 shorthand if names are e0, e1, etc.
            simple_name = "e" + "".join(str(i) for i in range(n) if (mask & (1 << i)))
            mapping[simple_name] = mask
        return mapping

    @property
    def blade_names(self) -> List[str]:
        n = self.signature.total_dim
        res = ["1"]
        for mask in range(1, 1 << n):
            gens = [self.signature.basis_names[i] for i in range(n) if (mask & (1 << i))]
            res.append("".join(gens))
        return res

    def get_blade(self, name: str) -> float:
        name_map = self._get_blade_names_map(self.signature)
        if name not in name_map:
            raise KeyError(f"Blade {name!r} not in signature")
        return self.components[name_map[name]]

    def __getitem__(self, idx: int) -> float:
        return self.components[idx]

    def __add__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            other = Multivector.scalar(self.signature, float(other))
        assert self.signature == other.signature, "Signature mismatch in multivector addition"
        return Multivector(
            self.signature,
            [a + b for a, b in zip(self.components, other.components)],
        )

    def __radd__(self, other: Union[float, int]) -> Multivector:
        return self.__add__(other)

    def __sub__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            other = Multivector.scalar(self.signature, float(other))
        assert self.signature == other.signature, "Signature mismatch in multivector subtraction"
        return Multivector(
            self.signature,
            [a - b for a, b in zip(self.components, other.components)],
        )

    def __neg__(self) -> Multivector:
        return Multivector(self.signature, [-c for c in self.components])

    def __mul__(self, other: Union[Multivector, float, int]) -> Multivector:
        if isinstance(other, (int, float)):
            return Multivector(self.signature, [c * float(other) for c in self.components])
        assert self.signature == other.signature, "Signature mismatch in geometric product"

        signs, targets = self._get_tables(self.signature)
        dim = self.signature.multivector_dim
        res = [0.0] * dim

        for a in range(dim):
            ca = self.components[a]
            if ca == 0.0:
                continue
            for b in range(dim):
                cb = other.components[b]
                if cb == 0.0:
                    continue
                s = signs[a][b]
                if s != 0:
                    t = targets[a][b]
                    res[t] += ca * cb * s

        return Multivector(self.signature, res)

    def __rmul__(self, other: Union[float, int]) -> Multivector:
        return self.__mul__(other)

    def reverse(self) -> Multivector:
        """Reversion involution ~A: reverses the order of vectors in each blade.
        
        For a k-blade, ~e_I = (-1)^(k*(k-1)//2) e_I.
        """
        dim = self.signature.multivector_dim
        res = [0.0] * dim
        for mask in range(dim):
            k = bin(mask).count("1")
            sign = -1 if (k * (k - 1) // 2) % 2 == 1 else 1
            res[mask] = self.components[mask] * sign
        return Multivector(self.signature, res)

    def involute(self) -> Multivector:
        """Main grade involution (parity involution) alpha(A):
        alpha(e_I) = (-1)^k e_I for a k-blade.
        """
        dim = self.signature.multivector_dim
        res = [0.0] * dim
        for mask in range(dim):
            k = bin(mask).count("1")
            sign = -1 if (k % 2 == 1) else 1
            res[mask] = self.components[mask] * sign
        return Multivector(self.signature, res)

    def trace(self) -> float:
        """Normalized Clifford trace tau(A) = <A>_0 = A[0]."""
        return self.components[0]

    def grade(self, k: int) -> Multivector:
        """Grade projection <A>_k: extracts terms of topological dimension / blade grade k."""
        dim = self.signature.multivector_dim
        res = [0.0] * dim
        for mask in range(dim):
            if bin(mask).count("1") == k:
                res[mask] = self.components[mask]
        return Multivector(self.signature, res)

    def wedge(self, other: Multivector) -> Multivector:
        """Outer / wedge product A ^ B: sum of grade(i+j) from <A>_i * <B>_j."""
        assert self.signature == other.signature
        n = self.signature.total_dim
        dim = self.signature.multivector_dim
        res = [0.0] * dim
        signs, targets = self._get_tables(self.signature)

        for a in range(dim):
            ca = self.components[a]
            if ca == 0.0:
                continue
            ga = bin(a).count("1")
            for b in range(dim):
                cb = other.components[b]
                if cb == 0.0:
                    continue
                gb = bin(b).count("1")
                t = targets[a][b]
                if bin(t).count("1") == ga + gb:
                    s = signs[a][b]
                    if s != 0:
                        res[t] += ca * cb * s
        return Multivector(self.signature, res)

    def __xor__(self, other: Multivector) -> Multivector:
        return self.wedge(other)

    def sandwich(self, rotor: Multivector) -> Multivector:
        """Versor sandwich transformation: S(M, A) = M * A * ~M."""
        return rotor * self * rotor.reverse()

    def exp_motor(self, steps: int = 12) -> Multivector:
        """Exponential map exp(B) for a bivector generator B (e.g. Lie algebra screw motion).
        
        Evaluated via series expansion exp(B) = sum_{k=0}^steps (1/k!) B^k.
        In PGA, higher powers of bivectors terminate quickly (nilpotent / periodic).
        """
        result = Multivector.scalar(self.signature, 1.0)
        term = Multivector.scalar(self.signature, 1.0)
        for k in range(1, steps + 1):
            term = (term * self) * (1.0 / k)
            result = result + term
            # Early break if term is negligible
            if max(abs(c) for c in term.components) < 1e-15:
                break
        return result

    def norm_squared(self) -> float:
        """Scalar product of multivector with its reverse <A * ~A>_0."""
        rev = self.reverse()
        prod = self * rev
        return prod.components[0]

    def norm(self) -> float:
        ns = self.norm_squared()
        return math.sqrt(max(0.0, ns))

    def to_array(self) -> List[float]:
        return list(self.components)

    def __repr__(self) -> str:
        terms = []
        names = self.blade_names
        for mask, val in enumerate(self.components):
            if abs(val) > 1e-12:
                terms.append(f"{val:+.4g}*{names[mask]}")
        if not terms:
            return "0"
        return " ".join(terms).lstrip("+")


    @classmethod
    def motor_pga301(
        cls,
        axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        angle: float = 0.0,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> Multivector:
        """Constructs standardized unit rigid-body motor M = T * R in Spin(3,0,1) = SE(3).
        
        Section 2.3 of SSRN-7618742:
        Rotor R = cos(theta/2) - sin(theta/2) * B_rot
          where B_rot = nx*e23 + ny*e31 + nz*e12
        Translator T = 1 - 0.5 * B_trans
          where B_trans = tx*e01 + ty*e02 + tz*e03
        Motor M = T * R satisfies unit manifold constraint M * ~M = 1.
        """
        # 1. Normalize axis
        ax, ay, az = axis
        norm_sq = ax * ax + ay * ay + az * az
        if norm_sq > 1e-18:
            inv_norm = 1.0 / math.sqrt(norm_sq)
            nx, ny, nz = ax * inv_norm, ay * inv_norm, az * inv_norm
        else:
            nx, ny, nz = 0.0, 0.0, 1.0

        # Rotor R: cos(theta/2) - sin(theta/2) * (nx*e23 + ny*e31 + nz*e12)
        # Note basis: e0 is bit 0, e1 is bit 1, e2 is bit 2, e3 is bit 3
        # e23 is e2^e3 -> bit 2 | bit 3 = 4 | 8 = 12
        # e31 is e3^e1 = -e1^e3 -> bit 1 | bit 3 = 2 | 8 = 10 with sign -1
        # e12 is e1^e2 -> bit 1 | bit 2 = 2 | 4 = 6
        half_theta = 0.5 * angle
        c_half = math.cos(half_theta)
        s_half = math.sin(half_theta)

        rotor_blades = {
            "1": c_half,
            "e2e3": -s_half * nx,
            "e1e3": +s_half * ny,  # e3^e1 = -e1^e3
            "e1e2": -s_half * nz,
        }
        R = cls.from_named_blades(PGA301, rotor_blades)


        # Translator T = 1 - 0.5 * (tx*e01 + ty*e02 + tz*e03)
        tx, ty, tz = translation
        trans_blades = {
            "1": 1.0,
            "e0e1": -0.5 * tx,
            "e0e2": -0.5 * ty,
            "e0e3": -0.5 * tz,
        }
        T = cls.from_named_blades(PGA301, trans_blades)

        # M = T * R
        M = T * R
        return M


@dataclass(frozen=True)
class PGA301Motor:
    """Rigid-body motor M in Spin(3,0,1) for coordinate-free SE(3) transformations."""
    motor: Multivector

    @classmethod
    def from_axis_angle_translation(
        cls,
        axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        angle: float = 0.0,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> PGA301Motor:
        m = Multivector.motor_pga301(axis, angle, translation)
        return cls(m)

    @property
    def multivector(self) -> Multivector:
        return self.motor

    def reverse(self) -> PGA301Motor:
        return PGA301Motor(self.motor.reverse())

    def transform(self, element: Multivector) -> Multivector:
        """Sandwich action M * X * ~M."""
        return element.sandwich(self.motor)


    def is_unit_motor(self, tolerance: float = 1e-12) -> bool:
        """Verifies unit motor manifold constraint M * ~M == 1."""
        prod = self.motor * self.motor.reverse()
        diff_scalar = abs(prod.components[0] - 1.0)
        diff_other = max(abs(c) for c in prod.components[1:])
        return diff_scalar < tolerance and diff_other < tolerance

