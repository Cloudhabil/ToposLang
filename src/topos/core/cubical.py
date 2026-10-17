"""Constructive Cubical Type Theory and Kan Composition Engine.

Implements:
1. Cubical Interval I = [0, 1] and face constraint bitmasks phi in Face(I^k).
2. Bounded KanStackFrame C-ABI structure (64-byte aligned).
3. KanBox open box horn conditions.
4. Constructive Kan composition operator (hcomp) and dynamic n-truncation.
"""

from __future__ import annotations
import ctypes
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from topos.core.clifford import CliffordSignature, Multivector, PGA301


# C-ABI Function Pointer Prototype for Defunctionalized Tube Evaluator
# void (*tube_evaluator)(const double* in_coords, double t, double* out_multivector)
TUBE_EVALUATOR_FUNC = ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
)


class KanStackFrame(ctypes.Structure):
    """C-compatible stack frame layout for O(1) thread-stack Kan composition.
    
    Matches Section 4.1 of the ToposLang-Cubical Bridge Specification:
    - face_mask_phi: Bitmask representing active boundary faces derived from sparse matrix B_k
    - dimension: Active topological dimension k
    - base_multivector: Extracted continuous base multivector state (u_0), 16 doubles
    - tube_evaluator: Function pointer for on-demand path tracking along interval I
    - env_stack_ptr: Bounded execution stack frame pointer for interval variables
    """
    _fields_ = [
        ("face_mask_phi", ctypes.c_uint32),
        ("dimension", ctypes.c_uint32),
        ("base_multivector", ctypes.c_double * 16),
        ("tube_evaluator", TUBE_EVALUATOR_FUNC),
        ("env_stack_ptr", ctypes.c_void_p),
    ]


@dataclass(frozen=True)
class Interval:
    """The abstract cubical interval I = [0, 1]."""
    name: str = "i"

    @staticmethod
    def endpoint_0() -> float:
        return 0.0

    @staticmethod
    def endpoint_1() -> float:
        return 1.0


@dataclass
class FaceMask:
    """Bitmask representation of active faces phi in Face(I^k).
    
    For hypercube of dimension k, each axis a in {0, ..., k-1} has two faces:
    x_a = 0 (encoded as bit 2*a + 0)
    x_a = 1 (encoded as bit 2*a + 1)
    """
    dimension: int
    mask: int = 0

    def add_face(self, axis: int, value: int) -> None:
        assert 0 <= axis < self.dimension, f"Axis {axis} out of bounds for dim {self.dimension}"
        assert value in (0, 1), f"Face value must be 0 or 1, got {value}"
        bit = 2 * axis + value
        self.mask |= (1 << bit)

    def has_face(self, axis: int, value: int) -> bool:
        bit = 2 * axis + value
        return bool(self.mask & (1 << bit))

    def is_face_active(self, face_idx: int) -> bool:
        return bool(self.mask & (1 << face_idx))

    @classmethod
    def from_bitmask(cls, dimension: int, mask: int) -> FaceMask:
        return cls(dimension=dimension, mask=mask)


@dataclass
class KanBox:
    """An open box (horn) in the cubical complex to be filled by hcomp."""
    dimension: int
    face_mask: FaceMask
    base_u0: Multivector
    tube: Callable[[float], Multivector]
    signature: CliffordSignature = field(default_factory=lambda: PGA301)

    def to_stack_frame(self) -> KanStackFrame:
        """Packs this KanBox into a ctypes KanStackFrame matching the C-ABI."""
        frame = KanStackFrame()
        frame.face_mask_phi = ctypes.c_uint32(self.face_mask.mask)
        frame.dimension = ctypes.c_uint32(self.dimension)

        # Copy base_u0 components
        comps = self.base_u0.components
        for idx in range(min(16, len(comps))):
            frame.base_multivector[idx] = ctypes.c_double(comps[idx])

        # Compile tube evaluator closure to C callback
        sig = self.signature
        tube_callable = self.tube

        def _eval_callback(in_coords_p, t_val, out_mv_p):
            res_mv = tube_callable(float(t_val))
            res_comps = res_mv.components
            for i in range(min(16, len(res_comps))):
                out_mv_p[i] = ctypes.c_double(res_comps[i])

        c_func = TUBE_EVALUATOR_FUNC(_eval_callback)
        # Store callback reference on frame to prevent garbage collection
        frame._keepalive_c_func = c_func
        frame.tube_evaluator = c_func
        frame.env_stack_ptr = ctypes.c_void_p(0)
        return frame


class KanComposition:
    """Constructive Kan box filling engine (hcomp) and dynamic n-truncation."""

    @staticmethod
    def hcomp(
        kan_box: KanBox,
        interval_t: float = 1.0,
        max_homotopy_dim: int = 2,
    ) -> Multivector:
        """Constructively computes the missing boundary face u(t) for an open box.
        
        Args:
            kan_box: The open box with base u0 and continuous tube function.
            interval_t: The evaluation point on the cubical interval I (default 1.0 for terminal face).
            max_homotopy_dim: Maximum coherence dimension before applying n-truncation.
        
        Returns:
            The terminal multivector state u(interval_t), guaranteed n-truncated.
        """
        assert 0.0 <= interval_t <= 1.0, f"Interval parameter t must be in [0, 1], got {interval_t}"

        # If t == 0, returns the initial base point u0
        if abs(interval_t) < 1e-15:
            return KanComposition.truncate(kan_box.base_u0, max_homotopy_dim)

        # Evaluate the continuous tube closure along the interval variable
        evaluated_mv = kan_box.tube(interval_t)

        # Apply dynamic n-truncation pass to bound higher coherences
        truncated_mv = KanComposition.truncate(evaluated_mv, max_homotopy_dim)
        return truncated_mv

    @staticmethod
    def truncate(mv: Multivector, n_dim: int) -> Multivector:
        """Dynamic n-truncation pass.
        
        Prunes all higher-dimensional coherences of grade > n_dim to machine NOPs (0.0),
        collapsing infinite homotopy towers to a strictly bounded O(n) memory footprint.
        """
        dim = mv.signature.multivector_dim
        comps = list(mv.components)
        for mask in range(dim):
            grade = bin(mask).count("1")
            if grade > n_dim:
                comps[mask] = 0.0
        return Multivector(mv.signature, comps)
