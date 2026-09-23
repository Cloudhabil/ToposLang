"""Topological Invariants & Stopping Predicates for Invariant-Driven Control Flow.

Computes Euler characteristics via the Euler-Poincare theorem and provides
stopping predicates (e.g., beta_1 == 0) for topological loop-collapsing processes.
"""

from __future__ import annotations
from typing import Dict

from topos.core.complex import CellComplex
from topos.topology.homology import HomologyEngine


def betti_numbers(complex_obj: CellComplex) -> Dict[int, int]:
    """Extracts the dictionary of Betti numbers {k: beta_k} for the complex."""
    return HomologyEngine(complex_obj).betti_profile()


def euler_characteristic(complex_obj: CellComplex) -> int:
    """Computes the topological Euler characteristic chi = sum (-1)^k beta_k."""
    profile = betti_numbers(complex_obj)
    chi = 0
    for k, beta in profile.items():
        chi += ((-1) ** k) * beta
    return chi


def verify_euler_poincare(complex_obj: CellComplex) -> bool:
    """Verifies the Euler-Poincare formula: sum (-1)^k |K_k| == sum (-1)^k beta_k."""
    combinatorial_chi = complex_obj.euler_characteristic()
    homological_chi = euler_characteristic(complex_obj)
    return combinatorial_chi == homological_chi


def is_connected(complex_obj: CellComplex) -> bool:
    """Returns True if the complex is path-connected (beta_0 == 1)."""
    return HomologyEngine(complex_obj).betti_number(0) == 1


def is_simply_connected(complex_obj: CellComplex) -> bool:
    """Returns True if the complex has no 1-dimensional topological holes (beta_1 == 0)."""
    return HomologyEngine(complex_obj).betti_number(1) == 0


def has_uncontracted_loops(complex_obj: CellComplex) -> bool:
    """Returns True if 1-dimensional cycles/loops remain (beta_1 > 0)."""
    return HomologyEngine(complex_obj).betti_number(1) > 0


def has_uncontracted_cavities(complex_obj: CellComplex) -> bool:
    """Returns True if 2-dimensional hollow cavities/voids remain (beta_2 > 0)."""
    return HomologyEngine(complex_obj).betti_number(2) > 0
