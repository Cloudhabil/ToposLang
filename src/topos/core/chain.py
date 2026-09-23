"""Formal Chain Group C_k(K; Z) implementation.

A chain is a formal linear combination of k-cells with integer coefficients.
Supports standard abelian group operations with automatic zero-coefficient pruning.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Iterator, Tuple, Union, Any

if TYPE_CHECKING:
    from topos.core.cell import Cell


class Chain:
    """Represents a formal k-chain: sum(a_i * c_i^k) with integer coefficients.
    
    Attributes:
        dim: The topological dimension k of the chain.
        terms: A dictionary mapping Cell -> integer coefficient (pruned of zeros).
    """

    __slots__ = ("_dim", "_terms")

    def __init__(self, dim: int, terms: Dict[Any, int] | None = None) -> None:
        self._dim = dim
        self._terms: Dict[Any, int] = {}
        if terms:
            for cell, coeff in terms.items():
                if cell.dim != dim:
                    raise ValueError(
                        f"Cell {cell} has dimension {cell.dim}, expected dimension {dim}"
                    )
                if coeff != 0:
                    self._terms[cell] = self._terms.get(cell, 0) + coeff
            # Clean up any terms that canceled out to zero
            self._terms = {c: a for c, a in self._terms.items() if a != 0}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def terms(self) -> Dict[Any, int]:
        return dict(self._terms)

    @classmethod
    def zero(cls, dim: int) -> Chain:
        """Returns the canonical zero chain of dimension dim."""
        return cls(dim, {})

    @classmethod
    def from_cell(cls, cell: Any, coeff: int = 1) -> Chain:
        """Creates an elementary chain consisting of a single cell."""
        if coeff == 0:
            return cls(cell.dim, {})
        return cls(cell.dim, {cell: coeff})

    def is_zero(self) -> bool:
        """Checks if the chain is identically zero."""
        return len(self._terms) == 0

    def __getitem__(self, cell: Any) -> int:
        return self._terms.get(cell, 0)

    def __contains__(self, cell: Any) -> bool:
        return cell in self._terms

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self) -> Iterator[Tuple[Any, int]]:
        return iter(self._terms.items())

    def __add__(self, other: Chain) -> Chain:
        if not isinstance(other, Chain):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError(
                f"Cannot add chains of different dimensions: {self._dim} and {other._dim}"
            )
        combined: Dict[Any, int] = dict(self._terms)
        for cell, coeff in other._terms.items():
            combined[cell] = combined.get(cell, 0) + coeff
        return Chain(self._dim, combined)

    def __sub__(self, other: Chain) -> Chain:
        if not isinstance(other, Chain):
            return NotImplemented
        if self._dim != other._dim:
            raise ValueError(
                f"Cannot subtract chains of different dimensions: {self._dim} and {other._dim}"
            )
        combined: Dict[Any, int] = dict(self._terms)
        for cell, coeff in other._terms.items():
            combined[cell] = combined.get(cell, 0) - coeff
        return Chain(self._dim, combined)

    def __neg__(self) -> Chain:
        return Chain(self._dim, {cell: -coeff for cell, coeff in self._terms.items()})

    def __mul__(self, scalar: int) -> Chain:
        if not isinstance(scalar, int):
            return NotImplemented
        if scalar == 0:
            return Chain.zero(self._dim)
        return Chain(self._dim, {cell: coeff * scalar for cell, coeff in self._terms.items()})

    def __rmul__(self, scalar: int) -> Chain:
        return self.__mul__(scalar)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Chain):
            return self._dim == other._dim and self._terms == other._terms
        if other == 0:
            return self.is_zero()
        return False

    def __hash__(self) -> int:
        # Sort terms by cell representation for stable hashing
        sorted_terms = tuple(sorted((hash(c), a) for c, a in self._terms.items()))
        return hash((self._dim, sorted_terms))

    def __repr__(self) -> str:
        if self.is_zero():
            return f"0_{self._dim}"
        parts = []
        for cell, coeff in self._terms.items():
            if coeff == 1:
                parts.append(f"{cell.name}")
            elif coeff == -1:
                parts.append(f"-{cell.name}")
            else:
                parts.append(f"{coeff}*{cell.name}")
        return " + ".join(parts).replace(" + -", " - ")
