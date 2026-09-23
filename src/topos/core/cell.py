"""Topological Cell Abstractions.

Defines the base Cell class along with Simplex (simplicial homology) and
GlobularCell (higher categories and globular rewrite rules).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Optional
from topos.core.chain import Chain


class Cell(ABC):
    """Abstract base class for all topological n-cells."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """The topological dimension k >= 0 of the cell."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical name or label of the cell."""
        pass

    @abstractmethod
    def boundary(self) -> Chain:
        """Computes the (dim - 1)-chain representing the oriented boundary of this cell."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dim}, name={self.name!r})"


@dataclass(frozen=True)
class Simplex(Cell):
    """An oriented k-simplex defined by an ordered tuple of vertices [v_0, v_1, ..., v_k].
    
    Dimension k = len(vertices) - 1.
    Boundary formula: d([v_0, ..., v_k]) = sum_{j=0}^k (-1)^j [v_0, ..., ^v_j, ..., v_k]
    """

    vertices: Tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) == 0:
            raise ValueError("A simplex must have at least one vertex.")
        if len(set(self.vertices)) != len(self.vertices):
            raise ValueError(f"Simplex vertices must be distinct, got: {self.vertices}")

    @property
    def dim(self) -> int:
        return len(self.vertices) - 1

    @property
    def name(self) -> str:
        return f"[{', '.join(self.vertices)}]"

    def boundary(self) -> Chain:
        k = self.dim
        if k == 0:
            return Chain.zero(-1)
        
        terms = {}
        for j in range(len(self.vertices)):
            face_vertices = self.vertices[:j] + self.vertices[j + 1:]
            face = Simplex(face_vertices)
            sign = (-1) ** j
            terms[face] = sign
            
        return Chain(k - 1, terms)


class GlobularCell(Cell):
    """A globular k-cell representing an n-morphism alpha : source => target.
    
    - 0-cell (object): source=None, target=None, boundary = 0
    - 1-cell (arrow): source (0-cell), target (0-cell), boundary = target - source
    - k-cell (k >= 2): source (k-1 cell or chain), target (k-1 cell or chain),
      boundary = target - source.
    """

    def __init__(
        self,
        name: str,
        dim: int,
        source: Optional[Union[Cell, Chain]] = None,
        target: Optional[Union[Cell, Chain]] = None,
    ) -> None:
        if dim < 0:
            raise ValueError(f"Cell dimension must be non-negative, got {dim}")
        
        self._name = name
        self._dim = dim

        if dim == 0:
            if source is not None or target is not None:
                raise ValueError("0-cells cannot have source or target boundaries.")
            self._source_chain = Chain.zero(-1)
            self._target_chain = Chain.zero(-1)
        else:
            if source is None or target is None:
                raise ValueError(f"{dim}-cells must specify both source and target boundaries.")
            
            # Convert to Chain if Cell
            if isinstance(source, Cell):
                if source.dim != dim - 1:
                    raise ValueError(
                        f"Source cell dimension {source.dim} does not match expected {dim - 1}"
                    )
                self._source_chain = Chain.from_cell(source)
            elif isinstance(source, Chain):
                if source.dim != dim - 1:
                    raise ValueError(
                        f"Source chain dimension {source.dim} does not match expected {dim - 1}"
                    )
                self._source_chain = source
            else:
                raise TypeError(f"Invalid source type: {type(source)}")

            if isinstance(target, Cell):
                if target.dim != dim - 1:
                    raise ValueError(
                        f"Target cell dimension {target.dim} does not match expected {dim - 1}"
                    )
                self._target_chain = Chain.from_cell(target)
            elif isinstance(target, Chain):
                if target.dim != dim - 1:
                    raise ValueError(
                        f"Target chain dimension {target.dim} does not match expected {dim - 1}"
                    )
                self._target_chain = target
            else:
                raise TypeError(f"Invalid target type: {type(target)}")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @property
    def source(self) -> Chain:
        return self._source_chain

    @property
    def target(self) -> Chain:
        return self._target_chain

    def boundary(self) -> Chain:
        if self._dim == 0:
            return Chain.zero(-1)
        return self._target_chain - self._source_chain

    def __hash__(self) -> int:
        return hash((self._dim, self._name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GlobularCell):
            return False
        return (
            self._dim == other._dim
            and self._name == other._name
            and self._source_chain == other._source_chain
            and self._target_chain == other._target_chain
        )
