"""Abstract Syntax Tree (AST) definitions for ToposLang (.tau)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 1
    col: int = 1


@dataclass
class PathExpr(ASTNode):
    """Represents a composite path of 1-cells: e.g. f or (f * g * h)."""
    elements: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if len(self.elements) == 1:
            return self.elements[0]
        return f"({' * '.join(self.elements)})"


@dataclass
class Statement(ASTNode):
    """Base class for all statements."""
    pass


@dataclass
class Cell0Decl(Statement):
    """cell0 A;"""
    name: str = ""


@dataclass
class Cell1Decl(Statement):
    """cell1 f : A -> B;"""
    name: str = ""
    source: str = ""
    target: str = ""


@dataclass
class Cell2Decl(Statement):
    """cell2 alpha : (f * g) => h;"""
    name: str = ""
    source: PathExpr = field(default_factory=PathExpr)
    target: PathExpr = field(default_factory=PathExpr)


@dataclass
class HITConstructor(ASTNode):
    """Constructor within a Higher Inductive Type."""
    pass


@dataclass
class PointDecl(HITConstructor):
    """point base;"""
    name: str = ""


@dataclass
class PathDecl(HITConstructor):
    """path loop : base = base; or path f : A -> B;"""
    name: str = ""
    source: str = ""
    target: str = ""


@dataclass
class SurfaceDecl(HITConstructor):
    """surface face : (p * q) = (q * p);"""
    name: str = ""
    lhs: PathExpr = field(default_factory=PathExpr)
    rhs: PathExpr = field(default_factory=PathExpr)


@dataclass
class HITDecl(Statement):
    """hit S1 { point base; path loop : base = base; }"""
    name: str = ""
    constructors: List[HITConstructor] = field(default_factory=list)


@dataclass
class RewriteDecl(Statement):
    """rewrite loop_contraction : (e1 * e2) => e3;"""
    name: str = ""
    lhs: PathExpr = field(default_factory=PathExpr)
    rhs: PathExpr = field(default_factory=PathExpr)


@dataclass
class ApplyStmt(Statement):
    """apply loop_contraction on complex;"""
    rule_name: str = ""
    target_name: str = ""


@dataclass
class BettiCondition(ASTNode):
    """betti(complex, dim=1) == 0"""
    target_name: str = ""
    dim: int = 1
    operator: str = "=="
    value: int = 0


@dataclass
class SheafDimKerCondition(ASTNode):
    """sheaf_dim_ker(SheafName.L2_down) == 0 or sheaf_dim_ker(SheafName) == 0"""
    sheaf_name: str = ""
    operator_name: str = "L2_down"
    operator: str = "=="
    value: int = 0


@dataclass
class UntilLoop(Statement):
    """until betti(...) == 0 { ... } or until sheaf_dim_ker(...) == 0 { ... }"""
    condition: Union[BettiCondition, SheafDimKerCondition] = field(default_factory=BettiCondition)
    body: List[Statement] = field(default_factory=list)


@dataclass
class StalkDecl(ASTNode):
    """stalk[2] = R^3; or stalk[1] = F_65537^2;"""
    cell_dim: int = 0
    field_type: str = "R"
    vector_dim: int = 1


@dataclass
class RestrictionDecl(ASTNode):
    """restriction(p0 -> e0) = [1.0, 0.0; 0.0, 1.0];"""
    source_cell: str = ""
    target_cell: str = ""
    matrix_values: List[List[float]] = field(default_factory=list)


@dataclass
class SheafDecl(Statement):
    """sheaf GaugeField over TorusLattice { stalk[2] = R^3; ... }"""
    name: str = ""
    complex_name: str = ""
    stalks: List[StalkDecl] = field(default_factory=list)
    restrictions: List[RestrictionDecl] = field(default_factory=list)


@dataclass
class ProcessDecl(Statement):
    """process contract_vortices(complex: CellComplex) { ... }"""
    name: str = ""
    param_name: str = ""
    param_type: str = "CellComplex"
    body: List[Statement] = field(default_factory=list)


@dataclass
class Program(ASTNode):
    """Root program node containing a sequence of statements."""
    statements: List[Statement] = field(default_factory=list)
