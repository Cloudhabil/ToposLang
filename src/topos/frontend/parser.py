"""Recursive-descent parser and lowering compiler for ToposLang (.tau)."""

from __future__ import annotations
from typing import List, Optional, Union

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.frontend.ast import (
    ApplyStmt,
    BettiCondition,
    Cell0Decl,
    Cell1Decl,
    Cell2Decl,
    HITConstructor,
    HITDecl,
    PathDecl,
    PathExpr,
    PointDecl,
    ProcessDecl,
    Program,
    RestrictionDecl,
    RewriteDecl,
    SheafDecl,
    SheafDimKerCondition,
    StalkDecl,
    Statement,
    SurfaceDecl,
    UntilLoop,
)
from topos.frontend.lexer import Lexer, Token, TokenType


class ParserError(Exception):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"ParserError at {line}:{col}: {message}")
        self.line = line
        self.col = col


class Parser:
    """Parses a sequence of tokens into an AST."""

    def __init__(self, tokens: List[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _match(self, *types: TokenType) -> bool:
        if self._peek().type in types:
            self._advance()
            return True
        return False

    def _advance(self) -> Token:
        tok = self._peek()
        if tok.type != TokenType.EOF:
            self.pos += 1
        return tok

    def _expect(self, token_type: TokenType, err_msg: str) -> Token:
        tok = self._peek()
        if tok.type != token_type:
            raise ParserError(
                f"{err_msg}, got {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        return self._advance()

    def parse(self) -> Program:
        statements: List[Statement] = []
        while self._peek().type != TokenType.EOF:
            stmt = self._parse_statement()
            if stmt is not None:
                statements.append(stmt)
        return Program(statements=statements)

    def _parse_statement(self) -> Statement:
        tok = self._peek()
        if tok.type == TokenType.CELL0:
            return self._parse_cell0()
        elif tok.type == TokenType.CELL1:
            return self._parse_cell1()
        elif tok.type == TokenType.CELL2:
            return self._parse_cell2()
        elif tok.type == TokenType.HIT:
            return self._parse_hit()
        elif tok.type == TokenType.PROCESS:
            return self._parse_process()
        elif tok.type == TokenType.REWRITE:
            return self._parse_rewrite()
        elif tok.type == TokenType.UNTIL:
            return self._parse_until()
        elif tok.type == TokenType.APPLY:
            return self._parse_apply()
        elif tok.type == TokenType.SHEAF:
            return self._parse_sheaf()
        else:
            raise ParserError(
                f"Unexpected token starting statement: {tok.type.name} ({tok.value!r})",
                tok.line,
                tok.col,
            )

    def _parse_cell0(self) -> Cell0Decl:
        tok = self._expect(TokenType.CELL0, "Expected 'cell0'")
        name_tok = self._expect(TokenType.IDENT, "Expected identifier for 0-cell name")
        self._expect(TokenType.SEMICOLON, "Expected ';' after cell0 declaration")
        return Cell0Decl(line=tok.line, col=tok.col, name=name_tok.value)

    def _parse_cell1(self) -> Cell1Decl:
        tok = self._expect(TokenType.CELL1, "Expected 'cell1'")
        name_tok = self._expect(TokenType.IDENT, "Expected identifier for 1-cell name")
        self._expect(TokenType.COLON, "Expected ':' after cell1 name")
        source_tok = self._expect(TokenType.IDENT, "Expected source 0-cell identifier")
        self._expect(TokenType.ARROW, "Expected '->' in 1-cell boundary specification")
        target_tok = self._expect(TokenType.IDENT, "Expected target 0-cell identifier")
        self._expect(TokenType.SEMICOLON, "Expected ';' after cell1 declaration")
        return Cell1Decl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            source=source_tok.value,
            target=target_tok.value,
        )

    def _parse_path_expr(self) -> PathExpr:
        tok = self._peek()
        elements: List[str] = []

        if self._match(TokenType.LPAREN):
            # (f * g * h)
            first_ident = self._expect(TokenType.IDENT, "Expected identifier in path expression")
            elements.append(first_ident.value)
            while self._match(TokenType.STAR):
                next_ident = self._expect(TokenType.IDENT, "Expected identifier after '*'")
                elements.append(next_ident.value)
            self._expect(TokenType.RPAREN, "Expected ')' ending path expression")
        elif tok.type == TokenType.IDENT:
            # Single ident or unparenthesized chain f * g
            first_ident = self._advance()
            elements.append(first_ident.value)
            while self._match(TokenType.STAR):
                next_ident = self._expect(TokenType.IDENT, "Expected identifier after '*'")
                elements.append(next_ident.value)
        else:
            raise ParserError("Expected path expression", tok.line, tok.col)

        return PathExpr(line=tok.line, col=tok.col, elements=elements)

    def _parse_cell2(self) -> Cell2Decl:
        tok = self._expect(TokenType.CELL2, "Expected 'cell2'")
        name_tok = self._expect(TokenType.IDENT, "Expected identifier for 2-cell name")
        self._expect(TokenType.COLON, "Expected ':' after cell2 name")
        source_expr = self._parse_path_expr()
        self._expect(TokenType.DOUBLE_ARROW, "Expected '=>' in 2-cell boundary specification")
        target_expr = self._parse_path_expr()
        self._expect(TokenType.SEMICOLON, "Expected ';' after cell2 declaration")
        return Cell2Decl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            source=source_expr,
            target=target_expr,
        )

    def _parse_hit(self) -> HITDecl:
        tok = self._expect(TokenType.HIT, "Expected 'hit'")
        name_tok = self._expect(TokenType.IDENT, "Expected HIT name")
        self._expect(TokenType.LBRACE, "Expected '{' to start HIT body")

        constructors: List[HITConstructor] = []
        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            c_tok = self._peek()
            if c_tok.type == TokenType.POINT:
                self._advance()
                p_name = self._expect(TokenType.IDENT, "Expected point constructor name")
                self._expect(TokenType.SEMICOLON, "Expected ';' after point declaration")
                constructors.append(PointDecl(line=c_tok.line, col=c_tok.col, name=p_name.value))
            elif c_tok.type == TokenType.PATH:
                self._advance()
                p_name = self._expect(TokenType.IDENT, "Expected path constructor name")
                self._expect(TokenType.COLON, "Expected ':' after path name")
                src = self._expect(TokenType.IDENT, "Expected path source point")
                if not (self._match(TokenType.EQUALS) or self._match(TokenType.ARROW)):
                    raise ParserError("Expected '=' or '->' in path constructor", self._peek().line, self._peek().col)
                tgt = self._expect(TokenType.IDENT, "Expected path target point")
                self._expect(TokenType.SEMICOLON, "Expected ';' after path declaration")
                constructors.append(
                    PathDecl(
                        line=c_tok.line,
                        col=c_tok.col,
                        name=p_name.value,
                        source=src.value,
                        target=tgt.value,
                    )
                )
            elif c_tok.type == TokenType.SURFACE:
                self._advance()
                s_name = self._expect(TokenType.IDENT, "Expected surface constructor name")
                self._expect(TokenType.COLON, "Expected ':' after surface name")
                lhs = self._parse_path_expr()
                if not (self._match(TokenType.EQUALS) or self._match(TokenType.DOUBLE_ARROW)):
                    raise ParserError("Expected '=' or '=>' in surface constructor", self._peek().line, self._peek().col)
                rhs = self._parse_path_expr()
                self._expect(TokenType.SEMICOLON, "Expected ';' after surface declaration")
                constructors.append(
                    SurfaceDecl(
                        line=c_tok.line,
                        col=c_tok.col,
                        name=s_name.value,
                        lhs=lhs,
                        rhs=rhs,
                    )
                )
            else:
                raise ParserError(
                    f"Unexpected token in HIT body: {c_tok.type.name} ({c_tok.value!r})",
                    c_tok.line,
                    c_tok.col,
                )

        self._expect(TokenType.RBRACE, "Expected '}' ending HIT body")
        return HITDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            constructors=constructors,
        )

    def _parse_rewrite(self) -> RewriteDecl:
        tok = self._expect(TokenType.REWRITE, "Expected 'rewrite'")
        name_tok = self._expect(TokenType.IDENT, "Expected rewrite rule name")
        self._expect(TokenType.COLON, "Expected ':' after rewrite rule name")
        lhs = self._parse_path_expr()
        self._expect(TokenType.DOUBLE_ARROW, "Expected '=>' in rewrite rule")
        rhs = self._parse_path_expr()
        self._expect(TokenType.SEMICOLON, "Expected ';' after rewrite rule")
        return RewriteDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            lhs=lhs,
            rhs=rhs,
        )

    def _parse_betti_condition(self) -> BettiCondition:
        tok = self._expect(TokenType.BETTI, "Expected 'betti'")
        self._expect(TokenType.LPAREN, "Expected '(' after 'betti'")
        target = self._expect(TokenType.IDENT, "Expected complex identifier in betti call")
        dim_val = 1
        if self._match(TokenType.COMMA):
            if self._match(TokenType.DIM):
                self._expect(TokenType.EQUALS, "Expected '=' after 'dim'")
            dim_tok = self._expect(TokenType.NUMBER, "Expected integer dimension")
            dim_val = int(dim_tok.value)
        self._expect(TokenType.RPAREN, "Expected ')' closing betti call")

        # Comparison operator
        op_tok = self._peek()
        if op_tok.type in (TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            self._advance()
            val_tok = self._expect(TokenType.NUMBER, "Expected number after comparison operator")
            return BettiCondition(
                line=tok.line,
                col=tok.col,
                target_name=target.value,
                dim=dim_val,
                operator=op_tok.value,
                value=int(val_tok.value),
            )
        else:
            raise ParserError("Expected comparison operator in betti condition", op_tok.line, op_tok.col)

    def _parse_sheaf_dim_ker_condition(self) -> SheafDimKerCondition:
        tok = self._expect(TokenType.SHEAF_DIM_KER, "Expected 'sheaf_dim_ker'")
        self._expect(TokenType.LPAREN, "Expected '(' after 'sheaf_dim_ker'")
        sheaf_tok = self._expect(TokenType.IDENT, "Expected sheaf identifier in sheaf_dim_ker call")
        operator_name = "L2_down"
        if self._match(TokenType.DOT):
            op_tok = self._expect(TokenType.IDENT, "Expected operator name after '.'")
            operator_name = op_tok.value
        self._expect(TokenType.RPAREN, "Expected ')' closing sheaf_dim_ker call")

        op_tok = self._peek()
        if op_tok.type in (TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT, TokenType.LTE, TokenType.GTE):
            self._advance()
            val_tok = self._expect(TokenType.NUMBER, "Expected number after comparison operator")
            return SheafDimKerCondition(
                line=tok.line,
                col=tok.col,
                sheaf_name=sheaf_tok.value,
                operator_name=operator_name,
                operator=op_tok.value,
                value=int(val_tok.value),
            )
        else:
            raise ParserError("Expected comparison operator in sheaf_dim_ker condition", op_tok.line, op_tok.col)

    def _parse_until(self) -> UntilLoop:
        tok = self._expect(TokenType.UNTIL, "Expected 'until'")
        peek_tok = self._peek()
        if peek_tok.type == TokenType.BETTI:
            cond: Union[BettiCondition, SheafDimKerCondition] = self._parse_betti_condition()
        elif peek_tok.type == TokenType.SHEAF_DIM_KER:
            cond = self._parse_sheaf_dim_ker_condition()
        else:
            raise ParserError(
                f"Expected 'betti' or 'sheaf_dim_ker' condition after 'until', got {peek_tok.type.name}",
                peek_tok.line,
                peek_tok.col,
            )
        self._expect(TokenType.LBRACE, "Expected '{' starting until body")
        body: List[Statement] = []
        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            stmt = self._parse_statement()
            if stmt is not None:
                body.append(stmt)
        self._expect(TokenType.RBRACE, "Expected '}' ending until body")
        return UntilLoop(line=tok.line, col=tok.col, condition=cond, body=body)

    def _parse_matrix_expr(self) -> List[List[float]]:
        self._expect(TokenType.LBRACKET, "Expected '[' starting matrix expression")
        rows: List[List[float]] = []
        while self._peek().type != TokenType.RBRACKET and self._peek().type != TokenType.EOF:
            row: List[float] = []
            val_tok = self._expect(TokenType.NUMBER, "Expected number in matrix row")
            row.append(float(val_tok.value))
            while self._match(TokenType.COMMA):
                val_tok = self._expect(TokenType.NUMBER, "Expected number after ',' in matrix row")
                row.append(float(val_tok.value))
            rows.append(row)
            if not self._match(TokenType.SEMICOLON):
                break
        self._expect(TokenType.RBRACKET, "Expected ']' ending matrix expression")
        return rows

    def _parse_stalk(self) -> StalkDecl:
        tok = self._expect(TokenType.STALK, "Expected 'stalk'")
        self._expect(TokenType.LBRACKET, "Expected '[' after 'stalk'")
        dim_tok = self._expect(TokenType.NUMBER, "Expected dimension number in stalk declaration")
        self._expect(TokenType.RBRACKET, "Expected ']' after stalk dimension")
        self._expect(TokenType.EQUALS, "Expected '=' in stalk declaration")
        field_tok = self._expect(TokenType.IDENT, "Expected field type in stalk declaration")
        vector_dim = 1
        if self._match(TokenType.CARET):
            vdim_tok = self._expect(TokenType.NUMBER, "Expected vector dimension after '^'")
            vector_dim = int(vdim_tok.value)
        self._expect(TokenType.SEMICOLON, "Expected ';' after stalk declaration")
        return StalkDecl(
            line=tok.line,
            col=tok.col,
            cell_dim=int(dim_tok.value),
            field_type=field_tok.value,
            vector_dim=vector_dim,
        )

    def _parse_restriction(self) -> RestrictionDecl:
        tok = self._expect(TokenType.RESTRICTION, "Expected 'restriction'")
        self._expect(TokenType.LPAREN, "Expected '(' after 'restriction'")
        source_tok = self._expect(TokenType.IDENT, "Expected source cell name in restriction")
        self._expect(TokenType.ARROW, "Expected '->' in restriction")
        target_tok = self._expect(TokenType.IDENT, "Expected target cell name in restriction")
        self._expect(TokenType.RPAREN, "Expected ')' after restriction mapping")
        self._expect(TokenType.EQUALS, "Expected '=' in restriction declaration")
        matrix_vals = self._parse_matrix_expr()
        self._expect(TokenType.SEMICOLON, "Expected ';' after restriction declaration")
        return RestrictionDecl(
            line=tok.line,
            col=tok.col,
            source_cell=source_tok.value,
            target_cell=target_tok.value,
            matrix_values=matrix_vals,
        )

    def _parse_sheaf(self) -> SheafDecl:
        tok = self._expect(TokenType.SHEAF, "Expected 'sheaf'")
        name_tok = self._expect(TokenType.IDENT, "Expected sheaf name identifier")
        self._expect(TokenType.OVER, "Expected 'over' after sheaf name")
        complex_tok = self._expect(TokenType.IDENT, "Expected complex identifier after 'over'")
        self._expect(TokenType.LBRACE, "Expected '{' starting sheaf body")

        stalks: List[StalkDecl] = []
        restrictions: List[RestrictionDecl] = []

        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            c_tok = self._peek()
            if c_tok.type == TokenType.STALK:
                stalks.append(self._parse_stalk())
            elif c_tok.type == TokenType.RESTRICTION:
                restrictions.append(self._parse_restriction())
            else:
                raise ParserError(
                    f"Unexpected token in sheaf body: {c_tok.type.name} ({c_tok.value!r})",
                    c_tok.line,
                    c_tok.col,
                )

        self._expect(TokenType.RBRACE, "Expected '}' closing sheaf body")
        return SheafDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            complex_name=complex_tok.value,
            stalks=stalks,
            restrictions=restrictions,
        )

    def _parse_apply(self) -> ApplyStmt:
        tok = self._expect(TokenType.APPLY, "Expected 'apply'")
        rule_tok = self._expect(TokenType.IDENT, "Expected rewrite rule name to apply")
        self._expect(TokenType.ON, "Expected 'on' after apply rule name")
        target_tok = self._expect(TokenType.IDENT, "Expected target complex name")
        self._expect(TokenType.SEMICOLON, "Expected ';' after apply statement")
        return ApplyStmt(
            line=tok.line,
            col=tok.col,
            rule_name=rule_tok.value,
            target_name=target_tok.value,
        )

    def _parse_process(self) -> ProcessDecl:
        tok = self._expect(TokenType.PROCESS, "Expected 'process'")
        name_tok = self._expect(TokenType.IDENT, "Expected process name")
        self._expect(TokenType.LPAREN, "Expected '(' for process parameters")
        param_name = self._expect(TokenType.IDENT, "Expected parameter name")
        param_type = "CellComplex"
        if self._match(TokenType.COLON):
            type_tok = self._expect(TokenType.IDENT, "Expected parameter type")
            param_type = type_tok.value
        self._expect(TokenType.RPAREN, "Expected ')' closing parameter list")
        self._expect(TokenType.LBRACE, "Expected '{' starting process body")

        body: List[Statement] = []
        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            stmt = self._parse_statement()
            if stmt is not None:
                body.append(stmt)

        self._expect(TokenType.RBRACE, "Expected '}' ending process body")
        return ProcessDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            param_name=param_name.value,
            param_type=param_type,
            body=body,
        )


def parse_source(source: str) -> Program:
    """Tokenizes and parses ToposLang source code into an AST."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def build_complex(program: Program, name: str = "TauComplex") -> CellComplex:
    """Lowers AST cell and HIT declarations into a validated CellComplex."""
    complex_obj = CellComplex(name=name)

    # First pass: collect 0-cells
    for stmt in program.statements:
        if isinstance(stmt, Cell0Decl):
            complex_obj.add_cell(GlobularCell(stmt.name, dim=0))
        elif isinstance(stmt, HITDecl):
            for c in stmt.constructors:
                if isinstance(c, PointDecl):
                    complex_obj.add_cell(GlobularCell(c.name, dim=0))

    # Second pass: collect 1-cells
    for stmt in program.statements:
        if isinstance(stmt, Cell1Decl):
            src_cell = complex_obj.get_cell(stmt.source, dim=0)
            tgt_cell = complex_obj.get_cell(stmt.target, dim=0)
            complex_obj.add_cell(GlobularCell(stmt.name, dim=1, source=src_cell, target=tgt_cell))
        elif isinstance(stmt, HITDecl):
            for c in stmt.constructors:
                if isinstance(c, PathDecl):
                    src_cell = complex_obj.get_cell(c.source, dim=0)
                    tgt_cell = complex_obj.get_cell(c.target, dim=0)
                    complex_obj.add_cell(GlobularCell(c.name, dim=1, source=src_cell, target=tgt_cell))

    # Third pass: collect 2-cells
    def _path_to_chain(expr: PathExpr) -> Chain:
        chain = Chain.zero(1)
        for elem in expr.elements:
            cell = complex_obj.get_cell(elem, dim=1)
            chain = chain + Chain.from_cell(cell)
        return chain

    for stmt in program.statements:
        if isinstance(stmt, Cell2Decl):
            src_chain = _path_to_chain(stmt.source)
            tgt_chain = _path_to_chain(stmt.target)
            complex_obj.add_cell(GlobularCell(stmt.name, dim=2, source=src_chain, target=tgt_chain))
        elif isinstance(stmt, HITDecl):
            for c in stmt.constructors:
                if isinstance(c, SurfaceDecl):
                    src_chain = _path_to_chain(c.lhs)
                    tgt_chain = _path_to_chain(c.rhs)
                    complex_obj.add_cell(GlobularCell(c.name, dim=2, source=src_chain, target=tgt_chain))

    # Validate nilpotency and boundary attachments
    complex_obj.validate(enforce_nilpotence=True)
    return complex_obj
