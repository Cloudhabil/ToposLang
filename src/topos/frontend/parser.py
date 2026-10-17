"""Recursive-descent parser and lowering compiler for ToposLang (.tau)."""

from __future__ import annotations
from typing import List, Optional, Union

from topos.core.cell import GlobularCell
from topos.core.chain import Chain
from topos.core.complex import CellComplex
from topos.frontend.ast import (
    ApplyStmt,
    AssertNilpotencyStmt,
    BettiCondition,
    Cell0Decl,
    Cell1Decl,
    Cell2Decl,
    CellsGradeDecl,
    ComplexBlockDecl,
    ComputeHodgeSpectrumStmt,
    ComputeScreeningGapStmt,
    HITConstructor,
    HITDecl,
    PathDecl,
    PathExpr,
    PointDecl,
    ProcessDecl,
    ProComplexDecl,
    Program,
    RestrictionDecl,
    RewriteDecl,
    SheafDecl,
    SheafDimKerCondition,
    StalkDecl,
    Statement,
    SurfaceDecl,
    TargetDecl,
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

    def _expect_ident(self, err_msg: str) -> Token:
        tok = self._peek()
        if tok.type in (TokenType.IDENT, TokenType.COMPLEX, TokenType.FIELD):
            return self._advance()
        raise ParserError(
            f"{err_msg}, got {tok.type.name} ({tok.value!r})",
            tok.line,
            tok.col,
        )

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
        elif tok.type == TokenType.CELL:
            return self._parse_bracketed_cell()
        elif tok.type == TokenType.CELLS:
            return self._parse_cells_grade()
        elif tok.type == TokenType.TARGET:
            return self._parse_target()
        elif tok.type == TokenType.COMPLEX:
            return self._parse_complex_block()
        elif tok.type == TokenType.PROCOMPLEX:
            return self._parse_procomplex()
        elif tok.type == TokenType.ASSERT:
            return self._parse_assert()
        elif tok.type == TokenType.COMPUTE_HODGE_SPECTRUM:
            return self._parse_compute_hodge_spectrum()
        elif tok.type == TokenType.COMPUTE_SCREENING_GAP:
            return self._parse_compute_screening_gap()
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

    def _parse_target(self) -> TargetDecl:
        tok = self._expect(TokenType.TARGET, "Expected 'target'")
        name_tok = self._expect(TokenType.IDENT, "Expected target name")
        self._expect(TokenType.LBRACE, "Expected '{' starting target block")
        props = {}
        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            k_tok = self._advance()
            self._expect(TokenType.COLON, "Expected ':'")
            v_parts = []
            while self._peek().type != TokenType.SEMICOLON and self._peek().type != TokenType.EOF:
                v_parts.append(self._advance().value)
            self._expect(TokenType.SEMICOLON, "Expected ';' after target property")
            props[k_tok.value] = "".join(v_parts)
        self._expect(TokenType.RBRACE, "Expected '}' closing target block")
        return TargetDecl(line=tok.line, col=tok.col, name=name_tok.value, properties=props)

    def _parse_complex_block(self) -> ComplexBlockDecl:
        tok = self._expect(TokenType.COMPLEX, "Expected 'complex'")
        name_tok = self._expect(TokenType.IDENT, "Expected complex name")
        self._expect(TokenType.LBRACE, "Expected '{' starting complex block")
        dim = 4
        cell_decls: List[CellsGradeDecl] = []
        sheaf_decl: Optional[SheafDecl] = None
        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            p = self._peek()
            if p.type == TokenType.IDENT and p.value == "dimension":
                self._advance()
                self._expect(TokenType.COLON, "Expected ':' after dimension")
                d_tok = self._expect(TokenType.NUMBER, "Expected integer dimension")
                dim = int(d_tok.value)
                self._expect(TokenType.SEMICOLON, "Expected ';'")
            elif p.type == TokenType.CELLS:
                cell_decls.append(self._parse_cells_grade())
            elif p.type == TokenType.SHEAF:
                sheaf_decl = self._parse_sheaf()
            else:
                self._advance()
        self._expect(TokenType.RBRACE, "Expected '}' closing complex block")
        return ComplexBlockDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            dimension=dim,
            cell_decls=cell_decls,
            sheaf_decl=sheaf_decl,
        )

    def _parse_cells_grade(self) -> CellsGradeDecl:
        tok = self._expect(TokenType.CELLS, "Expected 'cells'")
        dim_tok = self._expect(TokenType.NUMBER, "Expected cell dimension grade")
        dim = int(dim_tok.value)
        self._expect(TokenType.COLON, "Expected ':'")
        raw_entries = []
        while self._peek().type != TokenType.SEMICOLON and self._peek().type != TokenType.EOF:
            raw_entries.append(self._advance().value)
        self._expect(TokenType.SEMICOLON, "Expected ';'")
        return CellsGradeDecl(line=tok.line, col=tok.col, dim=dim, raw_entries=raw_entries)

    def _parse_bracketed_cell(self) -> Statement:
        tok = self._expect(TokenType.CELL, "Expected 'cell'")
        self._expect(TokenType.LBRACKET, "Expected '['")
        dim_tok = self._expect(TokenType.NUMBER, "Expected cell dimension")
        dim = int(dim_tok.value)
        self._expect(TokenType.RBRACKET, "Expected ']'")

        name_tok = self._expect(TokenType.IDENT, "Expected cell name")
        names = [name_tok.value]
        while self._match(TokenType.COMMA):
            n_tok = self._expect(TokenType.IDENT, "Expected cell name after comma")
            names.append(n_tok.value)

        if self._match(TokenType.SEMICOLON):
            if dim == 0:
                if len(names) == 1:
                    return Cell0Decl(line=tok.line, col=tok.col, name=names[0])
                return CellsGradeDecl(line=tok.line, col=tok.col, dim=0, raw_entries=names)
            return CellsGradeDecl(line=tok.line, col=tok.col, dim=dim, raw_entries=names)

        if self._match(TokenType.EQUALS) or self._match(TokenType.COLON):
            if dim == 1:
                src_tok = self._expect(TokenType.IDENT, "Expected source vertex")
                self._expect(TokenType.ARROW, "Expected '->'")
                tgt_tok = self._expect(TokenType.IDENT, "Expected target vertex")
                self._expect(TokenType.SEMICOLON, "Expected ';'")
                return Cell1Decl(line=tok.line, col=tok.col, name=names[0], source=src_tok.value, target=tgt_tok.value)
            elif dim == 2:
                src_expr = self._parse_path_expr()
                if not (self._match(TokenType.DOUBLE_ARROW) or self._match(TokenType.ARROW)):
                    raise ParserError("Expected '=>' or '->' in 2-cell boundary", self._peek().line, self._peek().col)
                tgt_expr = self._parse_path_expr()
                self._expect(TokenType.SEMICOLON, "Expected ';'")
                return Cell2Decl(line=tok.line, col=tok.col, name=names[0], source=src_expr, target=tgt_expr)

        self._expect(TokenType.SEMICOLON, "Expected ';'")
        return CellsGradeDecl(line=tok.line, col=tok.col, dim=dim, raw_entries=names)

    def _parse_assert(self) -> AssertNilpotencyStmt:
        tok = self._expect(TokenType.ASSERT, "Expected 'assert'")
        self._expect(TokenType.NILPOTENCY, "Expected 'nilpotency'")
        self._expect(TokenType.LPAREN, "Expected '('")
        self._expect(TokenType.DIM, "Expected 'dim'")
        self._expect(TokenType.EQUALS, "Expected '='")
        dim_tok = self._expect(TokenType.NUMBER, "Expected dimension number")
        self._expect(TokenType.RPAREN, "Expected ')'")
        self._expect(TokenType.EQ, "Expected '=='")
        expected = True
        if self._peek().type == TokenType.TRUE:
            self._advance()
            expected = True
        elif self._peek().type == TokenType.FALSE:
            self._advance()
            expected = False
        elif self._peek().type == TokenType.IDENT:
            val = self._advance().value.lower()
            expected = (val == "true")
        self._expect(TokenType.SEMICOLON, "Expected ';'")
        return AssertNilpotencyStmt(line=tok.line, col=tok.col, dim=int(dim_tok.value), expected=expected)

    def _parse_compute_hodge_spectrum(self) -> ComputeHodgeSpectrumStmt:
        tok = self._expect(TokenType.COMPUTE_HODGE_SPECTRUM, "Expected 'compute_hodge_spectrum'")
        self._expect(TokenType.LPAREN, "Expected '('")
        c_tok = self._expect_ident("Expected complex name")
        self._expect(TokenType.COMMA, "Expected ','")
        self._expect(TokenType.DIM, "Expected 'dim'")
        self._expect(TokenType.EQUALS, "Expected '='")
        dim_tok = self._expect(TokenType.NUMBER, "Expected dimension number")
        field_type = "ExactRational"
        if self._match(TokenType.COMMA):
            self._expect(TokenType.FIELD, "Expected 'field'")
            self._expect(TokenType.EQUALS, "Expected '='")
            f_tok = self._expect_ident("Expected field name")
            field_type = f_tok.value
        self._expect(TokenType.RPAREN, "Expected ')'")
        self._expect(TokenType.SEMICOLON, "Expected ';'")
        return ComputeHodgeSpectrumStmt(
            line=tok.line,
            col=tok.col,
            complex_name=c_tok.value,
            dim=int(dim_tok.value),
            field_type=field_type,
        )

    def _parse_compute_screening_gap(self) -> ComputeScreeningGapStmt:
        tok = self._expect(TokenType.COMPUTE_SCREENING_GAP, "Expected 'compute_screening_gap'")
        self._expect(TokenType.LPAREN, "Expected '('")
        c_tok = self._expect_ident("Expected complex name")
        gauge_group = "SU(2)"
        coupling = 0.8
        while self._match(TokenType.COMMA):
            key_tok = self._expect_ident("Expected parameter name")
            self._expect(TokenType.EQUALS, "Expected '='")
            if key_tok.value == "gauge_group":
                val_tok = self._advance()
                gauge_group = val_tok.value
                if self._match(TokenType.LPAREN):
                    num = self._expect(TokenType.NUMBER, "Expected group N")
                    self._expect(TokenType.RPAREN, "Expected ')'")
                    gauge_group = f"{gauge_group}({num.value})"
            elif key_tok.value == "coupling":
                val_tok = self._advance()
                coupling = float(val_tok.value)
            else:
                self._advance()
        self._expect(TokenType.RPAREN, "Expected ')'")
        self._expect(TokenType.SEMICOLON, "Expected ';'")
        return ComputeScreeningGapStmt(
            line=tok.line,
            col=tok.col,
            complex_name=c_tok.value,
            gauge_group=gauge_group,
            coupling=coupling,
        )

    def _parse_procomplex(self) -> ProComplexDecl:
        tok = self._expect(TokenType.PROCOMPLEX, "Expected 'procomplex'")
        name_tok = self._expect(TokenType.IDENT, "Expected procomplex name")
        base_name = ""
        if self._match(TokenType.OVER):
            base_name = self._expect(TokenType.IDENT, "Expected base complex name").value
        max_scale = 3
        if self._match(TokenType.LBRACE):
            while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
                p = self._advance()
                if p.value == "max_scale" and self._match(TokenType.COLON):
                    max_scale = int(self._expect(TokenType.NUMBER, "Expected scale integer").value)
                    self._match(TokenType.SEMICOLON)
            self._expect(TokenType.RBRACE, "Expected '}'")
        elif self._match(TokenType.SEMICOLON):
            pass
        return ProComplexDecl(line=tok.line, col=tok.col, name=name_tok.value, base_complex_name=base_name, max_scale=max_scale)

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

        if self._match(TokenType.LBRACKET):
            # [e01, e13]
            first_ident = self._expect(TokenType.IDENT, "Expected identifier in path expression")
            elements.append(first_ident.value)
            while self._match(TokenType.COMMA) or self._match(TokenType.PLUS) or self._match(TokenType.STAR):
                next_ident = self._expect(TokenType.IDENT, "Expected identifier after separator")
                elements.append(next_ident.value)
            self._expect(TokenType.RBRACKET, "Expected ']' ending bracketed path expression")
        elif self._match(TokenType.LPAREN):
            # (f * g * h) or (L01 + L12)
            first_ident = self._expect(TokenType.IDENT, "Expected identifier in path expression")
            elements.append(first_ident.value)
            while self._match(TokenType.STAR) or self._match(TokenType.PLUS) or self._match(TokenType.COMMA):
                next_ident = self._expect(TokenType.IDENT, "Expected identifier after operator")
                elements.append(next_ident.value)
            self._expect(TokenType.RPAREN, "Expected ')' ending path expression")
        elif tok.type == TokenType.IDENT:
            # Single ident or unparenthesized chain f * g or L01 + L12
            first_ident = self._advance()
            elements.append(first_ident.value)
            while self._match(TokenType.STAR) or self._match(TokenType.PLUS) or self._match(TokenType.COMMA):
                next_ident = self._expect(TokenType.IDENT, "Expected identifier after operator")
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
        if not (self._match(TokenType.DOUBLE_ARROW) or self._match(TokenType.ARROW)):
            raise ParserError("Expected '=>' or '->' in rewrite rule", self._peek().line, self._peek().col)
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
        target = "complex"
        dim_val = 1
        first_tok = self._peek()
        if first_tok.type == TokenType.NUMBER:
            dim_val = int(self._advance().value)
        elif first_tok.type == TokenType.DIM:
            self._advance()
            self._expect(TokenType.EQUALS, "Expected '='")
            dim_val = int(self._expect(TokenType.NUMBER, "Expected integer dimension").value)
        elif first_tok.type in (TokenType.IDENT, TokenType.COMPLEX):
            target = self._advance().value
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
                target_name=target,
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
        field_val = field_tok.value
        if self._match(TokenType.LPAREN):
            args = []
            while self._peek().type != TokenType.RPAREN and self._peek().type != TokenType.EOF:
                arg_tok = self._advance()
                args.append(arg_tok.value)
            self._expect(TokenType.RPAREN, "Expected ')' closing stalk type parameters")
            field_val = f"{field_val}({''.join(args)})"

        vector_dim = 1
        if self._match(TokenType.CARET):
            vdim_tok = self._expect(TokenType.NUMBER, "Expected vector dimension after '^'")
            vector_dim = int(vdim_tok.value)
        self._expect(TokenType.SEMICOLON, "Expected ';' after stalk declaration")
        return StalkDecl(
            line=tok.line,
            col=tok.col,
            cell_dim=int(dim_tok.value),
            field_type=field_val,
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
        complex_name = ""
        if self._match(TokenType.OVER):
            complex_tok = self._expect(TokenType.IDENT, "Expected complex identifier after 'over'")
            complex_name = complex_tok.value

        algebra: Optional[str] = None
        if self._match(TokenType.COLON):
            alg_tok = self._expect(TokenType.IDENT, "Expected algebra identifier after ':'")
            alg_str = alg_tok.value
            if self._match(TokenType.LPAREN):
                args = []
                while self._peek().type != TokenType.RPAREN and self._peek().type != TokenType.EOF:
                    arg_tok = self._advance()
                    args.append(arg_tok.value)
                self._expect(TokenType.RPAREN, "Expected ')' closing algebra parameters")
                alg_str = f"{alg_str}({''.join(args)})"
            algebra = alg_str

        self._expect(TokenType.LBRACE, "Expected '{' starting sheaf body")

        stalks: List[StalkDecl] = []
        restrictions: List[RestrictionDecl] = []

        while self._peek().type != TokenType.RBRACE and self._peek().type != TokenType.EOF:
            c_tok = self._peek()
            if c_tok.type == TokenType.STALK:
                stk = self._parse_stalk()
                if algebra and stk.field_type in ("motor", "rotor", "multivector"):
                    stk.field_type = algebra
                stalks.append(stk)
            elif c_tok.type == TokenType.RESTRICTION:
                restrictions.append(self._parse_restriction())
            elif c_tok.type == TokenType.IDENT:
                prop_key = self._advance().value
                if self._match(TokenType.COLON):
                    val_parts = []
                    while self._peek().type != TokenType.SEMICOLON and self._peek().type != TokenType.EOF:
                        val_parts.append(self._advance().value)
                    self._match(TokenType.SEMICOLON)
                    if prop_key == "stalk_algebra":
                        algebra = "".join(val_parts)
            else:
                self._advance()

        self._expect(TokenType.RBRACE, "Expected '}' closing sheaf body")
        return SheafDecl(
            line=tok.line,
            col=tok.col,
            name=name_tok.value,
            complex_name=complex_name,
            algebra=algebra,
            stalks=stalks,
            restrictions=restrictions,
        )

    def _parse_apply(self) -> ApplyStmt:
        tok = self._expect(TokenType.APPLY, "Expected 'apply'")
        rule_tok = self._expect(TokenType.IDENT, "Expected rewrite rule name to apply")
        target_name = "complex"
        if self._match(TokenType.ON):
            target_tok = self._expect_ident("Expected target complex name")
            target_name = target_tok.value
        self._expect(TokenType.SEMICOLON, "Expected ';' after apply statement")
        return ApplyStmt(
            line=tok.line,
            col=tok.col,
            rule_name=rule_tok.value,
            target_name=target_name,
        )

    def _parse_process(self) -> ProcessDecl:
        tok = self._expect(TokenType.PROCESS, "Expected 'process'")
        name_tok = self._expect_ident("Expected process name")
        param_name = "complex"
        param_type = "CellComplex"
        if self._match(TokenType.LPAREN):
            param_tok = self._expect_ident("Expected parameter name")
            param_name = param_tok.value
            if self._match(TokenType.COLON):
                type_tok = self._expect_ident("Expected parameter type")
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
            param_name=param_name,
            param_type=param_type,
            body=body,
        )


def parse_source(source: str) -> Program:
    """Tokenizes and parses ToposLang source code into an AST."""
    tokens = Lexer(source).tokenize()
    return Parser(tokens).parse()


def build_complex(program: Program, name: str = "TauComplex") -> CellComplex:
    """Lowers AST cell, HIT, complex block, and 4D declarations into a validated CellComplex."""
    complex_obj = CellComplex(name=name)

    # Check for ComplexBlockDecl override
    complex_blocks = [s for s in program.statements if isinstance(s, ComplexBlockDecl)]
    if complex_blocks:
        complex_obj.name = complex_blocks[0].name

    # 1. First pass: Collect 0-cells
    for stmt in program.statements:
        if isinstance(stmt, Cell0Decl):
            complex_obj.add_cell(GlobularCell(stmt.name, dim=0))
        elif isinstance(stmt, HITDecl):
            for c in stmt.constructors:
                if isinstance(c, PointDecl):
                    complex_obj.add_cell(GlobularCell(c.name, dim=0))
        elif isinstance(stmt, CellsGradeDecl) and stmt.dim == 0:
            for entry in stmt.raw_entries:
                if entry not in (",", ";", ":") and entry:
                    complex_obj.add_cell(GlobularCell(entry, dim=0))
        elif isinstance(stmt, ComplexBlockDecl):
            for c_decl in stmt.cell_decls:
                if c_decl.dim == 0:
                    for entry in c_decl.raw_entries:
                        if entry not in (",", ";", ":") and entry:
                            complex_obj.add_cell(GlobularCell(entry, dim=0))

    # 2. Second pass: Collect 1-cells
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
        elif isinstance(stmt, ComplexBlockDecl):
            for c_decl in stmt.cell_decls:
                if c_decl.dim == 1:
                    # Entries format: L01 : V0 -> V1 , L12 : V1 -> V2 ...
                    # Group by comma or colon
                    tokens = [t for t in c_decl.raw_entries if t != ","]
                    idx = 0
                    while idx < len(tokens):
                        if idx + 4 < len(tokens) and tokens[idx + 1] == ":" and tokens[idx + 3] in ("->", "→"):
                            l_name = tokens[idx]
                            s_name = tokens[idx + 2]
                            t_name = tokens[idx + 4]
                            s_cell = complex_obj.get_cell(s_name, dim=0)
                            t_cell = complex_obj.get_cell(t_name, dim=0)
                            complex_obj.add_cell(GlobularCell(l_name, dim=1, source=s_cell, target=t_cell))
                            idx += 5
                        elif idx < len(tokens):
                            if tokens[idx] not in (":", "->", "→"):
                                complex_obj.add_cell(GlobularCell(tokens[idx], dim=1, source=Chain.zero(0), target=Chain.zero(0)))
                            idx += 1

    # 3. Third pass: Collect 2-cells
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
        elif isinstance(stmt, ComplexBlockDecl):
            for c_decl in stmt.cell_decls:
                if c_decl.dim == 2:
                    # e.g. P0123 : { L01 : +1 , L12 : +1 , L23 : -1 , L30 : -1 }
                    tokens = [t for t in c_decl.raw_entries if t not in (",", "{", "}")]
                    if len(tokens) >= 1:
                        p_name = tokens[0]
                        src_chain = Chain.zero(1)
                        tgt_chain = Chain.zero(1)
                        i = 1
                        while i < len(tokens):
                            if tokens[i] == ":":
                                i += 1
                                continue
                            edge_name = tokens[i]
                            if i + 2 < len(tokens) and tokens[i + 1] == ":":
                                sign_str = tokens[i + 2]
                                try:
                                    sign_val = int(sign_str)
                                    edge_cell = complex_obj.get_cell(edge_name, dim=1)
                                    if sign_val > 0:
                                        src_chain = src_chain + Chain.from_cell(edge_cell) * sign_val
                                    elif sign_val < 0:
                                        tgt_chain = tgt_chain + Chain.from_cell(edge_cell) * (-sign_val)
                                except Exception:
                                    pass
                                i += 3
                            else:
                                i += 1
                        complex_obj.add_cell(GlobularCell(p_name, dim=2, source=src_chain, target=tgt_chain))

    # 4. Fourth pass: Collect 3-cells and 4-cells
    for stmt in program.statements:
        if isinstance(stmt, ComplexBlockDecl):
            for c_decl in stmt.cell_decls:
                if c_decl.dim == 3:
                    for raw in c_decl.raw_entries:
                        if raw not in (",", ";", ":") and raw:
                            complex_obj.add_cell(GlobularCell(raw, dim=3, source=Chain.zero(2), target=Chain.zero(2)))
                elif c_decl.dim == 4:
                    for raw in c_decl.raw_entries:
                        if raw not in (",", ";", ":") and raw:
                            complex_obj.add_cell(GlobularCell(raw, dim=4, source=Chain.zero(3), target=Chain.zero(3)))

    # Validate nilpotency and boundary attachments
    complex_obj.validate(enforce_nilpotence=True)
    return complex_obj
