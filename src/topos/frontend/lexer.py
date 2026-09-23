"""Lexer / Tokenizer for ToposLang (.tau)."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TokenType(Enum):
    # Keywords
    CELL0 = auto()
    CELL1 = auto()
    CELL2 = auto()
    HIT = auto()
    POINT = auto()
    PATH = auto()
    SURFACE = auto()
    PROCESS = auto()
    REWRITE = auto()
    UNTIL = auto()
    APPLY = auto()
    ON = auto()
    BETTI = auto()
    DIM = auto()
    SHEAF = auto()
    OVER = auto()
    STALK = auto()
    RESTRICTION = auto()
    SHEAF_DIM_KER = auto()
    SHEAF_BETTI = auto()

    # Literals
    IDENT = auto()
    NUMBER = auto()

    # Operators & Punctuation
    COLON = auto()          # :
    SEMICOLON = auto()      # ;
    COMMA = auto()          # ,
    DOT = auto()            # .
    ARROW = auto()          # ->
    DOUBLE_ARROW = auto()   # =>
    EQUALS = auto()         # =
    EQ = auto()             # ==
    NEQ = auto()            # !=
    LT = auto()             # <
    GT = auto()             # >
    LTE = auto()            # <=
    GTE = auto()            # >=
    STAR = auto()           # *
    CARET = auto()          # ^
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]

    EOF = auto()


KEYWORDS = {
    "cell0": TokenType.CELL0,
    "cell1": TokenType.CELL1,
    "cell2": TokenType.CELL2,
    "hit": TokenType.HIT,
    "point": TokenType.POINT,
    "path": TokenType.PATH,
    "surface": TokenType.SURFACE,
    "process": TokenType.PROCESS,
    "rewrite": TokenType.REWRITE,
    "until": TokenType.UNTIL,
    "apply": TokenType.APPLY,
    "on": TokenType.ON,
    "betti": TokenType.BETTI,
    "dim": TokenType.DIM,
    "sheaf": TokenType.SHEAF,
    "over": TokenType.OVER,
    "stalk": TokenType.STALK,
    "restriction": TokenType.RESTRICTION,
    "sheaf_dim_ker": TokenType.SHEAF_DIM_KER,
    "sheaf_betti": TokenType.SHEAF_BETTI,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:C{self.col})"


class LexerError(Exception):
    def __init__(self, message: str, line: int, col: int) -> None:
        super().__init__(f"LexerError at {line}:{col}: {message}")
        self.line = line
        self.col = col


class Lexer:
    """Scans ToposLang source code into tokens."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.col = 1

    def _peek(self, offset: int = 0) -> str:
        idx = self.pos + offset
        if idx < self.length:
            return self.source[idx]
        return ""

    def _advance(self) -> str:
        ch = self._peek()
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def tokenize(self) -> List[Token]:
        tokens: List[Token] = []

        while self.pos < self.length:
            ch = self._peek()

            # Skip whitespace
            if ch in (" ", "\t", "\r", "\n"):
                self._advance()
                continue

            # Skip single-line comments // ...
            if ch == "/" and self._peek(1) == "/":
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
                continue

            start_line = self.line
            start_col = self.col

            # Two-character operators
            if ch == "-" and self._peek(1) == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.ARROW, "->", start_line, start_col))
                continue

            if ch == "=" and self._peek(1) == ">":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.DOUBLE_ARROW, "=>", start_line, start_col))
                continue

            if ch == "=" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.EQ, "==", start_line, start_col))
                continue

            if ch == "!" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.NEQ, "!=", start_line, start_col))
                continue

            if ch == "<" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.LTE, "<=", start_line, start_col))
                continue

            if ch == ">" and self._peek(1) == "=":
                self._advance()
                self._advance()
                tokens.append(Token(TokenType.GTE, ">=", start_line, start_col))
                continue

            # Single-character tokens
            if ch == ":":
                self._advance()
                tokens.append(Token(TokenType.COLON, ":", start_line, start_col))
                continue
            if ch == ";":
                self._advance()
                tokens.append(Token(TokenType.SEMICOLON, ";", start_line, start_col))
                continue
            if ch == ",":
                self._advance()
                tokens.append(Token(TokenType.COMMA, ",", start_line, start_col))
                continue
            if ch == "=":
                self._advance()
                tokens.append(Token(TokenType.EQUALS, "=", start_line, start_col))
                continue
            if ch == "*":
                self._advance()
                tokens.append(Token(TokenType.STAR, "*", start_line, start_col))
                continue
            if ch == "<":
                self._advance()
                tokens.append(Token(TokenType.LT, "<", start_line, start_col))
                continue
            if ch == ">":
                self._advance()
                tokens.append(Token(TokenType.GT, ">", start_line, start_col))
                continue
            if ch == "(":
                self._advance()
                tokens.append(Token(TokenType.LPAREN, "(", start_line, start_col))
                continue
            if ch == ")":
                self._advance()
                tokens.append(Token(TokenType.RPAREN, ")", start_line, start_col))
                continue
            if ch == "{":
                self._advance()
                tokens.append(Token(TokenType.LBRACE, "{", start_line, start_col))
                continue
            if ch == "}":
                self._advance()
                tokens.append(Token(TokenType.RBRACE, "}", start_line, start_col))
                continue
            if ch == "[":
                self._advance()
                tokens.append(Token(TokenType.LBRACKET, "[", start_line, start_col))
                continue
            if ch == "]":
                self._advance()
                tokens.append(Token(TokenType.RBRACKET, "]", start_line, start_col))
                continue
            if ch == "^":
                self._advance()
                tokens.append(Token(TokenType.CARET, "^", start_line, start_col))
                continue
            if ch == ".":
                self._advance()
                tokens.append(Token(TokenType.DOT, ".", start_line, start_col))
                continue

            # Numbers (integers, floats, and negative numbers if preceded by minus without arrow)
            is_negative_num = (ch == "-" and self._peek(1).isdigit())
            if ch.isdigit() or is_negative_num:
                num_str = self._advance() if is_negative_num else ""
                while self._peek().isdigit():
                    num_str += self._advance()
                if self._peek() == "." and self._peek(1).isdigit():
                    num_str += self._advance()  # '.'
                    while self._peek().isdigit():
                        num_str += self._advance()
                tokens.append(Token(TokenType.NUMBER, num_str, start_line, start_col))
                continue

            # Identifiers or Keywords
            if ch.isalpha() or ch == "_":
                ident_str = ""
                while self._peek().isalnum() or self._peek() == "_":
                    ident_str += self._advance()
                token_type = KEYWORDS.get(ident_str, TokenType.IDENT)
                tokens.append(Token(token_type, ident_str, start_line, start_col))
                continue

            # Unrecognized character
            raise LexerError(f"Unexpected character: {ch!r}", start_line, start_col)

        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens
