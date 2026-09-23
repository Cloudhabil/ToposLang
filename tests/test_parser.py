"""Unit tests for ToposLang Lexer, Parser, AST, and lowering compiler."""

import unittest
from pathlib import Path

from topos.frontend.lexer import Lexer, TokenType, LexerError
from topos.frontend.parser import Parser, ParserError, parse_source, build_complex
from topos.frontend.ast import (
    Cell0Decl,
    Cell1Decl,
    Cell2Decl,
    HITDecl,
    PointDecl,
    PathDecl,
    SurfaceDecl,
    ProcessDecl,
    UntilLoop,
)
from topos.topology.homology import HomologyEngine


class TestLexer(unittest.TestCase):
    def test_tokenize_cells(self) -> None:
        code = "cell0 A;\ncell1 f : A -> B;\ncell2 alpha : (f * g) => h;"
        tokens = Lexer(code).tokenize()
        types = [t.type for t in tokens]
        expected = [
            TokenType.CELL0, TokenType.IDENT, TokenType.SEMICOLON,
            TokenType.CELL1, TokenType.IDENT, TokenType.COLON, TokenType.IDENT, TokenType.ARROW, TokenType.IDENT, TokenType.SEMICOLON,
            TokenType.CELL2, TokenType.IDENT, TokenType.COLON, TokenType.LPAREN, TokenType.IDENT, TokenType.STAR, TokenType.IDENT, TokenType.RPAREN, TokenType.DOUBLE_ARROW, TokenType.IDENT, TokenType.SEMICOLON,
            TokenType.EOF,
        ]
        self.assertEqual(types, expected)

    def test_skip_comments(self) -> None:
        code = "// This is a comment\ncell0 X; // trailing comment\n"
        tokens = Lexer(code).tokenize()
        self.assertEqual(len(tokens), 4)  # CELL0, IDENT, SEMICOLON, EOF
        self.assertEqual(tokens[1].value, "X")

    def test_lexer_error(self) -> None:
        with self.assertRaises(LexerError):
            Lexer("@invalid").tokenize()


class TestParser(unittest.TestCase):
    def test_parse_cells(self) -> None:
        code = """
        cell0 A;
        cell0 B;
        cell1 f : A -> B;
        """
        program = parse_source(code)
        self.assertEqual(len(program.statements), 3)
        self.assertIsInstance(program.statements[0], Cell0Decl)
        self.assertEqual(program.statements[0].name, "A")
        self.assertIsInstance(program.statements[2], Cell1Decl)
        self.assertEqual(program.statements[2].name, "f")
        self.assertEqual(program.statements[2].source, "A")
        self.assertEqual(program.statements[2].target, "B")

    def test_parse_hit(self) -> None:
        code = """
        hit S1 {
            point base;
            path loop : base = base;
        }
        """
        program = parse_source(code)
        self.assertEqual(len(program.statements), 1)
        hit_stmt = program.statements[0]
        self.assertIsInstance(hit_stmt, HITDecl)
        self.assertEqual(hit_stmt.name, "S1")
        self.assertEqual(len(hit_stmt.constructors), 2)
        self.assertIsInstance(hit_stmt.constructors[0], PointDecl)
        self.assertIsInstance(hit_stmt.constructors[1], PathDecl)

    def test_parse_process_and_until(self) -> None:
        code = """
        process test_proc(complex: CellComplex) {
            rewrite r : (e1 * e2) => e3;
            until betti(complex, dim=1) == 0 {
                apply r on complex;
            }
        }
        """
        program = parse_source(code)
        self.assertEqual(len(program.statements), 1)
        proc = program.statements[0]
        self.assertIsInstance(proc, ProcessDecl)
        self.assertEqual(proc.name, "test_proc")
        self.assertEqual(len(proc.body), 2)
        self.assertIsInstance(proc.body[1], UntilLoop)
        self.assertEqual(proc.body[1].condition.dim, 1)
        self.assertEqual(proc.body[1].condition.value, 0)

    def test_parser_syntax_error(self) -> None:
        # Missing semicolon
        code = "cell0 A"
        with self.assertRaises(ParserError) as ctx:
            parse_source(code)
        self.assertIn("Expected ';'", str(ctx.exception))


class TestCompilerLowering(unittest.TestCase):
    def test_build_circle_from_example_file(self) -> None:
        path = Path(__file__).parent.parent / "examples" / "01_circle_hit.tau"
        source = path.read_text()
        program = parse_source(source)
        complex_obj = build_complex(program, name="ParsedS1")

        engine = HomologyEngine(complex_obj)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 1)

    def test_build_torus_from_example_file(self) -> None:
        path = Path(__file__).parent.parent / "examples" / "02_torus_hit.tau"
        source = path.read_text()
        program = parse_source(source)
        complex_obj = build_complex(program, name="ParsedTorus")

        engine = HomologyEngine(complex_obj)
        self.assertEqual(engine.betti_number(0), 1)
        self.assertEqual(engine.betti_number(1), 2)
        self.assertEqual(engine.betti_number(2), 1)

    def test_build_homotopy_diagram(self) -> None:
        path = Path(__file__).parent.parent / "examples" / "03_loop_collapse.tau"
        source = path.read_text()
        program = parse_source(source)
        complex_obj = build_complex(program, name="LoopDiagram")

        # Initial complex has 3 0-cells, 3 1-cells, 0 2-cells (open 1-cycle)
        self.assertEqual(len(complex_obj.get_cells(0)), 3)
        self.assertEqual(len(complex_obj.get_cells(1)), 3)
        self.assertEqual(len(complex_obj.get_cells(2)), 0)
        self.assertTrue(complex_obj.validate(enforce_nilpotence=True))


if __name__ == "__main__":
    unittest.main()
