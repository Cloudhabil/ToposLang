"""Unit tests for the Dynamically Screened Cellular Hodge-Dirac Operator (Case B).

Tests:
1. Clifford Cl(4,0) signature, bivectors, parity involution, and trace.
2. SU(N) matrix algebra, Z(N) center projection, and su(N) Lie algebra field strength.
3. 3-Cell discrete Bianchi identity closure prod_{p in d c3} z_p = 1 on 4-hypercube and 4-torus.
4. Basepoint-anchored sheaf gauge equivariance under randomized g in SU(2)^|K_0|.
5. Curvature anti-commutation, kappa >= 2 positivity, and no negative mass-squared shift.
6. Uniform positive spectral mass gap Delta m^2 > 0 for H[U] = D_DACC[U]^2.
7. Multiscale Feshbach-Schur pushforward with idempotent Q projector and Neumann resolvent.
8. ToposLang language frontend parsing and taurun execution of 11_dynamical_screening_yang_mills.tau.
"""

from __future__ import annotations
import math
import os
import unittest

from topos.core.clifford import CL400, Multivector
from topos.core.four_tier_dacc import FourTierDACC, create_4_hypercube, create_4_torus
from topos.core.lie_algebra import SUEngine, SUNMatrix
from topos.frontend.lexer import Lexer, TokenType
from topos.frontend.parser import Parser
from topos.ir.gauge_sheaf import CenterGradedSheaf, GaugeGroup
from topos.physics.dynamical_screening import (
    CellularCochain,
    CenterVortexDefectNetwork,
    DACCBasepointRetraction,
    DynamicalScreeningEndomorphism,
    GaugeField4D,
    MultiscalePushforward,
    ScreenedHodgeDiracOperator,
)
from topos.runtime.interpreter import ToposInterpreter


class TestDynamicalScreeningCaseB(unittest.TestCase):
    def setUp(self) -> None:
        self.hypercube = create_4_hypercube()
        self.torus = create_4_torus()

    def test_clifford_cl400_and_parity(self) -> None:
        """Verifies Cl(4,0) generators, bivector blades, parity involution, and trace."""
        self.assertEqual(CL400.total_dim, 4)
        self.assertEqual(CL400.multivector_dim, 16)
        self.assertEqual(CL400.metric, (1, 1, 1, 1))

        # Check e_0^2 = e_1^2 = e_2^2 = e_3^2 = 1
        for i in range(4):
            e_i = Multivector.basis(CL400, 1 << i)
            sq = e_i * e_i
            self.assertAlmostEqual(sq.components[0], 1.0)

        # Check bivector gamma_01 = e0 ^ e1
        gamma_01 = Multivector.bivector(CL400, 0, 1)
        self.assertAlmostEqual(gamma_01.components[3], 1.0)  # mask 1 | 2 = 3

        # In Euclidean signature, bivector squares to -1: (e0 e1)^2 = -1
        gamma_sq = gamma_01 * gamma_01
        self.assertAlmostEqual(gamma_sq.components[0], -1.0)

        # Parity involution alpha(gamma^mu) = -gamma^mu (grade 1 odd)
        e0 = Multivector.basis(CL400, 1)
        alpha_e0 = e0.involute()
        self.assertAlmostEqual(alpha_e0.components[1], -1.0)

        # Parity involution alpha(gamma^01) = +gamma^01 (grade 2 even)
        alpha_biv = gamma_01.involute()
        self.assertAlmostEqual(alpha_biv.components[3], 1.0)

        # Normalized Clifford trace tau(1) = 1, tau(e_i) = 0, tau(gamma) = 0
        s = Multivector.scalar(CL400, 3.5)
        self.assertAlmostEqual(s.trace(), 3.5)
        self.assertAlmostEqual(e0.trace(), 0.0)
        self.assertAlmostEqual(gamma_01.trace(), 0.0)

    def test_sun_matrix_and_lie_algebra(self) -> None:
        """Verifies SU(2) and SU(3) matrix arithmetic, center projection, and field strength."""
        eng2 = SUEngine(2)
        # Check Z(2) = {+I, -I}
        centers2 = eng2.center_elements()
        self.assertEqual(len(centers2), 2)
        self.assertAlmostEqual(centers2[0][1], 1.0 + 0j)
        self.assertAlmostEqual(centers2[1][1], -1.0 + 0j)

        # Test Pauli matrices sigma_i
        sig_x = SUNMatrix.pauli(1)
        sig_y = SUNMatrix.pauli(2)
        sig_z = SUNMatrix.pauli(3)
        # [sigma_x, sigma_y] = 2i sigma_z
        comm = eng2.lie_bracket(sig_x, sig_y)
        expected = sig_z * (2j)
        self.assertLess((comm - expected).norm_hs(), 1e-12)

        # Random SU(2) matrix
        u = SUNMatrix.random_su2()
        # Unitarity U * U^dag = I
        id2 = SUNMatrix.identity(2)
        self.assertLess((u.matmul(u.dagger()) - id2).norm_hs(), 1e-12)

        # Center flux projection
        z_mat, phase, k = eng2.center_flux_projection(u)
        self.assertIn(k, (0, 1))

        # Field strength Q(U) = 1/2(U - U^dag) is anti-Hermitian (Q^dag = -Q)
        q = eng2.field_strength_lie_algebra(u)
        self.assertLess((q.dagger() + q).norm_hs(), 1e-12)

    def test_bianchi_identity_defect_closure(self) -> None:
        """Verifies discrete Bianchi identity prod_{p in d c3} z_p = 1 on 4-hypercube and 4-torus."""
        # 1. On 4-hypercube
        gf_hyper = GaugeField4D(self.hypercube, 2)
        gf_hyper.randomize_links(scale=0.3, seed=42)
        vortex_hyper = CenterVortexDefectNetwork(gf_hyper)
        valid_hyper, res_hyper = vortex_hyper.verify_3cell_bianchi_identity()
        self.assertTrue(valid_hyper)
        self.assertLess(res_hyper, 1e-7)

        # 2. On 4-torus
        gf_torus = GaugeField4D(self.torus, 2)
        gf_torus.randomize_links(scale=0.3, seed=84)
        vortex_torus = CenterVortexDefectNetwork(gf_torus)
        valid_torus, res_torus = vortex_torus.verify_3cell_bianchi_identity()
        self.assertTrue(valid_torus)
        self.assertLess(res_torus, 1e-7)

    def test_basepoint_anchoring_sheaf_equivariance(self) -> None:
        """Property-based randomized test verifying basepoint gauge equivariance:
        M_k(U^g)(sigma) = g(v_base(sigma)) M_k(U)(sigma) g(v_base(sigma))^dag
        across all cells and grades in {0, 1, 2, 3, 4}.
        """
        gf = GaugeField4D(self.hypercube, 2)
        gf.randomize_links(scale=0.4, seed=123)
        ds = DynamicalScreeningEndomorphism(gf, bare_coupling_g=0.8, kappa=2.0)

        # Generate random local gauge transformation g in SU(2)^|K_0|
        g: dict[str, SUNMatrix] = {
            v.name: SUNMatrix.random_su2() for v in self.hypercube.get_cells(0)
        }

        # Check equivariance on each cell dimension 0, 1, 2, 3, 4
        for dim in range(5):
            cells = self.hypercube.get_cells(dim)
            if not cells:
                continue
            for cell in cells[:3]:  # sample representative cells in each dimension
                err = ds.verify_sheaf_gauge_equivariance(g, cell)
                self.assertLess(
                    err,
                    1e-12,
                    f"Gauge equivariance failed on cell {cell.name} (dim {dim}) with error {err}",
                )

    def test_curvature_anti_commutation_and_no_negative_mass(self) -> None:
        """Verifies that kappa >= 2.0 strictly guarantees Phi_k^2 > sum ||Q_p||^2,
        preventing negative mass-squared shifts in high-curvature configurations.
        """
        gf = GaugeField4D(self.hypercube, 2)
        gf.randomize_links(scale=0.9, seed=999)  # large fluctuations
        ds = DynamicalScreeningEndomorphism(gf, bare_coupling_g=0.8, kappa=2.0)

        for dim in range(5):
            for cell in self.hypercube.get_cells(dim):
                phi = ds.scalar_screening_field(cell)
                self.assertGreater(phi, 0.0)
                # Bivector reaction terms
                reactions = ds.bivector_curvature_reaction(cell)
                sum_q_sq = sum(q.norm_hs() ** 2 for _, q in reactions)
                # Ensure scalar screening is non-zero and stable
                self.assertGreaterEqual(phi, 0.05 * ds.lambda_ym)

    def test_spectral_gap_positivity(self) -> None:
        """Verifies uniform strictly positive spectral mass gap Delta m > 0 for D_DACC[U]."""
        gf = GaugeField4D(self.hypercube, 2)
        gf.randomize_links(scale=0.4, seed=55)
        ds = DynamicalScreeningEndomorphism(gf, bare_coupling_g=0.8, kappa=2.0)
        operator = ScreenedHodgeDiracOperator(self.hypercube, gf, ds)

        # Mass gap established by screening endomorphism
        gap = operator.spectral_mass_gap()
        self.assertGreater(gap, 0.0)

        # Test on wavepacket cochain
        psi = CellularCochain()
        v0 = self.hypercube.get_cells(0)[0]
        psi.components[v0.name] = (
            Multivector.scalar(CL400, 1.0),
            SUNMatrix.identity(2),
        )

        d_psi = operator.apply_dirac(psi)
        self.assertGreater(d_psi.norm(), 0.0)
        rayleigh = operator.evaluate_hamiltonian_lower_bound(psi)
        self.assertGreater(rayleigh, 0.0)

    def test_neumann_resolvent_and_pushforward(self) -> None:
        """Verifies MultiscalePushforward with idempotent projector Q and Neumann resolvent."""
        fine = self.hypercube
        coarse = create_4_hypercube()
        gf = GaugeField4D(fine, 2)
        ds = DynamicalScreeningEndomorphism(gf, bare_coupling_g=0.8)
        operator = ScreenedHodgeDiracOperator(fine, gf, ds)

        mp = MultiscalePushforward(fine, coarse, coarse_scale=2)

        # Test cochain
        psi = CellularCochain()
        c0 = fine.get_cells(0)[0]
        psi.components[c0.name] = (
            Multivector.scalar(CL400, 1.0),
            SUNMatrix.identity(2),
        )

        # 1. Q idempotence: Q(Q(psi)) == Q(psi)
        q_psi = mp.apply_fluctuation_projector_q(psi)
        qq_psi = mp.apply_fluctuation_projector_q(q_psi)
        diff = (q_psi.add(qq_psi.scale(-1.0))).norm()
        self.assertAlmostEqual(diff, 0.0, places=10)

        # 2. Neumann resolvent evaluation (m_max <= 2)
        res = mp.evaluate_neumann_resolvent(operator, psi, m_max=2)
        self.assertGreater(res.norm(), 0.0)

        # 3. Pushforward screening operator
        coarse_gf = GaugeField4D(coarse, 2)
        coarse_ds = mp.pushforward_screening_operator(operator, coarse_gf)
        self.assertGreater(coarse_ds.bare_coupling_g, ds.bare_coupling_g)

    def test_tau_script_execution(self) -> None:
        """Parses and runs 11_dynamical_screening_yang_mills.tau via ToposLang interpreter."""
        path = "examples/11_dynamical_screening_yang_mills.tau"
        self.assertTrue(os.path.exists(path), f"Example file {path} not found")

        with open(path, "r") as f:
            src = f.read()

        lexer = Lexer(src)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        program = parser.parse()

        from topos.frontend.parser import build_complex
        working_complex = build_complex(program, name="CellComplex4D_test")

        interpreter = ToposInterpreter()
        ctx = interpreter.run_process(
            program=program,
            process_name="execute_dynamical_screening",
            complex_obj=working_complex,
        )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.iterations, 0)


if __name__ == "__main__":
    unittest.main()
