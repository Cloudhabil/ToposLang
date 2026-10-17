"""Unit tests for Multi-Prime CRT Engine, Galois Field F_{p^2}, and Garner Re-Lifting."""

import unittest
from topos.core.crt import (
    DEFAULT_CRT_PRIMES,
    GaloisField2,
    GF2Element,
    GarnersAlgorithm,
    CRTRing,
    CRTMatrixRREF,
    verify_roofline_machine_balance,
)


class TestCRTGalois(unittest.TestCase):
    def test_prime_pool_properties(self):
        self.assertEqual(len(DEFAULT_CRT_PRIMES), 4)
        for p in DEFAULT_CRT_PRIMES:
            # 64-bit bounds
            self.assertGreater(p, 1 << 62)
            self.assertLess(p, 1 << 63)
            # 3 mod 4 condition for quadratic extension irreducibility
            self.assertEqual(p % 4, 3)

    def test_galois_field_2_arithmetic(self):
        p = DEFAULT_CRT_PRIMES[0]
        gf = GaloisField2(p)
        a = gf.from_complex(3, 4)
        b = gf.from_complex(1, 2)

        # Addition: (3+4i) + (1+2i) = 4 + 6i
        c_add = a + b
        self.assertEqual(c_add.real, 4)
        self.assertEqual(c_add.imag, 6)

        # Multiplication: (3+4i)(1+2i) = (3 - 8) + (6 + 4)i = -5 + 10i
        c_mul = a * b
        self.assertEqual(c_mul.real, (p - 5) % p)
        self.assertEqual(c_mul.imag, 10)

        # Inversion: a * a^(-1) == 1
        a_inv = a.inv()
        prod = a * a_inv
        self.assertEqual(prod.real, 1)
        self.assertEqual(prod.imag, 0)

    def test_garners_algorithm_exact_lifting(self):
        garner = GarnersAlgorithm(DEFAULT_CRT_PRIMES)

        # Positive integer
        x_pos = 1234567890123456789
        residues_pos = [x_pos % p for p in DEFAULT_CRT_PRIMES]
        reconstructed_pos = garner.reconstruct_signed(residues_pos)
        self.assertEqual(reconstructed_pos, x_pos)

        # Negative integer
        x_neg = -987654321098765432
        residues_neg = [((x_neg % p) + p) % p for p in DEFAULT_CRT_PRIMES]
        reconstructed_neg = garner.reconstruct_signed(residues_neg)
        self.assertEqual(reconstructed_neg, x_neg)

    def test_crt_ring_sign_comparison(self):
        ring = CRTRing(DEFAULT_CRT_PRIMES)
        val_a = 50000
        val_b = 30000
        val_c = -20000

        res_a = ring.to_crt(val_a)
        res_b = ring.to_crt(val_b)
        res_c = ring.to_crt(val_c)

        # a > b
        self.assertEqual(ring.sign_compare(res_a, res_b), 1)
        # b < a
        self.assertEqual(ring.sign_compare(res_b, res_a), -1)
        # b == b
        self.assertEqual(ring.sign_compare(res_b, res_b), 0)
        # c < b
        self.assertEqual(ring.sign_compare(res_c, res_b), -1)
        # a > c
        self.assertEqual(ring.sign_compare(res_a, res_c), 1)

    def test_multi_prime_rref_and_rank_recovery(self):
        # Boundary matrix B_1 of triangle: rank over Q is 2
        # v0 -> v1, v1 -> v2, v0 -> v2
        B1 = [
            [-1, 0, -1],
            [1, -1, 0],
            [0, 1, 1],
        ]
        rref_engine = CRTMatrixRREF(DEFAULT_CRT_PRIMES)
        ranks, rat_rank = rref_engine.compute_ranks_across_lanes(B1)

        self.assertEqual(rat_rank, 2)
        for r in ranks:
            self.assertEqual(r, 2)

    def test_roofline_machine_balance_invariant(self):
        # L=4 prime lanes, d=6 average column density
        report = verify_roofline_machine_balance(prime_lanes_L=4, column_density_d=6.0)
        self.assertTrue(report.invariant_satisfied)
        self.assertTrue(report.is_compute_bound)
        self.assertAlmostEqual(report.arithmetic_intensity, 21.333333333333332)
        self.assertGreater(report.arithmetic_intensity, 16.5)

        # Low density violating roofline
        report_fail = verify_roofline_machine_balance(prime_lanes_L=2, column_density_d=2.0)
        self.assertFalse(report_fail.invariant_satisfied)
        self.assertFalse(report_fail.is_compute_bound)


if __name__ == "__main__":
    unittest.main()
