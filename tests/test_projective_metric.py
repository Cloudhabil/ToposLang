"""Unit tests for Projective Metric Engine, Non-Archimedean Dual-Lifting, and Meets."""

from fractions import Fraction
import unittest

from topos.core.projective_metric import (
    ProjectivePoint,
    ProjectiveMetricEngine,
)


class TestProjectiveMetric(unittest.TestCase):
    def setUp(self):
        self.engine = ProjectiveMetricEngine()

    def test_homogeneous_point_and_euclidean_projection(self):
        # Point (6, 8) with scale w=2 -> Euclidean (3, 4)
        p = ProjectivePoint(6, 8, z=0, w=2)
        ex, ey, ez = p.to_euclidean()
        self.assertEqual(ex, 3.0)
        self.assertEqual(ey, 4.0)
        self.assertEqual(ez, 0.0)

        # Ideal point at infinity
        p_inf = ProjectivePoint(1, 0, z=0, w=0)
        with self.assertRaises(ZeroDivisionError):
            p_inf.to_euclidean()

    def test_rational_quadratic_distance_query(self):
        # p1 = (0, 0), p2 = (3, 4) -> distance = 5
        p1 = ProjectivePoint(0, 0, z=0, w=1)
        p2 = ProjectivePoint(3, 4, z=0, w=1)

        d_sq = self.engine.squared_euclidean_distance_rational(p1, p2)
        self.assertEqual(d_sq, Fraction(25, 1))

        # Radius 6: d < 6 should be True
        self.assertTrue(self.engine.is_within_radius_rational(p1, p2, radius_num=6, radius_den=1))

        # Radius 5: d < 5 should be False (exact boundary, 5 is not < 5)
        self.assertFalse(self.engine.is_within_radius_rational(p1, p2, radius_num=5, radius_den=1))

        # Radius 4: d < 4 should be False
        self.assertFalse(self.engine.is_within_radius_rational(p1, p2, radius_num=4, radius_den=1))

        # Fractional radius test: radius 11/2 = 5.5 -> 5 < 5.5 is True
        self.assertTrue(self.engine.is_within_radius_rational(p1, p2, radius_num=11, radius_den=2))

    def test_non_trivial_homogeneous_scales(self):
        # p1 = (10, 20) with w=5 -> (2, 4)
        # p2 = (14, 21) with w=7 -> (2, 3)
        # Distance = 1
        p1 = ProjectivePoint(10, 20, z=0, w=5)
        p2 = ProjectivePoint(14, 21, z=0, w=7)

        d_sq = self.engine.squared_euclidean_distance_rational(p1, p2)
        self.assertEqual(d_sq, Fraction(1, 1))

        self.assertTrue(self.engine.is_within_radius_rational(p1, p2, radius_num=2, radius_den=1))
        self.assertFalse(self.engine.is_within_radius_rational(p1, p2, radius_num=1, radius_den=1))

    def test_2d_pga_line_intersections(self):
        # Proof 1 in paper:
        # L1: x = 1 => 1*e1 + 0*e2 - 1*e0 => (1, 0, -1)
        # L2: y = 2 => 0*e1 + 1*e2 - 2*e0 => (0, 1, -2)
        # Meet should be Euclidean point (1, 2)
        x, y, w, kind = ProjectiveMetricEngine.meet_lines_2d((1, 0, -1), (0, 1, -2))
        self.assertEqual(kind, "Point")
        # In PGA: inhomogeneous coordinates are (x/w, -y/w) or (x/w, y/w) depending on orientation
        self.assertEqual(w, 1)
        self.assertEqual(x, 1)
        self.assertEqual(y, 2)

        # Parallel lines: L2 (y = 2) and L3 (y = 1 => (0, 1, -1))
        # Parallel lines meet at ideal point at infinity (w == 0)
        x_par, y_par, w_par, kind_par = ProjectiveMetricEngine.meet_lines_2d((0, 1, -2), (0, 1, -1))
        self.assertEqual(kind_par, "Parallel_Ideal")
        self.assertEqual(w_par, 0)
        self.assertNotEqual(x_par, 0)

        # Coincident lines: L2 and L2
        x_coin, y_coin, w_coin, kind_coin = ProjectiveMetricEngine.meet_lines_2d((0, 1, -2), (0, 1, -2))
        self.assertEqual(kind_coin, "Coincident")
        self.assertEqual((x_coin, y_coin, w_coin), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
