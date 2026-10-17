"""Unit tests for Constructive Cubical Type Theory and Kan Composition Engine."""

import ctypes
import unittest
from topos.core.clifford import Multivector, PGA301
from topos.core.cubical import FaceMask, Interval, KanBox, KanComposition, KanStackFrame


class TestCubicalKanEngine(unittest.TestCase):
    """Verifies interval, face mask bit operations, KanStackFrame ABI, and hcomp."""

    def test_interval_and_face_mask(self) -> None:
        self.assertEqual(Interval.endpoint_0(), 0.0)
        self.assertEqual(Interval.endpoint_1(), 1.0)

        # 2-cube (square): 4 boundary faces: x0=0, x0=1, x1=0, x1=1
        fmask = FaceMask(dimension=2)
        fmask.add_face(axis=0, value=1)  # bit 2*0 + 1 = 1 -> mask 2
        fmask.add_face(axis=1, value=0)  # bit 2*1 + 0 = 2 -> mask 4
        self.assertTrue(fmask.has_face(axis=0, value=1))
        self.assertFalse(fmask.has_face(axis=0, value=0))
        self.assertTrue(fmask.has_face(axis=1, value=0))
        self.assertFalse(fmask.has_face(axis=1, value=1))
        self.assertEqual(fmask.mask, (1 << 1) | (1 << 2))

    def test_kan_stack_frame_layout(self) -> None:
        # Check ctypes layout and sizes
        frame = KanStackFrame()
        frame.face_mask_phi = 7
        frame.dimension = 2
        frame.base_multivector[0] = 1.0
        frame.base_multivector[1] = 0.5
        self.assertEqual(frame.face_mask_phi, 7)
        self.assertEqual(frame.dimension, 2)
        self.assertEqual(frame.base_multivector[0], 1.0)
        self.assertEqual(frame.base_multivector[1], 0.5)

    def test_kan_box_pack_and_evaluate(self) -> None:
        # Base multivector u0 at t=0
        u0 = Multivector.from_named_blades(PGA301, {"1": 1.0, "e1": 2.0})

        # Defunctionalized continuous tube closure: translation along e01 by t
        e01 = Multivector.from_named_blades(PGA301, {"e0e1": 1.0})
        def tube_fn(t: float) -> Multivector:
            motor = (e01 * (0.5 * t)).exp_motor()
            return u0.sandwich(motor)

        fmask = FaceMask(dimension=2, mask=3)
        kbox = KanBox(
            dimension=2,
            face_mask=fmask,
            base_u0=u0,
            tube=tube_fn,
            signature=PGA301,
        )

        # Convert to C-ABI stack frame
        frame = kbox.to_stack_frame()
        self.assertEqual(frame.face_mask_phi, 3)
        self.assertEqual(frame.dimension, 2)
        self.assertAlmostEqual(frame.base_multivector[0], 1.0)

        # Invoke tube evaluator directly via C callback
        out_buf = (ctypes.c_double * 16)()
        frame.tube_evaluator(None, ctypes.c_double(1.0), out_buf)
        self.assertAlmostEqual(out_buf[0], 1.0)

        # Execute hcomp
        u1 = KanComposition.hcomp(kbox, interval_t=1.0, max_homotopy_dim=2)
        self.assertAlmostEqual(u1.components[0], 1.0)

    def test_dynamic_n_truncation(self) -> None:
        # Multivector with grades 0, 1, 2, 3
        mv = Multivector.from_named_blades(
            PGA301,
            {"1": 1.0, "e1": 2.0, "e1e2": 3.0, "e1e2e3": 4.0},
        )
        # Truncate to dimension 1: grades > 1 become 0
        trunc1 = KanComposition.truncate(mv, n_dim=1)
        self.assertEqual(trunc1.get_blade("1"), 1.0)
        self.assertEqual(trunc1.get_blade("e1"), 2.0)
        self.assertEqual(trunc1.get_blade("e1e2"), 0.0)
        self.assertEqual(trunc1.get_blade("e1e2e3"), 0.0)


if __name__ == "__main__":
    unittest.main()
