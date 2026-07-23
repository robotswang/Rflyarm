#!/usr/bin/env python3
"""Unit checks for the ceiling-only virtual screw progress."""

import math
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.ceiling_bulb import (
    CeilingBulbProgress,
    CeilingBulbState,
    relative_twist_z,
)


def wrapped(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class CeilingBulbProgressTest(unittest.TestCase):
    def test_full_negative_turn_unlocks_across_angle_wrap(self):
        progress = CeilingBulbProgress()
        progress.reset(0.0)
        for angle_deg in (-90.0, -180.0, -270.0, -355.0):
            progress.update(wrapped(math.radians(angle_deg)))
        self.assertEqual(progress.state, CeilingBulbState.UNLOCKED)
        self.assertLessEqual(progress.remaining_angle_deg, 5.0)
        self.assertLessEqual(progress.logical_angle_deg, 5.0)
        self.assertGreaterEqual(progress.logical_angle_deg, 0.0)

    def test_tightening_cannot_pass_fully_locked_state(self):
        progress = CeilingBulbProgress()
        progress.reset(0.0)
        progress.update(math.radians(10.0))
        self.assertEqual(progress.state, CeilingBulbState.LOCKED)
        self.assertAlmostEqual(progress.loosened_angle_deg, 0.0)
        self.assertTrue(progress.rejected_tightening)

    def test_reverse_motion_relocks_a_partially_unscrewed_bulb(self):
        progress = CeilingBulbProgress()
        progress.reset(0.0)
        progress.update(math.radians(-40.0))
        self.assertEqual(progress.state, CeilingBulbState.UNSCREWING)
        progress.update(0.0)
        self.assertEqual(progress.state, CeilingBulbState.LOCKED)

    def test_unlocked_bulb_reengage_ignores_duplicate_physical_progress(self):
        progress = CeilingBulbProgress()
        progress.reset(0.0)
        for angle_deg in (-90.0, -180.0, -270.0, -355.0):
            progress.update(wrapped(math.radians(angle_deg)))
        self.assertEqual(progress.state, CeilingBulbState.UNLOCKED)

        loose_angle = wrapped(math.radians(-355.0))
        progress.engage_loose(loose_angle)
        self.assertEqual(progress.state, CeilingBulbState.SCREWING)
        progress.update(wrapped(loose_angle + math.radians(60.0)))
        self.assertEqual(progress.state, CeilingBulbState.SCREWING)
        self.assertAlmostEqual(progress.loosened_angle_deg, 180.0)
        for _ in range(3):
            progress.tighten_step(60.0)
        self.assertEqual(progress.state, CeilingBulbState.LOCKED)
        self.assertLessEqual(progress.loosened_angle_deg, 1.0)

    def test_reengage_requires_unlocked_state(self):
        progress = CeilingBulbProgress()
        with self.assertRaises(RuntimeError):
            progress.engage_loose(0.0)

    def test_complete_removal_forces_unlocked_endpoint(self):
        progress = CeilingBulbProgress()
        progress.complete_removal(0.0)
        self.assertEqual(progress.state, CeilingBulbState.UNLOCKED)
        self.assertAlmostEqual(progress.loosened_angle_deg, 180.0)

    def test_deterministic_tightening_steps_lock_after_full_stroke(self):
        progress = CeilingBulbProgress()
        progress.state = CeilingBulbState.UNLOCKED
        progress.engage_loose(0.0)
        progress.tighten_step(60.0)
        self.assertEqual(progress.state, CeilingBulbState.SCREWING)
        progress.tighten_step(60.0)
        self.assertEqual(progress.state, CeilingBulbState.SCREWING)
        progress.tighten_step(60.0)
        self.assertEqual(progress.state, CeilingBulbState.LOCKED)
        self.assertAlmostEqual(progress.logical_angle_deg, 180.0)

    def test_tightening_step_rejects_invalid_state_and_amount(self):
        progress = CeilingBulbProgress()
        with self.assertRaises(RuntimeError):
            progress.tighten_step(60.0)
        progress.state = CeilingBulbState.UNLOCKED
        progress.engage_loose(0.0)
        with self.assertRaises(ValueError):
            progress.tighten_step(0.0)

    def test_relative_twist_uses_socket_local_z(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        yaw_90 = (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
        self.assertAlmostEqual(relative_twist_z(identity, yaw_90), math.pi / 2.0)
        self.assertAlmostEqual(relative_twist_z(yaw_90, yaw_90), 0.0)


if __name__ == "__main__":
    unittest.main()
