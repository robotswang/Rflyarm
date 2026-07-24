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
    LOOSE_ENDPOINT_MARGIN_DEG,
    RELEASE_STROKE_DEG,
    loose_endpoint_quaternion_xyzw,
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

    def test_unlocked_bulb_reengage_uses_direct_observed_angle(self):
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
        observed_angle = progress.observe_tightening_angle(
            math.radians(-121.5),
            allow_lock=False,
        )
        self.assertAlmostEqual(observed_angle, -121.5)
        self.assertAlmostEqual(progress.loosened_angle_deg, 121.5)
        self.assertAlmostEqual(progress.logical_angle_deg, 58.5)

    def test_reengage_requires_unlocked_state(self):
        progress = CeilingBulbProgress()
        with self.assertRaises(RuntimeError):
            progress.engage_loose(0.0)

    def test_complete_removal_forces_unlocked_endpoint(self):
        progress = CeilingBulbProgress()
        progress.complete_removal(0.0)
        self.assertEqual(progress.state, CeilingBulbState.UNLOCKED)
        self.assertAlmostEqual(progress.loosened_angle_deg, 180.0)

    def test_four_direct_angle_checks_lock_only_after_fourth_stroke(self):
        progress = CeilingBulbProgress()
        progress.state = CeilingBulbState.UNLOCKED
        progress.engage_loose(0.0)
        actual_angles_deg = (-130.0, -80.0, -25.0, -0.5)
        for index, actual_angle_deg in enumerate(actual_angles_deg):
            measured_angle_deg = progress.observe_tightening_angle(
                math.radians(actual_angle_deg),
                allow_lock=index == len(actual_angles_deg) - 1,
            )
            self.assertAlmostEqual(measured_angle_deg, actual_angle_deg)
            if index < len(actual_angles_deg) - 1:
                self.assertEqual(progress.state, CeilingBulbState.SCREWING)
        self.assertEqual(progress.state, CeilingBulbState.LOCKED)
        self.assertAlmostEqual(progress.logical_angle_deg, 180.0)

    def test_tightening_angle_check_rejects_invalid_state(self):
        progress = CeilingBulbProgress()
        with self.assertRaises(RuntimeError):
            progress.observe_tightening_angle(0.0, allow_lock=True)

    def test_first_three_strokes_cannot_lock_early(self):
        progress = CeilingBulbProgress()
        progress.state = CeilingBulbState.UNLOCKED
        progress.engage_loose(0.0)
        progress.observe_tightening_angle(math.radians(-0.5), allow_lock=False)
        self.assertEqual(progress.state, CeilingBulbState.SCREWING)

    def test_relative_twist_uses_socket_local_z(self):
        identity = (0.0, 0.0, 0.0, 1.0)
        yaw_90 = (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0))
        self.assertAlmostEqual(relative_twist_z(identity, yaw_90), math.pi / 2.0)
        self.assertAlmostEqual(relative_twist_z(yaw_90, yaw_90), 0.0)

    def test_loose_endpoint_stays_inside_negative_joint_limit(self):
        locked = (0.0, -math.sqrt(0.5), 0.0, math.sqrt(0.5))
        loose = loose_endpoint_quaternion_xyzw(locked)
        self.assertAlmostEqual(
            math.degrees(relative_twist_z(locked, loose)),
            -(RELEASE_STROKE_DEG - LOOSE_ENDPOINT_MARGIN_DEG),
        )
        self.assertAlmostEqual(sum(value * value for value in loose), 1.0)

    def test_clockwise_installation_moves_from_red_endpoint_to_green_endpoint(self):
        loose_angle_deg = -(RELEASE_STROKE_DEG - LOOSE_ENDPOINT_MARGIN_DEG)
        physical_angles_deg = [loose_angle_deg, -130.0, -80.0, -25.0, 0.0]
        self.assertAlmostEqual(physical_angles_deg[0], loose_angle_deg)
        for previous, current in zip(physical_angles_deg, physical_angles_deg[1:]):
            self.assertGreater(current, previous)
            self.assertLess(current - previous, 60.0)
        self.assertAlmostEqual(physical_angles_deg[4], 0.0)


if __name__ == "__main__":
    unittest.main()
