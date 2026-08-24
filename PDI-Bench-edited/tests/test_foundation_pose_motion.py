import unittest
from unittest.mock import patch

import numpy as np

from pdi_eval.evaluator.motion_audit import (
    audit_foundation_pose_discontinuity,
    compose_metric_world_link_poses,
)
from pdi_eval.geometry.se3 import invert_rigid_transform, se3_exp, se3_log


def _constant_velocity_poses(frame_count: int, twist: np.ndarray) -> np.ndarray:
    return np.stack([se3_exp(twist * index) for index in range(frame_count)])


class Se3Tests(unittest.TestCase):
    def test_log_exp_round_trip(self):
        twist = np.asarray([0.2, -0.1, 0.3, 0.25, -0.4, 0.1])

        recovered = se3_log(se3_exp(twist))

        np.testing.assert_allclose(recovered, twist, atol=1e-12)


class FoundationPoseDiscontinuityTests(unittest.TestCase):
    def test_constant_fast_motion_has_zero_innovation(self):
        poses = _constant_velocity_poses(
            8, np.asarray([0.03, -0.01, 0.02, 0.0, 0.0, np.deg2rad(8.0)])
        )
        times = np.arange(len(poses), dtype=np.float64) / 30.0

        report = audit_foundation_pose_discontinuity(
            poses,
            times,
            np.ones(len(poses), dtype=bool),
            link_diameter=0.3,
        )

        self.assertFalse(report["pose_discontinuity"])
        self.assertLess(np.nanmax(report["severity"]), 1e-10)

    def test_unexpected_translation_crosses_exact_threshold(self):
        poses = _constant_velocity_poses(6, np.zeros(6))
        poses[3:, 0, 3] = 0.031
        times = np.arange(len(poses), dtype=np.float64) / 30.0

        report = audit_foundation_pose_discontinuity(
            poses,
            times,
            np.ones(len(poses), dtype=bool),
            link_diameter=0.3,
        )

        self.assertTrue(report["event"][3])
        self.assertEqual(report["classification"][3], "motion_discontinuity")
        self.assertGreater(report["severity"][3], 1.0)

    def test_small_translation_is_tolerated(self):
        poses = _constant_velocity_poses(6, np.zeros(6))
        poses[3:, 0, 3] = 0.01
        times = np.arange(len(poses), dtype=np.float64) / 30.0

        report = audit_foundation_pose_discontinuity(
            poses,
            times,
            np.ones(len(poses), dtype=bool),
            link_diameter=0.3,
        )

        self.assertFalse(report["event"][3])

    def test_camera_motion_is_removed_before_audit(self):
        scale = 2.0
        camera = _constant_velocity_poses(
            6, np.asarray([0.02, 0.0, 0.0, 0.0, np.deg2rad(2.0), 0.0])
        )
        metric_camera = camera.copy()
        metric_camera[:, :3, 3] *= scale
        static_link_world = np.repeat(np.eye(4)[None], len(camera), axis=0)
        link_in_camera = np.stack(
            [
                invert_rigid_transform(metric_camera[index])
                @ static_link_world[index]
                for index in range(len(camera))
            ]
        )

        composed = compose_metric_world_link_poses(camera, link_in_camera, scale)

        np.testing.assert_allclose(composed, static_link_world, atol=1e-12)

    def test_low_quality_jump_is_estimator_diagnostic(self):
        poses = _constant_velocity_poses(6, np.zeros(6))
        poses[3:, 0, 3] = 0.05
        times = np.arange(len(poses), dtype=np.float64) / 30.0
        quality = np.zeros(len(poses))
        quality[3] = 0.8

        report = audit_foundation_pose_discontinuity(
            poses,
            times,
            np.ones(len(poses), dtype=bool),
            link_diameter=0.3,
            pose_objective=quality,
        )

        self.assertEqual(report["classification"][3], "estimator_discontinuity")
        self.assertFalse(report["physical_event"][3])

    def test_composition_roundoff_is_projected_before_se3_log(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        poses[:, 0, 0] += 4e-7
        times = np.arange(len(poses), dtype=np.float64) / 30.0

        report = audit_foundation_pose_discontinuity(
            poses,
            times,
            np.ones(len(poses), dtype=bool),
            link_diameter=0.3,
        )

        self.assertEqual(report["valid_innovation_count"], 3)
        self.assertFalse(report["pose_discontinuity"])

    def test_predicted_pose_roundoff_is_projected_before_inversion(self):
        poses = np.repeat(np.eye(4)[None], 5, axis=0)
        times = np.arange(len(poses), dtype=np.float64) / 30.0
        derived_increment = np.eye(4)
        derived_increment[0, 0] += 8e-7

        with patch(
            "pdi_eval.evaluator.motion_audit.scale_rigid_increment",
            return_value=derived_increment,
        ):
            report = audit_foundation_pose_discontinuity(
                poses,
                times,
                np.ones(len(poses), dtype=bool),
                link_diameter=0.3,
            )

        self.assertEqual(report["valid_innovation_count"], 3)
        self.assertFalse(report["pose_discontinuity"])


if __name__ == "__main__":
    unittest.main()
