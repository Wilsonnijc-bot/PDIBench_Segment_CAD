import unittest
from unittest.mock import patch

import numpy as np

from infrastructure.shared.scoring.rigidity.rigidity import InsufficientRigidityEvidenceError, audit_3d_rigidity_cv


class RigidityEvidenceTests(unittest.TestCase):
    def test_legacy_policy_preserves_maximum_sentinel(self):
        pointmaps = np.zeros((3, 8, 8, 3), dtype=np.float64)
        tracks = np.zeros((3, 4, 2), dtype=np.float64)
        visibility = np.ones((3, 4), dtype=np.float64)
        masks = np.ones((3, 8, 8), dtype=bool)

        score, history = audit_3d_rigidity_cv(
            pointmaps, tracks, visibility, masks, insufficient_policy="maximum"
        )

        self.assertEqual(score, 1.0)
        np.testing.assert_array_equal(history, np.ones(3))

    def test_strict_policy_marks_insufficient_anchors_unavailable(self):
        pointmaps = np.zeros((3, 8, 8, 3), dtype=np.float64)
        tracks = np.zeros((3, 4, 2), dtype=np.float64)
        visibility = np.ones((3, 4), dtype=np.float64)
        masks = np.ones((3, 8, 8), dtype=bool)

        with self.assertRaises(InsufficientRigidityEvidenceError) as caught:
            audit_3d_rigidity_cv(
                pointmaps, tracks, visibility, masks, insufficient_policy="raise"
            )

        self.assertEqual(caught.exception.valid_anchor_count, 4)
        self.assertEqual(caught.exception.valid_pair_count, 0)

    def test_v2_does_not_recheck_depth_or_gradient_after_tracking(self):
        yy, xx = np.mgrid[:20, :20]
        frame = np.stack((xx / 20, yy / 20, np.ones_like(xx)), axis=-1)
        pointmaps = np.repeat(frame[None], 2, axis=0).astype(float)
        pointmaps[:, 10, 10, 2] = 4.0
        xy = np.array([[10, 10], [4, 4], [8, 4], [12, 4], [16, 4],
                       [4, 12], [8, 12], [12, 12], [16, 12]], dtype=float)
        tracks = np.repeat(xy[None], 2, axis=0)
        visibility = np.ones((2, len(xy)), dtype=float)
        masks = np.ones((2, 20, 20), dtype=bool)
        evidence = {}
        with patch("infrastructure.shared.scoring.rigidity.rigidity.cv2.Sobel", side_effect=AssertionError("V2 used Sobel")):
            score, _ = audit_3d_rigidity_cv(
                pointmaps, tracks, visibility, masks,
                point_filter_version="v2", insufficient_policy="raise",
                evidence=evidence,
            )
        self.assertEqual(score, 0.0)
        self.assertTrue(any(0 in (pair["track_i"], pair["track_j"])
                            for pair in evidence["selected_pairs"]))


if __name__ == "__main__":
    unittest.main()
