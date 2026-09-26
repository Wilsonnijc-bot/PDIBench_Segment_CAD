import unittest

import numpy as np

from pdi_eval.v1.rigidity import (
    InsufficientRigidityEvidenceError,
    audit_3d_rigidity_cv,
)


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


if __name__ == "__main__":
    unittest.main()
