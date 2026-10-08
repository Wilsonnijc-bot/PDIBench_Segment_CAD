import unittest
from unittest.mock import patch
import numpy as np

from robot.experiments.link5_pair_selection_ablation.filter_video_depth import filter_queries
from robot.experiments.link5_pair_selection_ablation.score_depth_v2 import audit_3d_rigidity_cv, InsufficientRigidityEvidenceError


class DepthSupport(unittest.TestCase):
    def inputs(self):
        xy = np.array([[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]], float)
        tracks = np.tile(xy, (3, 1, 1))
        pm = np.zeros((3, 8, 8, 3))
        for t in range(3):
            for i, (x, y) in enumerate(xy.astype(int)):
                pm[t, y, x] = [i * i if t else i, 0, 1]
        return pm, tracks, np.ones((3, 6)), np.ones((3, 8, 8), bool)

    def test_depth_rejection_preserves_raw_visibility_and_carries(self):
        pm, xy, vis, masks = self.inputs()
        support = np.ones(vis.shape, bool); support[0, 5] = False; support[2, 1:] = False
        evidence = {}
        pairs = [dict(i=0, j=j) for j in range(1, 5)]
        with patch('robot.experiments.link5_pair_selection_ablation.score_depth_v2.select_pairs', return_value=(pairs, {})) as selector:
            score, history = audit_3d_rigidity_cv(pm, xy, vis, masks, depth_support=support, evidence=evidence, insufficient_policy='raise')
        self.assertEqual(len(selector.call_args.args[1]), 5)
        self.assertEqual(evidence['eligible_track_ids'], list(range(5)))
        self.assertGreater(history[1], 0)
        self.assertEqual(history[2], history[1])
        self.assertEqual(evidence['carried_frames'], [2])
        self.assertEqual(score, np.mean(history[1:]))
        np.testing.assert_array_equal(vis, np.ones((3, 6)))

    def test_frame0_failure_has_no_visible_only_fallback(self):
        pm, xy, vis, masks = self.inputs()
        support = np.ones(vis.shape, bool); support[0, :2] = False
        with self.assertRaises(InsufficientRigidityEvidenceError):
            audit_3d_rigidity_cv(pm, xy, vis, masks, depth_support=support, insufficient_policy='raise')

    def test_future_support_cannot_change_initial_graph(self):
        pm, xy, vis, masks = self.inputs()
        support = np.ones(vis.shape, bool)
        pairs = [dict(i=0, j=j) for j in range(1, 6)]
        records = []
        with patch('robot.experiments.link5_pair_selection_ablation.score_depth_v2.select_pairs', return_value=(pairs, {})):
            for later in (True, False):
                support[1:] = later; evidence = {}
                audit_3d_rigidity_cv(pm, xy, vis, masks, depth_support=support, evidence=evidence, insufficient_policy='raise')
                records.append(evidence['selected_pairs'])
        self.assertEqual(records[0], records[1])

    def test_outside_image_is_rejected_without_clamping(self):
        support = np.ones((8, 8), bool)
        xy = np.array([[-1, 2], [8, 2], [2, 2]], float)
        np.testing.assert_array_equal(filter_queries(support, xy), [False, False, True])


if __name__ == '__main__':
    unittest.main()
