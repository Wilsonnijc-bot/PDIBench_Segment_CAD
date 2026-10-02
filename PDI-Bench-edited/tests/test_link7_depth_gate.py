import unittest

import numpy as np

from pdi_eval.v1.link7_depth_gate import (
    link7_depth_keep, link7_depth_region, link7_query_mask,
)


class Link7DepthGateTests(unittest.TestCase):
    def test_smooth_far_sheet_inside_mask_is_removed(self):
        pointmap = np.zeros((40, 40, 3), dtype=np.float32)
        pointmap[..., 2] = 1.0
        pointmap[:, 28:, 2] = 4.0
        mask = np.ones((40, 40), dtype=bool)
        xy = np.array([[10, 10], [20, 20], [34, 10], [35, 20]], dtype=np.float32)
        keep, stats = link7_depth_keep(pointmap, np.eye(4), mask, xy)
        self.assertEqual(keep.tolist(), [True, True, False, False])
        self.assertEqual(stats["retained_count"], 2)
        self.assertLess(stats["far_limit"], 4.0)

    def test_invalid_lift_is_removed(self):
        pointmap = np.zeros((40, 40, 3), dtype=np.float32)
        pointmap[..., 2] = 1.0
        pointmap[10, 10] = 0
        keep, _ = link7_depth_keep(pointmap, np.eye(4),
                                   np.ones((40, 40), dtype=bool),
                                   np.array([[10, 10], [20, 20]]))
        self.assertEqual(keep.tolist(), [False, True])

    def test_dense_region_is_filtered_before_query_sampling(self):
        pointmap = np.zeros((40, 40, 3), dtype=np.float32)
        pointmap[..., 2] = 1.0
        pointmap[:, 28:, 2] = 4.0
        region, stats = link7_depth_region(
            pointmap, np.eye(4), np.ones((40, 40), dtype=bool)
        )
        self.assertTrue(region[:, :28].all())
        self.assertFalse(region[:, 28:].any())
        self.assertEqual(stats["stage"], "before_query_sampling")
        self.assertEqual(stats["candidate_count"], 1600)
        self.assertEqual(stats["retained_count"], 1120)

    def test_tracker_mask_excludes_bilinear_far_depth_bleed(self):
        pointmap = np.zeros((40, 40, 3), dtype=np.float32)
        pointmap[..., 2] = 1.0
        pointmap[:, 28:, 2] = 4.0
        eligible, stats = link7_query_mask(
            pointmap, np.eye(4), np.ones((80, 80), dtype=bool), (80, 80)
        )
        self.assertTrue(eligible[20, 20])
        self.assertFalse(eligible[20, 55])
        self.assertGreater(stats["bilinear_lift_rejected_pixel_count"], 0)


if __name__ == "__main__":
    unittest.main()
