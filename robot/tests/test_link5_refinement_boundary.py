import unittest

import numpy as np

from robot.preprocessing.segmentation.sam3_dinov2_segment import _refine_link5_wrist_prompt


class _Predictor:
    def __init__(self, mask):
        self.mask = mask

    def handle_request(self, request):
        return {"outputs": {"out_obj_ids": [1],
                            "out_binary_masks": [self.mask]}}


class Link5RefinementBoundaryTests(unittest.TestCase):
    def test_point_mismatch_and_low_shaft_retention_are_recorded_without_rejection(self):
        initial = np.ones((10, 10), dtype=bool)
        points = np.array([[3, 5], [5, 5], [8, 5], [0, 2], [1, 6], [9, 7]])
        refined = initial.copy()
        for x, y in points[2:]:
            refined[y, x] = False
        mask, record = _refine_link5_wrist_prompt(
            _Predictor(refined), "test", 0, 1, initial, (0, 0, 10, 10),
            points_override=points, allow_distal_positive_dropout=True,
        )
        np.testing.assert_array_equal(mask, refined)
        self.assertTrue(record["distal_positive_dropped"])
        self.assertGreaterEqual(record["shaft_retained_fraction"], 0.85)

        normal_mask, normal_record = _refine_link5_wrist_prompt(
            _Predictor(refined), "test", 0, 1, initial,
            (0, 0, 10, 10), points_override=points,
        )
        np.testing.assert_array_equal(normal_mask, refined)
        self.assertFalse(normal_record["point_labels_accepted"])
        self.assertEqual(normal_record["mask_acceptance_policy"],
                         "no_point_membership_or_shaft_retention_gate")
        raw_mask, raw_record = _refine_link5_wrist_prompt(
            _Predictor(refined), "test", 0, 1, initial,
            (0, 0, 10, 10), points_override=points,
            diagnostic_allow_point_mismatch=True,
        )
        np.testing.assert_array_equal(raw_mask, refined)
        self.assertFalse(raw_record["point_labels_accepted"])
        self.assertTrue(raw_record["diagnostic_raw_mask"])

        initially_outside = initial.copy()
        initially_outside[5, 8] = False
        _, automatic = _refine_link5_wrist_prompt(
            _Predictor(refined), "test", 0, 1, initially_outside,
            (0, 0, 10, 10), points_override=points,
        )
        self.assertTrue(automatic["accepted_boundary_exception"])
        self.assertTrue(automatic["distal_positive_initially_outside"])

        middle_outside = initial.copy()
        middle_outside[5, 5] = False
        middle_refined = middle_outside.copy()
        for x, y in points[3:]:
            middle_refined[y, x] = False
        _, middle = _refine_link5_wrist_prompt(
            _Predictor(middle_refined), "test", 0, 1, middle_outside,
            (0, 0, 10, 10), points_override=points,
        )
        self.assertEqual(middle["preexisting_missing_positive_indices"], [1])
        self.assertTrue(middle["accepted_preexisting_positive_outside"])
        self.assertFalse(middle["accepted_boundary_exception"])

        _, mismatched = _refine_link5_wrist_prompt(
            _Predictor(middle_refined), "test", 0, 1, initial,
            (0, 0, 10, 10), points_override=points,
        )
        self.assertFalse(mismatched["point_labels_accepted"])

        sparse = np.zeros_like(initial)
        sparse[5, 3] = sparse[5, 5] = True
        sparse_mask, sparse_record = _refine_link5_wrist_prompt(
            _Predictor(sparse), "test", 0, 1, initial,
            (0, 0, 10, 10), points_override=points,
            allow_distal_positive_dropout=True,
        )
        np.testing.assert_array_equal(sparse_mask, sparse)
        self.assertLess(sparse_record["shaft_retained_fraction"], 0.85)


if __name__ == "__main__":
    unittest.main()
