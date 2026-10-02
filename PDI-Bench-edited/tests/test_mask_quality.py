import unittest

import numpy as np

from pdi_eval.perception.mask_quality import link7_area_validity


class MaskQualityTests(unittest.TestCase):
    def test_area_cutoff_inclusive_and_resolution_independent(self):
        masks = np.zeros((3, 20, 20), bool)
        masks[0, :4] = True
        masks[1, :5] = True
        masks[2, :6] = True
        valid, areas, fractions = link7_area_validity(masks)
        self.assertEqual(valid.tolist(), [True, False, False])
        self.assertEqual(areas.tolist(), [80, 100, 120])
        np.testing.assert_allclose(fractions, [.20, .25, .30])
        scaled = np.repeat(np.repeat(masks, 2, axis=1), 2, axis=2)
        np.testing.assert_array_equal(link7_area_validity(scaled)[0], valid)

    def test_counts_pixels_not_bounding_box_area(self):
        masks = np.zeros((1, 20, 20), bool)
        masks[0, 0, 0] = masks[0, -1, -1] = True
        self.assertTrue(link7_area_validity(masks)[0][0])

    def test_configurable_cutoff_and_invalid_inputs(self):
        masks = np.ones((1, 4, 4), bool)
        self.assertFalse(link7_area_validity(masks, .5)[0][0])
        for fraction in [0, -1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                link7_area_validity(masks, fraction)
        with self.assertRaises(ValueError):
            link7_area_validity(masks[0])


if __name__ == '__main__':
    unittest.main()
