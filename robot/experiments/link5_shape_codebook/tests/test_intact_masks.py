"""Raw masks stay intact while detector support permits two source pixels erosion."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot.experiments.link5_shape_codebook.prepare import save_observation


class IntactMaskPolicy(unittest.TestCase):
    def test_depth_filter_does_not_erode_the_guarded_mask(self):
        mask=np.ones((32,32),bool)
        depth=np.ones((32,32),np.float32)
        rgb=np.zeros((32,32,3),np.uint8)
        k=np.array([[50,0,16],[0,50,16],[0,0,1]],np.float32)
        with tempfile.TemporaryDirectory() as directory:
            row=save_observation(dict(id='fixture'),Path(directory),0,rgb,depth,k,mask,np.eye(4))
            with np.load(row['input_path']) as archive:
                np.testing.assert_array_equal(archive['source_mask'],mask)
                np.testing.assert_array_equal(archive['mask'],mask)
                self.assertFalse(archive['valid'][0].any())
                self.assertTrue(archive['valid'][2:-2,2:-2].all())
        self.assertEqual(row['mask_erosion_pixels'],2)
        self.assertFalse(row['mask_mapping_applied'])
        self.assertTrue(row['image_boundary_contact'])
        self.assertEqual(row['valid_point_count'],784)


if __name__=='__main__':unittest.main()
