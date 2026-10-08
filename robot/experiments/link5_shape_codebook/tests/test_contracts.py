"""Scientific coordinate/score regressions and source-adapter integration checks."""
import unittest

import numpy as np

from robot.experiments.link5_shape_codebook.common import normalize, observed_sample, raw_metrics, reference_align, transform_points
from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth


class CoordinateContracts(unittest.TestCase):
    def test_pose_equation_restores_arbitrary_reference_camera_pose(self):
        object_points=np.random.default_rng(4).normal(size=(256,3))*.05
        ref=np.eye(4);ref[:3,3]=[.1,.2,1.5]
        runtime=np.eye(4);runtime[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];runtime[:3,3]=[-.4,.1,1.8]
        camera_points=transform_points(object_points,runtime)
        np.testing.assert_allclose(reference_align(camera_points,ref,runtime),transform_points(object_points,ref),atol=1e-12)

    def test_partial_visibility_does_not_move_shared_points(self):
        points=np.random.default_rng(5).normal(size=(1000,3))
        fixed=dict(fixed_center=[.2,-.4,1],fixed_scale=.7)
        full=normalize(points,fixed);partial=normalize(points[200:400],fixed)
        np.testing.assert_array_equal(partial,full[200:400])

    def test_nonrigid_pose_is_rejected(self):
        wrong=np.eye(4);wrong[0,0]=1.1
        with self.assertRaises(ValueError):reference_align(np.zeros((200,3)),np.eye(4),wrong)

    def test_sparse_observations_are_not_repeated(self):
        points=np.arange(768,dtype=np.float32).reshape(256,3)
        sampled,ids=observed_sample(points,10000)
        self.assertEqual(len(sampled),256);self.assertEqual(len(set(ids)),256)
        np.testing.assert_array_equal(sampled,points[ids])

    def test_raw_top80_retains_cross_frame_amplitude(self):
        values=np.linspace(.01,.4,250)
        a=raw_metrics(values);b=raw_metrics(values*3)
        self.assertAlmostEqual(b['raw_mean_top80']/a['raw_mean_top80'],3)
        self.assertAlmostEqual(a['raw_mean_top80'],np.sort(values)[-80:].mean())

    def test_reused_filter_keeps_supported_step_and_rejects_isolated_spike(self):
        depth=np.ones((40,40),np.float32);depth[:,20:]=2;depth[10,10]=8
        mask=np.ones_like(depth,bool)
        k=np.array([[80,0,20],[0,80,20],[0,0,1]],np.float32)
        support=filter_depth(depth,mask,k,Config(erosion_pixels=0))
        valid,rejected,stats=(support[name] for name in ('valid','rejected','info'))
        self.assertTrue(rejected[10,10]);self.assertTrue(valid[20,19]);self.assertTrue(valid[20,20])
        self.assertEqual(stats['density_rejected_pixel_count'],1)
        self.assertFalse(stats['depth_smoothing'])


if __name__=='__main__':unittest.main()
