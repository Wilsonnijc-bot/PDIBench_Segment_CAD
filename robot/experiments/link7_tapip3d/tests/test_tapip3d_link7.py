"""Contracts at the existing Link 7 query to TAPIP3D boundary."""

import unittest
from types import SimpleNamespace

import numpy as np

from infrastructure.shared.contracts.base import SharedGeometryResult
from robot.experiments.link7_tapip3d.tapip3d_link7 import _input_geometry


class Tapip3DLink7InputTests(unittest.TestCase):
    def test_selected_ids_and_subpixel_locations_survive_world_lift(self):
        height, width = 5, 6
        focal_x, focal_y, cx, cy = 4.0, 4.2, 3.3, 2.1
        yy, xx = np.indices((height, width))
        depth = np.full((height, width), 2.0, dtype=np.float32)
        pointmap = np.stack(((xx - cx) * depth / focal_x,
                             (yy - cy) * depth / focal_y, depth), axis=-1)
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
        geometry = SharedGeometryResult(
            video_id="test", frames_count=2,
            pointmaps=np.repeat(pointmap[None], 2, axis=0),
            camera_poses=poses, focal_length=focal_x,
            object_depth_z=np.ones((2, 1)),
        )
        original = np.asarray([[0, 1.25, 1.5], [0, 4.1, 3.2]], dtype=np.float32)
        prepared = SimpleNamespace(
            object_names=("link7",), object_queries=(original,),
            scale_xy=(1.0, 1.0), frames_count=2,
            original_hw=(height, width),
        )
        depths, k, output_poses, world = _input_geometry(prepared, geometry)
        self.assertTrue(np.array_equal(output_poses, poses))
        self.assertTrue(np.allclose(depths, 2.0))
        self.assertTrue(np.allclose(k[0, [0, 2]], [focal_x, cx], atol=1e-4))
        self.assertTrue(np.allclose(k[1, [1, 2]], [focal_y, cy], atol=1e-4))
        self.assertTrue(np.array_equal(world[:, 0], original[:, 0]))
        projected_u = k[0, 0] * world[:, 1] / world[:, 3] + k[0, 2]
        projected_v = k[1, 1] * world[:, 2] / world[:, 3] + k[1, 2]
        self.assertTrue(np.allclose(projected_u, original[:, 1], atol=1e-4))
        self.assertTrue(np.allclose(projected_v, original[:, 2], atol=1e-4))


if __name__ == "__main__":
    unittest.main()
