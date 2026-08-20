import tempfile
import unittest
from pathlib import Path

import numpy as np

from pdi_eval.utils.reconstruct_replay import (
    lift_tracks_to_3d,
    load_segmentation_union,
    map_xy_between_grids,
    project_camera_points,
    resize_mask_nearest,
    transform_world_to_camera,
)


class ResizeMaskTests(unittest.TestCase):
    def test_nearest_resize_preserves_regions(self):
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        resized = resize_mask_nearest(mask, (4, 4))
        expected = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(resized, expected)

    def test_canonical_multi_object_archive_loads_as_union(self):
        object_masks = np.zeros((2, 3, 4, 5), dtype=bool)
        object_masks[:, 0, 0:2, 0:2] = True
        object_masks[:, 2, 2:4, 3:5] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "segmentation.npz"
            np.savez_compressed(path, object_masks=object_masks)
            union = load_segmentation_union(path)
        np.testing.assert_array_equal(union, np.any(object_masks, axis=1))


class CoordinateMappingTests(unittest.TestCase):
    def test_perspective_projection_matches_camera_image_plane(self):
        points = np.array(
            [[0.0, 0.0, 2.0], [1.0, -0.5, 2.0], [0.0, 0.0, -1.0], [np.nan, 0.0, 1.0]]
        )
        projected = project_camera_points(points, focal_length=100.0, image_hw=(200, 300))

        np.testing.assert_allclose(projected[:2], [[150.0, 100.0], [200.0, 75.0]])
        self.assertTrue(np.isnan(projected[2:]).all())

    def test_endpoint_mapping_handles_resolution_mismatch(self):
        xy = np.array([[0.0, 0.0], [4.0, 2.0], [8.0, 4.0]])
        mapped = map_xy_between_grids(xy, source_hw=(5, 9), target_hw=(3, 5))
        np.testing.assert_allclose(
            mapped,
            np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]]),
        )

    def test_lift_respects_visibility_bounds_and_zero_sentinel(self):
        pointmap = np.zeros((2, 3, 5, 3), dtype=np.float32)
        for t in range(2):
            for y in range(3):
                for x in range(5):
                    pointmap[t, y, x] = [x + 1, y + 1, 10 * t + x + y + 1]
        pointmap[1, 1, 2] = 0.0

        tracks = np.array(
            [
                [[0.0, 0.0], [8.0, 4.0], [4.0, 2.0], [-1.0, 2.0]],
                [[0.0, 0.0], [8.0, 4.0], [4.0, 2.0], [4.0, 2.0]],
            ]
        )
        visibility = np.array(
            [
                [1.0, 1.0, 0.5, 1.0],
                [1.0, 1.0, 1.0, np.nan],
            ]
        )

        lifted = lift_tracks_to_3d(
            pointmap,
            tracks,
            visibility,
            track_hw=(5, 9),
            visibility_threshold=0.5,
        )

        np.testing.assert_allclose(lifted[0, 0], [1.0, 1.0, 1.0])
        np.testing.assert_allclose(lifted[0, 1], [5.0, 3.0, 7.0])
        self.assertTrue(np.isnan(lifted[0, 2]).all())  # threshold is strict
        self.assertTrue(np.isnan(lifted[0, 3]).all())  # outside source grid
        np.testing.assert_allclose(lifted[1, 0], [1.0, 1.0, 11.0])
        np.testing.assert_allclose(lifted[1, 1], [5.0, 3.0, 17.0])
        self.assertTrue(np.isnan(lifted[1, 2]).all())  # invalid all-zero point
        self.assertTrue(np.isnan(lifted[1, 3]).all())  # non-finite visibility

    def test_world_points_transform_into_reference_camera(self):
        camera_c2w = np.eye(4)
        camera_c2w[:3, 3] = [10.0, 20.0, 30.0]
        points = np.array([[11.0, 22.0, 33.0], [0.0, 0.0, 0.0], [np.nan, 1.0, 2.0]])

        transformed = transform_world_to_camera(points, camera_c2w)

        np.testing.assert_allclose(transformed[0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(transformed[1], [0.0, 0.0, 0.0])
        self.assertTrue(np.isnan(transformed[2, 0]))


if __name__ == "__main__":
    unittest.main()
