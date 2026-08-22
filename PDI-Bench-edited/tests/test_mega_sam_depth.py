import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from pdi_eval.perception.mega_sam_wrapper import (
        InsufficientTargetDepthError,
        MegaSamWrapper,
        _target_depth_from_world_pointmaps,
        target_depth_from_world_pointmaps,
    )


def _fixture(depths):
    pointmaps = np.zeros((len(depths), 2, 2, 3), dtype=np.float64)
    masks = np.ones((len(depths), 2, 2), dtype=bool)
    for index, depth in enumerate(depths):
        pointmaps[index, ..., 2] = depth
    poses = np.repeat(np.eye(4, dtype=np.float64)[None], len(depths), axis=0)
    return pointmaps, poses, masks


@unittest.skipIf(torch is None, "MegaSAM depth tests require the PDI PyTorch environment")
class TargetDepthTests(unittest.TestCase):
    def test_interpolates_an_internal_invalid_frame(self):
        pointmaps, poses, masks = _fixture([2.0, np.nan, 4.0, 5.0, 6.0])

        depth, metadata = _target_depth_from_world_pointmaps(
            pointmaps, poses, masks
        )

        np.testing.assert_allclose(depth, [1.0, 1.5, 2.0, 2.5, 3.0])
        self.assertEqual(metadata["strategy"], "interpolation_fallback")
        self.assertEqual(metadata["valid_frame_count"], 4)
        self.assertEqual(metadata["interpolated_frame_count"], 1)
        self.assertEqual(metadata["interpolated_frame_indices"], [1])

    def test_fills_leading_and_trailing_gaps_with_nearest_depth(self):
        pointmaps, poses, masks = _fixture([np.nan, 2.0, 4.0, np.nan])

        depth = target_depth_from_world_pointmaps(
            pointmaps, poses, masks, maximum_interpolated_fraction=0.5
        )

        np.testing.assert_allclose(depth, [1.0, 1.0, 2.0, 2.0])

    def test_rejects_a_link_with_fewer_than_two_valid_frames(self):
        pointmaps, poses, masks = _fixture([np.nan, 2.0, np.nan])

        with self.assertRaisesRegex(
            InsufficientTargetDepthError, "only 1/3 frames"
        ):
            target_depth_from_world_pointmaps(pointmaps, poses, masks)

    def test_empty_mask_frames_are_interpolated(self):
        pointmaps, poses, masks = _fixture([2.0, 3.0, 4.0, 5.0, 6.0])
        masks[2] = False

        depth = target_depth_from_world_pointmaps(pointmaps, poses, masks)

        np.testing.assert_allclose(depth, [1.0, 1.5, 2.0, 2.5, 3.0])

    def test_rejects_when_fallback_would_replace_too_many_frames(self):
        pointmaps, poses, masks = _fixture([2.0, np.nan, 4.0, np.nan, 6.0])

        with self.assertRaisesRegex(
            InsufficientTargetDepthError, "need at least 4"
        ):
            target_depth_from_world_pointmaps(pointmaps, poses, masks)


@unittest.skipIf(torch is None, "MegaSAM depth tests require the PDI PyTorch environment")
class SharedGeometryIsolationTests(unittest.TestCase):
    def test_insufficient_depth_fails_only_the_affected_link(self):
        pointmaps, poses, _ = _fixture([2.0, 3.0, 4.0, 5.0, 6.0])
        object_masks = np.ones((5, 2, 2, 2), dtype=bool)
        object_masks[1:, 1] = False
        metadata = {"fixture": True}

        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "fixture.npz"
            np.savez_compressed(
                cache_path,
                pointmaps=pointmaps,
                camera_poses=poses,
                focal_length=np.asarray(100.0),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            wrapper = MegaSamWrapper.__new__(MegaSamWrapper)
            wrapper._geometry_cache_identity = lambda _: ("fixture", metadata)

            result = wrapper.infer_shared(
                "fixture.mp4", object_masks, cache_dir=temporary
            )

        self.assertTrue(np.all(np.isfinite(result.object_depth_z[:, 0])))
        self.assertTrue(np.all(np.isnan(result.object_depth_z[:, 1])))
        self.assertEqual(result.metadata["object_depth"][0]["status"], "complete")
        self.assertEqual(result.metadata["object_depth"][0]["strategy"], "direct")
        self.assertEqual(result.metadata["object_depth"][1]["status"], "failed")
        self.assertEqual(result.metadata["object_depth"][1]["valid_frame_count"], 1)

if __name__ == "__main__":
    unittest.main()
