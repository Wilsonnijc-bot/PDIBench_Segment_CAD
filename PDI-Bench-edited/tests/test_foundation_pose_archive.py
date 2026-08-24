import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pdi_eval.perception.foundation_pose_wrapper import (
    load_foundation_pose_archive,
)


class FoundationPoseArchiveTests(unittest.TestCase):
    def test_loads_numeric_pose_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poses.npz"
            transforms = np.repeat(np.eye(4)[None, None], 6, axis=0)
            transforms = np.repeat(transforms, 2, axis=1)
            np.savez_compressed(
                path,
                link_names=np.asarray(["link2", "link3"]),
                frame_indices=np.arange(6, dtype=np.int32),
                frame_times_seconds=np.arange(6) / 30.0,
                T_C_from_L=transforms,
                pose_valid=np.ones((6, 2), dtype=bool),
                pose_source=np.full((6, 2), 2, dtype=np.uint8),
                pose_objective=np.zeros((6, 2), dtype=np.float32),
                video_depth_scale=np.asarray(1.2),
                metadata_json=np.asarray(json.dumps({"revision": "fixture"})),
            )

            result = load_foundation_pose_archive(
                path,
                expected_link_names=("link2", "link3"),
                expected_frame_count=6,
            )

        self.assertEqual(result.link_names, ("link2", "link3"))
        self.assertEqual(result.T_C_from_L.shape, (6, 2, 4, 4))
        self.assertEqual(result.video_depth_scale, 1.2)
        self.assertEqual(result.metadata["revision"], "fixture")

    def test_rejects_invalid_pose_marked_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "poses.npz"
            transform = np.eye(4)
            transform[0, 0] = 2.0
            np.savez_compressed(
                path,
                link_names=np.asarray(["link2"]),
                frame_indices=np.arange(1, dtype=np.int32),
                frame_times_seconds=np.asarray([0.0]),
                T_C_from_L=transform[None, None],
                pose_valid=np.ones((1, 1), dtype=bool),
                pose_source=np.ones((1, 1), dtype=np.uint8),
                video_depth_scale=np.asarray(1.0),
            )

            with self.assertRaisesRegex(ValueError, "invalid rigid transform"):
                load_foundation_pose_archive(path)


if __name__ == "__main__":
    unittest.main()
