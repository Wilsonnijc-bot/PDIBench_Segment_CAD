import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from infrastructure.shared.replay.rigidity_replay import export_v1_replay
from infrastructure.shared.scoring.rigidity.rigidity import audit_3d_rigidity_cv


class V1ReplayEvidenceTests(unittest.TestCase):
    def test_export_reconstructs_exact_scored_history_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yy, xx = np.mgrid[:16, :16]
            frame = np.stack([xx / 10, yy / 10, np.ones_like(xx) * 2], axis=-1).astype(float)
            pointmaps = np.repeat(frame[None], 3, axis=0)
            pointmaps[1, 11, 11, 0] += 0.2
            pointmaps[2, 8, 8, 1] += 0.3
            coordinates = np.array([[4, 4], [11, 4], [4, 11], [11, 11], [8, 8]], dtype=float)
            tracks = np.repeat(coordinates[None], 3, axis=0)
            visibility = np.ones((3, 5), dtype=float)
            visibility[2, 2:] = 0.0
            masks = np.ones((3, 1, 16, 16), dtype=bool)
            evidence = {}
            score, history = audit_3d_rigidity_cv(
                pointmaps, tracks, visibility, masks[:, 0],
                insufficient_policy="raise", evidence=evidence,
            )
            tracks_path = root / "tracks.npz"
            masks_path = root / "masks.npz"
            geometry_path = root / "geometry.npz"
            metrics_path = root / "metrics.json"
            video_path = root / "source.mp4"
            np.savez_compressed(tracks_path, tracks=tracks, visibility=visibility,
                                object_names=np.array(["link2"]), object_offsets=np.array([0, 5]))
            np.savez_compressed(masks_path, object_masks=masks, object_names=np.array(["link2"]))
            np.savez_compressed(geometry_path, pointmaps=pointmaps,
                                camera_poses=np.repeat(np.eye(4)[None], 3, axis=0), focal_length=12.0)
            metrics = {"segmentation": {"object_names": ["link2"]}, "modes": {"exact-group": {"objects": {
                "link2": {"status": "complete", "pdi_score": 0.25, "breakdown": {
                    "rigidity_strategy": "Strategy 1 (3D rigid pairwise ratios)",
                    "volume_history": history.tolist(),
                }}
            }}}}
            metrics_path.write_text(json.dumps(metrics))
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (16, 16))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            writer.release()
            js = root / "plotly.min.js"
            js.write_text("// fixture")
            pages = export_v1_replay(metrics_path, tracks_path, masks_path, geometry_path,
                                     video_path, root / "output", plotly_js=js)
            self.assertEqual(len(pages), 1)
            self.assertNotIn("V2", pages[0].read_text())
            self.assertTrue((root / "output/source.mp4").is_file())
            saved = json.loads((root / "output/link2_exact-group_pairs.json").read_text())
            self.assertEqual(saved["selected_pairs"], evidence["selected_pairs"])
            self.assertEqual(saved["pair_frames"], evidence["pair_frames"])
            self.assertEqual(saved["rigidity_history"], history.tolist())
            self.assertEqual(saved["carried_frames"], [2])
            self.assertAlmostEqual(saved["final_rigidity_score"], score)
            replay_data = json.loads(pages[0].read_text().split("const data=", 1)[1].split(";", 1)[0])
            self.assertEqual(replay_data["observed"], [None, history[1], None])
            metrics["modes"]["exact-group"]["objects"]["link2"]["breakdown"]["volume_history"][1] += 0.1
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError, "differs from scored history"):
                export_v1_replay(metrics_path, tracks_path, masks_path, geometry_path,
                                 video_path, root / "invalid", plotly_js=js)


if __name__ == "__main__":
    unittest.main()
