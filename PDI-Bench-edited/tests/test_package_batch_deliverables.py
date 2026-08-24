import json
import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.package_batch_deliverables import package_available


class PackageBatchDeliverablesTests(unittest.TestCase):
    def test_packages_selected_video_as_hard_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            job = batch / "jobs/job-one"
            sources = {
                job / "output/metrics.json": json.dumps(
                    {
                        "modes": {
                            "exact-group": {
                                "objects": {
                                    "link2": {
                                        "pdi_score": 0.25,
                                        "cad_rigidity": {
                                            "status": "uncalibrated",
                                            "epsilon_cad_mean": 0.1,
                                            "epsilon_cad_p90": 0.2,
                                        },
                                        "pose_discontinuity": {
                                            "pose_discontinuity": False
                                        },
                                    }
                                }
                            }
                        },
                        "cad_canonicalization": {"video_depth_scale": 1.25},
                    }
                ).encode(),
                job / "output/replay/combined_exact-group.mp4": b"points",
                job / "output/replay/cad/cotracker_cad_replay.mp4": b"cad",
                job / "output/replay/cad/cad_replay.json": b"{}",
                job / "output/replay/cad/initial_sam_masks.png": b"masks",
                job / "output/replay/cad/point_cloud_frame_0000.png": b"cloud",
                job / "output/replay/cad/foundationpose_frame_0000.png": b"pose",
                job / "output/replay/cad/cotracker_cad_frame_0000.png": b"frame",
            }
            for path, value in sources.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
            manifest = batch / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "dataset": "COSMOS2.5",
                                "job_id": "job-one",
                                "relative_path": "0000.mp4",
                                "replay_selected": True,
                                "sha256": "a" * 64,
                                "staged_relative_path": "COSMOS2.5/0000.mp4",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = package_available(manifest, batch, batch / "deliverables")

            self.assertEqual(result, {"selected": 1, "packaged": 1, "pending": 0})
            deliverable = batch / "deliverables/COSMOS2.5_0000"
            (deliverable / "original.mp4").write_bytes(b"obsolete")
            (deliverable / "foundationpose_frame_0000.png").write_bytes(b"obsolete")
            result = package_available(manifest, batch, batch / "deliverables")

            self.assertFalse((deliverable / "original.mp4").exists())
            self.assertFalse((deliverable / "metrics.json").exists())
            self.assertFalse((deliverable / "bundle.json").exists())
            self.assertFalse((deliverable / "cad_replay.json").exists())
            self.assertFalse((deliverable / "foundationpose_frame_0000.png").exists())
            self.assertEqual(
                (deliverable / "point_cloud_replay.mp4").read_bytes(), b"points"
            )
            self.assertEqual(
                (deliverable / "cotracker_cad_replay.mp4").read_bytes(), b"cad"
            )
            self.assertEqual(
                {path.name for path in deliverable.iterdir()},
                {"point_cloud_replay.mp4", "cotracker_cad_replay.mp4"},
            )
            with (batch / "deliverables/metrics.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["dataset"], "COSMOS2.5")
            self.assertEqual(row["video"], "0000.mp4")
            self.assertEqual(row["link2_pdi_score"], "0.25")
            self.assertEqual(row["link2_cad_epsilon_mean"], "0.1")
            self.assertEqual(row["link2_pose_discontinuity"], "False")
            self.assertEqual(row["foundation_pose_video_depth_scale"], "1.25")


if __name__ == "__main__":
    unittest.main()
