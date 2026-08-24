import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.export_batch_metrics_csv import LINK_NAMES, export_batch_csv


class PartialLinkMetricsExportTests(unittest.TestCase):
    def test_exports_failed_link_and_interpolation_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "fixture-video"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "videos": [
                            {
                                "dataset": "fixture",
                                "relative_path": "video.mp4",
                                "job_id": job_id,
                                "sha256": "a" * 64,
                                "replay_selected": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            job_root = root / "jobs" / job_id
            (job_root / "output").mkdir(parents=True)
            (job_root / "status.json").write_text(
                json.dumps({"state": "complete"}), encoding="utf-8"
            )
            objects = {
                link: {
                    "object_name": link,
                    "status": "failed",
                    "error": "insufficient depth",
                    "depth": {
                        "strategy": "unavailable",
                        "valid_frame_count": 1,
                        "interpolated_frame_count": 0,
                        "interpolated_frame_fraction": 0.0,
                        "total_frame_count": 5,
                    },
                }
                for link in LINK_NAMES
            }
            objects["link2"] = {
                "object_name": "link2",
                "status": "complete",
                "pdi_score": 0.25,
                "grade": "B",
                "breakdown": {
                    "scale_component": 0.1,
                    "traj_component": 0.2,
                    "epsilon_rigidity": 0.3,
                    "vp_component": 0.4,
                },
                "depth": {
                    "strategy": "interpolation_fallback",
                    "valid_frame_count": 4,
                    "interpolated_frame_count": 1,
                    "interpolated_frame_fraction": 0.2,
                    "total_frame_count": 5,
                },
                "cad_rigidity": {
                    "status": "uncalibrated",
                    "method": "cad-canonical-v1",
                    "deformed": None,
                    "epsilon_cad_mean": 0.12,
                    "epsilon_cad_p90": 0.20,
                    "scored_frame_count": 5,
                    "scored_frame_fraction": 1.0,
                    "pose_valid_frame_count": 5,
                    "mask_present_frame_count": 5,
                },
                "pose_discontinuity": {
                    "pose_discontinuity": False,
                    "event_count": 0,
                    "event_rate": 0.0,
                    "valid_innovation_count": 3,
                    "severity_max": 0.1,
                    "severity_median": 0.05,
                    "severity_p95": 0.09,
                },
            }
            (job_root / "output/metrics.json").write_text(
                json.dumps(
                    {
                        "modes": {
                            "exact-group": {
                                "objects": objects,
                                "timing": {"tracking": {}},
                            }
                        },
                        "cad_canonicalization": {
                            "enabled": True,
                            "video_depth_scale": 1.25,
                            "foundation_pose": {"scale_policy": "video-global-cad"},
                        },
                        "timing": {"foundation_pose_seconds": 12.5},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "metrics.csv"

            export_batch_csv(manifest, root, output)

            with output.open(encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["link2_status"], "complete")
            self.assertEqual(row["link2_depth_strategy"], "interpolation_fallback")
            self.assertEqual(row["link2_depth_interpolated_frame_count"], "1")
            self.assertEqual(row["link2_cad_status"], "uncalibrated")
            self.assertEqual(row["link2_cad_epsilon_mean"], "0.12")
            self.assertEqual(row["link2_pose_discontinuity"], "False")
            self.assertEqual(row["foundation_pose_video_depth_scale"], "1.25")
            self.assertEqual(row["link3_status"], "failed")
            self.assertEqual(row["link3_error"], "insufficient depth")
            self.assertEqual(row["link3_pdi_score"], "")


if __name__ == "__main__":
    unittest.main()
