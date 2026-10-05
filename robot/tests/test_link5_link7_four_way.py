import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from robot.experiments.link7_tracker_filter_comparison import link5_link7_four_way as workflow


class FourWayRunnerTests(unittest.TestCase):
    def _case(self, root):
        source = root / "videos/LVP/0001.mp4"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"frozen-video")
        output = root / "results"
        output.mkdir()
        (output / "historical_references.json").write_text(
            json.dumps({"cases": {"LVP_ROBOWM_0001": {"status": "unavailable"}}}))
        entry = {"dataset": "LVP_ROBOWM", "video_number": "0001",
                 "workbook_row": 2, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "size_bytes": source.stat().st_size}
        return SimpleNamespace(output_root=output, video_root=root / "videos"), entry

    def _run(self, coverage):
        with tempfile.TemporaryDirectory() as directory:
            spec, entry = self._case(Path(directory))
            base = Path(directory) / "base.npz"
            base.write_bytes(b"selected-mask")
            calls = []

            def score(*args):
                targets = args[9]
                calls.append((args[6], args[7], targets))
                return {name: {"status": "complete", "epsilon_rigidity": 0.02}
                        for name in targets}

            with patch.object(workflow, "ensure_video_alias"), \
                 patch.object(workflow, "ensure_base_segmentation", return_value=base), \
                 patch.object(workflow, "persistent_mask", return_value=(Path(directory) / "work", {"status": "no_confirmed_deformation"})), \
                 patch.object(workflow, "mask_coverage", return_value=coverage), \
                 patch.object(workflow, "_score", side_effect=score), \
                 patch.object(workflow, "clean_megasam_intermediates"):
                self.assertTrue(workflow.run_case(spec, entry))
            status = json.loads((spec.output_root / "cases/LVP_ROBOWM_0001/status.json").read_text())
            return calls, status

    def test_four_paths_share_mask_and_score_link5_once(self):
        calls, status = self._run({"link5": 1.0, "link7": 1.0})
        self.assertEqual(len(calls), 4)
        self.assertEqual([item[2] for item in calls],
                         [("link5", "link7"), ("link7",), ("link7",), ("link7",)])
        self.assertEqual(set(status["paths"]), {path[0] for path in workflow.PATHS})
        self.assertEqual({path["selected_mask_sha256"] for path in status["paths"].values()},
                         {status["selected_mask_sha256"]})

    def test_low_link7_keeps_link5_result_and_marks_four_failures(self):
        calls, status = self._run({"link5": 1.0, "link7": 0.4})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], ("link5",))
        self.assertEqual(status["link5"]["status"], "complete")
        self.assertTrue(all(path["link7"]["error_type"] == "insufficient_sam3_coverage"
                            for path in status["paths"].values()))


if __name__ == "__main__":
    unittest.main()
