import json
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from pdi_eval.experiment.__main__ import main
from pdi_eval.experiment.batch_v1 import valid_metrics, valid_replay
from pdi_eval.experiment.contracts import page_index
from pdi_eval.experiment.runner import run_case, validate_mask_coverage, v1_outputs_complete
from pdi_eval.experiment.spec import ExperimentSpec, load_spec
from pdi_eval.experiment.contracts import sha256_file
import pdi_eval


class V1OnlyExperimentTests(unittest.TestCase):
    def test_old_versioned_commands_are_rejected(self):
        self.assertIsNone(importlib.util.find_spec("pdi_eval.v2"))
        self.assertFalse(hasattr(pdi_eval, "DeformationV2Pipeline"))
        with self.assertRaisesRegex(SystemExit, "no longer accepts --version"):
            main(["score", "--version", "v2", "--help"])
        with self.assertRaisesRegex(SystemExit, "no longer accepts --version"):
            main(["score", "--version", "v1", "--help"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(["batch-v2", "--help"])
        self.assertNotEqual(caught.exception.code, 0)

    def test_spec_rejects_old_version_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            payload = {
                "schema_version": 1,
                "output_root": temporary,
                "selection": f"{temporary}/selection.json",
                "workbook": f"{temporary}/workbook.xlsx",
                "video_root": temporary,
                "sam_python": "/usr/bin/python3",
                "qwen_python": "/usr/bin/python3",
                "qwen_model": temporary,
                "pdi_python": "/usr/bin/python3",
                "tracker_checkpoint": f"{temporary}/tracker.pth",
                "concurrency": 2,
                "gpu_slots": 1,
            }
            path.write_text(json.dumps(payload))
            spec = load_spec(path)
            self.assertEqual(spec.concurrency, 2)
            self.assertFalse(hasattr(spec, "versions"))
            payload["versions"] = ["v1", "v2"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "versions is no longer supported"):
                load_spec(path)

    def test_resume_requires_all_nonempty_v1_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "v1"
            expected = {"video_sha256": "video-hash", "segmentation_sha256": "mask-hash"}
            self.assertFalse(v1_outputs_complete(destination, **expected))
            paths = [
                destination / "segmentation.npz",
                destination / "cotracker_exact-group.npz",
                destination / "replay/combined_exact-group.mp4",
                destination / "replay/combined_exact-group_first_frame.png",
                destination / "replay/interactive_exact-group/index.html",
                destination / "replay/interactive_exact-group/source.mp4",
                destination / "replay/interactive_exact-group/plotly.min.js",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"ready")
            objects = {f"link{i}": {"status": "skipped"} for i in range(2, 8)}
            objects["link7"] = {"status": "complete"}
            (destination / "metrics.json").write_text(json.dumps({
                "modes": {"exact-group": {"objects": objects}}
            }))
            (destination / "replay/interactive_exact-group/link7_exact-group_pairs.json").write_text("{}")
            (destination / "timing.json").write_text(json.dumps({"status": "complete"}))
            (destination / "manifest.json").write_text(json.dumps({
                "status": "complete", "tracking_modes": ["exact-group"],
                "input": {"sha256": "video-hash"},
                "segmentation": {"sha256": "mask-hash"},
            }))
            (destination / "replay/combined_exact-group.json").write_text("{}")
            self.assertTrue(v1_outputs_complete(destination, **expected))
            self.assertFalse(v1_outputs_complete(
                destination, video_sha256="video-hash", segmentation_sha256="other"
            ))
            paths[-1].write_bytes(b"")
            self.assertFalse(v1_outputs_complete(destination, **expected))

    def test_batch_selected_replay_needs_interactive_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            replay = output / "replay"
            replay.mkdir()
            for name in ("combined_exact-group.mp4", "combined_exact-group_first_frame.png"):
                (replay / name).write_bytes(b"ready")
            (replay / "combined_exact-group.json").write_text("{}")
            self.assertFalse(valid_replay(output))
            interactive = replay / "interactive_exact-group"
            interactive.mkdir()
            for name in ("index.html", "source.mp4", "plotly.min.js"):
                (interactive / name).write_bytes(b"ready")
            self.assertTrue(valid_replay(output))

    def test_batch_resume_requires_final_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            names = [f"link{i}" for i in range(2, 8)]
            (output / "metrics.json").write_text(json.dumps({
                "modes": {"exact-group": {"objects": {name: {} for name in names}}}
            }))
            self.assertFalse(valid_metrics(output / "metrics.json", video_sha256="video-hash"))
            (output / "timing.json").write_text(json.dumps({"status": "complete"}))
            (output / "manifest.json").write_text(json.dumps({
                "status": "complete", "tracking_modes": ["exact-group"],
                "input": {"sha256": "video-hash"},
            }))
            self.assertTrue(valid_metrics(output / "metrics.json", video_sha256="video-hash"))
            self.assertFalse(valid_metrics(output / "metrics.json", video_sha256="other"))

    def test_runner_does_not_complete_when_scorer_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "videos/COSMOS3/0002.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            sample = "COSMOS3_0002"
            case = root / "output/cases" / sample
            case.mkdir(parents=True)
            base = case / "base_segmentation.npz"
            base.write_bytes(b"mask")
            entry = {
                "dataset": "COSMOS3", "video_number": "0002", "workbook_row": 3,
                "sha256": sha256_file(video), "size_bytes": video.stat().st_size,
                "labels": {"robot_deformation": 0},
            }
            (case / "base_segmentation_source.json").write_text(json.dumps({
                "source_video_sha256": entry["sha256"],
                "base_segmentation_sha256": sha256_file(base),
            }))
            spec = ExperimentSpec(
                output_root=root / "output", selection=root / "output/selection.json",
                workbook=root / "workbook.xlsx", video_root=root / "videos",
                sam_python=root / "sam-python", qwen_python=root / "qwen-python",
                qwen_model=root / "qwen", pdi_python=root / "pdi-python",
                tracker_checkpoint=root / "tracker.pth", concurrency=2, gpu_slots=1,
            )
            with patch("pdi_eval.experiment.runner.mask_is_valid", return_value=True), \
                 patch("pdi_eval.experiment.runner.mask_coverage", return_value={
                     f"link{i}": 1.0 for i in range(2, 8)
                 }), \
                 patch("pdi_eval.experiment.runner.persistent_mask", return_value=(
                     root / "work", {"status": "no_confirmed_deformation"})), \
                 patch("pdi_eval.experiment.runner.command"):
                self.assertFalse(run_case(spec, entry))
            status = json.loads((case / "status.json").read_text())
            self.assertEqual(status["state"], "failed")
            self.assertIn("without complete matching metrics", status["error"])

    def test_low_base_link7_reaches_persistent_decision_before_failing_scoring_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "videos/COSMOS2.5/0054.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            case = root / "output/cases/COSMOS2.5_0054"
            case.mkdir(parents=True)
            base = case / "base_segmentation.npz"
            base.write_bytes(b"mask")
            entry = {
                "dataset": "COSMOS2.5", "video_number": "0054", "workbook_row": 36,
                "sha256": sha256_file(video), "size_bytes": video.stat().st_size,
            }
            (case / "base_segmentation_source.json").write_text(json.dumps({
                "source_video_sha256": entry["sha256"],
                "base_segmentation_sha256": sha256_file(base),
            }))
            spec = ExperimentSpec(
                output_root=root / "output", selection=root / "output/selection.json",
                workbook=root / "workbook.xlsx", video_root=root / "videos",
                sam_python=root / "sam-python", qwen_python=root / "qwen-python",
                qwen_model=root / "qwen", pdi_python=root / "pdi-python",
                tracker_checkpoint=root / "tracker.pth", concurrency=2, gpu_slots=1,
            )
            coverage = {f"link{i}": 1.0 for i in range(2, 8)}
            coverage["link7"] = 36 / 93
            with patch("pdi_eval.experiment.runner.mask_is_valid", return_value=True), \
                 patch("pdi_eval.experiment.runner.mask_coverage", return_value=coverage), \
                 patch("pdi_eval.experiment.runner.persistent_mask", return_value=(
                     root / "work", {"status": "no_confirmed_deformation"})) as decision:
                self.assertFalse(run_case(spec, entry))
            decision.assert_called_once()
            status = json.loads((case / "status.json").read_text())
            self.assertEqual(status["persistent_mask_status"], "no_confirmed_deformation")
            self.assertIn("link7", status["error"])

    def test_low_link7_is_deferred_only_before_persistent_masking(self):
        coverage = {f"link{i}": 1.0 for i in range(2, 8)}
        coverage["link7"] = 36 / 93
        with patch("pdi_eval.experiment.runner.mask_coverage", return_value=coverage):
            self.assertIn("link7", validate_mask_coverage(Path("base"), defer_low_link7=True)[1])
            with self.assertRaisesRegex(ValueError, "link7"):
                validate_mask_coverage(Path("selected"))
        coverage["link5"] = 0.5
        with patch("pdi_eval.experiment.runner.mask_coverage", return_value=coverage):
            with self.assertRaisesRegex(ValueError, "link5"):
                validate_mask_coverage(Path("base"), defer_low_link7=True)

    def test_index_links_v1_even_when_historical_v2_files_exist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "cases/COSMOS3_0002"
            for version in ("v1", "v2"):
                path = case / version / "metrics.json"
                path.parent.mkdir(parents=True)
                path.write_text("{}")
            page_index(root, {"videos": [{"dataset": "COSMOS3", "video_number": "0002", "workbook_row": 3}]})
            html = (root / "index.html").read_text()
            self.assertIn("v1/metrics.json", html)
            self.assertNotIn("v2/metrics.json", html)
            self.assertNotIn("V2", html)


if __name__ == "__main__":
    unittest.main()
