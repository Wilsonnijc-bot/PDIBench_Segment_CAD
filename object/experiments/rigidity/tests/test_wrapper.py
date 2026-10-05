"""Small contract tests for workbook matching, grounding, and retry bounds."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from object.preprocessing.grounding.prompting import parse_grounding
from object.preprocessing.selection.selection import build_manifest


class WrapperContracts(unittest.TestCase):
    def test_selected_workbook_has_45_exact_prompt_matches(self):
        root = (_workspace_root())
        workbook = Path(os.environ.get("PDI_OBJECT_WORKBOOK", str(root / "documentation/data/labels/selected_45_matched_videos_styled.xlsx")))
        selection = Path(os.environ.get("PDI_OBJECT_SELECTION", str(root / "results/selected-45-v1/selection.json")))
        if not workbook.is_file() or not selection.is_file():
            self.skipTest("selected-45 workbook/selection not staged")
        manifest = build_manifest(workbook, selection)
        self.assertEqual(manifest["video_count"], 45)
        self.assertEqual(len({entry["case"] for entry in manifest["videos"]}), 45)
        self.assertTrue(all(entry["generation_prompt"] for entry in manifest["videos"]))

    def test_box_and_point_use_full_frame_coordinates(self):
        result = parse_grounding(
            '{"object_name":"blue cube","box_xyxy":[100,200,500,600],'
            '"point_xy":[300,400]}', 1000, 500)
        self.assertEqual(result.box_xyxy, (100, 100, 500, 300))
        self.assertEqual(result.point_xy, (300, 200))
        with self.assertRaisesRegex(ValueError, "strictly inside"):
            parse_grounding('{"object_name":"cube","box_xyxy":[100,200,500,600],'
                            '"point_xy":[700,400]}', 1000, 500)

    def test_segment_stops_after_one_gemini_retry(self):
        try:
            import cv2
            import numpy as np
            from PIL import Image
            from object.preprocessing.segmentation import segment as module
        except ImportError:
            self.skipTest("SAM3 environment dependencies unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
            answer = '{"object_name":"cube","box_xyxy":[100,100,900,900],"point_xy":[500,500]}'
            models = []

            def request(_image, _prompt, model):
                models.append(model)
                return {"model_requested": model, "answer": answer}

            with (patch.object(module, "_first_frame", return_value=(image, 2)),
                  patch.object(module, "call_302", side_effect=request),
                  patch.object(module, "_initialize_and_track", side_effect=ValueError("bad mask")),
                  patch.object(module, "sha256", return_value="test-hash")):
                result = module.segment(output / "video.mp4", "push the cube", output,
                                        output / "sam.pt", output / "bpe.gz")
            self.assertEqual(models, ["gpt-6-luna", "gemini-3.8-flash"])
            self.assertEqual(result["status"], "unscorable")
            self.assertEqual(len(result["vlm_attempts"]), 2)


if __name__ == "__main__":
    unittest.main()
