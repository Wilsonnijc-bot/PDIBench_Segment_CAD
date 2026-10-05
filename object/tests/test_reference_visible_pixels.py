import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from object.preprocessing.crop_pairs.reference_visible_pixels import available_pure_object, export_case, map_available_to_frame0, render_saved_examples


class ReferenceVisiblePixelsTests(unittest.TestCase):
    def test_translated_occlusion_keeps_exact_nonrectangular_shape(self):
        frame0 = np.zeros((20, 20), bool)
        frame0[4:10, 5:11] = True
        expected = np.zeros_like(frame0)
        expected[7:13, 9:15] = True
        object_mask = expected.copy()
        link7 = np.zeros_like(frame0)
        link7[7:10, 9:12] = True
        object_mask[7:10, 9:12] = False
        object_mask[11, 13] = False  # A separate visible-shape hole.
        available = available_pure_object(object_mask, link7)
        mapped, info = map_available_to_frame0(available, expected, frame0)
        wanted = frame0[4:10, 5:11].copy()
        wanted[:3, :3] = False
        wanted[4, 4] = False
        np.testing.assert_array_equal(mapped, wanted)
        self.assertEqual(info["available_area"], int(available.sum()))

    def test_link7_overlap_is_excluded_even_if_object_sam3_includes_it(self):
        obj = np.ones((3, 4), bool)
        grip = np.zeros_like(obj)
        grip[1, 2] = True
        available = available_pure_object(obj, grip)
        self.assertFalse(available[1, 2])
        self.assertEqual(int(available.sum()), 11)

    def test_mapping_clips_to_frame0_object(self):
        frame0 = np.zeros((12, 12), bool)
        frame0[2:8, 2:8] = True
        frame0[3, 4] = False
        expected = np.zeros_like(frame0)
        expected[4:10, 4:10] = True
        mapped, _ = map_available_to_frame0(expected, expected, frame0)
        np.testing.assert_array_equal(mapped, frame0[2:8, 2:8])

    def test_failed_link7_frame_clears_old_reference_and_example_crops(self):
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            (case / "occlusion").mkdir(parents=True)
            (case / "occlusion/detection.json").write_text("{}")
            output = Path(directory) / "output"
            example = output / "examples/frame_00000"
            example.mkdir(parents=True)
            (output / "selection.json").write_text('{"target_count": 5}')
            for path in (output / "frame0_reference.png",
                         example / "current_available.png",
                         example / "frame0_shape_crop.png"):
                path.write_bytes(b"stale")
            objects = np.zeros((2, 8, 8), bool)
            objects[:, 2:5, 2:5] = True
            grippers = np.zeros_like(objects)
            grippers[0] = True
            detection = {
                "config": {"max_gripper_area_fraction": .25},
                "frames": [
                    {"reference_frame": None, "expected_area": None, "mask_valid": False},
                    {"reference_frame": None, "expected_area": None, "mask_valid": True},
                ],
                "source_video_sha256": "video-hash",
                "inputs": {"object": {"sha256": "object-hash"},
                           "gripper": {"sha256": "gripper-hash"}},
            }
            with patch("object.preprocessing.crop_pairs.reference_visible_pixels.load_case",
                       return_value=(detection, objects, grippers, case / "missing.mp4")), \
                 patch("object.preprocessing.crop_pairs.reference_visible_pixels.cv2.VideoCapture"), \
                 patch("object.preprocessing.crop_pairs.reference_visible_pixels.read_frame",
                       return_value=np.zeros((8, 8, 3), np.uint8)), \
                 patch("object.preprocessing.crop_pairs.frame_selection.select_case") as refresh:
                export_case(case, output, [])
                refresh.assert_called_once_with(case, output, 5, strict_bins=False)
            self.assertFalse((output / "frame0_reference.png").exists())
            self.assertFalse((example / "current_available.png").exists())
            self.assertFalse((example / "frame0_shape_crop.png").exists())
            self.assertTrue((example / "preview.png").is_file())
            with np.load(output / "masks.npz", allow_pickle=False) as z:
                self.assertFalse(z["mask_valid"][0])
                self.assertFalse(z["available_pure_object"][0].any())
                self.assertFalse(z["mapping_valid"][0])
            (case / "occlusion/detection.json").write_text('{"updated": true}')
            with self.assertRaisesRegex(ValueError, "Occlusion audit changed"):
                render_saved_examples(case, output, [0])


if __name__ == "__main__":
    unittest.main()
