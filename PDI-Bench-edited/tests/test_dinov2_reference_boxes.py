import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pdi_eval.perception.dinov2_reference_boxes import (
    box_from_similarity,
    crop_reference_foreground,
    discover_reference_groups,
    guided_box_from_similarity,
    parse_reference_arguments,
    patch_grid_size,
    xyxy_to_normalized_xywh,
)
from pdi_eval.perception.sam3_dinov2_segment import (
    _select_prompt_result,
    _validate_franka_groups,
)


class ReferenceDiscoveryTests(unittest.TestCase):
    def test_discovers_subdirectories_as_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "target-a").mkdir()
            (root / "target-b").mkdir()
            cv2.imwrite(str(root / "target-a" / "one.png"), np.zeros((8, 8, 3), np.uint8))
            cv2.imwrite(str(root / "target-b" / "two.jpg"), np.zeros((8, 8, 3), np.uint8))
            (root / "ignored.txt").write_text("ignored", encoding="utf-8")

            groups = discover_reference_groups(root)

        self.assertEqual(list(groups), ["target-a", "target-b"])
        self.assertEqual([path.name for path in groups["target-a"]], ["one.png"])

    def test_parses_repeated_named_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            image = Path(temporary) / "reference.png"
            cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
            groups = parse_reference_arguments([f"part={image}", f"part={image}"])
        self.assertEqual(len(groups["part"]), 2)

    def test_discovers_nested_franka_groups_and_ignores_contact_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in range(1, 8):
                group = root / "by_link" / f"link_{index}"
                group.mkdir(parents=True)
                cv2.imwrite(str(group / "001.png"), np.ones((8, 8, 3), np.uint8))
                cv2.imwrite(
                    str(group / "contact_sheet_15.png"), np.ones((8, 8, 3), np.uint8)
                )

            groups = discover_reference_groups(root)

        self.assertEqual(list(groups), [f"link{index}" for index in range(1, 8)])
        self.assertTrue(all(len(paths) == 1 for paths in groups.values()))

    def test_crops_black_reference_canvas_to_foreground(self):
        pixels = np.zeros((100, 200, 3), dtype=np.uint8)
        pixels[30:70, 80:120] = 180
        cropped = crop_reference_foreground(
            Image.fromarray(pixels), padding_fraction=0.0
        )
        self.assertEqual(cropped.size, (40, 40))


class BoxExtractionTests(unittest.TestCase):
    def test_extracts_and_pads_high_similarity_component(self):
        similarity = np.zeros((10, 20), dtype=np.float32)
        similarity[2:6, 8:13] = 0.9
        box, metrics = box_from_similarity(
            similarity,
            (100, 200),
            top_fraction=0.08,
            padding_fraction=0.1,
        )
        self.assertEqual(box, (75, 16, 135, 64))
        self.assertGreaterEqual(metrics["component_patches"], 20)

    def test_falls_back_to_peak_when_component_minimum_is_unmet(self):
        similarity = np.zeros((4, 4), dtype=np.float32)
        similarity[1, 2] = 1.0
        box, _ = box_from_similarity(
            similarity,
            (40, 40),
            top_fraction=0.01,
            padding_fraction=0.0,
            minimum_component_patches=3,
        )
        self.assertEqual(box, (20, 10, 30, 20))

    def test_box_normalization_matches_sam3_xywh_contract(self):
        normalized = xyxy_to_normalized_xywh((20, 10, 100, 50), (100, 200))
        np.testing.assert_allclose(normalized, (0.1, 0.1, 0.4, 0.4))

    def test_patch_grid_preserves_aspect_and_patch_multiple(self):
        height, width = patch_grid_size((720, 1280), 840, 14)
        self.assertEqual((height % 14, width % 14), (0, 0))
        self.assertLess(abs(width / height - 1280 / 720), 0.03)

    def test_guided_box_stays_compact_around_reference_prior(self):
        similarity = np.zeros((10, 20), dtype=np.float32)
        similarity[5, 12] = 1.0
        similarity[5, 8] = 0.8
        box, _ = guided_box_from_similarity(
            similarity,
            (100, 200),
            (70, 40, 110, 70),
            padding_fraction=0.10,
        )
        self.assertEqual(box, (65, 37, 113, 73))


class Sam3PromptSelectionTests(unittest.TestCase):
    def test_requires_all_seven_canonical_franka_links(self):
        groups = {f"link{index}": [Path("reference.png")] for index in range(1, 8)}
        _validate_franka_groups(groups)
        with self.assertRaisesRegex(ValueError, "missing=.*link7"):
            _validate_franka_groups({name: paths for name, paths in groups.items() if name != "link7"})

    def test_selects_precise_mask_with_box_support(self):
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, 0:5, 0:5] = True
        masks[1, 6:14, 6:14] = True
        masks[2, 5:15, 5:15] = True
        object_id, selected, score = _select_prompt_result(
            np.asarray([1, 2, 3]),
            masks,
            np.asarray([0.99, 0.75, 0.60]),
            (5, 5, 15, 15),
        )
        self.assertEqual(object_id, 3)
        np.testing.assert_array_equal(selected, masks[2])
        self.assertEqual(score, 0.60)

    def test_rejects_inconsistent_sam3_candidate_arrays(self):
        with self.assertRaisesRegex(RuntimeError, "inconsistent candidate arrays"):
            _select_prompt_result(
                np.asarray([1, 2]),
                np.zeros((1, 20, 20), dtype=bool),
                np.asarray([0.9, 0.8]),
                (5, 5, 15, 15),
            )


if __name__ == "__main__":
    unittest.main()
