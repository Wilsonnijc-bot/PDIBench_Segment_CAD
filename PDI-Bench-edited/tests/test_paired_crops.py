import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from pdi_eval.anomaly_scoring.runner import prepare_selected, sha256
from pdi_eval.object_deformation_wrapper.paired_crops import METHOD, normalize_pair


class PairedCropTests(unittest.TestCase):
    def test_unequal_margins_do_not_change_visible_pixels_or_holes(self):
        current = np.zeros((8, 6, 4), np.uint8)
        current[1:7, 1:5] = [12, 34, 56, 255]
        current[3, 2] = [200, 100, 80, 0]
        reference = np.zeros((18, 20, 4), np.uint8)
        reference[9:15, 12:16] = current[1:7, 1:5]
        q, r, info = normalize_pair(current, reference)
        self.assertEqual(q.shape, (6, 4, 4))
        np.testing.assert_array_equal(q, r)
        self.assertEqual(info['roles']['query']['visible_pixel_count'], 23)
        self.assertEqual(info['roles']['reference']['visible_pixel_count'], 23)
        self.assertEqual(info['resampling'], 'none')
        self.assertFalse(q[2, 1].any())

    def test_real_shape_and_size_difference_survives_normalization(self):
        current = np.full((4, 8, 4), 255, np.uint8)
        reference = np.full((8, 4, 4), 255, np.uint8)
        q, r, info = normalize_pair(current, reference)
        self.assertEqual(info['canvas_size_wh'], [8, 8])
        self.assertEqual(q.shape, r.shape)
        np.testing.assert_array_equal(q[2:6, :, :], current)
        np.testing.assert_array_equal(r[:, 2:6, :], reference)
        self.assertFalse(np.array_equal(q[:, :, 3], r[:, :, 3]))

    def test_empty_crop_is_not_invented(self):
        empty = np.zeros((4, 5, 4), np.uint8)
        with self.assertRaisesRegex(ValueError, 'empty'):
            normalize_pair(empty, np.full_like(empty, 255))

    def test_partial_alpha_is_preserved(self):
        image = np.array([[[50, 90, 120, 64], [3, 6, 9, 255]]], np.uint8)
        q, r, _ = normalize_pair(image, image)
        np.testing.assert_array_equal(q, image)
        np.testing.assert_array_equal(r, image)

    def test_native_renderer_exports_the_same_normalized_model_pair(self):
        from pdi_eval.object_deformation_wrapper.reference_visible_pixels import render_saved_examples
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, output = root / 'case', root / 'crops'
            (case / 'occlusion').mkdir(parents=True)
            output.mkdir()
            audit = case / 'occlusion/detection.json'
            audit.write_text('{}')
            (output / 'manifest.json').write_text(json.dumps({
                'case': 'case', 'occlusion_detection_sha256': sha256(audit)}))
            available = np.zeros((2, 12, 12), bool)
            available[1, 4:8, 5:9] = True
            available[1, 5, 6] = False
            mapped = np.zeros((2, 8, 8), bool)
            mapped[1, 3:7, 3:7] = available[1, 4:8, 5:9]
            np.savez(output / 'masks.npz', available_pure_object=available,
                     mapped_to_frame0=mapped, mapping_valid=[False, True],
                     frame0_bbox_xyxy=[1, 1, 9, 9])
            frame0 = np.full((12, 12, 3), 50, np.uint8)
            frame1 = np.full((12, 12, 3), 100, np.uint8)
            with patch('pdi_eval.object_deformation_wrapper.reference_visible_pixels.cv2.VideoCapture'), \
                 patch('pdi_eval.object_deformation_wrapper.reference_visible_pixels.read_frame',
                       side_effect=[frame0, frame1]):
                render_saved_examples(case, output, [1], subdirectory='selected')
            folder = output / 'selected/frame_00001'
            q = np.asarray(Image.open(folder / 'current_available.png'))
            r = np.asarray(Image.open(folder / 'frame0_shape_crop.png'))
            self.assertEqual(q.shape, (4, 4, 4))
            self.assertEqual(q.shape, r.shape)
            np.testing.assert_array_equal(q[:, :, 3], r[:, :, 3])
            self.assertTrue((q[q[:, :, 3] > 0, :3] == 100).all())
            self.assertTrue((r[r[:, :, 3] > 0, :3] == 50).all())
            geometry = json.loads((output / 'pair_geometry.json').read_text())
            record = geometry['pairs']['selected/frame_00001']
            self.assertEqual(record['query_sha256'], sha256(folder / 'current_available.png'))
            self.assertEqual(record['reference_sha256'], sha256(folder / 'frame0_shape_crop.png'))

    def test_scoring_manifest_checks_normalized_pair_hashes_and_dimensions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'case/selected/frame_00001'
            folder.mkdir(parents=True)
            q, r, info = normalize_pair(np.full((4, 8, 4), 255, np.uint8),
                                        np.full((8, 4, 4), 255, np.uint8))
            paths = {'query': folder / 'current_available.png',
                     'reference': folder / 'frame0_shape_crop.png'}
            for role, image in (('query', q), ('reference', r)):
                Image.fromarray(image).save(paths[role])
                info[role + '_sha256'] = sha256(paths[role])
            (root / 'case/pair_geometry.json').write_text(json.dumps({
                'method': METHOD, 'pairs': {'selected/frame_00001': info}}))
            (root / 'selection_index.json').write_text(json.dumps([{
                'case': 'case', 'status': 'complete', 'selected_frames': [{
                    'frame': 1, 'crop_directory': 'selected/frame_00001',
                    'reason': 'largest_available_area_in_interval', 'occlusion_flagged': False}]}]))
            result = prepare_selected(root, [])
            self.assertEqual(result['pairs'][0]['crop_geometry'], METHOD)
            # An updated or independently resized image cannot use old geometry.
            Image.fromarray(q[:4]).save(paths['query'])
            with self.assertRaisesRegex(ValueError, 'crop changed'):
                prepare_selected(root, [])
            info['query_sha256'] = sha256(paths['query'])
            (root / 'case/pair_geometry.json').write_text(json.dumps({
                'method': METHOD, 'pairs': {'selected/frame_00001': info}}))
            with self.assertRaisesRegex(ValueError, 'canvas size differs'):
                prepare_selected(root, [])


if __name__ == '__main__':
    unittest.main()
