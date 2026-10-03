"""Check extracted inference against the unmodified official evaluation routine."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pdi_eval.anomaly_scoring import AnomalyDINOScorer
from pdi_eval.anomaly_scoring.anomalydino import load_upstream, rgb_array
from pdi_eval.anomaly_scoring.runner import DEFAULT_EXCLUSIONS, prepare_selected, run_pairs

UPSTREAM_COMMIT = 'b9d1c2648e3a5247437d4d953d907a8f3d994457'
UPSTREAM = Path(__file__).resolve().parents[1] / 'third_party/AnomalyDINO'


class TinyBackbone:
    """Deterministic test features; never used to generate experiment scores."""
    device = 'cpu'

    def __init__(self):
        self.forward_count = 0

    def prepare_image(self, image):
        return image, (10, 20)

    def extract_features(self, image):
        self.forward_count += 1
        pixels = image.reshape(-1, 3).astype(np.float32) / 255
        indices = np.linspace(0, len(pixels) - 1, 200).astype(int)
        return np.concatenate([pixels[indices], np.ones((200, 1), dtype=np.float32)], axis=1)

    def compute_background_mask(self, features, grid, threshold=10, masking_type=False):
        return np.arange(len(features)) % 3 != 0 if masking_type else np.ones(len(features), dtype=bool)


class TestAnomalyDINO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backbones, cls.inference = load_upstream()
        cls.reference = np.random.default_rng(5).integers(0, 256, (40, 36, 3), dtype=np.uint8)
        cls.query = np.random.default_rng(6).integers(0, 256, (38, 33, 3), dtype=np.uint8)

    def scorer(self, **kwargs):
        model = TinyBackbone()
        with patch.object(self.backbones, 'get_model', return_value=model):
            return AnomalyDINOScorer(device='cpu', **kwargs), model

    def test_parity_with_unmodified_upstream(self):
        # Compare to upstream itself, not a second hand-written anomaly formula.
        try:
            source = subprocess.check_output([
                'git', '-C', str(UPSTREAM), 'show', f'{UPSTREAM_COMMIT}:src/detection.py'], text=True)
        except subprocess.CalledProcessError:
            self.skipTest('Original upstream Git history required for parity check')
        original = types.ModuleType(self.inference.__package__ + '.original_detection')
        original.__package__ = self.inference.__package__
        source = source.replace('from src.utils import', 'from .utils import').replace(
            'from src.post_eval import', 'from .post_eval import')
        exec(compile(source, 'upstream-original-detection.py', 'exec'), original.__dict__)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_dir = root / 'object/train/good'
            query_dir = root / 'object/test/query'
            reference_dir.mkdir(parents=True)
            query_dir.mkdir(parents=True)
            Image.fromarray(self.reference).save(reference_dir / 'reference.png')
            Image.fromarray(self.query).save(query_dir / 'query.png')
            for rotation, masking in [(False, False), (True, False), (False, True), (True, True)]:
                with self.subTest(rotation=rotation, masking=masking):
                    scorer, model = self.scorer(rotation=rotation, masking=masking, mask_ref_images=True)
                    actual = scorer.score(self.reference, self.query)['anomaly_score']
                    with patch('torch.cuda.synchronize'):
                        scores, _, _ = original.run_anomaly_detection(
                            model, 'object', str(root), 1, {'object': ['query']}, str(root / 'output'),
                            save_examples=False, masking=masking, mask_ref_images=True,
                            rotation=rotation, faiss_on_cpu=True, save_patch_dists=False)
                    self.assertAlmostEqual(actual, float(scores['query/query.png']), places=7)

    def test_cache_and_input_types(self):
        scorer, model = self.scorer(rotation=False)
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / 'reference.png'
            Image.fromarray(self.reference).save(file)
            scorer.precompute_reference(file)
            self.assertEqual(model.forward_count, 1)
            result = scorer.score(Image.fromarray(self.reference), self.query, return_map=True)
            self.assertEqual(model.forward_count, 2)
            self.assertTrue(np.isfinite(result['anomaly_score']))
            self.assertEqual(result['anomaly_map'].shape, self.query.shape[:2])
            self.assertEqual(scorer.reference_cache_misses, 1)
            self.assertEqual(scorer.reference_cache_hits, 1)
            scorer.score(self.query, self.reference)
            self.assertEqual(scorer.reference_cache_misses, 2)

    def test_transparency_removes_hidden_scene_pixels(self):
        rgba = np.array([[[255, 0, 0, 0], [0, 200, 0, 255]]], dtype=np.uint8)
        np.testing.assert_array_equal(rgb_array(rgba), [[[0, 0, 0], [0, 200, 0]]])

    def test_modified_input_fails_before_inference(self):
        scorer, model = self.scorer(rotation=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(self.reference).save(root / 'r.png')
            Image.fromarray(self.query).save(root / 'q.png')
            pairs = [{'video_id': 'test', 'reference_crop': 'r.png', 'query_crop': 'q.png',
                      'reference_sha256': 'wrong'}]
            with self.assertRaises(ValueError):
                run_pairs(scorer, pairs, root, root / 'out')
            self.assertEqual(model.forward_count, 0)

    def test_selection_exclusions_and_unavailable(self):
        self.assertEqual(set(DEFAULT_EXCLUSIONS), {
            'COSMOS2.5_0001', 'COSMOS3_0001', 'LVP_ROBOWM_0001'})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'selection_index.json').write_text(json.dumps([
                {'case': 'skip', 'selected_frames': [], 'status': 'complete'},
                {'case': 'missing', 'selected_frames': [], 'status': 'no_reference'}]))
            manifest = prepare_selected(root, ['skip'])
            self.assertEqual(manifest['pair_count'], 0)
            self.assertEqual([v['status'] for v in manifest['videos']],
                             ['excluded_by_user', 'unavailable_reference'])


if __name__ == '__main__':
    unittest.main()
