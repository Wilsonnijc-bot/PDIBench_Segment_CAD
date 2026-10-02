import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from pdi_eval.perception.link5_point_guard import _parse_decision, review_link5_points


class Link5PointGuardTests(unittest.TestCase):
    def setUp(self):
        self.original = np.array([
            [488, 69], [569, 69], [767, 44],
            [495, 27], [478, 72], [807, 82],
        ], dtype=np.int32)
        self.box = (456, 0, 818, 126)

    def parse(self, task, answer):
        return _parse_decision(answer, task, 832, 480, self.original)

    def test_pass_keeps_task_defaults(self):
        for task in ('negative', 'positive'):
            self.assertEqual(self.parse(task, 'PASS'), ('PASS', None))

    def test_each_reject_has_only_its_own_two_points(self):
        decision, wrist = self.parse(
            'negative', '```json\n{"decision":"REJECT","negative_points_xy":'
            '[[495,27],[472,82]]}\n```'
        )
        self.assertEqual(decision, 'REJECT')
        self.assertEqual(wrist.tolist(), [[495, 27], [472, 82]])
        decision, forearm = self.parse(
            'positive', '{"decision":"REJECT","positive_points_xy":'
            '[[490,45],[734,42]]}'
        )
        self.assertEqual(decision, 'REJECT')
        # LVP010's Gemini proposal must pass without the old box-fraction veto.
        self.assertEqual(forearm.tolist(), [[490, 45], [734, 42]])

    def test_rejects_cross_task_fields_and_invalid_coordinates(self):
        for task, answer, error in (
            ('negative', '{"decision":"REJECT","negative_points_xy":'
             '[[495,27],[472,82]],"positive_points_xy":[[490,45],[734,42]]}',
             'unexpected fields'),
            ('negative', '{"decision":"REJECT","negative_points_xy":'
             '[[495,27],[900,82]]}', 'outside the full frame'),
            ('positive', '{"decision":"REJECT","positive_points_xy":'
             '[[490,45],[494,45]]}', 'too close'),
            ('positive', '{"decision":"REJECT","positive_points_xy":'
             '[[560,65],[807,82]]}', 'opposite-label'),
        ):
            with self.subTest(task=task, answer=answer), self.assertRaisesRegex(ValueError, error):
                self.parse(task, answer)

    def test_two_independent_calls_apply_only_their_points_without_material_check(self):
        # Dark pixels cannot veto coordinates validated from the VLM answers.
        image = Image.new('RGB', (832, 480), (20, 20, 20))
        calls = []

        class VLMRoute:
            model = 'test-model'

            def __init__(self, role, config):
                self.role = role
                self.config = config

            def ask(self, images, prompt, *, system_prompt):
                assert len(images) == 2
                assert 'Image 3' not in prompt + system_prompt
                assert 'Image 4' not in prompt + system_prompt
                assert all(im.size == (1664, 960) for im in images)
                calls.append((self.role, self.config, prompt, images))
                if self.role == 'link5_negative_guard':
                    assert 'N1' in prompt and 'N2' in prompt
                    assert 'P2' not in prompt and 'P3' not in prompt
                    assert 'N1' in system_prompt and 'N2' in system_prompt
                    assert 'P2' not in system_prompt and 'P3' not in system_prompt
                    # The two transmitted frames must not draw any positive or N3 marker.
                    for annotated, hidden in zip(images, (
                        ((570, 67), (770, 40), (807, 82)),
                        ((569, 69), (767, 44), (807, 82)),
                    )):
                        for x, y in hidden:
                            assert annotated.getpixel((2 * x, 2 * y)) == (20, 20, 20)
                    return {'answer': '{"decision":"REJECT","negative_points_xy":'
                                      '[[495,27],[472,82]]}'}
                assert self.role == 'link5_positive_guard'
                assert 'P2' in prompt and 'P3' in prompt
                assert 'N1' not in prompt and 'N2' not in prompt
                assert 'P2' in system_prompt and 'P3' in system_prompt
                assert 'N1' not in system_prompt and 'N2' not in system_prompt
                for annotated, hidden in zip(images, (
                    ((495, 27), (478, 92), (807, 82)),
                    ((495, 27), (478, 72), (807, 82)),
                )):
                    for x, y in hidden:
                        assert annotated.getpixel((2 * x, 2 * y)) == (20, 20, 20)
                return {'answer': '{"decision":"REJECT","positive_points_xy":'
                                  '[[490,45],[700,45]]}'}

        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('persistent_masking.interface.secrets.load_env_file'), \
             patch('persistent_masking.vlm_client.VLMClient', side_effect=VLMRoute):
            selected, result = review_link5_points(
                image, self.box, self.original, Path(temporary))
            files = {p.name for p in Path(temporary).iterdir()}
            self.assertTrue({'link5_guard_negative_reference.png',
                             'link5_guard_negative_current.png',
                             'link5_guard_positive_reference.png',
                             'link5_guard_positive_current.png',
                             'link5_guard_selected.png',
                             'link5_guard.json'} <= files)
            self.assertFalse(any('zoom' in name for name in files))
        self.assertEqual(len(calls), 2)
        self.assertEqual([call[0] for call in calls],
                         ['link5_negative_guard', 'link5_positive_guard'])
        for _, config, _, _ in calls:
            self.assertEqual(config['model'], 'gemini-3.8-flash')
            self.assertIsNone(config['reasoning_effort'])
            self.assertEqual(config['max_tokens'], 4096)
        self.assertEqual(result['decision'], 'REJECT')
        np.testing.assert_array_equal(selected[[0, 5]], self.original[[0, 5]])
        self.assertEqual(selected[1:3].tolist(), [[490, 45], [700, 45]])
        self.assertEqual(selected[3:5].tolist(), [[495, 27], [472, 82]])

    def test_one_failed_call_does_not_discard_other_replacement(self):
        image = Image.new('RGB', (832, 480), (170, 180, 200))

        class VLMRoute:
            model = 'test-model'

            def __init__(self, role, config):
                self.role = role

            def ask(self, images, prompt, *, system_prompt):
                if self.role == 'link5_negative_guard':
                    raise ValueError('VLM response is missing text')
                return {'answer': '{"decision":"REJECT","positive_points_xy":'
                                  '[[560,65],[700,45]]}'}

        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('persistent_masking.interface.secrets.load_env_file'), \
             patch('persistent_masking.vlm_client.VLMClient', side_effect=VLMRoute):
            selected, result = review_link5_points(
                image, self.box, self.original, Path(temporary))
        self.assertEqual(result['decision'], 'DEFAULT_UNREVIEWED')
        self.assertEqual(result['reviews']['negative']['decision'], 'DEFAULT_UNREVIEWED')
        self.assertEqual(result['reviews']['positive']['decision'], 'REJECT')
        self.assertEqual(len(result['attempts']), 2)
        np.testing.assert_array_equal(selected[3:5], self.original[3:5])
        self.assertEqual(selected[1:3].tolist(), [[560, 65], [700, 45]])


if __name__ == '__main__':
    unittest.main()
