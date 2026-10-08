import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from robot.preprocessing.link5_refinement.link5_point_guard import (
    _negative_box_check, _parse_decision, review_link5_points,
)


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
        self.assertEqual(self.parse('negative', 'PASS'), ('PASS', None))
        with self.assertRaisesRegex(ValueError, 'explicitly place P1'):
            self.parse('positive', 'PASS')

    def test_each_task_has_its_own_point_count(self):
        decision, wrist = self.parse(
            'negative', '```json\n{"decision":"REJECT","negative_points_xy":'
            '[[495,27],[472,82]]}\n```'
        )
        self.assertEqual(decision, 'REJECT')
        self.assertEqual(wrist.tolist(), [[495, 27], [472, 82]])
        decision, forearm = self.parse(
            'positive', '{"decision":"PLACE","positive_points_xy":'
            '[[510,65],[590,45],[734,42]]}'
        )
        self.assertEqual(decision, 'PLACE')
        # LVP010's Gemini proposal must pass without the old box-fraction veto.
        self.assertEqual(forearm.tolist(), [[510, 65], [590, 45], [734, 42]])

    def test_rejects_cross_task_fields_and_invalid_coordinates(self):
        for task, answer, error in (
            ('negative', '{"decision":"REJECT","negative_points_xy":'
             '[[495,27],[472,82]],"positive_points_xy":[[490,45],[734,42]]}',
             'unexpected fields'),
            ('negative', '{"decision":"REJECT","negative_points_xy":'
             '[[495,27],[900,82]]}', 'outside the full frame'),
            ('positive', '{"decision":"PLACE","positive_points_xy":'
             '[[510,65],[590,45],[594,45]]}', 'too close'),
            ('positive', '{"decision":"PLACE","positive_points_xy":'
             '[[510,65],[560,65],[807,82]]}', 'opposite-label'),
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
                assert images[1].size == (1664, 960)
                assert images[0].size == ((1664, 960) if self.role == 'link5_negative_guard' else (832, 480))
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
                assert 'P1' in prompt and 'P2' in prompt and 'P3' in prompt
                assert 'NO proposed or default' in prompt
                assert 'N1' not in prompt and 'N2' not in prompt
                assert 'P1' in system_prompt and 'P2' in system_prompt and 'P3' in system_prompt
                assert 'N1' not in system_prompt and 'N2' not in system_prompt
                for annotated, hidden in [(images[1], ((488, 69), (495, 27), (478, 72), (807, 82)))]:
                    for x, y in hidden:
                        assert annotated.getpixel((2 * x, 2 * y)) == (20, 20, 20)
                return {'answer': '{"decision":"PLACE","positive_points_xy":'
                                  '[[520,65],[590,45],[700,45]]}'}

        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link7_persistent.vlm_client.VLMClient', side_effect=VLMRoute):
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
            self.assertEqual(config['reasoning_effort'], 'high')
            self.assertEqual(config['max_tokens'], 65536)
            self.assertEqual(config['timeout_seconds'], 600)
        self.assertEqual(result['decision'], 'REJECT')
        np.testing.assert_array_equal(selected[[5]], self.original[[5]])
        self.assertEqual(selected[0].tolist(), [520, 65])
        self.assertFalse(result['p1_default_used'])
        self.assertEqual(result['original_points_xy'][0], [-1, -1])
        self.assertEqual(selected[1:3].tolist(), [[590, 45], [700, 45]])
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
                return {'answer': '{"decision":"PLACE","positive_points_xy":'
                                  '[[520,65],[560,65],[700,45]]}'}

        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link7_persistent.vlm_client.VLMClient', side_effect=VLMRoute):
            selected, result = review_link5_points(
                image, self.box, self.original, Path(temporary))
        self.assertEqual(result['decision'], 'DEFAULT_UNREVIEWED')
        self.assertEqual(result['reviews']['negative']['decision'], 'DEFAULT_UNREVIEWED')
        self.assertEqual(result['reviews']['positive']['decision'], 'PLACE')
        self.assertEqual(len(result['attempts']), 2)
        np.testing.assert_array_equal(selected[3:5], self.original[3:5])
        self.assertEqual(selected[1:3].tolist(), [[560, 65], [700, 45]])

    def test_missing_p1_cannot_fall_back_to_box_point(self):
        image = Image.new('RGB', (832, 480))
        class Client:
            def __init__(self, role, config):self.role = role
            def ask(self, *args, **kwargs):return {'answer': 'PASS'}
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link7_persistent.vlm_client.VLMClient', Client):
            with self.assertRaisesRegex(ValueError, 'no usable point review'):
                review_link5_points(image, self.box, self.original, Path(temporary), required=False)
            import json
            record = json.loads((Path(temporary) / 'link5_guard.json').read_text())
            self.assertEqual(record['selected_points_xy'][0], [-1, -1])
            self.assertFalse(record['p1_default_used'])
        self.assertEqual(self.original[0].tolist(), [488, 69])

    def test_box_check_catches_three_observed_background_failures(self):
        for points in ([[570, 65], [550, 155]], [[565, 60], [540, 160]],
                       [[570, 45], [552, 155]]):
            with self.subTest(points=points):
                check = _negative_box_check(points, (700, 0, 1200, 180), 5)
                self.assertFalse(check['passed'])
                self.assertEqual(check['outside_points'], ['N1', 'N2'])
        self.assertTrue(_negative_box_check([[695, 45], [720, 185]], (700, 0, 1200, 180), 5)['passed'])
        self.assertEqual(_negative_box_check([[694, 45], [720, 186]], (700, 0, 1200, 180), 5)['outside_points'], ['N1', 'N2'])
        # LVP0030 attempt 2: N2 outside must not veto an in-box N1.
        check = _negative_box_check([[470, 32], [448, 70]], (456, 0, 778, 122), 5)
        self.assertTrue(check['passed'])
        self.assertFalse(check['n2_inside_box'])

    def test_outside_replacement_retries_identical_inputs_then_keeps_valid_points(self):
        image = Image.new('RGB', (832, 480))
        replies = iter([
            '{"decision":"REJECT","negative_points_xy":[[300,27],[320,82]]}',
            '{"decision":"REJECT","negative_points_xy":[[495,27],[472,82]]}',
            '{"decision":"PLACE","positive_points_xy":[[520,65],[560,65],[700,45]]}',
        ])
        def review(task, images, prompt, width, height, original, **kwargs):
            answer = next(replies)
            decision, replacement = _parse_decision(answer, task, width, height, original)
            return decision, replacement, {'role': f'link5_{task}_guard', 'answer': answer}, None
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link5_refinement.link5_point_guard._review', side_effect=review) as mock:
            selected, result = review_link5_points(image, self.box, self.original, Path(temporary))
        self.assertEqual([call.args[0] for call in mock.call_args_list], ['negative', 'negative', 'positive'])
        first, retry = mock.call_args_list[:2]
        self.assertEqual(first.args[2], retry.args[2])
        self.assertEqual(first.kwargs, retry.kwargs)
        for before, after in zip(first.args[1], retry.args[1]):
            self.assertIs(before, after)
        self.assertEqual(selected[3:5].tolist(), [[495, 27], [472, 82]])
        self.assertFalse(result['attempts'][0]['accepted'])
        self.assertTrue(result['attempts'][1]['accepted'])
        self.assertEqual(result['reviews']['negative']['attempt_count'], 2)

    def test_box_retry_exhaustion_never_falls_back_to_defaults(self):
        import json
        image = Image.new('RGB', (832, 480))
        decision = ('REJECT', np.array([[300, 27], [320, 82]]), {'answer': 'outside'}, None)
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link5_refinement.link5_point_guard._review', side_effect=lambda *a, **kw: (decision[0], decision[1], dict(decision[2]), decision[3])) as mock:
            with self.assertRaisesRegex(ValueError, 'exhausted green-box retries'):
                review_link5_points(image, self.box, self.original, Path(temporary), required=False)
            result = json.loads((Path(temporary) / 'link5_guard.json').read_text())
        self.assertEqual(mock.call_count, 3)
        self.assertEqual(result['decision'], 'FAILED_BOX_CHECK')
        self.assertFalse(result['fallback_to_default'])
        self.assertEqual(result['selected_points_xy'][3:5], self.original[3:5].tolist())
        self.assertTrue(all(not call['accepted'] for call in result['attempts']))

    def test_pass_also_checks_original_negatives_against_box(self):
        image = Image.new('RGB', (832, 480))
        original = self.original.copy()
        original[3] = [300, 27]
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link5_refinement.link5_point_guard._review', side_effect=lambda *a, **kw: ('PASS', None, {'answer': 'PASS'}, None)) as mock:
            with self.assertRaisesRegex(ValueError, 'exhausted green-box retries'):
                review_link5_points(image, self.box, original, Path(temporary), negative_max_attempts=2)
        self.assertEqual(mock.call_count, 2)

    def test_negative_parse_failure_retries_same_prompt_and_does_not_call_positive(self):
        image = Image.new('RGB', (832, 480))
        calls = []
        class Client:
            model = 'test-model'
            def __init__(self, role, config): self.role = role
            def ask(self, images, prompt, *, system_prompt):
                self.assert_negative()
                calls.append((prompt, system_prompt, images))
                return {'role': self.role, 'answer': 'REJECT' if len(calls)==1 else
                        '{"decision":"REJECT","negative_points_xy":[[495,27],[448,82]]}'}
            def assert_negative(self):
                assert self.role == 'link5_negative_guard'
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link7_persistent.vlm_client.VLMClient', Client):
            selected, result = review_link5_points(image, self.box, self.original, Path(temporary),
                                                  required=True, negative_only=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][:2], calls[1][:2])
        for first, second in zip(calls[0][2], calls[1][2]): self.assertIs(first, second)
        np.testing.assert_array_equal(selected[:3], self.original[:3])
        self.assertEqual(selected[3:5].tolist(), [[495,27],[448,82]])
        self.assertFalse(result['positive_guard_called'])
        self.assertEqual(result['attempts'][0]['retry_reason'], 'parse_failure')
        self.assertTrue(result['attempts'][1]['accepted'])

    def test_negative_parse_failure_exhaustion_stops_without_fallback(self):
        import json
        image = Image.new('RGB', (832, 480))
        class Client:
            def __init__(self, role, config): self.role = role
            def ask(self, *args, **kwargs): return {'role':self.role,'answer':'REJECT'}
        with tempfile.TemporaryDirectory() as temporary, \
             patch('PIL.Image.open', return_value=image), \
             patch('robot.preprocessing.link7_persistent.interface.secrets.load_env_file'), \
             patch('robot.preprocessing.link7_persistent.vlm_client.VLMClient', Client):
            with self.assertRaisesRegex(ValueError, 'exhausted parsing retries'):
                review_link5_points(image,self.box,self.original,Path(temporary),negative_only=True)
            result=json.loads((Path(temporary)/'link5_guard.json').read_text())
        self.assertEqual(len(result['attempts']),3)
        self.assertEqual(result['decision'],'FAILED_PARSE_RETRIES')
        self.assertFalse(result['fallback_to_default'])
        self.assertTrue(all(a['retry_reason']=='parse_failure' for a in result['attempts']))


if __name__ == '__main__':
    unittest.main()
