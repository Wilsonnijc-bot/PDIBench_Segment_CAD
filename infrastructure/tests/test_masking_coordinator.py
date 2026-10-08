"""Masking scope completes and can resume into the full coordinator DAG."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

from infrastructure.deformation_detect.coordinator import coordinate, load_manifest, read, stages_for, write


class MaskingCoordinatorTests(unittest.TestCase):
    def test_masking_completion_resume_and_full_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'video.mp4'
            video.write_text('source identity')
            prompt = root / 'prompt.txt'
            prompt.write_text('pick up the banana')
            config = {'schema_version': 1, 'run_id': 'masking-test', 'mode': 'masking',
                      'output': str(root / 'run'), 'resources': {}, 'vlm': {},
                      'environments': dict.fromkeys(('sam3', 'vlm1', 'geometry', 'anomaly'), sys.executable),
                      'policy': {'vlm_attempts': 4, 'vlm_stage_timeout_seconds': 60, 'stage_timeout_seconds': 60},
                      'cases': [{'id': 'case', 'video': str(video), 'prompt_file': str(prompt)}]}
            calls = []

            def executor(path, interpreter, timeout):
                request = read(path)
                self.assertTrue(all((Path(dep) / 'result.json').exists() for dep in request['dependencies'].values()))
                calls.append(request['stage'])
                write(path.parent / 'result.json', {'status': 'complete', 'stage': request['stage']})
                return 0

            def run():
                return coordinate(config, executor=executor, check=lambda _: {}, source_identity={'code': 'fixed'})

            self.assertEqual(run()['status'], 'complete')
            self.assertEqual(calls, ['link7_initial', 'vlm1', 'vlm2', 'link7_masks', 'object_masks', 'mask_join'])
            calls.clear()
            self.assertEqual(run()['status'], 'complete')
            self.assertEqual(calls, [])
            config['mode'] = 'full'
            self.assertEqual(run()['status'], 'complete')
            self.assertEqual(calls, ['object_tracks', 'object_crops', 'robot_score', 'object_anomaly'])

    def test_default_keeps_full_pipeline(self):
        self.assertIn('robot_score', stages_for({}))
        self.assertIn('object_anomaly', stages_for({}))


    def test_unknown_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'schema_version': 1, 'mode': 'unknown'}))
            with self.assertRaisesRegex(ValueError, 'mode must be'):
                load_manifest(path)


if __name__ == '__main__':
    unittest.main()
