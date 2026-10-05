"""The automatic stop must reject incomplete, failed or mismatched runs."""


import json
from pathlib import Path
import tempfile
import unittest

from infrastructure.archive.simple3d.operations.simple3d_shutdown_after_completion import completion_ready, digest


class CompletionGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.save('metadata/run.json', {'started': 'bound-run'})
        self.schedule = dict(run_sha256=digest(self.root/'metadata/run.json'),
                             input_manifest_sha256='input', expected_pairs=1, expected_videos=1)
        self.receipt = dict(status='verified_complete', run_sha256=self.schedule['run_sha256'],
                            input_manifest_sha256='input', videos_processed=1, terminal_pairs=1,
                            sha256_verified_pairs=1, successful_scores=1, verified_successful_pairs=1,
                            artifact_audit_status='passed', correlation_status='complete',
                            replay_export_complete=True)
        self.save('cases/video/bin_00/comparison.json', dict(video_id='video', simple3d_status='complete'))
        path = self.root/'metadata/execution/run.log'
        path.parent.mkdir(parents=True)
        path.write_text('BATCH_FINISHED\n')

    def save(self, name, value):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def finish(self):
        (self.root/'metadata/execution/run.exit').write_text('0\n')
        self.save('metadata/completion_verified.json', self.receipt)

    def test_active_run_cannot_stop(self):
        self.save('metadata/completion_verified.json', self.receipt)
        self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_failed_run_cannot_stop(self):
        self.finish()
        (self.root/'metadata/execution/run.exit').write_text('1\n')
        self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_missing_collection_cannot_stop(self):
        self.finish()
        (self.root/'metadata/completion_verified.json').unlink()
        self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_each_incomplete_receipt_field_cannot_stop(self):
        self.finish()
        for key in self.receipt:
            with self.subTest(key=key):
                bad = dict(self.receipt)
                bad.pop(key)
                self.save('metadata/completion_verified.json', bad)
                self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_changed_run_cannot_stop(self):
        self.finish()
        self.save('metadata/run.json', {'started': 'different-run'})
        self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_incomplete_remote_outputs_cannot_stop(self):
        self.finish()
        (self.root/'cases/video/bin_00/comparison.json').unlink()
        self.assertFalse(completion_ready(self.root, self.schedule)[0])

    def test_fully_verified_completion_is_ready(self):
        self.finish()
        self.assertTrue(completion_ready(self.root, self.schedule)[0])


if __name__ == '__main__':
    unittest.main()
