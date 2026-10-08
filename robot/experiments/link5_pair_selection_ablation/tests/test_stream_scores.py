import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot.experiments.link5_pair_selection_ablation.stream_scores_depth import finish_video, save_progress


class StreamingScores(unittest.TestCase):
    def test_failed_filter_never_scores_without_support(self):
        entry = dict(video_id='example', cohort='selected45')
        with patch('robot.experiments.link5_pair_selection_ablation.stream_scores_depth.filter_video',
                   return_value=dict(video_id='example', status='failed', error='Provenance mismatch')):
            with patch('robot.experiments.link5_pair_selection_ablation.stream_scores_depth.score_video') as scorer:
                result = finish_video((entry, '/tmp/example', '/tmp/cache'))
                scorer.assert_not_called()
        self.assertEqual(len(result['scores']), 2)
        self.assertTrue(all(row['status'] == 'failed' for row in result['scores']))

    def test_video_is_filtered_before_both_method_scores(self):
        entry = dict(video_id='example', cohort='selected45'); calls = []
        def filtering(job):
            calls.append('filter'); return dict(video_id='example', status='complete')
        def scoring(job):
            calls.append('both scores'); return [dict(method=m, status='complete') for m in ('balanced_v0', 'refine_v1')]
        with patch('robot.experiments.link5_pair_selection_ablation.stream_scores_depth.filter_video', side_effect=filtering), \
             patch('robot.experiments.link5_pair_selection_ablation.stream_scores_depth.score_video', side_effect=scoring):
            result = finish_video((entry, '/tmp/example', '/tmp/cache'))
        self.assertEqual(calls, ['filter', 'both scores'])
        self.assertEqual(len(result['scores']), 2)

    def test_waiting_videos_are_pending_not_failures(self):
        entries = [dict(video_id='waiting', cohort='selected45')]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); save_progress(root, entries, {}, {})
            rows = json.loads((root / 'rigidity/summary.json').read_text())
            progress = json.loads((root / 'metadata/stream_progress.json').read_text())
        self.assertEqual([row['status'] for row in rows], ['pending', 'pending'])
        self.assertEqual(progress['failed_scores'], 0)
        self.assertEqual(progress['successful_scores'], 0)


if __name__ == '__main__':
    unittest.main()
