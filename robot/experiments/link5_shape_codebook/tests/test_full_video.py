"""Every-frame accounting and requested sum must not inherit sparse Simple3D."""
import unittest
from robot.experiments.link5_shape_codebook.full_video import aggregate_all_frames


class FullVideoAggregate(unittest.TestCase):
    def test_sum_includes_frame0_and_every_frame(self):
        rows=[dict(frame_id=i,status='complete',raw_mean_top80=v) for i,v in enumerate((8,2,3,4))]
        result=aggregate_all_frames(rows,4)
        self.assertEqual(result['sum_all_frame_raw_mean_top80'],17)
        self.assertEqual(result['scored_frame_count'],4)
        self.assertTrue(result['include_frame0'])

    def test_missing_pose_is_not_a_zero_or_complete_sum(self):
        rows=[dict(frame_id=0,status='complete',raw_mean_top80=2),dict(frame_id=1,status='failed')]
        result=aggregate_all_frames(rows,2)
        self.assertIsNone(result['sum_all_frame_raw_mean_top80'])
        self.assertEqual(result['available_frame_score_sum'],2)
        self.assertEqual(result['missing_frame_ids'],[1])

    def test_sparse_or_duplicate_frames_rejected(self):
        for ids in ((0,3),(0,0)):
            with self.assertRaises(ValueError):
                aggregate_all_frames([dict(frame_id=i,status='complete',raw_mean_top80=1) for i in ids],2)


if __name__=='__main__':unittest.main()
