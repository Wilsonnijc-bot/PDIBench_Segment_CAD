"""Protect video aggregates from frame0 leakage and hidden pose failures."""
from pathlib import Path
import tempfile
import unittest

from robot.experiments.link5_shape_codebook.score import summarize


class ScoringSummary(unittest.TestCase):
    def test_later_frame_aggregates_keep_failures_and_heldout_membership(self):
        visibility=dict(valid_point_count=1000,mask_area=1500,mask_bbox_coverage=.5,image_boundary_contact=False)
        rows=[]
        for frame,value in enumerate([9999,1,5,2]):
            rows.append(dict(video_id='heldout_example',frame_id=frame,status='complete',reference_split='heldout',
                             raw_mean_top80=value,visibility=visibility))
        rows.append(dict(video_id='heldout_example',frame_id=4,status='failed',reference_split='heldout',
                         error='catastrophic pose',visibility=visibility))
        with tempfile.TemporaryDirectory() as directory:
            result=summarize(Path(directory),rows)[0]
        self.assertEqual(result['sampled_count'],4)
        self.assertEqual(result['scored_count'],3)
        self.assertEqual(result['failed_count'],1)
        self.assertEqual(result['max_sampled_frame_score'],5)
        self.assertEqual(result['median_sampled_frame_score'],2)
        self.assertAlmostEqual(result['mean_top3_sampled_frame_scores'],8/3)
        self.assertTrue(result['primary_evaluation_case'])


if __name__=='__main__':unittest.main()
