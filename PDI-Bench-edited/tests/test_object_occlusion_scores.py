import unittest
from pdi_eval.object_deformation_wrapper.occlusion_scores import filtered_score

class OcclusionScoreTests(unittest.TestCase):
    def calc(self,flags):
        score={'rigidity_history':[0.,.1,.9,.2],'rigidity_score':.4}
        detection={'method':'test','config':{},'frames':[{'frame':i,'flagged':v,'status':'assessed','rigidity_carried':i==3} for i,v in enumerate(flags)]}
        return filtered_score(score,detection)
    def test_exclusion_preserves_naive(self):
        r=self.calc([False,False,True,False]);self.assertAlmostEqual(r['filtered_rigidity_score'],.15)
        self.assertEqual(r['naive_rigidity_score'],.4);self.assertEqual(r['filtered_rigidity_history'],[None,.1,None,.2])
        self.assertEqual(r['retained_carried_frames'],[3])
    def test_no_exclusion(self):self.assertAlmostEqual(self.calc([False]*4)['filtered_rigidity_score'],.4)
    def test_all_excluded_is_null(self):self.assertIsNone(self.calc([False,True,True,True])['filtered_rigidity_score'])
    def test_reference_never_counts(self):self.assertAlmostEqual(self.calc([True,False,True,False])['filtered_rigidity_score'],.15)
    def test_failed_mask_is_excluded_without_being_called_occlusion(self):
        score={'rigidity_history':[0.,.1,.9,.2],'rigidity_score':.4}
        rows=[{'frame':i,'flagged':False,'mask_valid':i!=2,
               'status':'failed_link7_mask_area' if i==2 else 'assessed'} for i in range(4)]
        result=filtered_score(score,{'method':'test','config':{},'frames':rows})
        self.assertEqual(result['failed_mask_frames'],[2])
        self.assertEqual(result['excluded_frames'],[2])
        self.assertEqual(result['filtered_rigidity_history'],[None,.1,None,.2])
        self.assertAlmostEqual(result['filtered_rigidity_score'],.15)
        self.assertEqual(result['naive_rigidity_score'],.4)

    def test_all_failed_masks_produce_null_score(self):
        score={'rigidity_history':[0.,.1,.9,.2],'rigidity_score':.4}
        rows=[{'frame':i,'flagged':False,'mask_valid':False,
               'status':'failed_link7_mask_area'} for i in range(4)]
        result=filtered_score(score,{'method':'test','config':{},'frames':rows})
        self.assertIsNone(result['filtered_rigidity_score'])
        self.assertEqual(result['status'],'no_retained_frames')
        self.assertEqual(result['filtered_rigidity_history'],[None]*4)
if __name__=='__main__':unittest.main()
