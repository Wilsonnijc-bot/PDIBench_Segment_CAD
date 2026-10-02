import unittest
import numpy as np
from pdi_eval.object_deformation_wrapper.occlusion import detect, sample


class OcclusionTests(unittest.TestCase):
    def scene(self):
        obj = np.zeros((5, 80, 100), bool); obj[:,30:50,30:50]=True
        grip = np.zeros_like(obj)
        xy = np.array([(x,y) for y in range(32,50,4) for x in range(32,50,4)],float)
        return obj, grip, np.tile(xy,(5,1,1)), np.ones((5,len(xy)),bool)

    def test_substantial_replacement(self):
        o,g,t,v=self.scene();o[2:,30:50,30:40]=False;g[2:,30:50,30:40]=True
        self.assertEqual([r['flagged'] for r in detect(o,g,t,v)],[False,False,True,True,True])

    def test_small_contact_ignored(self):
        o,g,t,v=self.scene();o[2:,30:50,30:32]=False;g[2:,30:50,30:32]=True
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_shrink_without_gripper_ignored(self):
        o,g,t,v=self.scene();o[2:,30:50,30:40]=False;g[2:,10:20,10:20]=True
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_translation_ignored(self):
        o,g,t,v=self.scene();o[2:]=np.roll(o[2:],20,axis=2);t[2:,:,0]+=20
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_one_frame_glitch_ignored(self):
        o,g,t,v=self.scene();o[2,30:50,30:40]=False;g[2,30:50,30:40]=True
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_overlapping_segmentations(self):
        o,g,t,v=self.scene();g[2:,30:50,30:40]=True
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_unaffected_tracks_ignored(self):
        o,g,t,v=self.scene();g[2:,30:50,30:40]=True;t[:,:,0]=45
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_established_episode_survives_mask_regrowth_then_releases(self):
        o,g,t,v=self.scene();o[1:3,30:50,30:40]=False;g[1:4,30:50,30:40]=True
        rows=detect(o,g,t,v)
        self.assertEqual([r['flagged'] for r in rows],[False,True,True,True,False])
        self.assertTrue(rows[3]['continued'])
        self.assertFalse(rows[3]['onset'])

    def test_severity_waits_for_major_loss_then_continues_without_tracks(self):
        o,g,t,v=self.scene();t[1:,:,0]=45
        for f,width in [(1,8),(2,10),(3,8),(4,8)]:
            o[f,30:50,30:30+width]=False;g[f,30:50,30:30+width]=True
        rows=detect(o,g,t,v)
        self.assertEqual([r['flagged'] for r in rows],[False,False,True,True,True])
        self.assertFalse(any(r['legacy_flagged'] for r in rows))
        self.assertTrue(rows[2]['severity_onset'])
        self.assertTrue(rows[3]['severity_continued'])

    def test_moderate_replacement_needs_reduced_track_corroboration(self):
        o,g,t,v=self.scene();o[2:,30:50,30:36]=False;g[2:,30:50,30:36]=True
        t[2:,:,0]=45;t[2:,:3,0]=32  # 3/25 = 12%, below the legacy gate
        rows=detect(o,g,t,v)
        self.assertEqual([r['flagged'] for r in rows],[False,False,True,True,True])
        self.assertFalse(any(r['legacy_flagged'] for r in rows))
        t[2:,:,0]=45
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))

    def test_severity_requires_three_frames_independently_of_legacy(self):
        o,g,t,v=self.scene();o[3:,30:50,30:40]=False;g[3:,30:50,30:40]=True
        t[3:,:,0]=45
        self.assertFalse(any(r['flagged'] for r in detect(o,g,t,v)))
        # Existing two-frame catches must survive even though severity rejects them.
        t[3:,:,0]=32
        rows=detect(o,g,t,v)
        self.assertTrue(rows[3]['legacy_flagged'])
        self.assertFalse(rows[3]['severity_flagged'])
        self.assertTrue(rows[3]['flagged'])

    def test_severity_cannot_seed_legacy_mask_regrowth_continuation(self):
        o,g,t,v=self.scene();o[1:4,30:50,30:40]=False;g[1:,30:50,30:40]=True
        t[1:4,:,0]=45;t[4,:,0]=32  # Tracks enter gripper only after mask regrowth.
        rows=detect(o,g,t,v)
        self.assertEqual([r['flagged'] for r in rows],[False,True,True,True,False])
        self.assertFalse(rows[4]['continued'])
        self.assertFalse(rows[4]['severity_candidate'])

    def test_severity_requires_strong_attribution(self):
        o,g,t,v=self.scene();o[2:,30:50,30:44]=False;g[2:,30:50,30:40]=True
        t[2:,:,0]=47
        rows=detect(o,g,t,v)
        self.assertTrue(all(r['occluded_fraction']>=.5 for r in rows[2:]))
        self.assertTrue(all(r['explained_fraction']<.8 for r in rows[2:]))
        self.assertFalse(any(r['flagged'] for r in rows))

    def test_unassessable_frame_resets_severity_episode(self):
        o,g,t,v=self.scene();t[1:,:,0]=45
        o[1,30:50,30:40]=False;g[1,30:50,30:40]=True
        o[2]=False
        o[3:,30:50,30:38]=False;g[3:,30:50,30:38]=True
        rows=detect(o,g,t,v)
        self.assertEqual(rows[2]['status'],'insufficient_visible_object')
        self.assertFalse(rows[3]['severity_candidate'])
        self.assertFalse(any(r['flagged'] for r in rows))

    def test_missing_reference_unknown(self):
        o,g,t,v=self.scene();g[:]=o
        self.assertTrue(all(r['status']=='insufficient_reference' for r in detect(o,g,t,v)))

    def test_table_sized_masks_fail_before_reference_or_alignment(self):
        o,g,t,v=self.scene();g[:]=True
        rows=detect(o,g,t,v)
        self.assertTrue(all(not r['mask_valid'] for r in rows))
        self.assertTrue(all(r['status']=='failed_link7_mask_area' for r in rows))
        self.assertTrue(all(r['reference_frame'] is None and r['expected_area'] is None for r in rows))
        self.assertFalse(any(r['candidate'] or r['flagged'] for r in rows))

    def test_failed_first_frame_cannot_supply_reference(self):
        o,g,t,v=self.scene();g[0]=True
        rows=detect(o,g,t,v)
        self.assertEqual(rows[0]['status'],'failed_link7_mask_area')
        self.assertEqual(rows[1]['reference_frame'],1)
        self.assertEqual(rows[1]['status'],'assessed')

    def test_failed_frame_resets_both_episodes_without_mask_refresh(self):
        o,g,t,v=self.scene()
        o[1:4,30:50,30:40]=False;g[1:4,30:50,30:40]=True
        g[2]=True
        # Regrown object after the failure cannot continue the earlier episode.
        g[3,30:50,30:40]=True;o[3]=o[0]
        rows=detect(o,g,t,v)
        self.assertTrue(rows[1]['candidate'])
        self.assertEqual(rows[2]['status'],'failed_link7_mask_area')
        self.assertEqual(rows[3]['reference_frame'],0)
        self.assertFalse(rows[3]['continued'] or rows[3]['severity_continued'])
        self.assertFalse(rows[3]['candidate'])

    def test_bad_alignment_rejected(self):
        o,g,t,v=self.scene()
        with self.assertRaises(ValueError): detect(o,g[:-1],t,v)

    def test_out_of_bounds_not_clipped(self):
        m=np.ones((10,10),bool)
        self.assertEqual(sample(m,np.array([[-1,5],[5,20],[np.nan,0],[5,5]])).tolist(),[False,False,False,True])

if __name__=='__main__': unittest.main()
