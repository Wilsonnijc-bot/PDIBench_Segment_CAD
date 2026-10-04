"""Geometry-independent regression checks for support and conservative filtering."""
import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'PDI-Bench-edited/src')]
from experiments.simple3d_pair_visibility import Config, common_visible, depth_filter, erode, select_frames, restore_cleaned_mask


class VisibilityTests(unittest.TestCase):
    def test_anchor_preserves_observed_edges_outside_sparse_track_hull(self):
        reference=np.zeros((100,180),bool);reference[20:80,10:130]=True
        test=np.zeros_like(reference);test[20:80,80:180]=True
        xy=np.array([(x,y) for y in range(30,71,8) for x in range(25,90,8)],np.float32)
        visible=np.ones(len(xy),bool)
        a,b,info=common_visible(reference,test,xy,xy+[70,0],visible,visible,Config())
        self.assertEqual(info['anchor_role'],'test')
        np.testing.assert_array_equal(b,test)
        self.assertEqual(a.sum(),test.sum())
        self.assertFalse(info['hull_used_for_cropping'])
        strict_a,strict_b,_=common_visible(reference,test,xy,xy+[70,0],visible,visible,
            Config(visibility_mode='strict_hulls'))
        self.assertGreater(a.sum(),strict_a.sum());self.assertGreater(b.sum(),strict_b.sum())

    def test_anchor_area_accounts_for_image_scale(self):
        reference=np.zeros((220,240),bool);reference[20:100,20:100]=True
        test=np.zeros_like(reference);test[40:160,40:160]=True
        xy=np.array([(x,y) for y in range(30,76,8) for x in range(30,76,8)],np.float32)
        visible=np.ones(len(xy),bool)
        a,b,info=common_visible(reference,test,xy,xy*2,visible,visible,Config())
        self.assertLess(reference.sum(),test.sum())
        self.assertEqual(info['anchor_role'],'test')
        np.testing.assert_array_equal(b,test)
        self.assertTrue((a<=reference).all())

    def test_boundary_cut_is_not_stretched_to_full_reference(self):
        reference = np.zeros((100, 180), bool)
        reference[20:80, 10:130] = True
        test = np.zeros_like(reference)
        test[20:80, 80:180] = True  # Same surface translated +70, clipped right.
        xy = np.array([(x, y) for y in range(22, 80, 8) for x in range(12, 108, 8)], np.float32)
        common0, common1, info = common_visible(reference, test, xy, xy+[70, 0],
            np.ones(len(xy), bool), np.ones(len(xy), bool), Config())
        self.assertFalse(common0[:, 110:].any())
        self.assertTrue((common0 <= reference).all())
        self.assertTrue((common1 <= test).all())
        self.assertAlmostEqual(info['image_scale'], 1, places=4)
        reverse1, reverse0, _ = common_visible(test, reference, xy+[70, 0], xy,
            np.ones(len(xy), bool), np.ones(len(xy), bool), Config())
        np.testing.assert_array_equal(common0, reverse0)
        np.testing.assert_array_equal(common1, reverse1)

    def test_spikes_rejected_but_supported_depth_step_preserved(self):
        depth = np.ones((30, 40), np.float32)
        depth[:, 20:] = 2
        depth[10, 10] = 4
        depth[15, 15] = np.nan
        final, rejected, info = depth_filter(depth, np.ones_like(depth, bool), Config())
        self.assertFalse(final[10, 10]); self.assertFalse(final[15, 15])
        self.assertTrue(final[:, 19:22].all())
        self.assertEqual(info['isolated_spike_pixel_count'], 1)
        self.assertEqual(rejected.sum(), 2)

    def test_sampling_ten_disjoint_later_bins(self):
        masks = np.zeros((100, 50, 80), bool)
        masks[:, 10:40, 10:50] = True
        for t in [12, 22, 30, 40, 49, 58, 66, 75, 84, 99]:
            masks[t, 10:40, 50:60] = True
        base, selection, _, _ = select_frames(masks, Config())
        self.assertEqual(base, 0); self.assertEqual(len(selection), 10)
        self.assertEqual(len(set(s['test_frame_id'] for s in selection)), 10)
        for s in selection:
            a, b = s['bin_interval_frames']
            self.assertTrue(a <= s['test_frame_id'] < b)

    def test_camera_edge_is_eroded(self):
        mask = np.ones((30, 30), bool)
        result = erode(mask, 3)
        self.assertFalse(result[:3].any()); self.assertFalse(result[:, :3].any())
        self.assertTrue(result[3:-3, 3:-3].all())

    def test_low_correspondence_never_falls_back(self):
        with self.assertRaisesRegex(ValueError, 'mutually visible'):
            common_visible(np.ones((50,50), bool), np.ones((50,50), bool),
                np.zeros((3,2)), np.zeros((3,2)), np.ones(3, bool), np.ones(3, bool), Config(track_quality_gate=True))

    def test_submajority_tracks_do_not_veto_usable_mask_mapping(self):
        mask=np.ones((100,180),bool)
        points=np.array([(x,y) for y in range(15,86,14) for x in range(20,161,20)],np.float32)
        targets=points.copy()
        targets[::2]=np.random.default_rng(3).uniform([2,2],[178,98],size=targets[::2].shape)
        visible=np.ones(len(points),bool)
        with self.assertRaisesRegex(ValueError,'weak 2D correspondence'):
            common_visible(mask,mask,points,targets,visible,visible,Config(track_quality_gate=True))
        reference,test,info=common_visible(mask,mask,points,targets,visible,visible,Config())
        self.assertLess(info['inlier_fraction'],.6)
        self.assertFalse(info['track_quality_gate'])
        self.assertGreater(reference.sum(),.9*mask.sum());self.assertGreater(test.sum(),.9*mask.sum())

    def test_unestimable_correspondence_still_has_no_full_mask_fallback(self):
        with self.assertRaises(ValueError):
            common_visible(np.ones((50,50),bool),np.ones((50,50),bool),
                np.zeros((1,2)),np.zeros((1,2)),np.ones(1,bool),np.ones(1,bool),Config())

    def test_depth_grid_roundtrip_without_rejections_preserves_source(self):
        mask = np.zeros((53,91), bool);mask[4:48,8:86]=True;mask[11:20,25:33]=False
        cleaned,rejected = restore_cleaned_mask(mask,np.zeros((24,48),bool),(8,4,86,48),(44,78))
        np.testing.assert_array_equal(cleaned,mask)
        self.assertFalse(rejected.any())

    def test_one_upsampled_depth_spike_still_rejects_a_source_pixel(self):
        mask=np.ones((20,30),bool);grid=np.zeros((60,90),bool);grid[1,1]=True
        cleaned,rejected=restore_cleaned_mask(mask,grid,(0,0,30,20),(20,30))
        self.assertTrue(rejected[0,0]);self.assertEqual(rejected.sum(),1)
        self.assertEqual(cleaned.sum(),mask.sum()-1)


if __name__ == '__main__':
    unittest.main()
