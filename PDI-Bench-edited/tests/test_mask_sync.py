import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pdi_eval.experiment.mask_merge import build_repaired_segmentation, sha256_file
from pdi_eval.object_deformation_wrapper.mask_sync import sync_case


class AcceptedMaskSyncTests(unittest.TestCase):
    def fixture(self, root):
        case = root/'cases/example'; (case/'masking').mkdir(parents=True)
        (case/'replay').mkdir(); (case/'occlusion').mkdir()
        video = case/'replay/source.mp4'; video.write_bytes(b'source video')
        objects = case/'masking/segmentation.npz'
        np.savez_compressed(objects, object_masks=np.ones((2, 1, 8, 8), bool))
        base = root/'base.npz'
        before = np.zeros((2, 6, 8, 8), bool); before[:, 0, :2] = True
        before[:, 5, 2:4] = True
        np.savez_compressed(base, object_masks=before, object_names=['link2','link3','link4','link5','link6','link7'], object_ids=[2,3,4,5,6,7])
        masks = root/'masks.npz'; after = np.zeros((2,8,8), bool); after[:,4:6] = True
        np.savez_compressed(masks,masks=after)
        repair = root/'repair.json'
        repair.write_text(json.dumps(dict(case='example', accepted=True, status='accepted',
            source_video_sha256=sha256_file(video), output_masks=str(masks),
            output_masks_sha256=sha256_file(masks),
            object_input={'sha256':sha256_file(objects)},
            link7_input={'path':str(base), 'sha256':sha256_file(base)})))
        return case, base, repair, before, after

    def test_accepted_archive_is_identical_for_detector_viewer_and_crops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();case,base,repair,before,after=self.fixture(root)
            gripper=root/'gripper'; selected=gripper/'cases/example/v1_cotracker3/segmentation.npz'
            events=[]
            def process(c,g,config):
                self.assertEqual(g,gripper);events.append('detect')
                with np.load(selected) as a:
                    np.testing.assert_array_equal(a['object_masks'][:,:5],before[:,:5])
                    np.testing.assert_array_equal(a['object_masks'][:,5],after)
                (c/'occlusion/detection.json').write_text(json.dumps({'inputs':{'gripper':{'sha256':sha256_file(selected)}}}))
                return {'case':'example'}
            def replay(c):
                events.append('replay');(c/'occlusion/replay.html').write_text('viewer')
                return json.loads((c/'occlusion/detection.json').read_text())
            def crops(c,o,frames):events.append('crops');return {'case':'example'}
            def select(c,o,count):
                events.append('select');self.assertEqual(count,10)
                return dict(selected_frames=[{'frame':1}],required_frames=[],method='v5',final_interval_policy={})
            with patch('pdi_eval.object_deformation_wrapper.mask_sync.process',process), patch('pdi_eval.object_deformation_wrapper.mask_sync.export_replay',replay), patch('pdi_eval.object_deformation_wrapper.mask_sync.export_case',crops), patch('pdi_eval.object_deformation_wrapper.mask_sync.select_case',select):
                result=sync_case(case=case,repair_record=repair,base_segmentation=base,gripper_root=gripper,crop_root=root/'crops')
            self.assertEqual(events,['detect','replay','crops','select'])
            self.assertEqual(result['selected_segmentation_sha256'],sha256_file(selected))

    def test_rejected_and_tampered_repairs_cannot_replace_link7(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case,base,repair,_,_=self.fixture(root)
            args=dict(video=case/'replay/source.mp4',base_segmentation=base,repair_record=repair,output_npz=root/'out.npz')
            data=json.loads(repair.read_text());data['accepted']=False;repair.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Only accepted'):build_repaired_segmentation(**args)
            data['accepted']=True;data['output_masks_sha256']='wrong';repair.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'output mask hash'):build_repaired_segmentation(**args)
            self.assertFalse((root/'out.npz').exists())


if __name__ == '__main__':unittest.main()
