import unittest
import hashlib
import io
import json
from PIL import Image
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from persistent_masking.vlm3_overmask import audit_gate, parse_points, repair_case, repair_persistent_run, digest, crop_policy


class VLM3Tests(unittest.TestCase):
    def test_residual_overmask_excludes_frames_instead_of_rejecting_video(self):
        obj=np.ones((3,10,10),bool);mask=np.zeros_like(obj)
        mask[1]=True
        post=audit_gate(mask,obj,include_area_guard=False)
        sam={'membership_ok':True,'empty_frames':[]}
        policy=crop_policy(post,sam)
        self.assertTrue(policy['accepted'])
        self.assertEqual(policy['status'],'accepted_with_frame_exclusions')
        self.assertEqual(policy['crop_excluded_frames'],[1])
        self.assertEqual(policy['crop_eligible_frames'],[0,2])
        self.assertFalse(crop_policy(post,{'membership_ok':False,'empty_frames':[]})['accepted'])
        self.assertFalse(crop_policy(audit_gate(obj,obj,include_area_guard=False),sam)['accepted'])

    def test_optional_gate_runs_even_when_qwen_finds_no_deformation(self):
        with TemporaryDirectory() as tmp:
            work=Path(tmp);video=work/'source.mp4';video.write_bytes(b'video')
            source=work/'naive/Cosmos25_0005/masks.npz';source.parent.mkdir(parents=True)
            masks=np.ones((2,10,10),bool);np.savez_compressed(source,masks=masks)
            original={'status':'no_confirmed_deformation','source_sha256':digest(video),
                      'video':str(video),'naive_masks_sha256':digest(source),'diagnoses':[{'state':'normal'}]}
            (work/'provenance.json').write_text(json.dumps({'results':{'Cosmos25_0005':original}}))
            result={'case':'COSMOS2.5_0005','status':'repair_failed','accepted':False,
                    'frame':0,'gate':{'frame':0},'attempts':[]}
            with patch('persistent_masking.vlm3_overmask.load_objects',return_value=(masks,{'target_object':'cube'})),patch('persistent_masking.vlm3_overmask.repair_case',return_value=result) as repair,patch('persistent_masking.vlm3_overmask.export_review'):
                repair_persistent_run(work,['Cosmos25_0005'],work/'objects')
            repair.assert_called_once()
            saved=json.loads((work/'provenance.json').read_text())['results']['Cosmos25_0005']
            for key,value in original.items():self.assertEqual(saved[key],value)
            self.assertEqual(saved['vlm3']['status'],'repair_failed')
            self.assertEqual(result['link7_input']['mask_policy'],'old_pipeline_naive_fallback')

    def test_repair_uses_exact_first_gate_frame_and_keeps_original_masks(self):
        obj=np.zeros((3,100,100),bool);obj[:,60:80,60:80]=True
        before=np.zeros_like(obj);before[:,0,0]=True;before[2]|=obj[2]
        frozen=before.copy()
        after=np.zeros_like(obj);after[:,10,[10,20,30]]=True
        class Client:
            def ask(self, images, prompt, **kwargs):
                hashes=[]
                for image in images:
                    b=io.BytesIO();image.save(b,format='PNG');hashes.append(hashlib.sha256(b.getvalue()).hexdigest())
                return {'image_sha256':hashes,'answer':'{"positive_points":[[100,100],[200,100],[300,100]],"negative_points":[[100,300],[200,300],[700,700]]}'}
        with TemporaryDirectory() as tmp:
            p=Path(tmp);video=p/'source.mp4';video.write_bytes(b'source')
            reference=p/'reference.png';Image.new('RGB',(100,100)).save(reference)
            with patch('persistent_masking.vlm3_overmask.read_original_frame',return_value=np.zeros((100,100,3),np.uint8)) as read,patch('persistent_masking.vlm3_overmask.propagate',return_value=(after,{'membership_ok':True,'membership':[True,True,True,False,False,False],'empty_frames':[]})) as sam,patch('persistent_masking.vlm3_overmask.compare_video'):
                r=repair_case('Cosmos25_0005',video,before,obj,{'target_object':'cube'},p/'review',client=Client(),reference=reference)
                self.assertTrue(r['accepted']);self.assertEqual(r['frame'],2)
                self.assertEqual(read.call_args.args[1],2)
                self.assertEqual(sam.call_args.args[1],2)
                np.testing.assert_array_equal(before,frozen)
                self.assertEqual(r['attempts'][0]['seed']['labels'],[1,1,1,0,0,0])

    def test_strict_gate_first_frame_including_zero_and_empty_objects(self):
        obj=np.ones((4,10,10),bool);grip=np.zeros_like(obj)
        for t,n in enumerate((95,96,99,0)):grip[t].flat[:n]=True
        g=audit_gate(grip,obj,include_area_guard=False)
        self.assertEqual(g['frame'],1)
        self.assertEqual(g['overlap_frames'],[1,2])
        obj[0]=False
        self.assertEqual(audit_gate(grip,obj,include_area_guard=False)['frame'],1)
        grip[0]=True;obj[0]=True
        self.assertEqual(audit_gate(grip,obj,include_area_guard=False)['frame'],0)

    def test_existing_area_guard_catches_table_mask_without_object_overlap(self):
        obj=np.zeros((2,10,10),bool);obj[:,9,9]=True
        grip=np.zeros_like(obj);grip[0,:5,:5]=True
        self.assertIsNone(audit_gate(grip,obj,include_area_guard=False)['frame'])
        g=audit_gate(grip,obj)
        self.assertEqual(g['frame'],0)
        self.assertEqual(g['reasons'],['link7-frame-area-v1'])

    def test_ratio_uses_object_area_not_image_area(self):
        obj=np.zeros((1,100,100),bool);obj[0,:5,:5]=True
        grip=obj.copy()
        self.assertEqual(audit_gate(grip,obj)['frame'],0)
        self.assertEqual(audit_gate(grip,obj)['max_object_covered_fraction'],1)

    def test_six_points_require_object_negative_and_exclude_object_positives(self):
        obj=np.zeros((100,100),bool);obj[60:80,60:80]=True
        raw={'answer':'{"positive_points":[[100,100],[200,100],[300,100]],"negative_points":[[100,300],[200,300],[700,700]]}'}
        p=parse_points(raw,100,100,obj)
        self.assertEqual(p['labels'],[1,1,1,0,0,0])
        with self.assertRaisesRegex(ValueError,'positive point'):
            parse_points({'answer':raw['answer'].replace('[100,100]','[650,650]')},100,100,obj)
        with self.assertRaisesRegex(ValueError,'third negative'):
            parse_points({'answer':raw['answer'].replace('[700,700]','[900,900]')},100,100,obj)
        with self.assertRaises(ValueError):
            parse_points({'answer':'{"positive_points":null,"negative_points":null}'},100,100,obj)

    def test_no_trigger_calls_neither_cloud_nor_sam(self):
        before=np.zeros((2,10,10),bool);before[:,0,0]=True
        objects=np.zeros_like(before);objects[:,9,9]=True
        with TemporaryDirectory() as tmp, patch('persistent_masking.vlm3_overmask.digest',return_value='hash'),patch('persistent_masking.vlm3_overmask.VLMClient') as client,patch('persistent_masking.vlm3_overmask.propagate') as sam:
            r=repair_case('Cosmos25_0005','unused-video',before,objects,{'target_object':'cube'},Path(tmp))
            self.assertFalse(r['accepted']);self.assertEqual(r['status'],'not_triggered')
            client.assert_not_called();sam.assert_not_called()

    def test_invalid_grids_and_threshold_rejected(self):
        obj=np.ones((1,5,5),bool)
        with self.assertRaises(ValueError):audit_gate(obj,obj,threshold=1)
        with self.assertRaises(ValueError):audit_gate(obj,obj[:,:,:4])


if __name__=='__main__':unittest.main()
