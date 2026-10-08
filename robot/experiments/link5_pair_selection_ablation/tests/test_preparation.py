import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot.experiments.link5_pair_selection_ablation.prepare_gpu import sha, scale_queries, validate_sources


class PreparationContracts(unittest.TestCase):
    def test_anisotropic_resize_preserves_query_order_and_zero_time(self):
        xy=np.array([[100.,200.],[5.,8.]])
        queries,scale=scale_queries(xy,(400,1000),(100,300))
        np.testing.assert_allclose(scale,[.3,.25])
        np.testing.assert_allclose(queries,[[0,30,50],[0,1.5,2]])

    def test_hashes_use_selected_attempt_and_reject_query_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);case='LVP_ROBOWM_0001'
            video=root/'videos/LVP/0001.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'video')
            mask=root/'cache/cases/attempt-0003/segmentation.npz';mask.parent.mkdir(parents=True);mask.write_bytes(b'selected mask')
            init=root/'queries'/case/'pairs.json';init.parent.mkdir(parents=True)
            entry=dict(video_id=case,mask='cases/attempt-0003/segmentation.npz',source_video_sha256=sha(video),mask_sha256=sha(mask))
            data=dict(source_video_sha256=sha(video),mask_sha256=sha(mask),point_ids=[9,4,7,2,15],query_points_xy=[[1,2]]*5)
            init.write_text(json.dumps(data))
            _,selected,_,_,ids=validate_sources(entry,root/'cache',root/'videos',root/'queries')
            self.assertEqual(selected,mask);self.assertEqual(ids.tolist(),data['point_ids'])
            data['mask_sha256']='wrong';init.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                validate_sources(entry,root/'cache',root/'videos',root/'queries')
