"""Audit native saved depths, pair masks, normalization and top80 evidence."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = (_workspace_root())
sys.path[:0] = [str(ROOT), str((_workspace_root()))]
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import sha, write_json


def verify(root):
    root=Path(root)
    checked=0; hashes={}
    for evidence in sorted(root.glob('cases/*/bin_*/comparison.json')):
        row=json.loads(evidence.read_text());pair=evidence.parent
        if row['simple3d_status']!='complete':
            assert row['scalar_anomaly_score'] is None and row['failure_reason'], str(evidence)
            continue
        scores=np.load(pair/'point_scores.npy',allow_pickle=False)
        top=np.load(pair/'top80.npz',allow_pickle=False)
        assert len(top['point_indices'])==80 and len(np.unique(top['point_indices']))==80
        np.testing.assert_array_equal(scores[top['point_indices']],top['scores'])
        assert np.isclose(top['scores'].mean(),row['scalar_anomaly_score'],rtol=2e-6,atol=1e-6)
        assert np.isfinite(scores).all() and scores.dtype==np.float32
        bootstrap=json.loads((pair/'bootstrap_geometry.json').read_text())
        final=json.loads((pair/'final_geometry.json').read_text())
        assert bootstrap['scene_name']!=final['scene_name'] and not bootstrap['cache_hit'] and not final['cache_hit']
        assert row['reference_memory_rebuilt'] and bootstrap['status']==final['status']=='complete'
        # Freshness comes from independent native scenes and cache_hit=False.
        # With no source-grid rejection the input legitimately remains identical;
        # lossy video encoding may also erase a tiny changed pixel. A hash change
        # is not a valid test of whether a fresh reconstruction actually ran.
        with np.load(pair/'final_reconstruction.npz',allow_pickle=False) as z:
            depths,poses,k=z['depths'],z['cam_c2w'],z['intrinsic']
        if k.ndim==1:fx,fy,cx,cy=k[:4]
        else:fx,fy,cx,cy=k[0,0],k[1,1],k[0,2],k[1,2]
        for index,role in enumerate(('reference','test')):
            path=pair/f'{role}_cloud.npz'
            with np.load(path,allow_pickle=False) as z:
                xyz,raw,pixels=z['xyz'],z['raw_xyz'],z['pixels_yx']
                source_pixels=z['source_pixels_xy']
            info=json.loads(path.with_suffix('.json').read_text())
            assert xyz.shape==raw.shape==(len(pixels),3) and len(xyz)>=128
            assert np.isfinite(xyz).all() and np.isfinite(raw).all()
            np.testing.assert_allclose((raw-np.array(info['centroid']))/info['scale_divisor'],xyz,rtol=5e-4,atol=1e-4)
            valid=cv2.imread(str(pair/f'{role}_final_valid_grid.png'),cv2.IMREAD_GRAYSCALE)>0
            yy,xx=pixels.T;assert valid[yy,xx].all()
            depth=depths[index][yy,xx].astype(np.float32)
            camera=np.column_stack(((xx-cx)*depth/fx,(yy-cy)*depth/fy,depth))
            world=camera@poses[index,:3,:3].T+poses[index,:3,3]
            assert (depth>0).all();np.testing.assert_allclose(world,raw,rtol=1e-5,atol=1e-5)
            h,w=valid.shape;ih,iw=info['input_canvas_hw'];x0,y0,_,_=info['crop_bbox_xyxy']
            np.testing.assert_allclose(source_pixels,np.column_stack(((xx+.5)*iw/w+x0-.5,(yy+.5)*ih/h+y0-.5)),atol=5e-5)
            for stage in ('bootstrap','final'):
                rgb=cv2.imread(str(pair/f'{role}_{stage}_input.png'))
                mask=cv2.imread(str(pair/f'{role}_{stage}_input_mask.png'),cv2.IMREAD_GRAYSCALE)>0
                assert not rgb[~mask].any()
                stage_depth=json.loads((pair/f'{stage}_geometry.json').read_text())['depth_filter'][index]
                source_rejected=cv2.imread(str(pair/f'{role}_{stage}_rejected.png'),cv2.IMREAD_GRAYSCALE)>0
                if stage_depth['rejected_depth_pixel_count']==0:
                    assert not source_rejected.any(), 'resampling loss falsely marked as depth rejection'
            original=cv2.imread(str(pair/f'{role}_original.png'),cv2.IMREAD_GRAYSCALE)>0
            common=cv2.imread(str(pair/f'{role}_common.png'),cv2.IMREAD_GRAYSCALE)>0
            eroded=cv2.imread(str(pair/f'{role}_eroded.png'),cv2.IMREAD_GRAYSCALE)>0
            assert (common<=original).all() and (eroded<=common).all()
            hashes[str(path.relative_to(root))]=sha(path)
            if role=='test':assert len(scores)==len(xyz)
        with np.load(pair/'reference_prototypes.npz',allow_pickle=False) as z:
            assert len(z['features'])==204 and np.isfinite(z['features']).all()
        checked+=1
    result=dict(status='passed' if checked else 'no_successes_yet',verified_successful_pairs=checked,
        cloud_sha256=hashes,checks=['independent bootstrap/final scenes with no cache',
            'foreground-only RGB inputs','original/common/eroded mask containment',
            'native depth/intrinsics/pose back-projection','native-grid final-valid membership',
            'source-image crop/resize pixel coordinates','official cloud normalization',
            'new 204-prototype reference coreset','continuous point scores and exact top80 scalar'])
    write_json(root/'metadata/artifact_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=(_workspace_root() / 'results/simple3d-20261003-run2/link5_pair_visible_anchor'))
    print(json.dumps(verify(p.parse_args().root),indent=2))
