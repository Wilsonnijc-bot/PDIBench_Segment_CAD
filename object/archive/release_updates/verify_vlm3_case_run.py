#!/usr/bin/env python3
"""Verify GPU masks, synchronized audits/replays and exact selected crop bytes."""
from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/verify_vlm3_case_run.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break

import argparse,hashlib,json,re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pdi_eval.object_deformation_wrapper.frame_selection import select_frames
from pdi_eval.object_deformation_wrapper.reference_visible_pixels import mask_bbox, rgba_crop
from pdi_eval.object_deformation_wrapper.paired_crops import normalize_pair

ROOT=_SOURCE_PATH.parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run: Path):
    run=run.resolve();remote='/root/autodl-tmp/pdi/experiments/'+run.name
    def relocate(path):
        if str(path).startswith(remote+'/'):return run/str(path)[len(remote)+1:]
        p=Path(path)
        if p.is_file():return p
        raise FileNotFoundError(path)
    hashes=json.loads((run/'metadata/artifact_hashes.json').read_text())
    for relative,expected in hashes.items():assert digest(run/relative)==expected,relative
    assert (run/'metadata/run.exit').read_text().strip()=='0'
    assert (run/'metadata/finalize_score.exit').read_text().strip()=='0'
    assert 'SEVEN_CASES_PIPELINE_COMPLETE' in (run/'metadata/run.log').read_text()
    assert 'SEVEN_EXPORTS_AND_SCORES_COMPLETE' in (run/'metadata/finalize_score.log').read_text()
    summary=json.loads((run/'summary.json').read_text());assert summary['cases']==7
    cases=[r['case'] for r in json.loads((run/'metadata/inputs.json').read_text())]
    gpu_pixels=json.loads((run/'metadata/pixel_verification.json').read_text())
    assert gpu_pixels['status']=='passed' and gpu_pixels['exact_source_crop_pairs']==70 and gpu_pixels['exact_vlm3_targets']==7
    checked=[];pairs=0;video_frames=0
    baseline=ROOT/'results/object-deformation-selected45-20260929/cases'
    for name in cases:
        repair_path=run/name/'repair.json';repair=json.loads(repair_path.read_text())
        case=run/'inputs/object/cases'/name
        for relative in ['masking/segmentation.npz','masking/grounding.json','replay/source.mp4','score/cotracker_exact-group.npz','score/rigidity.json']:
            assert digest(case/relative)==digest(baseline/name/relative),(name,relative)
        video=case/'replay/source.mp4';assert digest(video)==repair['source_video_sha256']
        assert digest(case/'masking/segmentation.npz')==repair['object_input']['sha256']
        source=relocate(repair['link7_input']['path']);assert digest(source)==repair['link7_input']['sha256']
        with np.load(source,allow_pickle=False) as a:before=a['object_masks'].astype(bool);names=a['object_names'].tolist()
        selected=run/'downstream/gripper/cases'/name/'v1_cotracker3/segmentation.npz'
        with np.load(selected,allow_pickle=False) as a:after=a['object_masks'].astype(bool)
        target=names.index('link7');others=[i for i in range(len(names)) if i!=target]
        np.testing.assert_array_equal(before[:,others],after[:,others])
        if repair['accepted']:
            masks=relocate(repair['output_masks']);assert digest(masks)==repair['output_masks_sha256']
            with np.load(masks,allow_pickle=False) as a:np.testing.assert_array_equal(after[:,target],a['masks'])
            attempt=repair['attempts'][repair['selected_attempt']-1]
            assert attempt['sam']['membership']==[True,True,True,False,False,False]
            assert not attempt['sam']['empty_frames']
        detection_path=case/'occlusion/detection.json';d=json.loads(detection_path.read_text())
        assert d['inputs']['gripper']['sha256']==digest(selected)
        for item in d['inputs'].values():assert digest(relocate(item['path']))==item['sha256']
        html=(case/'occlusion/replay.html').read_text()
        replay=json.loads(re.search(r'const D=(.*?);\nconst \$',html,re.S)[1])
        assert replay['inputs']==d['inputs'] and replay['frames']==d['frames']
        assert len(replay['overlays'])==d['frame_count']
        assert replay['comparison'] is None
        assert replay['gripper_mask_method']==('vlm3-mask-link-replacement' if repair['accepted'] else 'original-mask-retained')
        assert (case/'replay/plotly.min.js').is_file()
        crop=run/'crops'/name;manifest=json.loads((crop/'manifest.json').read_text())
        selection=json.loads((crop/'selection.json').read_text())
        assert manifest['occlusion_detection_sha256']==digest(detection_path)
        assert manifest['link7_segmentation_sha256']==digest(selected)
        assert selection['crop_manifest_sha256']==digest(crop/'manifest.json')
        calculated=select_frames(manifest,d,10)
        assert [r['frame'] for r in selection['selected_frames']]==[r['frame'] for r in calculated['selected_frames']]
        assert selection['required_frames']==calculated['required_frames']
        assert selection['interval_quotas']==[0,2,2,2,4]
        with np.load(case/'masking/segmentation.npz',allow_pickle=False) as a:objects=a['object_masks'][:,0].astype(bool)
        with np.load(crop/'masks.npz',allow_pickle=False) as a:
            available=a['available_pure_object'];mapped=a['mapped_to_frame0'];valid=a['mask_valid'];bbox=tuple(a['frame0_bbox_xyxy'])
        expected=objects&~after[:,target];expected[~valid]=False
        np.testing.assert_array_equal(available,expected)
        # GPU decoding is checked exactly in pixel_verification.json. Use saved
        # original PNGs here to avoid platform-specific FFmpeg color rounding.
        reference_rgb=np.asarray(Image.open(crop/'frame0_reference.png').convert('RGB'))
        for row in selection['selected_frames']:
            t=row['frame'];directory=crop/row['crop_directory']
            rgb=np.asarray(Image.open(directory/'original_frame.png').convert('RGB'))
            expected_current = rgba_crop(rgb,available[t],mask_bbox(available[t]))
            x0,y0,x1,y1=bbox
            expected_reference = np.dstack((reference_rgb,mapped[t].astype(np.uint8)*255))
            if (crop/'pair_geometry.json').is_file():
                expected_current, expected_reference, _ = normalize_pair(expected_current, expected_reference)
            np.testing.assert_array_equal(np.asarray(Image.open(directory/'current_available.png')), expected_current)
            np.testing.assert_array_equal(np.asarray(Image.open(directory/'frame0_shape_crop.png')), expected_reference)
            assert t not in selection['final_interval_policy']['excluded_frames']
            if repair['accepted']:assert t not in repair['crop_excluded_frames']
            pairs+=1
        comparison=run/name/'comparison.mp4'
        if comparison.exists():
            cap=cv2.VideoCapture(str(comparison));count=0
            while True:
                ok,_=cap.read()
                if not ok:break
                count+=1
            cap.release();assert count==len(objects),(name,count,len(objects));video_frames+=count
        checked.append(dict(case=name,vlm3_frame=repair['gate']['frame'],status=repair['status'],
            selected_frames=[r['frame'] for r in selection['selected_frames']],
            occlusion_intervals=d['flagged_intervals'],frame0_reference_quality=manifest['frame0_reference_quality'],
            same_masks_without_late_gate=[r['frame'] for r in select_frames(manifest,d,10,last_interval_min_area_ratio=0)['selected_frames']],
            late_gate_policy=selection['final_interval_policy']))
    result=dict(status='passed',verified_remote_files=len(hashes),verified_crop_pairs=pairs,
                decoded_comparison_frames=video_frames,exact_gpu_pixel_verification=gpu_pixels,cases=checked)
    (run/'metadata/verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);verify(p.parse_args().run)
