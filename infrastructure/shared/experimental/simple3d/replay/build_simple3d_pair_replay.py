"""Export the pair-visible experiment into the existing offline replay style."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = (_workspace_root())
sys.path[:0] = [str(ROOT), str((_workspace_root()))]
from infrastructure.shared.experimental.simple3d.simple3d_pipeline import write_json


def packed(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array,dtype=dtype).tobytes()).decode('ascii')


def build(root):
    root=Path(root).resolve()
    (root/'metadata').mkdir(parents=True,exist_ok=True)
    # The background collector and an interactive refresh may export together.
    # Serialize their temporary asset/link/index writes.
    with (root/'metadata/replay_export.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _build(root)


def _build(root):
    root = Path(root).resolve()
    output = root/'replay'
    output.mkdir(exist_ok=True)
    removed=root/'metadata/trial_clouds_removed.json'
    if removed.exists():
        replacement=root.parent/'link5_pair_visible_anchor/replay/index.html'
        relative=os.path.relpath(replacement,output)
        (output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={relative}"><title>Trial clouds removed</title><a href="{relative}">Open the current point-cloud replay</a>')
        return dict(status='superseded_trial_clouds_removed')
    (output/'cases').mkdir(exist_ok=True)
    manifest = json.loads((root/'inputs/manifest.json').read_text())
    preview_path=root/'metadata/preview.json'
    preview=json.loads(preview_path.read_text()) if preview_path.exists() else None
    preview_active=bool(preview and preview.get('status') not in ('approved_full_run','full_run_complete'))
    if preview_active:
        manifest=dict(manifest,rows=[r for r in manifest['rows'] if r['video_id'] in preview['video_ids']],
                      expected_comparisons=len(preview['video_ids'])*10)
    old_replay = (_workspace_root() / 'results/simple3d-20261003-run2/replay')
    shutil.copy2(old_replay/'plotly.min.js', output/'plotly.min.js')
    all_scores = []; audit = dict(clouds=0, points=0, scores=0, invalid=0, pending=0)

    def recorded(path):
        if not path:
            return None
        p = Path(path)
        if p.exists():
            return p
        marker = '/'+root.name+'/'
        if marker in str(path):
            return root/str(path).split(marker,1)[1]
        return None

    def href(path):
        if not path or not path.is_file():
            return None
        if path.suffix.lower() in ('.png','.jpg','.jpeg'):
            # Safari restricts file:// images outside the opened HTML's folder.
            # Keep display images within the replay, sharing their bytes through
            # hard links where available; scientific source paths stay intact.
            relative=path.resolve().relative_to(root)
            asset=output/'media'/relative
            asset.parent.mkdir(parents=True,exist_ok=True)
            if not asset.exists() or not os.path.samefile(path,asset):
                temporary=asset.with_name(asset.name+'.tmp')
                temporary.unlink(missing_ok=True)
                try:
                    os.link(path,temporary)
                except OSError:
                    shutil.copy2(path,temporary)
                temporary.replace(asset)
            return os.path.relpath(asset,output)
        return os.path.relpath(path,output)

    def cloud(path):
        if not path or not path.is_file():
            return None
        with np.load(path,allow_pickle=False) as z:
            xyz,pixels = z['xyz'],z['source_pixels_xy']
            assert xyz.shape==(len(pixels),3) and np.isfinite(xyz).all() and np.isfinite(pixels).all()
            c = dict(count=len(xyz),xyz=packed(xyz,'<f4'),pixels=packed(pixels,'<f4'),
                rgb=packed(np.round(z['rgb'].clip(0,1)*255),'u1'),source=href(path),
                metadata=json.loads(path.with_suffix('.json').read_text()))
        audit['clouds']+=1; audit['points']+=len(xyz)
        return c

    cases = []
    for row in manifest['rows']:
        video = row['video_id']; case = root/'cases'/video
        # Saved replay source is the hash-identified original video, already local.
        # Export only the requested eleven unannotated source frames when remote
        # processing has not reached this case yet; these are display evidence.
        with np.load(root/'inputs'/row['selected_masks_path'],allow_pickle=False) as z:
            masks = dict(zip(z['frame_ids'].tolist(),z['masks']))
        cap = None
        for t in masks:
            dest = case/'frames'/f'{t:06d}'
            if (dest/'frame.jpg').is_file():
                continue
            if cap is None:
                source = (_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/cases')/video/'v1_cotracker3/replay/interactive_exact-group/source.mp4'
                cap = cv2.VideoCapture(str(source))
            cap.set(cv2.CAP_PROP_POS_FRAMES,t); ok,bgr = cap.read()
            if not ok:
                raise ValueError(f'missing existing source display frame {video}/{t}')
            dest.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(dest/'frame.jpg'),bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
        if cap is not None:
            cap.release()
        pairs = []
        for selected in row['tests']:
            b = selected['temporal_bin']; pair = case/f'bin_{b:02d}'
            evidence = pair/'comparison.json'
            r = json.loads(evidence.read_text()) if evidence.is_file() else dict(simple3d_status='invalid' if selected['status']=='invalid' else 'pending',
                scalar_anomaly_score=None,failure_reason=selected.get('failure_reason'),megasam_status='not_attempted')
            entry = dict(temporal_bin=b,reference_frame_id=selected['reference_frame_id'], test_frame_id=selected['test_frame_id'],
                timestamp_seconds=selected['timestamp_seconds'],reference_timestamp_seconds=selected['reference_timestamp_seconds'],
                raw_mask_area=selected['raw_mask_area'], eroded_mask_area=selected['eroded_mask_area'],
                status=r['simple3d_status'],reason=r.get('failure_reason'),scalar=r['scalar_anomaly_score'],
                geometry_status=r.get('megasam_status'),correspondence=selected.get('correspondence'),
                evidence=href(evidence),depth_filter=r.get('depth_filter'),roles={})
            for role,key in (('reference','reference_frame_id'),('test','test_frame_id')):
                t = selected[key]
                stage = {}; mask_counts = {}
                for name,suffix in [('original','original'),('common','common'),('eroded','eroded'),
                                    ('bootstrap-rejected','bootstrap_rejected'),('rejected','final_rejected'),('valid','final_valid')]:
                    path = pair/f'{role}_{suffix}.png'
                    if not path.is_file():
                        path = root/'inputs'/selected['pair_inputs']/f'{role}_{suffix}.png'
                    # Embedded masks allow offline Safari canvas reads without
                    # changing local-file security preferences.
                    stage[name] = ('data:image/png;base64,'+
                        base64.b64encode(path.read_bytes()).decode('ascii')) if path.is_file() else None
                    mask_counts[name] = int((cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)>127).sum()) if path.is_file() else None
                if selected['status']=='ready':
                    assert all(stage[k] for k in ('original','common','eroded')),f'{video}/{b}/{role}: missing mask stage'
                entry['roles'][role] = dict(frame=href(case/'frames'/f'{t:06d}'/'frame.jpg') if t is not None else None,
                    masks=stage,mask_counts=mask_counts,cloud=cloud(recorded(r.get(f'{role}_point_cloud_path'))),
                    model_input=href(pair/f'{role}_final_input.png'),
                    bootstrap_input=href(pair/f'{role}_bootstrap_input.png'))
            if r['simple3d_status']=='complete':
                s = np.load(recorded(r['point_anomaly_scores_path']),allow_pickle=False)
                assert len(s)==entry['roles']['test']['cloud']['count'] and np.isfinite(s).all()
                top=np.argsort(-s,kind='stable')[:320]
                assert len(top)==320, 'aggregation replay requires at least 320 scored points'
                assert np.isclose(s[top[:80]].mean(),r['scalar_anomaly_score'],rtol=2e-6,atol=1e-6)
                entry.update(scores=packed(s,'<f4'),score_source=href(recorded(r['point_anomaly_scores_path'])),
                    top80=top[:80].tolist(), top_indices={k:top[:k].tolist() for k in (80,160,320)},
                    aggregation_scores={80:r['scalar_anomaly_score'],
                        **{k:float(s[top[:k]].astype(np.float64).mean()) for k in (160,320)}})
                audit['scores']+=1; all_scores.append(s)
            elif r['simple3d_status']=='pending':
                audit['pending']+=1
            else:
                audit['invalid']+=1
            pairs.append(entry)
        script='window.Simple3DPairCases['+json.dumps(video)+']='+json.dumps(dict(video_id=video,pairs=pairs),separators=(',',':'))+';\n'
        dest=output/'cases'/f'{video}.js'
        temporary=dest.with_suffix('.tmp');temporary.write_text(script);temporary.replace(dest)
        cases.append(dict(video_id=video,ready_pairs=sum(s['status']=='ready' for s in row['tests']),
                          successful_scores=sum(s['status']=='complete' for s in pairs)))
    state=dict(cases=cases,config=manifest['config'],expected_pairs=manifest['expected_comparisons'],
               preview=preview,
               counts=audit,heatmap_max=float(np.quantile(np.concatenate(all_scores),.99)) if all_scores else 1,
               earlier_replay=os.path.relpath(old_replay/'index.html',output))
    template=((_workspace_root() / 'infrastructure/shared/experimental/simple3d/replay/simple3d_pair_replay.html')).read_text()
    # Preserve the established dark point-cloud palette/layout.
    old=((_workspace_root() / 'infrastructure/shared/experimental/simple3d/replay/simple3d_replay.html')).read_text()
    css=old.split('<style>',1)[1].split('</style>',1)[0]
    html=template.replace('__CSS__',css).replace('__MANIFEST__',json.dumps(state,separators=(',',':')))
    if audit['scores']==0:
        html=html.replace('<option value="anomaly" selected>', '<option value="anomaly">').replace(
            '<option value="common">','<option value="common" selected>')
    if preview_active:
        html=html.replace('45 existing videos · 10 later-bin tests',f"{len(cases)} preview videos · 10 later-bin tests")
    html=html.replace('eroded by 3 source-image pixels by default',
        f"eroded by {manifest['config']['erosion_px']} source-image pixels")
    if manifest['config'].get('visibility_mode')=='smaller_view_anchor':
        previous=root.parent/'link5_pair_visible/replay/index.html'
        if previous.is_file() and not (root.parent/'link5_pair_visible/metadata/trial_clouds_removed.json').exists():
            html=html.replace('<a id="earlier">Earlier object / CAD replay</a>',
                '<a id="earlier">Earlier object / CAD replay</a> <a href="'+os.path.relpath(previous,output)+'">Previous stricter crop preview</a>')
        html=html.replace('Reliable track hulls prevent extrapolation. Each original-grid mask is intersected with the other’s mapped support.',
            'Track hull coverage checks mapping reliability without cropping the mask to the hull. The view with less visible mask area, corrected for image scale, anchors the crop. Its mask is mapped into the fuller view and clipped to that view’s original Link5 mask; unsupported anchor pixels are removed. There are no repeated hull intersections. This permits extrapolation beyond the tracked hull and can retain errors near mask edges.')
    if manifest['config'].get('track_quality_gate') is False:
        html=html.replace('Track hull coverage checks mapping reliability without cropping the mask to the hull.',
            'Track counts, RANSAC inlier percentages and hull coverage are diagnostics only; they do not reject a pair.')
        html=html.replace('Weak correspondence, insufficient area, excessive depth rejection and unusable reconstructions are explicit invalid pairs.',
            'The previous track-quality gate is disabled. Every selected pair with a computable mask map enters reconstruction and scoring. Empty masks, excessive depth rejection or unusable reconstructions remain explicit failures.')
    temporary=output/'index.html.tmp';temporary.write_text(html);temporary.replace(output/'index.html')
    write_json(output/'metadata.json',state)
    index=(_workspace_root() / 'results/simple3d-20261003-run2/index.html')
    relative=os.path.relpath(output/'index.html',index.parent)
    index.write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={relative}"><title>Simple3D replay</title><a href="{relative}">Updated pair-visible replay</a> · <a href="replay/index.html">Earlier object/CAD evaluation</a>\n')
    print('REPLAY_EXPORTED',json.dumps(audit),flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=(_workspace_root() / 'results/simple3d-20261003-run2/link5_pair_visible_anchor'))
    build(p.parse_args().root)
