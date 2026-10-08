"""Create an independent rigidity comparison and synchronized evidence viewer."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from .prepare_gpu import sha, write
from .selectors import METHODS


def build(root, sources):
    manifest=json.loads((root/'manifest.json').read_text())
    scores=json.loads((root/'rigidity/summary.json').read_text())
    by_key={(r['case'],r['method']):r for r in scores}
    videos=[];paired=[];diagnostics=[];graph_comparisons=[]
    viewer=root/'report';viewer.mkdir(exist_ok=True)
    playback_receipt=root/'metadata/playback_provenance.json'
    playback_by_case={r['case']:r for r in json.loads(playback_receipt.read_text())['files']} if playback_receipt.exists() else {}
    playback_receipt=root/'metadata/playback_provenance.json'
    playback_by_case={r['case']:r for r in json.loads(playback_receipt.read_text())['files']} if playback_receipt.exists() else {}
    for entry in manifest['entries']:
        case=entry['video_id'];record=dict(case=case,cohort=entry['cohort'],methods={})
        pairrow=dict(video_id=case,cohort=entry['cohort'])
        for method in METHODS:
            result=by_key.get((case,method),dict(status='failed',error='Missing score row'))
            pairrow[method+'_status']=result['status'];pairrow[method+'_score']=result.get('rigidity_score')
            pairrow[method+'_pairs']=result.get('pairs');pairrow[method+'_carried_frames']=result.get('carried_frames')
            if result['status']!='complete':
                record['methods'][method]=result;continue
            folder=root/'rigidity'/case/method
            evidence=json.loads((folder/'evidence.json').read_text())
            with np.load(folder/'trajectories.npz',allow_pickle=False) as archive:
                xy=archive['tracks_2d'];vis=archive['raw_visibility'];ids=archive['point_ids'];indices=archive['pair_track_indices'];depth_supported=archive['query_depth_support']
            raw=root/'shared_inputs'/(case+'.json')
            receipt=json.loads(raw.read_text())
            h,w=receipt['tracker_pointmap_hw']
            record.update(fps=receipt['fps'],frame_count=len(xy),grid_hw=[h,w])
            record['methods'][method]=dict(result=result,history=evidence['rigidity_history'],frames=evidence['frame_diagnostics'],
                coverage=evidence['selection_stats'],pairs=evidence['selected_pairs'],geometric_selection=evidence['geometric_selection'],
                query_ids=ids.tolist(),xy=np.round(xy.astype(float),3).tolist(),visible=((vis>.5)&depth_supported).tolist(),depth_supported=depth_supported.tolist(),pair_indices=indices.tolist())
            if case in ('COSMOS3_0010','COSMOS3_0015','LVP_ROBOWM_0015','COSMOS2.5_0015'):
                center={'COSMOS3_0010':101,'COSMOS3_0015':73}.get(case,int(np.argmax(evidence['rigidity_history'])))
                for frame in range(max(0,center-2),min(len(xy),center+3)):
                    diag=evidence['frame_diagnostics'][frame]
                    diagnostics.append(dict(video_id=case,method=method,**diag,center_frame_zero_based=center))
        if all(pairrow[m+'_status']=='complete' for m in METHODS):
            pairrow['refine_minus_balanced']=pairrow['refine_v1_score']-pairrow['balanced_v0_score']
            a,b=[record['methods'][m] for m in METHODS]
            sets=[{tuple(sorted((p['point_id_i'],p['point_id_j']))) for p in d['pairs']} for d in (a,b)]
            endpoints=[{i for pair in pairs for i in pair} for pairs in sets]
            pairrow['shared_pairs']=len(sets[0]&sets[1]);pairrow['shared_endpoints']=len(endpoints[0]&endpoints[1])
            pairrow['availability_difference_frames']=sum(x['available_pair_count']!=y['available_pair_count'] for x,y in zip(a['frames'][1:],b['frames'][1:]))
            pairrow['carry_difference_frames']=sum(x['carried']!=y['carried'] for x,y in zip(a['frames'][1:],b['frames'][1:]))
            comparison=dict(video_id=case,cohort=entry['cohort'],score_difference=pairrow['refine_minus_balanced'],
                shared_pairs=pairrow['shared_pairs'],shared_endpoints=pairrow['shared_endpoints'],
                availability_difference_frames=pairrow['availability_difference_frames'],carry_difference_frames=pairrow['carry_difference_frames'],
                methods={m:dict(selected_pair_count=len(record['methods'][m]['pairs']),
                    mean_available_pairs=float(np.mean([f['available_pair_count'] for f in record['methods'][m]['frames'][1:]])),
                    carried_frames=record['methods'][m]['result']['carried_frames'],
                    quota_deficits=record['methods'][m]['coverage'].get('deficits',{}),
                    geometry_flags=record['methods'][m]['coverage'].get('geometry_flags',[])) for m in METHODS},
                attribution='Identical raw inputs and gate; differences arise from graph selection and its visibility/carry consequences. Depth or tracking error versus real deformation requires visual assessment.')
            graph_comparisons.append(comparison)
        paired.append(pairrow)
        if 'fps' in record:
            generator,number=case.rsplit('_',1)
            source=sources/('LVP' if generator=='LVP_ROBOWM' else generator)/(number+'.mp4')
            if sha(source)!=entry['source_video_sha256']:raise ValueError('Viewer source mismatch: '+case)
            localvideo=viewer/'videos'/(case+'.mp4');localvideo.parent.mkdir(exist_ok=True)
            if not localvideo.exists():shutil.copy2(source,localvideo)
            elif sha(localvideo)!=entry['source_video_sha256']:raise ValueError('Existing viewer source mismatch')
            record['video']='videos/'+case+'.mp4'
            if case in playback_by_case:
                display_copy=viewer/'playback'/(case+'.mp4')
                provenance=playback_by_case[case]
                if provenance['source_sha256']!=entry['source_video_sha256'] or sha(display_copy)!=provenance['playback_sha256']:
                    raise ValueError('Playback provenance mismatch: '+case)
                record['original_video']=record['video']
                record['video']='playback/'+case+'.mp4'
            if case in playback_by_case:
                display_copy=viewer/'playback'/(case+'.mp4')
                provenance=playback_by_case[case]
                if provenance['source_sha256']!=entry['source_video_sha256'] or sha(display_copy)!=provenance['playback_sha256']:
                    raise ValueError('Playback provenance mismatch: '+case)
                record['original_video']=record['video']
                record['video']='playback/'+case+'.mp4'
            cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();cap.release()
            if not ok:raise ValueError('Cannot decode frame0 for overlay: '+case)
            h,w=record['grid_hw'];frame=cv2.resize(bgr,(w,h))
            for method in METHODS:
                data=record['methods'][method]
                if 'xy' not in data:continue
                overlay=frame.copy();coords=np.rint(data['xy'][0]).astype(int)
                for i,j in data['pair_indices']:
                    cv2.line(overlay,tuple(coords[i]),tuple(coords[j]),(55,210,255) if method=='balanced_v0' else (210,100,225),2,cv2.LINE_AA)
                for i in np.unique(data['pair_indices']):
                    cv2.circle(overlay,tuple(coords[i]),3,(255,255,255),-1)
                    cv2.putText(overlay,str(data['query_ids'][i]),tuple(coords[i]+[4,-4]),cv2.FONT_HERSHEY_SIMPLEX,.3,(255,255,255),1,cv2.LINE_AA)
                destination=root/'rigidity'/case/method/'frame0_actual_graph.png'
                if not cv2.imwrite(str(destination),overlay):raise RuntimeError('Overlay write failed')
        videos.append(record)
    keys=list(dict.fromkeys(k for row in paired for k in row))
    with (root/'rigidity/paired_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(paired)
    write(root/'rigidity/paired_scores.json',paired)
    write(root/'rigidity/neighboring_frame_diagnostics.json',diagnostics)
    write(root/'rigidity/graph_comparison.json',graph_comparisons)
    summary=dict(primary_videos=manifest['primary_count'],additional_videos=manifest['additional_count'],
        successful_method_video_scores=sum(r['status']=='complete' for r in scores),expected_method_video_scores=2*len(manifest['entries']),
        excluded=manifest['excluded'],published=False,metric='Mean per-frame MAD(distance ratio)/(median(distance ratio)+1e-6), excluding frame0',
        depth_filter='refined ray-normalized 32NN density on all frames',interpretation='Paired graph ablation with shared refined depth support;  no detection improvement inferred without labeled full-video validation.',
        cohorts={cohort:{method:dict(successful=sum(r['status']=='complete' for r in scores if r['cohort']==cohort and r['method']==method),
            failed=sum(r['status']!='complete' for r in scores if r['cohort']==cohort and r['method']==method),
            mean_score=float(np.mean([r['rigidity_score'] for r in scores if r['cohort']==cohort and r['method']==method and r['status']=='complete']))
            if any(r['status']=='complete' for r in scores if r['cohort']==cohort and r['method']==method) else None)
            for method in METHODS} for cohort in ('selected45','additional')})
    write(root/'rigidity/comparison_summary.json',summary)
    (root/'REPORT.md').write_text('\n'.join([
        '# Link5 pair-selection rigidity comparison','',
        f"{summary['successful_method_video_scores']}/{summary['expected_method_video_scores']} method/video scores complete; {manifest['primary_count']} primary videos and {manifest['additional_count']} additional video.",'',
        'Metric: '+summary['metric']+'. Both methods use identical cached masks, saved frame-zero query IDs, raw CoTracker visibility/tracks and full-sequence CVD world pointmaps. Both methods use the same refined ray-normalized 32-neighbor density filter on every frame, including two source-pixel mask erosion. Frame-zero eligibility is depth support plus raw visibility, without the old gradient/fallback gate. Tiny-baseline rejection, fewer-than-three-pair carry and temporal mean are unchanged.','',
        'The five explicit handoff exclusions include COSMOS2.5_0010, so the older 41-primary count is superseded by 40. Its old artifacts remain preserved; no new diagnostic score is invented for the excluded video.','',
        'Actual graphs may differ from CPU source-resolution previews because native-grid resizing and the common frame-zero depth-support/visibility gate affect eligibility. Quota deficits are retained.','',
        'Interpretation: graph choice changes which spatial deformations and reconstruction/tracking errors enter MAD. More pairs alone do not establish detection improvement. A coherent scale change produces nearly equal distance ratios and can be suppressed by MAD; minority pair changes can also be suppressed. Occlusion can carry a previous score. The unchanged temporal mean can dilute brief events.','',
        'Inspect neighboring_frame_diagnostics.json and the synchronized viewer around zero-based101/display102 for COSMOS3_0010 and zero-based73/display74 for COSMOS3_0015. For the other two guide videos, automatically centered neighborhoods are score peaks, not manually labeled deformation onset.','',
        'Per-method evidence includes coverage/geometry flags, actual point IDs/baselines, frame diagnostics, distance-ratio trajectories and selected graph PNGs. Finite 3D samples are numerical availability, separate from raw tracker visibility and filtered depth support; finite depth can still be geometrically wrong.','',
        'Sources, model/checkpoint hashes, preprocessing, query coordinate transforms, seeds and environment are in shared_inputs/*.json. Original high-precision arrays and trajectories are preserved; viewer coordinate rounding is display-only.','',
        'Viewer: report/index.html. Tables: rigidity/paired_scores.csv and rigidity/rigidity_scores.csv. This report is local and has not been published.','']))
    from .export_replay_context import run as export_context
    from .build_previous_replay import build as build_replay
    from .prepare_playback import prepare as prepare_playback
    if not (root/'replay_context/completion.json').exists():
        export_context(root, 4)
    if not (root/'metadata/playback_provenance.json').exists():
        prepare_playback(root)
    build_replay(root)
    print('RIGIDITY_REPORT_COMPLETE',summary['successful_method_video_scores'],summary['expected_method_video_scores'],flush=True)




def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--sources',type=Path,required=True)
    args=parser.parse_args();build(args.root,args.sources)


if __name__=='__main__':main()
