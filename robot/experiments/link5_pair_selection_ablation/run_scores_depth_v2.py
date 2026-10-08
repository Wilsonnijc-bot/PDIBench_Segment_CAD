"""Score both methods from shared unfiltered CoTracker/geometry NPZs, five lanes.

NPZ: pointmaps [T,H,W,3], tracks_2d [T,N,2] on that grid,
visibility [T,N] raw tracker confidence, masks [T,H,W] current cached masks.
Companion .json must contain video_id, source_video_sha256, mask_sha256.
GPU preparation is separate and must preserve these raw arrays and query IDs.
"""
import argparse,csv,json,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from .score_depth_v2 import audit_3d_rigidity_cv
from .selectors import METHODS
from .prepare_gpu import sha

def process(job):
    entry,inputs,output,support_root=job;case=entry['video_id'];path=Path(inputs)/(case+'.npz')
    rows=[]
    try:
        receipt=json.loads(path.with_suffix('.json').read_text())
        for key in ('video_id','source_video_sha256','mask_sha256'):
            if receipt[key]!=entry[key]:raise ValueError(f'Provenance mismatch: {key}')
        if receipt.get('raw_sha256') and receipt['raw_sha256']!=sha(path):raise ValueError('Raw array checksum mismatch')
        with np.load(path,allow_pickle=False) as data:
            pm,xy,vis,masks=[data[k] for k in ['pointmaps','tracks_2d','visibility','masks']]
            point_ids=data['point_ids'].copy() if 'point_ids' in data else np.arange(xy.shape[1])
        if pm.ndim!=4 or pm.shape[-1]!=3 or xy.shape!=(len(pm),vis.shape[1],2) or vis.shape!=xy.shape[:2] or masks.shape!=pm.shape[:3]:
            raise ValueError('Array shape/grid mismatch')
        if not all(np.isfinite(a).all() for a in [pm,xy,vis]):raise ValueError('Nonfinite inputs; diagnose, do not silently turn into scores')
        support_path=Path(support_root)/(case+'.npz')
        support_receipt=json.loads(support_path.with_suffix('.json').read_text())
        if support_receipt['raw_input_sha256'] != receipt['raw_sha256'] or support_receipt['support_sha256'] != sha(support_path):
            raise ValueError('Depth-support provenance/checksum mismatch')
        with np.load(support_path,allow_pickle=False) as support_archive:
            depth_support=support_archive['query_support'];frame_qc=support_archive['frame_qc_passed']
            if not np.array_equal(support_archive['point_ids'],point_ids):raise ValueError('Depth-support point ID mismatch')
        if depth_support.shape!=vis.shape:raise ValueError('Depth-support query grid mismatch')
        for method in METHODS:
            folder=Path(output)/case/method;folder.mkdir(parents=True,exist_ok=True)
            evidence={}
            try:
                score,history=audit_3d_rigidity_cv(pm,xy,vis,masks,insufficient_policy='raise',pair_method=method,evidence=evidence,depth_support=depth_support)
                pairs=evidence['selected_pairs'];pi=np.array([p['track_i'] for p in pairs]);pj=np.array([p['track_j'] for p in pairs])
                u=np.clip(np.rint(xy[...,0]).astype(int),0,pm.shape[2]-1);v=np.clip(np.rint(xy[...,1]).astype(int),0,pm.shape[1]-1)
                xyz=pm[np.arange(len(pm))[:,None],v,u]
                distances=np.linalg.norm(xyz[:,pi]-xyz[:,pj],axis=-1)
                raw_pair_visible=(vis[:,pi]>.5)&(vis[:,pj]>.5)
                available=raw_pair_visible & depth_support[:,pi] & depth_support[:,pj]
                evidence['query_ids']=point_ids.tolist()
                evidence['depth_filter_provenance']=support_receipt
                for pair in pairs:
                    pair.update(point_id_i=int(point_ids[pair['track_i']]),point_id_j=int(point_ids[pair['track_j']]))
                evidence['frame_diagnostics']=[dict(frame=t,display_frame=t+1,score=float(history[t]),
                    selected_pair_count=len(pairs),available_pair_count=int(available[t].sum()),
                    raw_visible_point_count=int((vis[t]>.5).sum()),depth_supported_point_count=int(depth_support[t].sum()),
                    depth_and_visible_point_count=int(((vis[t]>.5)&depth_support[t]).sum()),raw_visible_pair_count=int(raw_pair_visible[t].sum()),
                    depth_support_qc_passed=bool(frame_qc[t]),finite_sampled_3d_point_count=int(np.isfinite(xyz[t]).all(axis=-1).sum()),
                    carried=t in evidence['carried_frames']) for t in range(len(pm))]
                np.savez(folder/'trajectories.npz',point_ids=point_ids,tracks_2d=xy,sampled_world_xyz=xyz,
                    raw_visibility=vis,pair_track_indices=np.c_[pi,pj],pair_point_ids=np.c_[point_ids[pi],point_ids[pj]],
                    baseline_distances=np.array([p['baseline_distance'] for p in pairs]),distance_ratios=distances/np.array([p['baseline_distance'] for p in pairs]),
                    pair_available=available,raw_pair_visible=raw_pair_visible,query_depth_support=depth_support)
                result=dict(case=case,cohort=entry['cohort'],method=method,status='complete',depth_filter='refined_all_frames',rigidity_score=score,pairs=len(evidence['selected_pairs']),carried_frames=len(evidence['carried_frames']),**{k:receipt[k] for k in ['source_video_sha256','mask_sha256']})
                (folder/'evidence.json').write_text(json.dumps(evidence,indent=2));np.save(folder/'history.npy',history)
            except Exception as exc:result=dict(case=case,cohort=entry['cohort'],method=method,status='failed',error=str(exc))
            (folder/'result.json').write_text(json.dumps(result,indent=2));rows.append(result)
    except Exception as exc:
        rows=[dict(case=case,cohort=entry['cohort'],method=m,status='failed',error=str(exc)) for m in METHODS]
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--workers',type=int,default=5);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:rows=[r for group in pool.map(process,[(e,str(a.inputs),str(a.output),str(a.support)) for e in manifest['entries']]) for r in group]
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2))
    keys=sorted({k for row in rows for k in row})
    with (a.output/'rigidity_scores.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print(f'{sum(r["status"]=="complete" for r in rows)}/{len(rows)} scores complete')
    if any(r['status']!='complete' for r in rows):raise SystemExit(1)
if __name__=='__main__':main()
