"""Compare immediate SAM masks with three positives, with/without negatives."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    from persistent_masking.v1_mask import predictor
    manifest=json.loads((out/'manifest.json').read_text());correction=json.loads((out/'correction.json').read_text())
    if correction['status']!='ready_for_sam':raise ValueError('No well-formed three-positive proposal')
    points=np.asarray(correction['proposal']['points_xy']);labels=np.asarray(correction['proposal']['labels'])
    masks=np.load(manifest['masks'])['masks'];f=correction['frame'];h,w=masks.shape[1:]
    p=predictor();results={};report={}
    try:
        for variant,keep in [('positives_only',labels==1),('with_negatives',np.ones(len(labels),bool))]:
            sid=p.handle_request(dict(type='start_session',resource_path=manifest['video'],offload_video_to_cpu=True))['session_id']
            try:
                d=p.handle_request(dict(type='add_prompt',session_id=sid,frame_index=f,obj_id=0,
                    points=(points[keep]/[w,h]).tolist(),point_labels=labels[keep].tolist()))['outputs']
                ids=list(d['out_obj_ids']);mask=np.zeros((h,w),bool)
                if 0 in ids:mask=np.asarray(d['out_binary_masks'][ids.index(0)],bool)
                xy=np.floor(points).astype(int);membership=mask[xy[:,1],xy[:,0]]
                results[variant]=mask
                report[variant]=dict(area=int(mask.sum()),added=int((mask&~masks[f]).sum()),
                    removed=int((masks[f]&~mask).sum()),membership_at_all_prompts=membership.tolist(),
                    supplied_labels=labels[keep].tolist(),supplied_prompt_membership_ok=bool(np.array_equal(membership[keep],labels[keep].astype(bool))))
            finally:p.handle_request(dict(type='close_session',session_id=sid))
    finally:p.shutdown()
    np.savez_compressed(out/'seed_masks.npz',**results,original=masks[f],frame=f)
    (out/'seed_ablation.json').write_text(json.dumps(report,indent=2))
    print(report)


if __name__=='__main__':main()
