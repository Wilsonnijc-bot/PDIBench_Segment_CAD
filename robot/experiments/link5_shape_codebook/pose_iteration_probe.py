"""Diagnose native independent-registration iteration sensitivity on failed QC."""
import argparse
from pathlib import Path

import numpy as np

from .audit_alignment import pair_distance
from .common import observed_sample,read,reference_align,write
from .pose import Link5Pose


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);output=Path(config['output'])
    checks=read(output/'sanity/pose_checks.json')
    failed=[r for r in checks['pose_checks'] if not r['passed']]
    if len(failed)!=1:raise ValueError('this diagnostic expects exactly one failing real-pose candidate')
    row=failed[0];case=output/'cases'/row['video_id']/'observations'
    input_path=case/f"frame_{row['frame_id']:05d}"/'input.npz'
    normalization=read(output/'link5_normalization.json')
    with np.load(case/'frame_00000/input.npz') as archive:first=observed_sample(archive['xyz_camera'],2000)[0]
    with np.load(input_path) as archive:
        points=archive['xyz_camera'].copy();k=archive['K'].copy();mask=archive['mask'].copy()
    scratch=output/'metadata/foundationpose-debug/iteration-probe'
    adapter=Link5Pose({**config['pose'],'debug_directory':str(scratch)})
    records=[]
    for iterations in (1,3,5):
        adapter.registration_iterations=iterations;adapter.provenance['registration_iterations']=iterations
        pose=adapter.estimate(input_path,scratch/f'iterations_{iterations}')
        result=dict(iterations=iterations,pose=pose,passed=False)
        if pose['status']=='ok':
            aligned=reference_align(points,normalization['T_ref'],pose['T_camera_from_link5'])
            median,p95=pair_distance(first,observed_sample(aligned,2000)[0])
            result.update(median_distance=median,p95_distance=p95,
                passed=median<=config['sanity']['pose_alignment_median_diagonal']*checks['reference_bbox_diagonal']
                and p95<=config['sanity']['pose_alignment_p95_diagonal']*checks['reference_bbox_diagonal'])
            candidates,ious=adapter.candidate_mask_qc(k,mask)
            rank=int(ious.argmax().item());candidate=candidates[rank].cpu().numpy()
            candidate_aligned=reference_align(points,normalization['T_ref'],candidate)
            candidate_median,candidate_p95=pair_distance(first,observed_sample(candidate_aligned,2000)[0])
            result['best_mask_native_hypothesis']=dict(native_rank=rank,cad_mask_iou=float(ious[rank]),
                native_pose_score=float(adapter.estimator.scores[rank]),T_camera_from_link5=candidate.tolist(),
                median_distance=candidate_median,p95_distance=candidate_p95,
                passed=candidate_median<=config['sanity']['pose_alignment_median_diagonal']*checks['reference_bbox_diagonal']
                and candidate_p95<=config['sanity']['pose_alignment_p95_diagonal']*checks['reference_bbox_diagonal'])
            print('LINK5_NATIVE_CANDIDATE_MASK_QC',iterations,result['best_mask_native_hypothesis'],flush=True)
        records.append(result);print('LINK5_NATIVE_POSE_ITERATION_PROBE',iterations,result['passed'],pose.get('cad_mask_iou'),flush=True)
    write(output/'metadata/pose_iteration_probe.json',dict(video_id=row['video_id'],frame_id=row['frame_id'],
        policy='native register only; independent each call; no ICP/tracking/nonrigid alignment; does not change production setting',records=records))
    print('LINK5_POSE_ITERATION_PROBE_FINISHED',flush=True)


if __name__=='__main__':main()
