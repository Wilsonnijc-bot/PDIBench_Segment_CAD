"""Re-filter saved MegaSAM observations on CPU and export an exact local replay."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .common import ROOT, normalize, read, reference_align, sha, write
from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth, settings, xyz_from_depth
from .refine_guard_local import successful_masking
from .replay_framework import render as render_existing_replay


def packed(array,dtype):
    value=np.ascontiguousarray(array,dtype=dtype)
    encoded=base64.b64encode(value.tobytes()).decode('ascii')
    if not np.array_equal(np.frombuffer(base64.b64decode(encoded),dtype=dtype).reshape(value.shape),value):
        raise ValueError('replay binary roundtrip failed')
    return encoded


def run(root,destination=None,workers=4):
    root=root.resolve();destination=(destination or root/'refinement').resolve();replay=destination/'replay'
    replay.mkdir(parents=True,exist_ok=True)
    train=[r['video_id'] for r in read(root/'training/train20_manifest.json')];train_ids=set(train)
    scope=set(read(root/'full_video/metadata/evaluation_scope.json')['selected_video_ids'])
    normalization=read(root/'link5_normalization.json');cfg=Config().validate()
    def href(path):return os.path.relpath(path,replay)
    cases=[];tasks=[]
    for case in read(root/'config.json')['cases']:
        video=case['id'];folders=[root/'cases'/video/'observations/frame_00000']
        dense=root/'full_video/cases'/video
        if video in scope:
            info=read(dense/'preparation.json')
            folders.extend(dense/'observations'/f'frame_{i:05d}' for i in range(1,info['frame_count']))
        rows=[]
        for folder in folders:
            fid=int(folder.name.removeprefix('frame_'))
            row=dict(frame_id=fid,packet=None,status='missing saved observation')
            if (folder/'input.npz').exists():tasks.append((video,folder,row))
            rows.append(row)
        cases.append(dict(video_id=video,training_reference=video in train_ids,full_video=video in scope,frames=rows))
    receipts=[]
    def export(task):
        video,folder,row=task;fid=row['frame_id'];source=folder/'input.npz'
        with np.load(source,allow_pickle=False) as a:
            depth=a['depth'].copy();rgb=a['rgb'].copy();k=a['K'].copy();mask=a['source_mask'].copy();old_valid=a['valid'].copy()
            old_xyz=a['xyz_camera'].copy();old_pixels=a['pixels_yx'].copy()
        support=filter_depth(depth,mask,k,cfg);valid=support['valid'];info=support['info']
        old_reconstructed,_=xyz_from_depth(depth,old_valid,k)
        if not np.array_equal(old_xyz,old_reconstructed):raise ValueError('saved camera cloud differs from its depth backprojection')
        candidates=support['mask']&np.isfinite(depth)&(depth>0);camera,pixels=xyz_from_depth(depth,candidates,k)
        reasons=support['rejection_reason'][pixels[:,0],pixels[:,1]]
        old=old_valid[pixels[:,0],pixels[:,1]].astype(np.uint8)
        ids=np.arange(len(camera)) if len(camera)<=10000 else np.sort(np.random.default_rng(0).choice(len(camera),10000,replace=False))
        pose_path=folder/'pose/pose.json';pose=read(pose_path) if pose_path.exists() else None
        aligned=normalize(camera[ids],normalization) if fid==0 else None
        if fid and pose and pose['status']=='ok':aligned=normalize(reference_align(camera[ids],normalization['T_ref'],pose['T_camera_from_link5']),normalization)
        colors=rgb[pixels[ids,0],pixels[ids,1]]
        packet=dict(video_id=video,frame_id=fid,count=len(ids),camera=packed(normalize(camera[ids],normalization),'<f4'),
            aligned=packed(aligned,'<f4') if aligned is not None else None,rgb=packed(colors,'u1'),reason=packed(reasons[ids],'u1'),old=packed(old[ids],'u1'),
            pixels=packed(pixels[ids],'<u2'),observed_indices=packed(pixels[ids,0]*depth.shape[1]+pixels[ids,1],'<u4'))
        out=destination/'cases'/video/'depth';out.mkdir(parents=True,exist_ok=True)
        support_path=out/f'frame_{fid:05d}.npz'
        np.savez_compressed(support_path,valid=valid,rejected=support['rejected'],rejection_reason=support['rejection_reason'],
            source_input_sha256=np.array(sha(source)),settings_json=np.array(json.dumps(settings(cfg),sort_keys=True)))
        packet_path=replay/'data'/video/f'frame_{fid:05d}.js';packet_path.parent.mkdir(parents=True,exist_ok=True)
        packet_path.write_text('window.LINK5_REFINEMENT_FRAME('+json.dumps(packet,separators=(',',':'))+');\n')
        # Compact inspection overlay is generated from exact RGB; inputs remain unchanged.
        overlay=rgb.copy();labels=support['rejection_reason'];kept=valid
        overlay[kept]=(overlay[kept]*.6+np.array([70,220,200])*.4).astype(np.uint8)
        for reason,color in [(2,[242,181,66]),(3,[244,83,65]),(1,[180,95,230])]:
            selected=labels==reason;overlay[selected]=(overlay[selected]*.25+np.array(color)*.75).astype(np.uint8)
        overlay_path=out/f'frame_{fid:05d}.png';cv2.imwrite(str(overlay_path),overlay[...,::-1])
        usable=info['final_valid_pixel_count']>=cfg.minimum_pixels and info['retained_fraction']>=cfg.minimum_depth_retained_fraction
        row.update(status='usable' if usable else 'insufficient depth support',packet=href(packet_path),images=dict(rgb=href(folder/'rgb.png'),mask=href(folder/'mask.png'),overlay=href(overlay_path)),
            original_points=len(old_xyz),raw_points=len(camera),displayed_points=len(ids),filter=info,source_input=href(source),support=href(support_path),
            pose_status=pose['status'] if pose else None)
        if fid==0 and video in train_ids:
            filtered,_=xyz_from_depth(depth,valid,k);normal_path=out/'normal_camera.npy';np.save(normal_path,filtered,allow_pickle=False)
            row['normal_cloud']=href(normal_path)
        return dict(video_id=video,frame_id=fid,source_input=str(source),source_input_sha256=sha(source),support_sha256=sha(support_path),
            retained=info['final_valid_pixel_count'],original=len(old_xyz),raw=len(camera),erosion_removed=info['erosion_rejected_pixel_count'],
            density_removed=info['density_rejected_pixel_count'],usable=usable,
            z_before=np.quantile(camera[:,2],[.5,.95,.99,1]).tolist() if len(camera) else None,
            z_after=np.quantile(depth[valid],[.5,.95,.99,1]).tolist() if valid.any() else None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(export,task) for task in tasks]
        for future in as_completed(futures):
            receipts.append(future.result())
            if len(receipts)%100==0:print('LINK5_CPU_DEPTH_REPLAY',len(receipts),'/',len(tasks),flush=True)
    guard_cases=[]
    for video in train:
        old=successful_masking(root,video);original=read(old/'link5_guard.json')
        variants=[dict(name='Previous run',record=href(old/'link5_guard.json'),image=href(old/original['selected_image']),calls=original['attempts'],points=original['selected_points_xy'],reviews=original['reviews'])]
        local=destination/'cases'/video/'guard'
        for label,guard_folder in [('Refined initial',local),('Refined 64K',destination/'validation_64k/cases'/video/'guard')]:
            for model in ['gemini','luna']:
                receipt=guard_folder/model/'link5_guard.json'
                if receipt.exists():
                    g=read(receipt)
                    variants.append(dict(name=label+' '+model.capitalize(),record=href(receipt),image=href(receipt.parent/g['selected_image']),
                        calls=g['attempts'],points=g['selected_points_xy'],reviews=g['reviews'],failed=g['fallback_to_default'],
                        inputs={role:href(receipt.parent/f'link5_guard_{role}_current.png') for role in ['positive','negative']}))
        guard_cases.append(dict(video_id=video,variants=variants,mask=href(root/'cases'/video/'observations/frame_00000/source_mask.png')))
    receipt=dict(status='complete',CPU_only=True,filter_settings=settings(cfg),filter_source_sha256=sha(ROOT / 'robot/preprocessing/depth/link5_depth_filter.py'),
        expected_observations=len(tasks),exported_observations=len(receipts),full_test_videos=len(scope),training_references=len(train),
        normal20_retained_points=sum(r['retained'] for r in receipts if r['frame_id']==0 and r['video_id'] in train_ids),
        insufficient_support=sum(not r['usable'] for r in receipts),source_depth_changed=False,prior_inputs_and_scores_changed=False,
        existing_replay_interface_updated=True,
        new_sam_masks=False,new_pose_inference=False,new_detector_scores=False,observations=sorted(receipts,key=lambda r:(r['video_id'],r['frame_id'])))
    write(destination/'metadata/depth_summary.json',receipt)
    manifest=dict(cases=cases,train20=train,guards=guard_cases,settings=settings(cfg),summary={k:v for k,v in receipt.items() if k!='observations'})
    (replay/'manifest.js').write_text('window.LINK5_REFINEMENT='+json.dumps(manifest,separators=(',',':'))+';\n')
    existing=root/'full_video/replay'
    (existing/'index.html').write_text(render_existing_replay(ROOT))
    (existing/'app.js').write_text(Path(__file__).with_name('replay_app.js').read_text())
    print('LINK5_REFINEMENT_REPLAY_CREATED',len(receipts),'observations',receipt['normal20_retained_points'],'normal points',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();run(args.root,args.destination,args.workers)
