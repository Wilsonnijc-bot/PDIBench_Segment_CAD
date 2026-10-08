"""Every-frame Link5 evaluation with multiple video workers sharing each GPU.

This is an explicitly authorized exploratory evaluation of the unchanged final
checkpoints. Failed sensitivity receipts remain failed and are never overwritten.
"""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import cv2
import numpy as np

from .common import ROOT, MODES, cloud_metrics, mode_folder, read, sha, write, split_path
from .prepare import MASK_POLICY, filtering_settings, native_mask, save_observation
from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth, xyz_from_depth


def location(config):
    return Path(config['output'])/'full_video'


def validate_contract(config):
    policy=config['full_video']
    if policy['frame_selection']!='every_decoded_frame' or not policy['include_frame0']:
        raise ValueError('Link5 testing requires every frame including frame0')
    if policy['primary_video_metric']!='sum_all_frame_raw_mean_top80':
        raise ValueError('video score must be an unnormalized sum of all frame scores')
    refined=bool(config.get('normal_reference'))
    if refined:
        workers=policy.get('workers_per_gpu')
        if type(workers) is not int or workers<1 or policy['workers']!=workers or policy.get('gpu_count')!=1:
            raise ValueError('refined round requires one GPU with a positive shared video-worker count')
    elif policy['workers']!=10:raise ValueError('legacy round requires ten parallel video workers')
    output=Path(config['output'])
    if policy['normalization_sha256']!=sha(output/'link5_normalization.json') or policy['split_sha256']!=sha(split_path(output)):
        raise ValueError('training reference changed')
    if not refined and policy['exploratory_authorization']!='User requested full-video GPU testing after the failed sensitivity results were disclosed.':
        raise ValueError('explicit exploratory authorization missing')
    for mode in MODES:
        if policy['checkpoint_sha256'][mode]!=sha(mode_folder(output,'training',mode)/'link5/final.pt'):
            raise ValueError('detector checkpoint changed')
        if policy['sanity_receipt_sha256'][mode]!=sha(mode_folder(output,'sanity',mode)/'checks.json'):
            raise ValueError('failed sanity receipt changed')
        if refined:
            receipt=read(mode_folder(output,'sanity',mode)/'checks.json')
            if receipt['status']!='passed':raise ValueError('new detector validation failed; full evaluation disabled')
            if receipt['heldout_manifest_sha256']!=sha(output/'splits/heldout_frame0.json'):raise ValueError('heldout manifest changed')
    groups={}
    for case in config['cases']:
        generator,number=case['id'].rsplit('_',1)
        groups.setdefault(number,set()).add(generator)
    if len(config['cases'])!=45 or len(groups)!=15 or any(g!={'LVP_ROBOWM','COSMOS2.5','COSMOS3'} for g in groups.values()):
        raise ValueError('fifteen matching video numbers across all three generators required')


def aggregate_all_frames(rows, expected_count):
    ids=[int(r['frame_id']) for r in rows]
    if sorted(ids)!=list(range(expected_count)):
        raise ValueError('full-video report must retain exactly every source frame')
    good=[r for r in rows if r['status']=='complete']
    values=[float(r['raw_mean_top80']) for r in good]
    complete=len(good)==expected_count
    return dict(expected_frame_count=expected_count,scored_frame_count=len(good),failed_frame_count=expected_count-len(good),
        include_frame0=True,primary_metric='sum_all_frame_raw_mean_top80',
        sum_all_frame_raw_mean_top80=float(sum(values)) if complete else None,
        available_frame_score_sum=float(sum(values)),sum_is_complete=complete,
        raw_frame_mean=float(np.mean(values)) if values else None,
        raw_frame_max=float(max(values)) if values else None,
        raw_frame_median=float(np.median(values)) if values else None,
        raw_frame_mean_top3=float(np.mean(sorted(values,reverse=True)[:3])) if values else None,
        missing_frame_ids=[r['frame_id'] for r in rows if r['status']!='complete'])


def failed_observation(case, folder, index, rgb, depth, k, mask, cam_c2w, error):
    """Keep real missing/low-quality observations; never invent a score or mask."""
    h,w=depth.shape
    cfg=Config().validate();support=filter_depth(depth,mask,k,cfg)
    grid,valid,rejected,info=(support[name] for name in ('mask','valid','rejected','info'))
    xyz,pixels=xyz_from_depth(depth,valid,k)
    obs=folder/'observations'/f'frame_{index:05d}';obs.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(obs/'input.npz',xyz_camera=xyz,pixels_yx=pixels,rgb=rgb,depth=depth,K=k,
                        source_mask=mask,mask=grid,valid=valid,rejected=rejected,eroded_mask=support['eroded_mask'],
                        rejection_reason=support['rejection_reason'],density_score=support['density_score'],cam_c2w=cam_c2w,frame_id=index)
    y,x=np.where(mask);bbox=(y.max()-y.min()+1)*(x.max()-x.min()+1) if len(y) else 0
    row=dict(video_id=case['id'],frame_id=index,status='failed',error=str(error),input_path=str(obs/'input.npz'),
        valid_point_count=len(xyz),mask_area=int(mask.sum()),mask_bbox_coverage=float(mask.sum()/bbox) if bbox else 0.,
        image_boundary_contact=bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        source_image_hw=list(mask.shape),native_depth_hw=[h,w],depth_filter=info,
        mask_erosion_pixels=cfg.erosion_pixels,mask_mapping_applied=False,observation_policy=MASK_POLICY)
    write(obs/'observation.json',row)
    cv2.imwrite(str(obs/'rgb.png'),rgb[...,::-1]);cv2.imwrite(str(obs/'mask.png'),grid.astype(np.uint8)*255)
    cv2.imwrite(str(obs/'source_mask.png'),mask.astype(np.uint8)*255)
    return row


def prepare(config,case):
    output=Path(config['output']);folder=location(config)/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    receipt=folder/'preparation.json'
    if receipt.exists() and read(receipt)['status']=='complete':
        prior=read(receipt)
        if prior['source_sha256']!=sha(case['video']) or prior['mask_policy']!=MASK_POLICY:raise ValueError('dense inputs changed')
        rows=read(folder/'observations.json')
        if [r['frame_id'] for r in rows]!=list(range(prior['frame_count'])):raise ValueError('dense observation missing')
        return
    if sha(case['video'])!=case['video_sha256']:raise ValueError('video identity changed')
    # These are the fresh current-runtime masks already produced for this trial,
    # covering the whole video. They are not historical artifact Link5 masks.
    mask_path=native_mask(config,case,output/'cases'/case['id'])
    with np.load(mask_path,allow_pickle=False) as a:
        masks=a['object_masks'][:,a['object_names'].tolist().index('link5')].astype(bool)
    cap=cv2.VideoCapture(case['video']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
    if len(masks)!=count:raise ValueError('full guarded mask coverage mismatch')
    work=folder/'geometry-work';work.mkdir(exist_ok=True)
    from infrastructure.deformation_detect.worker import isolate_mega
    if not (work/'mega_sam').exists():isolate_mega(dict(config=config,directory=str(work)))
    else:os.environ['PDI_MEGA_SAM_ROOT']=str(work/'mega_sam')
    from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
    wrapper=MegaSamWrapper(device='cuda')
    # The original full video is passed directly, with every real decoded frame.
    # Per-case mutable directories isolate native scene-name collisions (0001).
    scene=Path(case['video']).stem
    native=Path(wrapper.mega_sam_root)/'outputs_cvd'/f'{scene}_sgd_cvd_hr.npz'
    native_receipt=folder/'metadata/full_sequence_geometry.json'
    if not native.exists() or not native_receipt.exists():
        geometry=wrapper.infer_shared(case['video'],masks[:,None],cache_dir=None)
        if geometry.metadata['cache_hit'] or geometry.frames_count!=count or not native.is_file():
            raise ValueError('full-sequence native CVD geometry incomplete; no raw-depth fallback')
        write(native_receipt,dict(status='complete',frame_count=count,source_sha256=sha(case['video']),
            mask_sha256=sha(mask_path),native_cvd_sha256=sha(native),geometry_metadata=geometry.metadata,
            reconstruction_code_identity=wrapper._reconstruction_code_identity()))
        del geometry
    else:
        prior=read(native_receipt)
        if prior['source_sha256']!=sha(case['video']) or prior['mask_sha256']!=sha(mask_path) or prior['native_cvd_sha256']!=sha(native):
            raise ValueError('owned dense CVD provenance mismatch')
    with np.load(native,allow_pickle=False) as a:
        depths=a['depths'].copy();intrinsic=a['intrinsic'].copy();poses=a['cam_c2w'].copy()
    if len(depths)!=count or len(poses)!=count:raise ValueError('CVD output is not every frame')
    if intrinsic.shape not in ((4,),(3,3)):raise ValueError('unknown native intrinsics')
    fx,fy,cx,cy=wrapper._parse_intrinsic(intrinsic);k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
    cap=cv2.VideoCapture(case['video']);rows=[]
    for index in range(count):
        ok,bgr=cap.read()
        if not ok:raise ValueError(f'cannot decode source frame {index}')
        depth=depths[index].astype(np.float32);h,w=depth.shape;rgb=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_LINEAR)[...,::-1].copy()
        if index==0:
            # Freeze the actual frame0 normal reference used for training/tests.
            frozen=output/'cases'/case['id']/'observations/frame_00000';dest=folder/'observations/frame_00000';dest.mkdir(parents=True,exist_ok=True)
            for name in ('input.npz','rgb.png','mask.png','source_mask.png'):shutil.copy2(frozen/name,dest/name)
            row=read(frozen/'observation.json');row.update(input_path=str(dest/'input.npz'),status='complete',
                frame0_policy='exact frozen original camera reference; no pose registration; no retraining')
            write(dest/'observation.json',row)
            # Record any reconstruction-scale change without correcting scale or
            # changing the twenty training clouds/shared normalization.
            try:
                dense=save_observation(case,folder/'geometry-audit',0,rgb,depth,k,masks[0],poses[0])
                with np.load(frozen/'input.npz',allow_pickle=False) as a:normal=cloud_metrics(a['xyz_camera'])
                ratios=np.array(dense['bbox_dimensions'])/np.maximum(normal['bbox_dimensions'],1e-8)
                write(folder/'metadata/dense_frame0_comparison.json',dict(frozen=normal,dense=dense,
                    bbox_dimension_ratios=ratios.tolist(),obvious_scale_inconsistency=bool(np.any((ratios<.65)|(ratios>1.5))),
                    scale_correction_applied=False,training_cloud_unchanged=True))
            except Exception as error:write(folder/'metadata/dense_frame0_comparison.json',dict(status='failed',error=str(error),scale_correction_applied=False))
        else:
            try:
                row=save_observation(case,folder,index,rgb,depth,k,masks[index],poses[index]);row['status']='complete'
                write(Path(row['input_path']).parent/'observation.json',row)
            except ValueError as error:row=failed_observation(case,folder,index,rgb,depth,k,masks[index],poses[index],error)
        rows.append(row)
    ok,_=cap.read();cap.release()
    if ok:raise ValueError('source frame-count metadata omitted decoded frames')
    write(folder/'observations.json',rows)
    write(receipt,dict(status='complete',video_id=case['id'],frame_count=count,fps=fps,
        source_sha256=sha(case['video']),mask_sha256=sha(mask_path),mask_source=str(mask_path),
        mask_policy=MASK_POLICY,frame_selection='every_decoded_frame',input_failure_count=sum(r['status']!='complete' for r in rows),
        filter_settings=filtering_settings(Config()),filter_source_sha256=sha(ROOT / 'robot/preprocessing/depth/link5_depth_filter.py')))
    print('LINK5_FULL_VIDEO_PREPARED',case['id'],count,flush=True)


def poses(config,case):
    from .pose import Link5Pose
    folder=location(config)/'cases'/case['id']
    adapter=Link5Pose({**config['pose'],'debug_directory':str(folder/'metadata/foundationpose-debug')})
    rows=read(folder/'observations.json');records=[]
    for row in rows:
        if row['frame_id']==0:continue
        if row['status']!='complete':continue
        destination=Path(row['input_path']).parent/'pose';path=destination/'pose.json'
        if path.exists():
            record=read(path)
            if record['input_sha256']!=sha(row['input_path']) or record['provenance']!=adapter.provenance:raise ValueError('pose cache provenance changed')
        else:record=adapter.estimate(row['input_path'],destination)
        records.append(dict(frame_id=row['frame_id'],status=record['status'],cad_mask_iou=record.get('cad_mask_iou')))
        print('LINK5_FULL_VIDEO_POSE',case['id'],row['frame_id'],record['status'],flush=True)
    write(folder/'metadata/pose_exits.json',records)
    print('LINK5_FULL_VIDEO_POSES_FINISHED',case['id'],flush=True)


def score(config_path,config,case):
    from .score import load_detector, score_row
    output=Path(config['output']);folder=location(config)/'cases'/case['id']
    split=read(split_path(output));train_ids={r['video_id'] for r in split['train']}
    observations=read(folder/'observations.json');summaries={}
    for mode in MODES:
        model,modules,normalization=load_detector(output,mode);results=[]
        for row in observations:
            if row['status']=='complete':result=score_row(config_path,config,row,model,modules,normalization,train_ids,mode)
            else:
                result=dict(video_id=case['id'],frame_id=row['frame_id'],status='failed',error=row['error'],
                    augmentation_mode=mode,reference_split='train' if case['id'] in train_ids else 'heldout',
                    visibility={k:row[k] for k in ('valid_point_count','mask_area','mask_bbox_coverage','image_boundary_contact')})
                suffix='' if mode=='paper_original' else '_robot_structural'
                write(Path(row['input_path']).parent/f'score{suffix}.json',result)
            results.append(result)
            print('LINK5_FULL_VIDEO_SCORED',case['id'],mode,row['frame_id'],result['status'],result.get('raw_mean_top80'),flush=True)
        write(folder/f'frames_{mode}.json',results)
        summary=dict(video_id=case['id'],generator=case['id'].rsplit('_',1)[0],video_number=case['id'].rsplit('_',1)[1],
            augmentation_mode=mode,reference_split='train' if case['id'] in train_ids else 'heldout',
            **aggregate_all_frames(results,len(observations)),exploratory=True,sensitivity_gate_status='failed')
        write(folder/f'video_{mode}.json',summary);summaries[mode]=summary
        del model
        import torch
        torch.cuda.empty_cache()
    write(folder/'completion.json',dict(status='complete' if all(x['sum_is_complete'] for x in summaries.values()) else 'complete_with_failures',
        video_id=case['id'],frame_count=len(observations),configuration_sha256=sha(config_path),summaries=summaries))
    print('LINK5_FULL_VIDEO_COMPLETE',case['id'],flush=True)


def summarize(config):
    root=location(config);root.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        videos=[];frames=[]
        for case in config['cases']:
            folder=root/'cases'/case['id']
            if (folder/f'video_{mode}.json').exists():
                videos.append(read(folder/f'video_{mode}.json'));frames.extend(read(folder/f'frames_{mode}.json'))
        destination=root/'scores'/mode;destination.mkdir(parents=True,exist_ok=True)
        write(destination/'videos.json',videos);write(destination/'frames.json',frames)
        for name,items in (('videos',videos),('frames',[{k:v for k,v in x.items() if k!='visibility'}|x.get('visibility',{}) for x in frames])):
            if not items:continue
            keys=list(dict.fromkeys(k for x in items for k in x))
            with (destination/f'{name}.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(items)
        matched=[]
        for number in sorted({c['id'].rsplit('_',1)[1] for c in config['cases']}):
            item=dict(video_number=number)
            for generator in ('LVP_ROBOWM','COSMOS2.5','COSMOS3'):
                v=next((v for v in videos if v['video_id']==generator+'_'+number),None)
                item[generator+'_sum']=v['sum_all_frame_raw_mean_top80'] if v else None
                item[generator+'_frames']=v['expected_frame_count'] if v else None
                item[generator+'_scored']=v['scored_frame_count'] if v else None
            matched.append(item)
        with (destination/'matched_generators.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(matched[0]));writer.writeheader();writer.writerows(matched)
        from scipy.stats import spearmanr
        good=[x for x in frames if x['status']=='complete'];correlations={}
        for key in ('valid_point_count','mask_area','mask_bbox_coverage','image_boundary_contact'):
            x=np.array([r['visibility'][key] for r in good],float);y=np.array([r['raw_mean_top80'] for r in good],float)
            rho,p=spearmanr(x,y) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else (np.nan,np.nan)
            correlations[key]=dict(count=len(x),spearman_rho=float(rho) if np.isfinite(rho) else None,p_value=float(p) if np.isfinite(p) else None)
        write(destination/'visibility_correlations.json',correlations)


def preflight(config):
    validate_contract(config)
    import torch
    if not torch.cuda.is_available() or (torch.ones(2,device='cuda')+1).sum().item()!=4:raise RuntimeError('CUDA preflight failed')
    counts=[]
    for case in config['cases']:
        if sha(case['video'])!=case['video_sha256']:raise ValueError('source changed')
        cap=cv2.VideoCapture(case['video']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=float(cap.get(cv2.CAP_PROP_FPS));cap.release()
        mask=native_mask(config,case,Path(config['output'])/'cases'/case['id'])
        with np.load(mask,allow_pickle=False) as a:
            if len(a['object_masks'])!=count:raise ValueError('mask frame coverage mismatch')
        counts.append(dict(video_id=case['id'],frame_count=count,fps=fps,mask_sha256=sha(mask)))
    write(location(config)/'metadata/preflight.json',dict(status='passed',gpu=torch.cuda.get_device_name(0),
        configuration_contract='every decoded frame; sum all raw top80 including frame0; multiple video workers per GPU',
        video_count=len(counts),frame_count=sum(x['frame_count'] for x in counts),cases=counts))
    print('LINK5_FULL_VIDEO_PREFLIGHT_PASSED',flush=True)


def batch(config_path,config):
    validate_contract(config)
    devices=os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')
    expected_gpus=config['full_video'].get('gpu_count',2)
    workers_per_gpu=config['full_video'].get('workers_per_gpu',5)
    if len(devices)!=expected_gpus:raise ValueError('allocated GPU count differs from configured shared-worker count')
    root=location(config);root.mkdir(parents=True,exist_ok=True)
    subprocess.run([config['environments']['geometry'],'-u','-m',__spec__.name,'--config',str(config_path),'--phase','preflight'],cwd=ROOT,check=True)
    subprocess.run([config['environments']['foundationpose'],'-u','-m','robot.experiments.link5_shape_codebook.verify_pose_runtime',
                    '--config',str(config_path)],cwd=ROOT,check=True)
    results=[]
    def worker(case,device):
        folder=root/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=device
        result=dict(video_id=case['id'],device=device,status='running')
        try:
            for phase,environment in (('prepare','geometry'),('pose','foundationpose'),('score','shape_codebook')):
                attempt=len(list((folder/'metadata').glob(phase+'-*.log')))+1;log=folder/'metadata'/f'{phase}-{attempt:04d}.log';log.parent.mkdir(exist_ok=True)
                with log.open('x') as stream:
                    proc=subprocess.run([config['environments'][environment],'-u','-m',__spec__.name,
                        '--config',str(config_path),'--video-id',case['id'],'--phase',phase],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
                write(folder/'metadata'/f'{phase}-{attempt:04d}.exit.json',dict(exit_status=proc.returncode,log=str(log)))
                if proc.returncode:raise RuntimeError(f'{phase} failed with exit{proc.returncode}; see {log}')
            result['status']=read(folder/'completion.json')['status']
        except Exception as error:result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        write(folder/'worker.json',result)
        return result
    # Separate fixed-size pools prevent one GPU from receiving more than five
    # concurrent video pipelines as fast/slow generator videos finish.
    with ExitStack() as stack:
        pools=[stack.enter_context(ThreadPoolExecutor(max_workers=workers_per_gpu)) for _ in devices]
        futures=[pools[i%len(devices)].submit(worker,c,devices[i%len(devices)]) for i,c in enumerate(config['cases'])]
        for future in as_completed(futures):
            result=future.result();results.append(result);write(root/'metadata/worker_exits.json',results);summarize(config)
            print('LINK5_FULL_VIDEO_WORKER_FINISHED',result['video_id'],result['status'],flush=True)
    summarize(config)
    complete=all(x['status']=='complete' for x in results)
    write(root/'completion.json',dict(status='complete' if complete else 'complete_with_failures',video_count=len(results),
        workers=len(devices)*workers_per_gpu,gpus=len(devices),workers_per_gpu=workers_per_gpu,expected_frame_count=read(root/'metadata/preflight.json')['frame_count'],
        primary_video_metric='sum_all_frame_raw_mean_top80',include_frame0=True,exploratory=not bool(config.get('normal_reference')),
        failed_sensitivity_checks_preserved=not bool(config.get('normal_reference')),configuration_sha256=sha(config_path),worker_results=results))
    if any(x['status']=='failed' for x in results):raise RuntimeError('full-video pipeline failures retained; see worker receipts')
    print('LINK5_FULL_VIDEO_EVALUATION_FINISHED',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=('preflight','prepare','pose','score','batch','summarize'),required=True);parser.add_argument('--video-id')
    args=parser.parse_args();config=read(args.config);validate_contract(config)
    if args.phase=='preflight':preflight(config)
    elif args.phase=='batch':batch(args.config,config)
    elif args.phase=='summarize':summarize(config)
    else:
        case=next(c for c in config['cases'] if c['id']==args.video_id)
        if args.phase=='prepare':prepare(config,case)
        elif args.phase=='pose':poses(config,case)
        else:score(args.config,config,case)


if __name__=='__main__':main()
