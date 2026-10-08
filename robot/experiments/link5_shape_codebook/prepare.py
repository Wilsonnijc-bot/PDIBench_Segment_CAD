"""Fresh native guarded masks and MegaSAM; observed-only Link5 depth filtering."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import traceback
import uuid
import time

import cv2
import numpy as np

from .common import ROOT, cloud_metrics, read, sha, write
from robot.preprocessing.depth.link5_depth_filter import Config, filter_depth, settings as filtering_settings, xyz_from_depth


def validate_mask_guard(config,directory):
    if not config.get('normal_reference'):return
    guard=read(directory/'link5_guard.json')
    if guard.get('positive_placement_policy')!='vlm_places_all_three_no_default_p1' or guard.get('p1_default_used') is not False or guard.get('fallback_to_default'):
        raise ValueError('fresh three-positive VLM placement required; historical masks cannot be reused')
    diagnostics=read(directory/'sam3_prompt_diagnostics.json')
    refinement=next(r for r in diagnostics if r['target']=='link5')['point_refinement']
    if refinement['frame_index']!=0 or refinement['points_xy']!=guard['selected_points_xy']:
        raise ValueError('fresh SAM mask did not use the selected frame0 VLM points')
    reference=Path(config['resources']['link5_positive_guard_reference'])
    if guard.get('positive_reference_sha256')!=sha(reference):
        raise ValueError('SAM guard used a different positive reference example')


def completed_mask_path(config, case, directory, record):
    """Reuse the verified mask beside its receipt, even after cloud cleanup/sync."""
    validate_mask_guard(config, directory)
    mask = directory / 'segmentation.npz'
    provenance = read(directory / 'provenance.json')
    expected_source = case.get('video_sha256') or sha(case['video'])
    if provenance.get('video_sha256') != expected_source:
        raise ValueError('saved Link5 mask belongs to a different source video')
    if not mask.is_file() or provenance.get('segmentation_sha256') != sha(mask):
        raise ValueError('saved Link5 mask is missing or its content changed')
    if Path(record['segmentation']).name != mask.name:
        raise ValueError('saved Link5 receipt names an unexpected mask archive')
    return mask


def native_mask(config, case, folder):
    directory = folder / 'masking'
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve old owned attempts and use the runtime's native worker/executor
    # and configured four-attempt primary/alternate routes. Never accept the
    # guard's DEFAULT_UNREVIEWED candidate.
    prior=[]
    if (directory/'result.json').exists():prior.append(directory)
    prior.extend(sorted(directory.glob('attempt-*')))
    for previous in prior:
        if (previous/'result.json').exists():
            record=read(previous/'result.json')
            if record.get('status')=='complete':
                return completed_mask_path(config, case, previous, record)
    from infrastructure.deformation_detect.coordinator import execute
    maximum=config['policy']['vlm_attempts'];started=time.monotonic()
    previous=str(prior[-1]) if prior else None
    route='alternate' if prior else 'primary'
    latest=read(prior[-1]/'result.json') if prior and (prior[-1]/'result.json').exists() else {}
    if not latest or latest.get('failure',{}).get('category') in ('interrupted','cancelled'):
        route='primary'
    if latest.get('failure',{}).get('category')=='vlm_timeout':route='primary'
    for attempt in range(len(prior)+1,maximum+1):
        destination=directory/f'attempt-{attempt:04d}';destination.mkdir()
        request=dict(config=config,case=case,stage='link5_masks',directory=str(destination),attempt=attempt,
                     vlm_route=route,previous_attempt=previous,dependencies={})
        path=destination/'request.json';write(path,request)
        remaining=config['policy']['vlm_stage_timeout_seconds']-(time.monotonic()-started)
        if remaining<=0:raise TimeoutError('native Link5 VLM stage budget exhausted')
        code=execute(path,config['environments']['sam3'],remaining)
        result=read(destination/'result.json')
        if code==0 and result.get('status')=='complete':
            return completed_mask_path(config, case, destination, result)
        previous=str(destination)
        route='primary' if result.get('failure',{}).get('category')=='vlm_timeout' else 'alternate'
    raise ValueError('current native guarded masking exhausted configured attempt budget')


MASK_POLICY = 'guarded_mask_2px_source_erosion__ray_normalized_knn_v1'


def observation_ids(masks, minimum=256):
    areas=masks.sum(axis=(1,2))
    if areas[0]<minimum:raise ValueError('frame0 has insufficient raw guarded Link5 mask support')
    frames=np.arange(len(masks));edges=np.linspace(.1*len(masks),len(masks),11)
    chosen=[0]
    for low,high in zip(edges[:-1],edges[1:]):
        candidates=np.flatnonzero((frames>=low)&(frames<high)&(areas>=minimum)&(frames>0))
        if len(candidates):chosen.append(int(candidates[np.argmax(areas[candidates])]))
    return sorted(set(chosen))


def save_observation(case,folder,index,rgb,depth,k,source_mask,cam_c2w):
    cfg=Config().validate();h,w=depth.shape
    # Preserve the raw guarded masks. Erode source support by two pixels and
    # reject sparse 3D depth tails only for the observed detector cloud.
    support=filter_depth(depth,source_mask,k,cfg)
    grid_mask,valid,rejected,info=(support[name] for name in ('mask','valid','rejected','info'))
    xyz,pixels=xyz_from_depth(depth,valid,k)
    if len(xyz)<cfg.minimum_pixels or info['retained_fraction']<cfg.minimum_depth_retained_fraction:
        raise ValueError(f'frame {index}: insufficient filtered observed Link5 depth support')
    y,x=np.where(source_mask);bbox_area=(y.max()-y.min()+1)*(x.max()-x.min()+1)
    obs=folder/'observations'/f'frame_{index:05d}';obs.mkdir(parents=True,exist_ok=True)
    temporary=obs/f'.input-{os.getpid()}.tmp'
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,xyz_camera=xyz,pixels_yx=pixels,rgb=rgb,depth=depth,K=k,
            source_mask=source_mask,mask=grid_mask,valid=valid,rejected=rejected,
            eroded_mask=support['eroded_mask'],rejection_reason=support['rejection_reason'],density_score=support['density_score'],
            cam_c2w=cam_c2w,frame_id=index)
    temporary.replace(obs/'input.npz')
    metrics=dict(video_id=case['id'],frame_id=index,**cloud_metrics(xyz),mask_area=int(source_mask.sum()),
        mask_bbox_coverage=float(source_mask.sum()/bbox_area),image_boundary_contact=bool(source_mask[0].any() or source_mask[-1].any() or source_mask[:,0].any() or source_mask[:,-1].any()),
        source_image_hw=list(source_mask.shape),native_depth_hw=[h,w],depth_filter=info,input_path=str(obs/'input.npz'),
        coordinate_system='camera; no centering, scale change or FoundationPose',mask_erosion_pixels=cfg.erosion_pixels,mask_mapping_applied=False,
        observation_policy=MASK_POLICY,mask_depth_grid_alignment='nearest-neighbor resolution alignment only; original source_mask retained verbatim')
    write(obs/'observation.json',metrics)
    cv2.imwrite(str(obs/'rgb.png'),rgb[...,::-1]);cv2.imwrite(str(obs/'mask.png'),grid_mask.astype(np.uint8)*255)
    cv2.imwrite(str(obs/'source_mask.png'),source_mask.astype(np.uint8)*255)
    cv2.imwrite(str(obs/'depth_support.png'),valid.astype(np.uint8)*255)
    return metrics


def refresh_owned_observations(config,case,folder,receipt):
    output=Path(config['output'])
    if (output/'training/link5/final.pt').exists() or (output/'training_robot_structural/link5/final.pt').exists():
        raise ValueError('trained inputs are immutable; refine into a new run output and retrain')
    mask_path=native_mask(config,case,folder)
    with np.load(mask_path,allow_pickle=False) as archive:
        masks=archive['object_masks'][:,archive['object_names'].tolist().index('link5')].astype(bool)
    expected=observation_ids(masks)
    if expected!=receipt['selected_frames']:return False
    records=[];previous=[]
    for row in read(folder/'observations.json'):
        with np.load(row['input_path'],allow_pickle=False) as archive:
            values={key:archive[key].copy() for key in ('rgb','depth','K','cam_c2w')}
        previous.append(dict(frame_id=row['frame_id'],input_sha256=sha(row['input_path']),valid_point_count=row['valid_point_count']))
        records.append(save_observation(case,folder,row['frame_id'],values['rgb'],values['depth'],values['K'],masks[row['frame_id']],values['cam_c2w']))
    write(folder/'metadata/depth_filter_refresh_provenance.json',dict(previous_observations=previous,previous_policy=receipt.get('observation_policy'),
        new_policy=MASK_POLICY,mask_sha256=sha(mask_path),MegaSAM_rerun=False,reason='CPU-only refresh of untrained observed depth support'))
    write(folder/'observations.json',records)
    geometry=read(folder/'geometry.json');geometry.update(filtering=filtering_settings(Config()),observation_policy=MASK_POLICY,mask_erosion_pixels=Config().erosion_pixels,mask_mapping_applied=False)
    write(folder/'geometry.json',geometry)
    receipt.update(observation_policy=MASK_POLICY,mask_erosion_pixels=Config().erosion_pixels,mask_mapping_applied=False)
    write(folder/'status.json',receipt)
    print('OWNED_UNTRAINED_DEPTH_SUPPORT_REFRESHED',case['id'],flush=True)
    return True


def prepare_case(config, case, output):
    folder = output/'cases'/case['id']
    folder.mkdir(parents=True, exist_ok=True)
    existing=folder/'status.json'
    if existing.is_file() and read(existing).get('status')=='complete':
        receipt=read(existing)
        native_mask(config,case,folder)
        if receipt['source_sha256']!=sha(case['video']):raise ValueError('owned fresh preparation source identity changed')
        for row in read(folder/'observations.json'):
            with np.load(row['input_path'],allow_pickle=False) as archive:
                if not np.isfinite(archive['xyz_camera']).all():raise ValueError('owned fresh cloud corrupted')
        if receipt.get('observation_policy')==MASK_POLICY:
            print('OWNED_FRESH_OBSERVATIONS_RETAINED',case['id'],flush=True)
            return
        if refresh_owned_observations(config,case,folder,receipt):return
    status = dict(video_id=case['id'], status='running', source_video=case['video'], source_sha256=sha(case['video']),
                  historical_masks_used=False, historical_geometry_used=False, frame0_pose_normalization=False)
    write(folder/'status.json', status)
    try:
        mask_path = native_mask(config, case, folder)
        with np.load(mask_path, allow_pickle=False) as archive:
            names = archive['object_names'].tolist()
            masks = archive['object_masks'][:, names.index('link5')].astype(bool)
        cfg = Config().validate()
        ids = observation_ids(masks,cfg.minimum_pixels)
        if len(ids) < 2:
            raise ValueError('need frame0 and real temporal context for MegaSAM')
        cap = cv2.VideoCapture(case['video'])
        frames = {}
        for index in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'cannot decode source frame {index}')
            frames[index] = frame
        cap.release()
        height, width = frames[0].shape[:2]
        scene = 'l5cb_' + case['id'].replace('.', '_') + '_' + uuid.uuid4().hex[:12]
        sequence = folder/(scene+'.mp4')
        writer = cv2.VideoWriter(str(sequence), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
        if not writer.isOpened():
            raise ValueError('cannot write fresh MegaSAM temporal input')
        try:
            for index in ids:
                writer.write(frames[index])
        finally:
            writer.release()
        # Isolate upstream mutable scratch while retaining its implementation.
        from infrastructure.deformation_detect.worker import isolate_mega
        geom_dir = folder/'geometry-work'
        geom_dir.mkdir(exist_ok=True)
        if not (geom_dir/'mega_sam').exists():isolate_mega(dict(config=config, directory=str(geom_dir)))
        else:os.environ['PDI_MEGA_SAM_ROOT']=str(geom_dir/'mega_sam')
        from infrastructure.shared.inference.mega_sam_wrapper import MegaSamWrapper
        wrapper = MegaSamWrapper(device='cuda')
        geometry = wrapper.infer_shared(str(sequence), masks[ids, None], cache_dir=None)
        if geometry.metadata['cache_hit'] or geometry.frames_count != len(ids):
            raise ValueError('fresh MegaSAM inference incomplete or cached')
        root = Path(wrapper.mega_sam_root)
        native = root/'outputs_cvd'/f'{scene}_sgd_cvd_hr.npz'
        if not native.is_file():
            raise ValueError('CVD refined depth missing; no silent raw-depth fallback')
        with np.load(native, allow_pickle=False) as archive:
            depths = archive['depths'].copy()
            intrinsic = archive['intrinsic'].copy()
            cam_c2w = archive['cam_c2w'].copy()
        if not ((intrinsic.shape == (4,)) or (intrinsic.shape == (3,3))):
            raise ValueError(f'unknown actual MegaSAM intrinsics layout: {intrinsic.shape}')
        fx, fy, cx, cy = wrapper._parse_intrinsic(intrinsic)
        k = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
        observations = []
        for position, index in enumerate(ids):
            depth = depths[position].astype(np.float32)
            h, w = depth.shape
            rgb = cv2.resize(frames[index], (w,h), interpolation=cv2.INTER_LINEAR)[..., ::-1].copy()
            observations.append(save_observation(case,folder,index,rgb,depth,k,masks[index],cam_c2w[position]))
        write(folder/'observations.json', observations)
        write(folder/'geometry.json', dict(input_frame_ids=ids, geometry_metadata=geometry.metadata,
                                          reconstruction_code_identity=wrapper._reconstruction_code_identity(),
                                          K=k.tolist(), filtering=filtering_settings(cfg), observation_policy=MASK_POLICY, mask_erosion_pixels=Config().erosion_pixels, mask_mapping_applied=False, filter_source_sha256=sha(ROOT / 'robot/preprocessing/depth/link5_depth_filter.py'),
                                          full_scene_RGB=True, input_video_sha256=sha(sequence)))
        status.update(status='complete', mask_sha256=sha(mask_path), selected_frames=ids, observation_count=len(observations),
                      observation_policy=MASK_POLICY, mask_erosion_pixels=Config().erosion_pixels, mask_mapping_applied=False)
    except Exception as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write(folder/'status.json', status)


def prepare_locked(config,case,output):
    """Allow shared-GPU capacity expansion without duplicate mutable case work."""
    folder=output/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    with (folder/'preparation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return prepare_case(config,case,output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--video-id', required=True)
    args = parser.parse_args()
    config = read(args.config)
    matches = [c for c in config['cases'] if c['id']==args.video_id]
    if len(matches) != 1:
        raise ValueError('video id must identify exactly one frozen case')
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or (torch.ones(2,device='cuda')+1).sum().item()!=4:
        raise RuntimeError('native GPU preflight failed')
    prepare_locked(config, matches[0], Path(config['output']))
    print('LINK5_OBSERVATIONS_COMPLETE', args.video_id, flush=True)


if __name__ == '__main__':
    main()
