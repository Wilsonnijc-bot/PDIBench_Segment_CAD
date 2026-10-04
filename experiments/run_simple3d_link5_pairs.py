"""Fresh pair-visible link5 MegaSaM reconstruction and native Simple3D evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import traceback
import uuid

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'PDI-Bench-edited/src')]
from experiments.simple3d_pipeline import distribution, preprocess, sha, write_csv, write_json
from experiments.simple3d_pair_visibility import Config, depth_filter, erode, restore_cleaned_mask
from pdi_eval.object_deformation_wrapper.reference_visible_pixels import mask_bbox, rgba_crop, write_png


def read_mask(path):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(path)
    return mask > 0


def save_mask(path, mask):
    if not cv2.imwrite(str(path), mask.astype(np.uint8)*255):
        raise OSError(path)


def pair_geometry(row, selection, inputs, case, pair, frames, selected_masks, pair_masks, config, stage):
    """Reconstruct all 11 selected images jointly, keeping only this pair's depth.

    Nine real selected images provide the native temporal tracker's context.
    Only reference and test views are evaluated; their input supports are this
    pair's eroded (bootstrap) or depth-cleaned (final) corresponding masks.
    All context RGB is masked to its eroded Link5 support. No old XYZ is read.
    """
    from pdi_eval.perception.mega_sam_wrapper import MegaSamWrapper
    from pdi_eval.experiment.runner import clean_megasam_intermediates
    ids = sorted(frames)
    ref, test = selection['reference_frame_id'], selection['test_frame_id']
    masks = {t: erode(selected_masks[t], config.erosion_px) for t in ids}
    masks[ref], masks[test] = pair_masks
    # One integer crop shared by the sequence, preserving all real pixel content
    # and source coordinates. Hard foreground masks determine inference pixels.
    bbox = mask_bbox(np.logical_or.reduce(list(masks.values())))
    x0, y0, x1, y1 = bbox
    width, height = x1-x0, y1-y0
    width += width % 2; height += height % 2
    scene = 's3dp_'+row['video_id'].replace('.', '_')+'_'+uuid.uuid4().hex[:12]
    sequence = pair/f'{stage}_input.mp4'
    writer = cv2.VideoWriter(str(sequence), cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    if not writer.isOpened():
        raise ValueError('cannot create pair-specific MegaSaM video')
    cropped_masks = []
    try:
        for t in ids:
            bgr = frames[t]
            image = np.zeros((height, width, 3), np.uint8)
            cmask = np.zeros((height, width), bool)
            sh, sw = bgr[y0:y1, x0:x1].shape[:2]
            cmask[:sh, :sw] = masks[t][y0:y1, x0:x1]
            image[:sh, :sw] = bgr[y0:y1, x0:x1]
            image[~cmask] = 0
            cropped_masks.append(cmask)
            writer.write(image)
            if t in (ref, test):
                role = 'reference' if t == ref else 'test'
                cv2.imwrite(str(pair/f'{role}_{stage}_input.png'), image)
                save_mask(pair/f'{role}_{stage}_input_mask.png', cmask)
    finally:
        writer.release()
    cropped_masks = np.asarray(cropped_masks)
    # Native MegaSaM uses video basename as scratch scene identity.
    native_sequence = pair/f'{scene}.mp4'
    sequence.rename(native_sequence)
    wrapper = MegaSamWrapper(device='cuda')
    meta = dict(status='running', cache_hit=False, scene_name=scene, input_frame_ids=ids,
                evaluated_frame_ids=[ref, test], crop_bbox_xyxy=list(bbox), input_canvas_hw=[height, width],
                input_video_path=str(sequence), input_pixel_policy='RGB outside each foreground mask set to black',
                context_policy='same eleven selected real images; nine unscored Link5-masked context views',
                geometry_code_identity=wrapper._reconstruction_code_identity())
    try:
        geometry = wrapper.infer_shared(str(native_sequence), cropped_masks[:, None], cache_dir=None)
        if geometry.frames_count != len(ids) or geometry.metadata['cache_hit']:
            raise ValueError('fresh cropped sequence geometry incomplete or cached')
        root = Path(wrapper.mega_sam_root)
        cvd = root/'outputs_cvd'/f'{scene}_sgd_cvd_hr.npz'
        droid = root/'outputs'/f'{scene}_droid.npz'
        native = cvd if cvd.is_file() else droid
        with np.load(native, allow_pickle=False) as z:
            positions = [ids.index(ref), ids.index(test)]
            depths = z['depths'][positions].copy()
            poses = z['cam_c2w'][positions].copy()
            intrinsic = z['intrinsic'].copy()
        points = geometry.pointmaps[positions].copy()
        depth_path = pair/f'{stage}_reconstruction.npz'
        np.savez_compressed(depth_path, depths=depths, cam_c2w=poses, intrinsic=intrinsic,
            frame_ids=[ref, test], crop_bbox_xyxy=bbox, input_canvas_hw=[height, width])
        grids, clean_source, statistics = [], [], []
        for i, (role, t) in enumerate((('reference', ref), ('test', test))):
            h, w = depths[i].shape
            m = cv2.resize(cropped_masks[ids.index(t)].astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST).astype(bool)
            final, rejected, info = depth_filter(depths[i].astype(np.float32), m, config)
            save_mask(pair/f'{role}_{stage}_rejected_grid.png', rejected)
            save_mask(pair/f'{role}_{stage}_valid_grid.png', final)
            source_final, source_rejected = restore_cleaned_mask(masks[t], rejected, bbox, [height,width])
            save_mask(pair/f'{role}_{stage}_rejected.png', source_rejected)
            save_mask(pair/f'{role}_{stage}_valid.png', source_final)
            info.update(source_final_pixel_count=int(source_final.sum()), source_rejected_pixel_count=int(source_rejected.sum()),
                        pixel_grid_hw=[h,w], mask_grid_resampling='nearest neighbor; depth cleanup on native MegaSaM depth grid')
            if final.sum() < config.minimum_pixels or info['retained_fraction'] < config.minimum_depth_retained_fraction:
                raise ValueError(f'{role}: insufficient {stage} depth support: {info}')
            grids.append(final); clean_source.append(source_final); statistics.append(info)
        meta.update(status='complete', refinement='CVD' if native == cvd else 'raw_DROID_CVD_unavailable',
                    reconstruction_path=str(depth_path), pointmap_shape=list(points.shape), depth_filter=statistics,
                    geometry_metadata=geometry.metadata)
        return points, grids, clean_source, meta
    except Exception as exc:
        meta.update(status='failed', failure_reason=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(pair/f'{stage}_geometry.json', meta)
        native_sequence.rename(sequence)
        clean_megasam_intermediates(scene)


def score_pair(row, selected, inputs, case, frames, masks, config):
    from experiments.simple3d_adapter import build_simple3d_reference, score_simple3d
    pair = case/f"bin_{selected['temporal_bin']:02d}"
    pair.mkdir(exist_ok=True)
    result = dict(video_id=row['video_id'], reference_type='pair_specific_first_observed_frame',
        reference_frame_id=selected['reference_frame_id'], test_frame_id=selected['test_frame_id'],
        temporal_bin=selected['temporal_bin'], timestamp_seconds=selected['timestamp_seconds'],
        raw_mask_area=selected['raw_mask_area'], selected_eroded_mask_area=selected['eroded_mask_area'],
        correspondence=selected.get('correspondence'), pair_path=str(pair),
        preprocessing_config=asdict(config),
        preprocessing_source_sha256=sha(ROOT/'experiments/simple3d_pair_visibility.py'),
        megasam_status='not_attempted', simple3d_status='invalid', scalar_anomaly_score=None,
        failure_reason=selected.get('failure_reason'), point_anomaly_scores_path=None,
        reference_point_cloud_path=None, test_point_cloud_path=None)
    source = inputs/selected['pair_inputs']
    for p in source.glob('*.png'):
        shutil.copy2(p, pair/p.name)
    if selected['status'] != 'ready':
        write_json(pair/'comparison.json', result)
        return result
    try:
        ref, test = selected['reference_frame_id'], selected['test_frame_id']
        initial = [read_mask(pair/f'{role}_eroded.png') for role in ('reference','test')]
        _, _, cleaned, bootstrap = pair_geometry(row, selected, inputs, case, pair, frames, masks, initial, config, 'bootstrap')
        result['bootstrap_megasam_status'] = bootstrap['refinement']
        points, valid, _, final = pair_geometry(row, selected, inputs, case, pair, frames, masks, cleaned, config, 'final')
        result['megasam_status'] = final['refinement']
        result['depth_filter'] = dict(bootstrap=bootstrap['depth_filter'], final=final['depth_filter'])
        result['clouds'] = {}
        x0,y0,x1,y1 = final['crop_bbox_xyxy']
        height,width = final['input_canvas_hw']
        for i, (role,t) in enumerate((('reference',ref), ('test',test))):
            yy,xx = np.where(valid[i])
            h,w = valid[i].shape
            rgb = cv2.imread(str(pair/f'{role}_final_input.png'))
            rgb = cv2.resize(rgb, (w,h))[yy,xx,::-1]/255
            cloud, info = preprocess(points[i,yy,xx], pixels=np.column_stack((yy,xx)), rgb=rgb)
            # Continuous source positions are retained separately from native grid.
            source_xy = np.column_stack(((cloud['pixels_yx'][:,1]+.5)*width/w+x0-.5,
                                         (cloud['pixels_yx'][:,0]+.5)*height/h+y0-.5)).astype(np.float32)
            cloud['source_pixels_xy'] = source_xy
            cloud_path = pair/f'{role}_cloud.npz'
            np.savez_compressed(cloud_path, **cloud)
            info.update(status='complete', frame_id=t, pixel_grid_hw=[h,w], source_hw=list(masks[t].shape),
                        crop_bbox_xyxy=final['crop_bbox_xyxy'], input_canvas_hw=[height,width],
                        depth_filter=final['depth_filter'][i], megasam_status=final['refinement'],
                        cloud_path=str(cloud_path), point_pixels='native cropped MegaSaM grid; source_pixels_xy includes crop/resize transform')
            write_json(cloud_path.with_suffix('.json'), info)
            result[f'{role}_point_cloud_path'] = str(cloud_path)
            result[f'valid_{role}_point_count'] = info['valid_point_count']
            result[f'sampled_{role}_point_count'] = len(cloud['xyz'])
            result[f'{role}_original_mask_pixel_count'] = int(masks[t].sum())
            result[f'{role}_common_mask_pixel_count'] = int(read_mask(pair/f'{role}_common.png').sum())
            result[f'{role}_eroded_mask_pixel_count'] = int(initial[i].sum())
            result[f'{role}_final_valid_pixel_count'] = int(valid[i].sum())
            result[f'{role}_rejected_depth_pixel_count'] = final['depth_filter'][i]['rejected_depth_pixel_count']
            result[f'{role}_bootstrap_rejected_depth_pixel_count'] = bootstrap['depth_filter'][i]['rejected_depth_pixel_count']
            result['clouds'][role] = info
        with np.load(pair/'reference_cloud.npz', allow_pickle=False) as z:
            reference = build_simple3d_reference(z['xyz'])
        np.savez_compressed(pair/'reference_prototypes.npz', features=reference.method.patch_lib.cpu().numpy(),
                            coreset_indices=reference.method.coreset_idx.cpu().numpy())
        with np.load(pair/'test_cloud.npz', allow_pickle=False) as z:
            scores, scalar = score_simple3d(reference, z['xyz'])
        top = np.argsort(-scores, kind='stable')[:80]
        if not np.isclose(scores[top].mean(), scalar, rtol=2e-6, atol=1e-6):
            raise ValueError('official top-80 scalar verification failed')
        np.save(pair/'point_scores.npy', scores, allow_pickle=False)
        np.savez_compressed(pair/'top80.npz', point_indices=top, scores=scores[top])
        result.update(simple3d_status='complete', failure_reason=None, scalar_anomaly_score=float(scalar),
            point_anomaly_scores_path=str(pair/'point_scores.npy'), top80_path=str(pair/'top80.npz'),
            reference_memory_path=str(pair/'reference_prototypes.npz'), reference_memory_rebuilt=True)
        del reference
    except Exception as exc:
        result.update(simple3d_status='failed', failure_reason=str(exc), traceback=traceback.format_exc())
        if (pair/'final_geometry.json').exists():
            result['megasam_status'] = json.loads((pair/'final_geometry.json').read_text()).get('status')
        elif (pair/'bootstrap_geometry.json').exists():
            result['megasam_status'] = 'bootstrap_'+json.loads((pair/'bootstrap_geometry.json').read_text()).get('status')
    write_json(pair/'comparison.json', result)
    print('PAIR_FINISHED', row['video_id'], selected['temporal_bin'], result['simple3d_status'],
          result['scalar_anomaly_score'], result['failure_reason'], flush=True)
    return result


def run_case(row, inputs, output, config, resume):
    import torch
    torch.set_num_threads(2)
    case = output/'cases'/row['video_id']
    case.mkdir(parents=True, exist_ok=True)
    terminal = case/'comparisons.json'
    if terminal.exists() and resume:
        return json.loads(terminal.read_text())
    write_json(case/'status.json', dict(status='running', video_id=row['video_id']))
    if sha(row['video_path']) != row['video_sha256']:
        raise ValueError('source video differs from frozen selection')
    mask_path = inputs/row['selected_masks_path']
    if sha(mask_path) != row['selected_masks_sha256']:
        raise ValueError('selected masks differ from frozen manifest')
    with np.load(mask_path, allow_pickle=False) as z:
        masks = dict(zip(z['frame_ids'].tolist(), z['masks'].astype(bool)))
    frames = {}
    cap = cv2.VideoCapture(row['video_path'])
    try:
        for t in masks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok,bgr = cap.read()
            if not ok or bgr.shape[:2] != masks[t].shape:
                raise ValueError('unreadable source frame or mismatched mask grid')
            frames[t] = bgr
            dest = case/'frames'/f'{t:06d}'
            dest.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dest/'frame.jpg'), bgr, [cv2.IMWRITE_JPEG_QUALITY,95])
            save_mask(dest/'mask.png', masks[t])
            write_png(dest/'crop.png', rgba_crop(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), masks[t], mask_bbox(masks[t])))
    finally:
        cap.release()
    write_json(case/'selection.json', row)
    results = []
    for selected in row['tests']:
        path = case/f"bin_{selected['temporal_bin']:02d}"/'comparison.json'
        if resume and path.exists():
            results.append(json.loads(path.read_text())); continue
        results.append(score_pair(row, selected, inputs, case, frames, masks, config))
        torch.cuda.empty_cache()
    write_json(terminal, results)
    write_json(case/'status.json', dict(status='completed', video_id=row['video_id'],
        successful_scores=sum(r['simple3d_status']=='complete' for r in results), expected_scores=10))
    print('VIDEO_FINISHED', row['video_id'], flush=True)
    return results


def aggregate(output, manifest):
    records = []
    completed = 0
    for row in manifest['rows']:
        case = output/'cases'/row['video_id']
        terminal = case/'comparisons.json'
        if terminal.exists():
            records.extend(json.loads(terminal.read_text())); completed += 1
        else:
            for p in sorted(case.glob('bin_*/comparison.json')):
                records.append(json.loads(p.read_text()))
    scores = [r['scalar_anomaly_score'] for r in records if r['simple3d_status']=='complete']
    summary = dict(status='completed' if completed==len(manifest['rows']) else 'running',
        videos_processed=completed, video_count=len(manifest['rows']), expected_comparisons=10*len(manifest['rows']),
        terminal_pairs=len(records), successful_simple3d_scores=len(scores),
        successful_megasam_pair_reconstructions=sum(r['megasam_status'] in ('CVD','raw_DROID_CVD_unavailable') for r in records),
        visibility_accepted_pairs=sum(s['status']=='ready' for r in manifest['rows'] for s in r['tests']),
        score_distribution=distribution(scores), failures=[{k:r.get(k) for k in ('video_id','temporal_bin','test_frame_id','failure_reason')}
            for r in records if r['simple3d_status']!='complete'])
    write_json(output/'comparisons.json', records)
    # Keep nested evidence in JSON; flat CSV remains easy to analyze.
    flat = [{k:v for k,v in r.items() if k not in ('traceback','clouds','correspondence','depth_filter','preprocessing_config')} for r in records]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    write_csv(output/'comparisons.csv', flat, fields=fields or None)
    write_json(output/'metadata/summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'results/simple3d-20261003-run2/link5_pair_visible_anchor')
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--limit', type=int)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    output = a.root.resolve(); inputs = output/'inputs'
    manifest = json.loads((inputs/'manifest.json').read_text())
    saved_config = dict(manifest['config'])
    # The first preview predates explicit visibility modes.
    saved_config.setdefault('visibility_mode', 'strict_hulls')
    saved_config.setdefault('track_quality_gate', True)
    config = Config(**saved_config).validate()
    if a.limit:
        manifest = dict(manifest, rows=manifest['rows'][:a.limit])
    if a.preflight:
        import torch
        from experiments.simple3d_adapter import build_simple3d_reference, score_simple3d, provenance
        from pdi_eval.perception.mega_sam_wrapper import MegaSamWrapper
        assert torch.cuda.is_available() and float(torch.ones(2, device='cuda').sum())==2
        wrapper = MegaSamWrapper()
        required = [Path(wrapper.da_ckpt), Path(wrapper.megasam_weights), Path(wrapper.raft_weights)]
        for f in required:
            assert f.is_file() and f.stat().st_size, str(f)
        for row in manifest['rows']:
            assert sha(row['video_path'])==row['video_sha256'], row['video_id']
            assert sha(inputs/row['selected_masks_path'])==row['selected_masks_sha256']
            assert len(row['tests'])==10
        x = np.random.default_rng(1).normal(size=(256,3)).astype(np.float32)
        bank = build_simple3d_reference(x); scores, scalar = score_simple3d(bank,x)
        assert np.isfinite(scores).all() and np.isfinite(scalar)
        write_json(output/'metadata/preflight.json', dict(status='passed', gpu=torch.cuda.get_device_name(),
            python=sys.executable, cuda=torch.version.cuda, torch=torch.__version__, method=provenance(),
            source_files_sha256={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__), ROOT/'experiments/simple3d_pair_visibility.py')},
            manifest_sha256=sha(inputs/'manifest.json'), checkpoint_paths=[str(f) for f in required]))
        print('PREFLIGHT_PASSED', flush=True); return 0
    if (output/'metadata/run.json').exists() and not a.resume:
        raise FileExistsError('existing experiment; use --resume to finish its missing pairs')
    write_json(output/'metadata/run.json', dict(started=datetime.now(timezone.utc).isoformat(), config=asdict(config),
        expected_videos=len(manifest['rows']), expected_pairs=len(manifest['rows'])*10,
        input_manifest_sha256=sha(inputs/'manifest.json'), command=sys.argv,
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (Path(__file__),ROOT/'experiments/simple3d_pair_visibility.py',ROOT/'experiments/simple3d_adapter.py')}))
    unexpected = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_case, row, inputs, output, config, a.resume):row for row in manifest['rows']}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                row = futures[future]
                unexpected.append(dict(video_id=row['video_id'], reason=str(exc), traceback=traceback.format_exc()))
                case = output/'cases'/row['video_id']
                write_json(case/'status.json', dict(status='failed', reason=str(exc)))
                failed = [dict(video_id=row['video_id'], temporal_bin=s['temporal_bin'], test_frame_id=s['test_frame_id'],
                    reference_frame_id=s['reference_frame_id'], simple3d_status='failed', megasam_status='not_attempted',
                    scalar_anomaly_score=None, failure_reason='case setup failed: '+str(exc)) for s in row['tests']]
                write_json(case/'comparisons.json', failed)
            print('PROGRESS', json.dumps(aggregate(output, manifest), default=str)[:500], flush=True)
    summary = aggregate(output, manifest)
    write_json(output/'metadata/unexpected_failures.json', unexpected)
    print('BATCH_FINISHED', json.dumps({k:v for k,v in summary.items() if k!='failures'}), flush=True)
    return 1 if unexpected else 0


if __name__ == '__main__':
    raise SystemExit(main())
