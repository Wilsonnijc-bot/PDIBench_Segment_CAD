"""Explicit-input adapters for persistent masking and native V1 scoring."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from infrastructure.deformation_detect.coordinator import digest, read, write
from infrastructure.shared.contracts.vlm_failure import raise_timeout


def clone_work(source, target):
    """Each attempt owns its mutable native provenance; predecessors stay frozen."""
    shutil.copytree(source,target)
    old=str(source);new=str(target)
    # Native provenance contains absolute paths. Rebase only this owned snapshot.
    for p in target.rglob('*.json'):
        def replace(value):
            if isinstance(value,str):return new+value[len(old):] if value==old or value.startswith(old+os.sep) else value
            if isinstance(value,list):return [replace(x) for x in value]
            if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
            return value
        write(p,replace(read(p)))


def persistent(request):
    from robot.preprocessing.link7_persistent import pipeline, naive_sam3
    out=Path(request['directory']);case=request['case'];name=case['id'];resources=request['config']['resources'];stage=request['stage']
    work=out/'work'
    if stage=='link7_initial':
        naive_sam3.run(SimpleNamespace(work=work/'naive',cases=[name],prompt='robot gripper',initializer='reference-box',ffmpeg=resources['ffmpeg']))
        d=read(work/'naive'/name/'provenance.json')
        if d['status']!='propagated' or d['returned_frames']!=d['source_frame_count']:raise ValueError('Incomplete initial SAM propagation')
        return {'work':str(work)}
    dependency={'vlm1':'link7_initial','vlm2':'vlm1','link7_masks':'vlm2'}[stage]
    previous=request.get('previous_attempt')
    # VLM1 retries preserve successful frame diagnoses, retrying only errors.
    reuse=stage=='vlm1' and previous and (Path(previous)/'work/provenance.json').is_file()
    source=Path(previous)/'work' if reuse else Path(request['dependencies'][dependency])/'work'
    clone_work(source,work)
    refs=out/'references'
    if case.get('vlm1_reference'):
        target=refs/name/'inputs/reference.png';target.parent.mkdir(parents=True);shutil.copy2(case['vlm1_reference'],target)
    args=SimpleNamespace(work=work,cases=[name],level='local',examples=[Path(p) for p in resources['vlm2_examples']],reference_root=refs,ffmpeg=resources['ffmpeg'],vlm2_fallback_attempts=0)
    if stage=='vlm1':
        if not (work/'provenance.json').exists():pipeline.prepare(args)
        record=pipeline.load(args);r=record['results'][name]
        if r['status']=='failed_prepare':raise ValueError(r.get('error','Failed preparation'))
        if reuse:
            r['status']='vlm1_pending'
            for d in r.get('diagnoses',[]):
                if d.get('parse_error') or d.get('state')=='unclear':
                    d['retry_feedback']={'error':d.get('parse_error','Previously unclear comparison'),
                                         'answer':d.get('raw_response','')}
                    d.pop('raw_response',None);d.pop('parse_error',None);d.pop('failure',None)
            pipeline.commit(args,record)
        pipeline.select_frames(args)
        r=pipeline.load(args)['results'][name]
        for diagnosis in r.get('diagnoses',[]):raise_timeout(diagnosis)
        if any(d.get('parse_error') for d in r.get('diagnoses',[])):raise ValueError('VLM1 frame responses exhausted this attempt; invalid response is not no-deformation')
        if r.get('diagnoses') and not any(d.get('state') in {'normal','deformed'} for d in r['diagnoses']):
            raise ValueError('VLM1 has no assessable frames; unclear is not no-deformation')
        allowed={'vlm2_pending','no_confirmed_deformation','no_naive_surge'}
    elif stage=='vlm2':
        pipeline.prompt_sam(args);r=pipeline.load(args)['results'][name]
        allowed={'sam_pending','no_confirmed_deformation','no_naive_surge'}
    else:
        pipeline.segment(args);pipeline.validate(args);r=pipeline.load(args)['results'][name]
        allowed={'completed_checks','no_confirmed_deformation','no_naive_surge'}
    raise_timeout(r)
    if r['status'] not in allowed:raise ValueError(f'{stage}: {r["status"]}: {r.get("error", "")}')
    return {'work':str(work),'native_status':r['status']}


def mask_join(request):
    import numpy as np
    from robot.preprocessing.link7_persistent.vlm3_overmask import repair_case
    from robot.preprocessing.segmentation.mask_merge import _replace_named_mask, build_repaired_segmentation
    from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
    out=Path(request['directory']);case=request['case'];name=case['id'];video=Path(case['video'])
    work=Path(request['dependencies']['link7_masks'])/'work';r=read(work/'provenance.json')['results'][name]
    if r['source_sha256']!=digest(video):raise ValueError('Persistent video identity changed')
    source=Path(r['mask_source']) if r['status']=='completed_checks' else work/'naive'/name/'masks.npz'
    expected=r['masks_sha256'] if r['status']=='completed_checks' else r['naive_masks_sha256']
    if digest(source)!=expected:raise ValueError('Persistent mask identity changed')
    with np.load(source,allow_pickle=False) as z:before=z['masks'].astype(bool)
    object_dir=Path(request['dependencies']['object_masks'])/'masking';objp=object_dir/'segmentation.npz';ground=read(object_dir/'grounding.json')
    if ground['video_sha256']!=digest(video) or ground['segmentation_sha256']!=digest(objp):raise ValueError('Object input identity mismatch')
    archive=load_multi_object_segmentation(objp,video)
    objects=archive.object_masks[:,archive.object_names.index('task_object')]
    if before.shape!=objects.shape:raise ValueError('Mask frame grid mismatch')
    core=out/'core_segmentation.npz'
    if case.get('base_segmentation'):
        load_multi_object_segmentation(case['base_segmentation'],video)
        _replace_named_mask(Path(case['base_segmentation']),core,before,'link7')
    else:
        np.savez_compressed(core,object_masks=before[:,None],object_names=np.array(['link7']),object_ids=np.array([0]),masks=before)
    object_input={'path':str(objp),'sha256':digest(objp),'target_object':ground['target_object'],'video_sha256':digest(video)}
    feedback=''
    if request.get('previous_attempt'):
        old=Path(request['previous_attempt'])/'vlm3/repair.json'
        if old.is_file():
            candidate=(read(old).get('attempts') or [{}])[-1]
            if not candidate.get('failure',{}).get('defer'):
                feedback='\nPrevious candidate failed validation. '+json.dumps(candidate.get('error','Check all six point roles.'))
    repair=repair_case(name,video,before,objects,object_input,out/'vlm3',max_attempts=1,
        reference=Path(request['config']['resources']['vlm3_reference']),initial_feedback=feedback)
    repair['link7_input']={'path':str(core),'sha256':digest(core),'video_sha256':digest(video)}
    write(out/'vlm3/repair.json',repair)
    if not repair['accepted']:
        for candidate in repair['attempts']:raise_timeout(candidate)
    if not repair['accepted'] and repair['status']!='not_triggered':raise ValueError('Required VLM3 repair failed; downstream disabled rather than using suspect core mask')
    selected=out/'segmentation.npz'
    if repair['accepted']:build_repaired_segmentation(video=video,base_segmentation=core,repair_record=out/'vlm3/repair.json',output_npz=selected)
    else:shutil.copy2(core,selected)
    write(out/'selection.json',{'case':name,'source_video_sha256':digest(video),'core_sha256':digest(core),'selected_sha256':digest(selected),'policy':'accepted_vlm3' if repair['accepted'] else 'core_not_triggered','repair_status':repair['status']})
    return {'selected_segmentation':str(selected),'repair_status':repair['status']}


def single_link_masks(request, link):
    """Use maintained single-target SAM3; Link5 adds its independent point guard."""
    from robot.preprocessing.segmentation.sam3_dinov2_segment import build_parser, run
    from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
    out=Path(request['directory']);resources=request['config']['resources'];video=Path(request['case']['video'])
    command=[
        '--input',str(video),'--reference-dir',resources['robot_references'],
        '--output-npz',str(out/'segmentation.npz'),'--dinov2-model',resources['dino_directory'],
        '--sam3-checkpoint',resources['sam3_checkpoint'],'--sam3-bpe',resources['sam3_bpe'],
        '--selected-target',link,'--text-prompt','visual','--require-franka-links',
        '--reference-spatial-priors','--padding-fraction','0.10','--minimum-tracked-fraction','0.80']
    if link=='link5':
        command.extend(['--link5-vlm-guard','--link5-negative-points','2','--link5-guard-required',
                        '--link5-guard-reference',resources['link5_guard_reference']])
        if resources.get('link5_positive_guard_reference'):
            command.extend(['--link5-positive-guard-reference',resources['link5_positive_guard_reference']])
    args=build_parser().parse_args(command)
    metadata=run(args)
    archive=load_multi_object_segmentation(out/'segmentation.npz',video)
    if archive.object_names != (link,) or not archive.object_masks.any():
        raise ValueError(f'{link} segmenter did not produce a usable named mask')
    if metadata['targets'][0]['sam3_status'] != 'complete':
        raise ValueError(f'{link} SAM propagation was skipped')
    write(out/'provenance.json',{'video_sha256':digest(video),'segmentation_sha256':digest(out/'segmentation.npz')})
    return {'segmentation':str(out/'segmentation.npz')}


def link2_masks(request):
    return single_link_masks(request,'link2')


def link5_masks(request):
    return single_link_masks(request,'link5')


def scoring_segmentation(request):
    """Assemble robot inputs without modifying the object/link7 selected archive."""
    import numpy as np
    from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation, frame_measurements
    selected=Path(request['dependencies']['mask_join'])/'segmentation.npz'
    extras=[link for link in ('link2','link5') if link in request['config'].get('robot_links',['link7'])]
    if not extras:return selected
    video=Path(request['case']['video'])
    base=load_multi_object_segmentation(selected,video)
    names=list(base.object_names);ids=list(base.object_ids);masks=base.object_masks.copy()
    sources={}
    for link in extras:
        folder=Path(request['dependencies'][f'{link}_masks'])
        provenance=read(folder/'provenance.json')
        if provenance['video_sha256']!=digest(video) or provenance['segmentation_sha256']!=digest(folder/'segmentation.npz'):
            raise ValueError(f'{link.capitalize()} mask identity mismatch')
        extra=load_multi_object_segmentation(folder/'segmentation.npz',video)
        if extra.object_names != (link,):raise ValueError(f'Expected single-{link} segmentation')
        if link in names:
            masks[:,names.index(link)]=extra.object_masks[:,0]
        else:
            names.append(link);ids.append(max(ids)+1)
            masks=np.concatenate((masks,extra.object_masks),axis=1)
        sources[f'{link}_sha256']=digest(folder/'segmentation.npz')
    union=masks.any(axis=1);heights,centers,truncated=frame_measurements(union)
    target=Path(request['directory'])/'segmentation.npz'
    np.savez_compressed(target,object_masks=masks,object_names=np.asarray(names),object_ids=np.asarray(ids),
        masks=union,h_pixel=heights,x_center=centers,is_truncated=truncated)
    write(target.with_suffix('.json'),{'video_sha256':digest(video),'selected_link7_sha256':digest(selected),
        **sources,'segmentation_sha256':digest(target),'object_names':names})
    return target



def scoring_outcomes(metrics, links):
    """Expected track insufficiency completes measurement with an unavailable score."""
    outcomes = {}
    for link in links:
        item = metrics[link]
        if item['status'] == 'complete':
            outcomes[link] = {'outcome': 'scored', 'rigidity': item['breakdown']['epsilon_rigidity']}
        elif item.get('error_type') == 'insufficient_cotracker_tracks':
            outcomes[link] = {'outcome': 'insufficient_cotracker_tracks', 'rigidity': None,
                              'tracking': item.get('tracking', {}), 'reason': item.get('error')}
        else:
            raise ValueError(f'{link} scoring unavailable: {item.get("error_type",item["status"])}')
    return outcomes


def robot_score(request):
    from infrastructure.deformation_detect.worker import isolate_mega
    out=Path(request['directory']);resources=request['config']['resources'];case=request['case']
    segmentation=scoring_segmentation(request)
    links=request['config'].get('robot_links',['link7'])
    isolate_mega(request)
    # Dataset IDs with dots must not collapse to the same MegaSAM scene name.
    video=out/'input.mp4';video.symlink_to(Path(case['video']))
    import yaml
    config=yaml.safe_load(Path(resources['robot_config']).read_text())
    config.setdefault('multi_object_replay',{})['enabled']=True
    score_config=out/'robot-config.yaml';score_config.write_text(yaml.safe_dump(config))
    command=[sys.executable,'-u','-m','robot.workflows.score_v1','--config',str(score_config),'--input',str(video),
        '--segmentation-npz',str(segmentation),'--output-dir',str(out/'score'),
        '--geometry-cache-dir',str(out/'geometry-cache'),'--tracker-checkpoint',resources['tracker_checkpoint'],
        '--link7-tracker','cotracker3','--link7-point-filter','v1','--tracking-mode','exact-group',
        '--mask-label','Coordinated SAM masks + selected Link7 VLM3; V1 CoTracker']
    for link in links:command.extend(['--score-link',link])
    subprocess.run(command,check=True)
    metrics=read(out/'score/metrics.json')['modes']['exact-group']['objects']
    outcomes = scoring_outcomes(metrics, links)
    metric=metrics['link7']
    # Scratch is owned by this attempt; model symlinks are not result artifacts.
    shutil.rmtree(out/'mega_sam')
    return {'metrics':str(out/'score/metrics.json'),'rigidity':outcomes['link7']['rigidity'],'rigidity_by_link':{link:outcomes[link]['rigidity'] for link in links},'link_outcomes':outcomes,'reconstruction_audit':metric.get('reconstruction_audit')}


def run_stage(request):
    if request['stage']=='link2_masks':return link2_masks(request)
    if request['stage']=='link5_masks':return link5_masks(request)
    if request['stage']=='mask_join':return mask_join(request)
    if request['stage']=='robot_score':return robot_score(request)
    return persistent(request)
