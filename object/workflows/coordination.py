"""Object preprocessing and anomaly adapters; no object geometry/rigidity stage."""
from dataclasses import asdict
import csv
import io
from pathlib import Path
import shutil

from infrastructure.pdibench.coordinator import digest, read, write


def named_masks(path, name):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:
        names=z['object_names'].tolist()
        if names.count(name)!=1:raise ValueError('Expected exactly one '+name)
        return z['object_masks'][:,names.index(name)].astype(bool)


def object_masks(request):
    from object.preprocessing.segmentation.segment import segment
    out=Path(request['directory']);r=request['config']['resources'];case=request['case']
    models=request['config']['vlm'].get('object_models',['gpt-6-luna','gemini-3.8-flash'])
    model=models[min(request['attempt']-1,len(models)-1)]
    result=segment(Path(case['video']),Path(case['prompt_file']).read_text(),out/'masking',Path(r['sam3_checkpoint']),Path(r['sam3_bpe']),models=(model,))
    if result['status']!='complete':raise ValueError('Object grounding/SAM failed: '+str(result['vlm_attempts'][-1].get('error','unknown')))
    return {'masking':str(out/'masking'),'target_object':result['target_object']}


def object_tracks(request):
    from infrastructure.shared.inference.tracking import TrackWrapper
    from robot.workflows.pipeline import save_track_result
    from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
    out=Path(request['directory']);video=Path(request['case']['video']);seg=Path(request['dependencies']['object_masks'])/'masking/segmentation.npz'
    a=load_multi_object_segmentation(seg,video)
    tracker=TrackWrapper(checkpoint=request['config']['resources']['tracker_checkpoint'],device='cuda')
    prepared=tracker.prepare_multi(str(video),a.object_masks[0],a.object_names,grid_size=10,bg_grid_size=15,background_dilation=5,max_dim=880,object_query_counts={'task_object':100})
    tracks=tracker.track_prepared(prepared,'exact-group');save_track_result(out/'cotracker_exact-group.npz',tracks)
    return {'tracks':str(out/'cotracker_exact-group.npz'),'retained_tracks':tracks.object_tracks[0].shape[1]}


def object_crops(request):
    import numpy as np
    from object.preprocessing.occlusion import occlusion
    from object.preprocessing.crop_pairs.reference_visible_pixels import export_case
    from object.preprocessing.crop_pairs.frame_selection import select_case
    from object.replay.frame_selection_gallery import write_gallery
    from object.replay.occlusion.occlusion_replay import export as replay
    from infrastructure.shared.contracts.mask_quality import MASK_QUALITY_METHOD
    out=Path(request['directory']);case=request['case'];name=case['id'];folder=out/'cases'/name
    objp=Path(request['dependencies']['object_masks'])/'masking/segmentation.npz'
    gripp=Path(request['dependencies']['mask_join'])/'segmentation.npz'
    trackp=Path(request['dependencies']['object_tracks'])/'cotracker_exact-group.npz'
    objects=named_masks(objp,'task_object');grippers=named_masks(gripp,'link7')
    with np.load(trackp,allow_pickle=False) as z:
        i=z['object_names'].tolist().index('task_object');a,b=z['object_offsets'][i:i+2];tracks=z['tracks'][:,a:b];visibility=z['visibility'][:,a:b]
        import json
        if tuple(json.loads(str(z['metadata_json']))['source_hw'])!=objects.shape[1:]:raise ValueError('Track coordinate grid mismatch')
    config=occlusion.Config();rows=occlusion.detect(objects,grippers,tracks,visibility,config)
    data={'method':'gripper-occlusion-v2','case':name,'frame_index_base':0,'mask_quality_method':MASK_QUALITY_METHOD,
        'failed_mask_frames':np.flatnonzero([not x['mask_valid'] for x in rows]).tolist(),'failed_mask_intervals':occlusion.runs([not x['mask_valid'] for x in rows]),
        'detector_source_sha256':digest(Path(occlusion.__file__)),'source_video_sha256':digest(case['video']),'config':asdict(config),
        'inputs':{k:{'path':str(p),'sha256':digest(p)} for k,p in [('object',objp),('gripper',gripp),('tracks',trackp)]},
        'flagged_intervals':occlusion.runs([x['flagged'] for x in rows]),'flagged_frames':sum(x['flagged'] for x in rows),'frame_count':len(rows),'frames':rows}
    write(folder/'occlusion/detection.json',data)
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows);(folder/'occlusion/frames.csv').write_text(stream.getvalue())
    (folder/'replay').mkdir();(folder/'replay/source.mp4').symlink_to(Path(case['video']))
    crop=out/'crops'/name;export_case(folder,crop,[]);selection=select_case(folder,crop,10)
    replay(folder);write_gallery(out/'crops')
    if selection['status']!='complete':raise ValueError(f'Crop selection incomplete: {selection["selected_count"]}/10, unsatisfied recovery {selection["unsatisfied_recovery_count"]}')
    return {'crops':str(out/'crops'),'selected_frames':[r['frame'] for r in selection['selected_frames']]}


def object_anomaly(request):
    from object.scoring.anomalydino.anomalydino import AnomalyDINOScorer
    from object.scoring.anomalydino.runner import prepare_selected, run_pairs, read_pairs
    from object.replay.annotate_crops import annotate
    out=Path(request['directory']);resources=request['config']['resources']
    # Annotation modifies selections; keep the preprocessing stage immutable.
    shutil.copytree(Path(request['dependencies']['object_crops'])/'crops',out/'crops')
    manifest=prepare_selected(out/'crops',[]);write(out/'pair_manifest.json',manifest)
    _,pairs=read_pairs(out/'pair_manifest.json')
    scorer=AnomalyDINOScorer(model_name='dinov2_vitb14',device='cuda:0',resolution=448,rotation=True,masking=False,faiss_on_cpu=True,dino_repo=resources['dino_repo'],checkpoint_path=Path(resources['anomaly_checkpoint']))
    run_pairs(scorer,pairs,out/'crops',out/'anomaly',manifest=manifest)
    result=annotate(out/'crops',out/'anomaly')
    return {'scored_pairs':result['scored_pairs'],'gallery':str(out/'crops/selection_gallery.html')}


def run_stage(request):
    return globals()[request['stage']](request)
