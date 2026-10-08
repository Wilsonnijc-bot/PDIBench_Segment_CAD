"""Show a newly constructed four-frame0 reference before detector training."""
import argparse
import base64
import json
import os
import shutil
from pathlib import Path

import numpy as np

from .build_reference_replay import packed
from .common import FOUR_REFERENCE_IDS, MODES, ROOT, mode_folder, read, sha, split_path


def run(root,destination=None):
    root=Path(root);destination=Path(destination or root/'full_video/reference_replay')
    destination.mkdir(parents=True,exist_ok=True)
    receipt=read(root/'normal_reference/construction.json')
    if receipt['normal_video_ids']!=list(FOUR_REFERENCE_IDS) or receipt['normal_frame_ids']!=[0]*4 or receipt['later_frames_in_reference']:
        raise ValueError('replay reference must contain exactly the four requested frame0 clouds')
    pooled_path=root/'normal_reference/pooled_frame0.npz'
    if sha(pooled_path)!=receipt['pooled_sha256']:raise ValueError('constructed reference identity changed')
    norm_path=root/'link5_normalization.json'
    normalization=read(norm_path) if norm_path.exists() else dict(fixed_center=[0,0,0],fixed_scale=1,available=False)
    href=lambda path:os.path.relpath(path,destination)
    trained={};sizes={};phase1=[]
    for mode in MODES:
        training=mode_folder(root,'training',mode)
        done=training/'completion.json'
        trained[mode]=done.exists() and read(done)['status']=='complete'
        if trained[mode]:
            if read(done)['checkpoint_sha256']!=sha(training/'link5/final.pt'):raise ValueError('completed detector checkpoint changed')
            manifest=read(training/'normal_manifest.json')
            if {r['video_id'] for r in manifest}!=set(FOUR_REFERENCE_IDS) or any(r['frame_id']!=0 for r in manifest):raise ValueError('trained detector uses different normals')
            initial=read(training/'phase1_normal_codebook.json')
            if initial['normal_observations']!=4 or initial['split_sha256']!=sha(split_path(root)) or initial['normalization_sha256']!=sha(norm_path):raise ValueError('trained codebook reference changed')
            if not phase1:phase1=[r['size'] for r in initial['statistics']]
            elif phase1!=[r['size'] for r in initial['statistics']]:raise ValueError('paired normal codebook sizes differ')
            sizes[mode]=[r['size'] for r in read(training/'codebook_statistics.json')]
    training_complete=all(trained.values())
    if training_complete:
        comparison=read(root/'training_robot_structural/phase1_paired_comparison.json')
        if not comparison['tensors_match_within_tolerance'] or not comparison['hash_state_identical']:raise ValueError('paired codebooks differ')
    rows=[];all_camera=[]
    for entry in receipt['observations']:
        video=entry['video_id'];folder=root/'cases'/video/'observations/frame_00000';input_path=folder/'input.npz'
        if sha(input_path)!=entry['input_sha256']:raise ValueError('constructed frame0 input changed')
        with np.load(input_path,allow_pickle=False) as a:camera=a['xyz_camera'].copy()
        all_camera.append(camera)
        normalized=((camera-np.asarray(normalization['fixed_center']))/normalization['fixed_scale']).astype(np.float32)
        training_array=root/'normal_training_data/link5/train/normal'/(video+'.npy')
        if training_array.exists() and not np.array_equal(np.load(training_array,allow_pickle=False),normalized):raise ValueError('stored training normal differs from displayed input')
        observation=read(folder/'observation.json')
        successes=[p for p in (root/'cases'/video/'masking').glob('attempt-*')
                   if (p/'result.json').exists() and read(p/'result.json').get('status')=='complete']
        if len(successes)!=1:raise ValueError('ambiguous current masking attempt')
        selected=successes[0];guard=read(selected/'link5_guard.json')
        if sha(selected/'link5_guard.json')!=entry['guard_sha256']:raise ValueError('constructed guard changed')
        diagnostic=next(d for d in read(selected/'sam3_prompt_diagnostics.json') if d['target']=='link5')
        refinement=diagnostic['point_refinement']
        if refinement['frame_index']!=0 or refinement['points_xy']!=guard['selected_points_xy']:
            raise ValueError('SAM did not use the displayed VLM coordinates')
        segmentation=selected/'segmentation.npz'
        if sha(segmentation)!=entry['segmentation_sha256']:raise ValueError('constructed segmentation changed')
        with np.load(segmentation,allow_pickle=False) as a:
            mask=a['object_masks'][0,a['object_names'].tolist().index('link5')].astype(bool)
        with np.load(input_path,allow_pickle=False) as a:
            if not np.array_equal(mask,a['source_mask']):raise ValueError('source mask changed')
        images={k:href(selected/v) for k,v in dict(selected=guard['selected_image'],
            positive='link5_guard_positive_current.png',negative='link5_guard_negative_current.png',
            positive_reference='link5_guard_positive_reference.png',negative_reference='link5_guard_negative_reference.png').items()}
        if any(not (destination/p).is_file() for p in images.values()):raise ValueError('guard image missing')
        prompt=dict(decision=guard['decision'],original=guard['original_points_xy'],selected=guard['selected_points_xy'],
            labels=refinement['point_labels'],reviews=guard['reviews'],images=images,p1_vlm=True,
            calls=[{k:a.get(k) for k in ['role','model','requested_model','reasoning_effort','output_token_limit','timeout_seconds','elapsed_seconds','answer']} for a in guard['attempts']],
            record=href(selected/'link5_guard.json'),diagnostics=href(selected/'sam3_prompt_diagnostics.json'),
            successful_attempt=selected.name)
        rows.append(dict(video_id=video,count=len(camera),camera=packed(camera),normalized=packed(normalized),
            rgb=href(folder/'rgb.png'),mask=href(folder/'mask.png'),source_mask=href(folder/'source_mask.png'),
            support=href(folder/'depth_support.png'),source_image_hw=observation['source_image_hw'],
            native_depth_hw=observation['native_depth_hw'],prompt=prompt,
            prompt_variants=[dict(name='Fresh GPU SAM · three VLM positives',prompt=prompt)],
            input=href(input_path),input_sha256=sha(input_path),training_npy=href(training_array if training_array.exists() else input_path),training_sha256=sha(training_array if training_array.exists() else input_path),
            mask_area=observation['mask_area'],mask_erosion_pixels=observation['mask_erosion_pixels']))
    with np.load(pooled_path,allow_pickle=False) as a:
        if not np.array_equal(a['xyz_camera'],np.concatenate(all_camera)) or a['frame_ids'].tolist()!=[0]*4:
            raise ValueError('pooled cloud is not the exact four-frame0 concatenation')
    examples=[];geometry=None
    validation_path=root/'normal_reference/structural_examples.json'
    if validation_path.exists():
        validation=read(validation_path)
        if validation['status']!='passed' or validation['pooled_sha256']!=receipt['pooled_sha256']:
            raise ValueError('deformation examples belong to a different constructed reference')
        geometry=validation['geometry'];example_dir=root/'normal_reference/examples'
        normal=np.load(example_dir/'normal.npy',allow_pickle=False)
        if not np.array_equal(normal,np.concatenate(all_camera)):raise ValueError('deformation normal differs from constructed union')
        examples.append(dict(name='normal',count=len(normal),points=packed(normal),offset=packed(np.zeros_like(normal)),
            mask=base64.b64encode(np.zeros(len(normal),np.uint8)).decode(),parameters={},
            ply=href(example_dir/'normal.ply'),archive=href(example_dir/'normal.npy')))
        for check in validation['checks']:
            name=check['example'];path=example_dir/(name+'.npz')
            with np.load(path,allow_pickle=False) as a:
                if not np.array_equal(a['normal'],normal) or not np.array_equal(a['gt_offset'],normal-a['anomalous']):
                    raise ValueError('deformation example correspondence changed')
                examples.append(dict(name=name,count=len(normal),points=packed(a['anomalous']),offset=packed(a['gt_offset']),
                    mask=base64.b64encode(a['gt_mask'].astype(np.uint8).tobytes()).decode(),parameters=check['parameters'],checks=check,
                    ply=href(example_dir/(name+'.ply')),archive=href(path)))
    payload=dict(normals=rows,examples=examples,normalization=normalization,geometry=geometry,
        phase1_sizes=phase1,baseline_sizes=sizes.get('paper_original',[]),structural_sizes=sizes.get('robot_structural',[]),constructed=True,
        phase1_archive=href(root/'training/phase1_normal_codebook.pt'),trained_modes=trained,
        pooled=dict(archive=href(pooled_path),ply=href(root/'normal_reference/pooled_frame0.ply'),
            receipt=href(root/'normal_reference/construction.json'),point_count=receipt['point_count']),
        training_complete=training_complete,example_source='union of four frame0 clouds')
    (destination/'data.js').write_text('window.LINK5_REFERENCE='+json.dumps(payload,separators=(',',':'))+';\n')
    template=(ROOT/'infrastructure/shared/replay/rigidity_replay.html').read_text().split('<body>')[0]
    shutil.copy2(ROOT/'infrastructure/shared/replay/assets/plotly.min.js',destination/'plotly.min.js')
    template=template.replace('Rigidity evidence replay','Link5 four-frame0 normal reference')
    body=Path(__file__).with_name('reference_replay_body.html').read_text()
    body=body.replace('Original frame0 training observations and the local depth-filter preview. Updated VLM points have not been segmented; filtered reference clouds have not been used to retrain the codebooks.',
        'New reference: exactly four frame0 clouds, regenerated SAM masks with all three positives placed by the VLM, and the new depth filter. Constructed before detector retraining. Later frames are excluded.')
    body=body.replace('The VLM reviews P2/P3 and N1/N2 independently, keeping or replacing their proposals. Earlier runs kept P1 deterministic.', 'The VLM explicitly places P1/P2/P3 and reviews N1/N2. All five selected points are supplied to fresh SAM segmentation.')
    body=body.replace('20 normal references','4 frame0 references').replace('exact 20 codebook references','four new reference clouds').replace('all 20','all 4').replace('same 20','same four').replace('exact 20 previous training observations','four freshly constructed frame0 observations')
    body=body.replace('Previous normalized training array','Fresh filtered frame0 archive').replace('Filtered camera cloud · preview','Constructed pooled cloud')
    body=body.replace('Select the previous training clouds to see every point in the 20 stored normal arrays, or the new filter to inspect their retained observations. The filtered preview has not been used in training. Each original training visit sampled up to 10,000 observed points without replacement; historical minibatch indices were not saved.',
        'The overlay is the exact pooled reference: every retained point from the four frame0 clouds. No later frame, shape completion, individual rescaling or point replication is included. The codebook will receive these as four separate normal examples.')
    # Historical endpoint/prompt comparisons belong to the previous run.
    body='\n'.join(line for line in body.splitlines() if not line.startswith('<p class="muted">GPT-6.1') and not line.startswith('<p class="muted">New three-positive-point'))
    if training_complete:
        body=body.replace('Constructed before detector retraining. Later frames are excluded.','Both detector variants have been retrained on these four frame0 inputs. Later frames are excluded.')
        body=body.replace('The codebook will receive these as four separate normal examples.','Both codebooks received these as four separate normal examples.')
    (destination/'index.html').write_text(template+body)
    (destination/'app.js').write_text(Path(__file__).with_name('reference_replay_app.js').read_text())
    (destination/'verification.json').write_text(json.dumps(dict(status='verified',normal_files=4,
        frame_ids=[0]*4,video_ids=list(FOUR_REFERENCE_IDS),fresh_sam_points_match_guard=True,
        source_masks_match_segmentation=True,pooled_exact_concatenation=True,
        observed_normal_point_count=receipt['point_count'],detector_retrained=training_complete,
        later_frames_in_reference=False,construction_sha256=sha(root/'normal_reference/construction.json')),indent=2)+'\n')
    print('LINK5_FOUR_FRAME0_REFERENCE_REPLAY_READY',receipt['point_count'],destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--destination',type=Path)
    a=p.parse_args();run(a.root,a.destination)
