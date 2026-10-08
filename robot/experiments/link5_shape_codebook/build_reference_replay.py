"""Replay exact stored train20 normals and saved correspondence-preserving examples."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def packed(points):return base64.b64encode(np.ascontiguousarray(points,dtype='<f4').tobytes()).decode('ascii')


def run(root):
    active=root/'round_four_frame0'
    if (active/'normal_reference/construction.json').exists():
        from .build_constructed_reference_replay import run as constructed_replay
        return constructed_replay(active,root/'full_video/reference_replay')
    if (root/'normal_reference/construction.json').exists():
        from .build_constructed_reference_replay import run as constructed_replay
        return constructed_replay(root)
    destination=root/'full_video/reference_replay';destination.mkdir(parents=True,exist_ok=True)
    normalization=read(root/'link5_normalization.json');manifest=read(root/'training/train20_manifest.json')
    rows=[]
    def href(path):return os.path.relpath(path,destination)
    for row in manifest:
        video=row['video_id'];input_path=root/'cases'/video/'observations/frame_00000/input.npz'
        stored=root/'normal_training_data/link5/train/normal'/(video+'.npy')
        if sha(input_path)!=row['input_sha256']:raise ValueError('training observation changed')
        with np.load(input_path,allow_pickle=False) as a:camera=a['xyz_camera'].copy()
        normal=np.load(stored,allow_pickle=False)
        expected=((camera-np.asarray(normalization['fixed_center']))/normalization['fixed_scale']).astype(np.float32)
        if not np.array_equal(normal,expected):raise ValueError('stored training normal differs from exact fixed normalization')
        observation=read(input_path.parent/'observation.json')
        masking=root/'cases'/video/'masking'
        candidates=([masking] if (masking/'result.json').exists() else [])+sorted(masking.glob('attempt-*'))
        completed=[p for p in candidates if (p/'result.json').exists() and read(p/'result.json').get('status')=='complete']
        if len(completed)!=1:raise ValueError(f'{video}: ambiguous successful masking attempt')
        selected=completed[0];guard=read(selected/'link5_guard.json')
        diagnostic=next(d for d in read(selected/'sam3_prompt_diagnostics.json') if d['target']=='link5')
        refinement=diagnostic['point_refinement']
        if guard['fallback_to_default'] or guard['decision']=='DEFAULT_UNREVIEWED':raise ValueError('unreviewed training prompt')
        if refinement['frame_index']!=0 or refinement['points_xy']!=guard['selected_points_xy']:
            raise ValueError('SAM points differ from selected frame0 guard output')
        with np.load(selected/'segmentation.npz',allow_pickle=False) as a:
            source_mask=a['object_masks'][0,a['object_names'].tolist().index('link5')].astype(bool)
        with np.load(input_path,allow_pickle=False) as a:
            if not np.array_equal(source_mask,a['source_mask']):raise ValueError('training source mask differs from guarded segmentation')
        images={name:href(selected/filename) for name,filename in dict(
            selected=guard['selected_image'],positive='link5_guard_positive_current.png',negative='link5_guard_negative_current.png',
            positive_reference='link5_guard_positive_reference.png',negative_reference='link5_guard_negative_reference.png').items()}
        for path in images.values():
            if not (destination/path).is_file():raise ValueError('missing guard image')
        prompt=dict(decision=guard['decision'],original=guard['original_points_xy'],selected=guard['selected_points_xy'],
            labels=refinement['point_labels'],reviews=guard['reviews'],images=images,
            calls=[dict(role=a['role'],model=a.get('model') or a.get('requested_model'),answer=a.get('answer')) for a in guard['attempts']],
            record=href(selected/'link5_guard.json'),diagnostics=href(selected/'sam3_prompt_diagnostics.json'),
            record_sha256=sha(selected/'link5_guard.json'),segmentation_sha256=sha(selected/'segmentation.npz'),
            successful_attempt=selected.name,source_mask_matches=True)
        variants=[dict(name='Previous GPU run',prompt=prompt)]
        for name,base in [('Refined initial',root/'refinement'),('Refined 64K',root/'refinement/validation_64k'),
                          ('P1/P2/P3 guard',root/'refinement/three_positive_guard')]:
            for model in ['gemini','luna']:
                receipt=base/'cases'/video/'guard'/model/'link5_guard.json'
                if not receipt.exists():continue
                g=read(receipt)
                p=dict(decision=g['decision'],original=g['original_points_xy'],selected=g['selected_points_xy'],
                    labels=refinement['point_labels'],reviews=g['reviews'],
                    images={key:href(receipt.parent/filename) for key,filename in dict(selected=g['selected_image'],
                        positive='link5_guard_positive_current.png',negative='link5_guard_negative_current.png',
                        positive_reference='link5_guard_positive_reference.png',negative_reference='link5_guard_negative_reference.png').items()
                        if (receipt.parent/filename).exists()},
                    calls=[{k:a.get(k) for k in ['role','requested_model','model','reasoning_effort','output_token_limit','timeout_seconds','elapsed_seconds','answer','response_error','parse_error','usage']} for a in g['attempts']],
                    record=href(receipt),diagnostics=prompt['diagnostics'],preview_only=True,failed=g['fallback_to_default'],
                    p1_vlm=g.get('positive_placement_policy')=='vlm_places_all_three_no_default_p1')
                for view,filename in [('positive_output','positive_points.png'),('negative_output','negative_points.png')]:
                    if (receipt.parent/filename).is_file():p['images'][view]=href(receipt.parent/filename)
                variants.append(dict(name=name+' '+model.capitalize(),prompt=p,
                    id='three-positive-'+model if p['p1_vlm'] else None))
        receipt=root/'refinement/endpoint_probe/cases'/video/'guard/gpt-6.1-sol/link5_guard.json'
        if receipt.exists():
            g=read(receipt)
            images={key:href(receipt.parent/filename) for key,filename in g.get('output_images',{}).items()}
            for call in g['attempts']:
                task=call['role'].removeprefix('link5_').removesuffix('_guard')
                paths=[Path(path) for path in call['input_paths']]
                if [sha(path) for path in paths]!=call['image_sha256']:
                    raise ValueError('Endpoint probe input hash mismatch')
                images[task+'_reference']=href(paths[0]);images[task]=href(paths[1])
            if any(not (destination/path).is_file() for path in images.values()):
                raise ValueError('Missing endpoint probe replay image')
            p=dict(decision=g['decision'],original=g['original_points_xy'],selected=g['selected_points_xy'],
                labels=refinement['point_labels'],reviews=g['reviews'],images=images,
                calls=[{k:a.get(k) for k in ['role','requested_model','model','reasoning_effort','output_token_limit',
                    'timeout_seconds','elapsed_seconds','answer','response_error','parse_error','usage','api_base','api_style']}
                    for a in g['attempts']],record=href(receipt),diagnostics=prompt['diagnostics'],
                preview_only=True,failed=g['fallback_to_default'])
            variants.append(dict(id='gpt-6.1-sol-medium',name='GPT-6.1 Sol · medium · 64K',prompt=p))
        filtered_path=root/'refinement/cases'/video/'depth/normal_camera.npy'
        filtered=None
        if filtered_path.exists():
            xyz=np.load(filtered_path,allow_pickle=False);q=((xyz-np.asarray(normalization['fixed_center']))/normalization['fixed_scale']).astype(np.float32)
            filtered=dict(count=len(xyz),camera=packed(xyz),normalized=packed(q),archive=href(filtered_path),
                overlay=href(filtered_path.parent/'frame_00000.png'))
        rows.append(dict(video_id=video,count=len(normal),camera=packed(camera),normalized=packed(normal),
            rgb=href(input_path.parent/'rgb.png'),mask=href(input_path.parent/'mask.png'),
            source_mask=href(input_path.parent/'source_mask.png'),source_image_hw=observation['source_image_hw'],
            native_depth_hw=observation['native_depth_hw'],prompt=prompt,prompt_variants=variants,refined=filtered,
            input=href(input_path),input_sha256=sha(input_path),training_npy=href(stored),training_sha256=sha(stored),
            mask_area=observation['mask_area'],mask_erosion_pixels=observation['mask_erosion_pixels']))
    validation=read(root/'augmentation/robot_structural_validation.json')
    examples_dir=root/'augmentation/robot_structural_examples'
    normal=np.load(examples_dir/'normal.npy',allow_pickle=False)
    source=validation['source_video'];training=np.load(root/'normal_training_data/link5/train/normal'/(source+'.npy'),allow_pickle=False)
    ids=np.random.default_rng(0).choice(len(training),min(10000,len(training)),replace=False)
    if not np.array_equal(training[ids],normal):raise ValueError('saved augmentation example normal is not its exact training reference')
    examples=[dict(name='normal',count=len(normal),points=packed(normal),offset=packed(np.zeros_like(normal)),
        mask=base64.b64encode(np.zeros(len(normal),dtype=np.uint8)).decode('ascii'),parameters={},ply=href(examples_dir/'normal.ply'),archive=href(examples_dir/'normal.npy'))]
    for item in validation['checks']:
        name=item['example'];path=examples_dir/(name+'.npz')
        with np.load(path,allow_pickle=False) as a:
            if not np.array_equal(a['normal'],normal) or not np.array_equal(a['gt_offset'],normal-a['anomalous']):
                raise ValueError('saved synthetic correspondence/offset differs')
            mask=a['gt_mask'].astype(np.uint8)
            if not np.array_equal(mask,(a['gt_offset']!=0).any(1).astype(np.uint8)):raise ValueError('actual-modification mask differs')
            examples.append(dict(name=name,count=len(normal),points=packed(a['anomalous']),offset=packed(a['gt_offset']),
                mask=base64.b64encode(mask.tobytes()).decode('ascii'),parameters=item['parameters'],checks=item,
                ply=href(examples_dir/(name+'.ply')),archive=href(path)))
    payload=dict(normals=rows,examples=examples,example_source=source,normalization=normalization,
        geometry=read(root/'augmentation/robot_structural_geometry.json'),
        phase1_sizes=[r['size'] for r in read(root/'training/phase1_normal_codebook.json')['statistics']],
        baseline_sizes=[r['size'] for r in read(root/'training/codebook_statistics.json')],
        structural_sizes=[r['size'] for r in read(root/'training_robot_structural/codebook_statistics.json')])
    (destination/'data.js').write_text('window.LINK5_REFERENCE='+json.dumps(payload,separators=(',',':'))+';\n')
    workspace=Path(__file__).resolve().parents[3]
    template=(workspace/'infrastructure/shared/replay/rigidity_replay.html').read_text().split('<body>')[0]
    template=template.replace('src="plotly.min.js"','src="../replay/plotly.min.js"').replace('Rigidity evidence replay','Link5 normal reference and structural deformation')
    body=Path(__file__).with_name('reference_replay_body.html').read_text()
    (destination/'index.html').write_text(template+body)
    (destination/'app.js').write_text(Path(__file__).with_name('reference_replay_app.js').read_text())
    (destination/'verification.json').write_text(json.dumps(dict(status='verified',normal_files=20,
       exact_fixed_normalization_matches=20,example_source=source,exact_example_correspondence=True,
       guarded_frame0_source_mask_matches=len(rows),sam_points_match_guard=len(rows),unreviewed_guard_fallbacks=0,
       guard_unchanged=sum(r['prompt']['decision']=='PASS' for r in rows),
       guard_corrected=sum(r['prompt']['decision']=='REJECT' for r in rows),
       observed_normal_point_count=sum(r['count'] for r in rows),example_point_count=len(normal),
       normal_minibatch_indices_saved=False,source='unchanged stored training arrays and pretraining example archives',
       train20_manifest_sha256=sha(root/'training/train20_manifest.json'),
       normalization_sha256=sha(root/'link5_normalization.json'),
       structural_geometry_sha256=sha(root/'augmentation/robot_structural_geometry.json'),
       structural_validation_sha256=sha(root/'augmentation/robot_structural_validation.json')),indent=2)+'\n')
    print('LINK5_EXACT_REFERENCE_REPLAY_CREATED',len(rows),sum(r['count'] for r in rows),source,len(normal),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
