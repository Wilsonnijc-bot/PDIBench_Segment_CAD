"""Optional post-persistent repair of link7 masks that absorb the task object.

VLM1/VLM2 selection and their masks remain unchanged. This stage selects the
first strict >95% object-mask intersection (or the existing oversized-mask
failure), asks the VLM2 backend for six points, then reseeds the existing SAM3
predictor at THAT frame and propagates both directions. No frame+1 shift.
"""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw

from infrastructure.shared.contracts.mask_quality import link7_area_validity, MASK_QUALITY_METHOD
from robot.preprocessing.link7_persistent.frame_lineage import read_original_frame
from robot.preprocessing.link7_persistent.interface.config import PROJECT_ROOT, role_config
from robot.preprocessing.link7_persistent.interface.secrets import load_env_file
from robot.preprocessing.link7_persistent.vlm_client import VLMClient
from robot.preprocessing.link7_persistent.vlm2_sam_prompting import to_pixels

REFERENCE = (_workspace_root() / 'robot/preprocessing/link7_persistent/vlm3_interface/reference.png')
SYSTEM_PROMPT = '''You are a SAM3 prompting expert repairing gripper overmasking.
Image 1 is the exact source-video frame on which SAM3 will be seeded. Image 2
is an annotated reference: THREE green inclusion points are on the gripper;
THREE red exclusion points are on the forearm, wrist, and manipulated object.
The gripper includes its white palm housing and lower dark fingers. The held or
manipulated object is NEVER part of the gripper, even when touching or enclosed
by its fingers. Inspect original detail. Return only the requested JSON schema.
Coordinates are integer x,y values normalized 0 through 1000 relative to Image 1.
Do not copy reference coordinates or infer points on invisible material.'''
POINT_PROMPT = '''Place THREE positive points and THREE negative points in Image 1.
The task object is {target_object}. The old link7 segmentation wrongly includes
object or unrelated scene pixels; generate fresh gripper points from the RGB.
Positive points, in order: (1) well inside lower dark/black GRIPPER material below
the white palm/collar; (2) inside another spatially separated dark/black gripper
region; (3) at the center of the hanging WHITE gripper palm below the shiny
collar. Never put a positive on the task object, rim, forearm, collar, wrist,
background, or a thin outline. Object color is not gripper anatomy.
Negative points, in order: (1) inside the long forearm link, in its elbow-side
half; (2) inside the upright wrist/end-cap above the gripper, away from joints;
(3) DEEP INSIDE THE VISIBLE TASK OBJECT ({target_object}), preferably its broad
central surface, away from its boundary and from gripping fingers. This last
negative is essential: exclude the object while retaining the gripper around it.
The object may still be on the table before contact; exclude the SAME task
object then too. Ignore other cups or props when naming the task object.
Match the anatomical roles illustrated by Image 2, not its particular pose.
Return only {{"positive_points":[[x,y],[x,y],[x,y]],
"negative_points":[[x,y],[x,y],[x,y]]}}. If a required role is truly not visible,
return {{"positive_points":null,"negative_points":null}} rather than guessing.'''


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def canonical_case(case):
    if '_' not in case:
        return case
    dataset, number = case.rsplit('_', 1)
    return {'Cosmos25': 'COSMOS2.5', 'Cosmos3': 'COSMOS3', 'LVP': 'LVP_ROBOWM'}.get(dataset, dataset)+'_'+number


def named_masks(path, channel):
    with np.load(path, allow_pickle=False) as a:
        names = a['object_names'].tolist()
        if names.count(channel) != 1:
            raise ValueError(f'Expected exactly one {channel} channel')
        return a['object_masks'][:, names.index(channel)].astype(bool)


def audit_gate(link7, objects, threshold=.95, *, include_area_guard=True):
    link7, objects = np.asarray(link7, bool), np.asarray(objects, bool)
    if link7.shape != objects.shape or link7.ndim != 3 or not len(link7):
        raise ValueError('Expected matching nonempty [T,H,W] masks')
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Overlap threshold must be between zero and one')
    object_area = objects.sum((1, 2))
    shared = (objects & link7).sum((1, 2))
    fraction = shared / np.maximum(1, object_area)
    valid, link_area, image_fraction = link7_area_validity(link7)
    overlap_hits = (object_area > 0) & (fraction > threshold)
    oversized = ~valid if include_area_guard else np.zeros(len(link7), bool)
    hits = np.flatnonzero(overlap_hits | oversized)
    frame = int(hits[0]) if len(hits) else None
    reasons = [] if frame is None else ([f'object_overlap_gt_{threshold:g}'] if overlap_hits[frame] else []) + ([MASK_QUALITY_METHOD] if oversized[frame] else [])
    return {'threshold': threshold, 'comparison': 'strictly greater than',
            'fraction_denominator': 'current task-object SAM3 mask pixels',
            'frame': frame, 'reasons': reasons, 'overlap_frames': np.flatnonzero(overlap_hits).tolist(),
            'oversized_frames': np.flatnonzero(oversized).tolist(),
            'max_object_covered_fraction': float(fraction.max()),
            'frames': [{'frame': t, 'object_area': int(object_area[t]), 'link7_area': int(link_area[t]),
                        'shared_pixels': int(shared[t]), 'object_covered_fraction': float(fraction[t]),
                        'link7_image_fraction': float(image_fraction[t])} for t in range(len(link7))]}


def load_objects(object_root, case, video_sha, shape):
    folder = Path(object_root)/'cases'/canonical_case(case)/'masking'
    record = json.loads((folder/'grounding.json').read_text())
    path = folder/'segmentation.npz'
    if record['status'] != 'complete' or record['video_sha256'] != video_sha:
        raise ValueError('Object mask is incomplete or uses another video')
    if record['segmentation_sha256'] != digest(path):
        raise ValueError('Object mask hash differs from provenance')
    objects = named_masks(path, 'task_object')
    if objects.shape != tuple(shape) or record['source_hw'] != list(shape[1:]) or record['source_frame_count'] != shape[0]:
        raise ValueError('Object mask and link7 frame grids differ')
    return objects, {'path': str(path.resolve()), 'sha256': digest(path),
                     'grounding_sha256': digest(folder/'grounding.json'),
                     'target_object': record['target_object']}


def load_existing_case(object_root, gripper_root, case):
    folder = Path(gripper_root)/'cases'/canonical_case(case)
    status = json.loads((folder/'status.json').read_text())
    path = folder/('refined_segmentation.npz' if status['mask_policy'] == 'persistent_refined_link7' else 'base_segmentation.npz')
    mask_sha = digest(path)
    if mask_sha != status['selected_mask_sha256']:
        raise ValueError('Link7 mask differs from the selected persistent run')
    video = Path(object_root)/'cases'/canonical_case(case)/'replay/source.mp4'
    video_sha = digest(video)
    if video_sha != status['source_sha256']:
        raise ValueError('Object and link7 source videos differ')
    link7 = named_masks(path, 'link7')
    objects, obj = load_objects(object_root, case, video_sha, link7.shape)
    return video, link7, objects, obj, {'path': str(path.resolve()), 'sha256': mask_sha,
                                      'mask_policy': status['mask_policy'], 'video_sha256': video_sha}


def parse_points(raw, width, height, object_mask):
    text = raw['answer']
    start = text.find('{')
    if start < 0:
        raise ValueError('VLM3 returned no JSON')
    data, _ = json.JSONDecoder().raw_decode(text[start:])
    if set(data) != {'positive_points', 'negative_points'}:
        raise ValueError('Expected positive_points and negative_points only')
    for key in data:
        pts = data[key]
        if not isinstance(pts, list) or len(pts) != 3:
            raise ValueError(f'Exactly three visible {key} are required')
        if any(not isinstance(p, list) or len(p) != 2 or any(type(v) is not int or not 0 <= v < 1000 for v in p) for p in pts):
            raise ValueError('Expected interior integer coordinates normalized 0–1000')
    pixels = to_pixels(data['positive_points'], data['negative_points'], width, height)
    xy = np.floor(pixels).astype(int)
    if object_mask[xy[:3, 1], xy[:3, 0]].any():
        raise ValueError('A positive point falls inside the task-object mask; move all positives onto robot material')
    if not object_mask[xy[5, 1], xy[5, 0]]:
        raise ValueError('The third negative is not inside the task-object mask; locate the named object more precisely')
    return {**data, 'points': pixels, 'labels': [1, 1, 1, 0, 0, 0]}


def propagate(video, frame, points, labels):
    """Use the same point-prompt SAM3 predictor and bidirectional propagation."""
    from robot.preprocessing.link7_persistent.v1_mask import predictor
    cap = cv2.VideoCapture(str(video)); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));cap.release()
    p = predictor()
    sid = p.handle_request({'type': 'start_session', 'resource_path': str(video), 'offload_video_to_cpu': True})['session_id']
    masks = np.zeros((count, h, w), bool)
    prompt = {'type': 'add_prompt', 'session_id': sid, 'frame_index': frame, 'obj_id': 0,
              'points': (np.asarray(points)/[w, h]).tolist(), 'point_labels': labels}
    try:
        response = p.handle_request(prompt)['outputs']
        masks[frame] = response['out_binary_masks'][list(response['out_obj_ids']).index(0)]
        for direction, limit in [('forward', count-frame-1), ('backward', frame)]:
            if not limit:
                continue
            for response in p.handle_stream_request({'type': 'propagate_in_video', 'session_id': sid,
                    'propagation_direction': direction, 'start_frame_index': frame, 'max_frame_num_to_track': limit}):
                out = response['outputs'];ids = list(out['out_obj_ids'])
                if 0 in ids:
                    masks[int(response['frame_index'])] = out['out_binary_masks'][ids.index(0)]
        xy = np.floor(points).astype(int)
        member = masks[frame, xy[:, 1], xy[:, 0]].tolist()
        return masks, {'prompt': {k: v for k, v in prompt.items() if k != 'session_id'},
                       'membership': member, 'membership_ok': member == [bool(v) for v in labels],
                       'empty_frames': np.flatnonzero(~masks.any((1, 2))).tolist()}
    finally:
        p.handle_request({'type': 'close_session', 'session_id': sid})
        p.shutdown()


def draw_points(image, points, labels, path):
    image = image.copy();draw = ImageDraw.Draw(image)
    for i, ((x, y), label) in enumerate(zip(points, labels)):
        color = 'lime' if label else 'red'
        draw.ellipse((x-6, y-6, x+6, y+6), fill=color, outline='white', width=1)
        draw.text((x+8, y+2), str(i+1), fill=color)
    image.save(path)


def crop_policy(post_gate, sam):
    """Keep a useful repair and exclude residual mask failures frame by frame."""
    excluded = set(post_gate['overlap_frames']) | set(post_gate['oversized_frames'])
    excluded.update(sam['empty_frames'])
    excluded.update(row['frame'] for row in post_gate['frames'] if not row['object_area'])
    eligible = [row['frame'] for row in post_gate['frames'] if row['frame'] not in excluded]
    accepted = bool(sam['membership_ok'] and not sam['empty_frames'] and eligible)
    return {'accepted': accepted,
            'status': ('accepted_with_frame_exclusions' if excluded else 'accepted') if accepted else 'repair_failed',
            'crop_excluded_frames': sorted(excluded), 'crop_eligible_frames': eligible,
            'acceptance_policy': 'six-point membership and nonempty SAM masks; residual overlap/area failures excluded from cropping'}


def compare_video(video, before, after, objects, output, name, *, after_label='VLM3 CANDIDATE'):
    count, h, w = before.shape;strip = 48
    cap = cv2.VideoCapture(str(video));fps = cap.get(cv2.CAP_PROP_FPS)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
           '-s', f'{2*w}x{h+strip}', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        try:
            for t in range(count):
                ok, rgb = cap.read()
                if not ok:
                    raise ValueError(f'Cannot decode comparison frame {t}')
                canvas = np.zeros((h+strip, 2*w, 3), np.uint8)
                for i, (mask, label) in enumerate([(before[t], 'BEFORE'), (after[t], after_label)]):
                    view = rgb.copy();view[mask] = np.rint(view[mask]*.6 + np.array([215, 90, 210])*.4).astype(np.uint8)
                    contours = cv2.findContours(objects[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    cv2.drawContours(view, contours, -1, (35, 215, 245), 1)
                    fraction = (mask & objects[t]).sum()/max(1, objects[t].sum())
                    canvas[strip:, i*w:(i+1)*w] = view
                    cv2.putText(canvas, f'{name} | {label} | f{t} | object overlap {fraction:.1%}',
                                (i*w+12, 32), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 240, 240), 1, cv2.LINE_AA)
                process.stdin.write(canvas.tobytes())
        finally:
            cap.release();process.stdin.close()
        if process.wait():
            raise RuntimeError('FFmpeg comparison export failed')


def repair_case(case, video, before, objects, object_input, output, *, threshold=.95,
                include_area_guard=True, max_attempts=2, client=None, reference=REFERENCE, initial_feedback=''):
    output = Path(output);output.mkdir(parents=True, exist_ok=True)
    gate = audit_gate(before, objects, threshold, include_area_guard=include_area_guard)
    result = {'case': canonical_case(case), 'status': 'not_triggered', 'accepted': False,
              'gate': gate, 'object_input': object_input, 'attempts': [],
              'original_mask_array_sha256': hashlib.sha256(np.asarray(before, bool).tobytes()).hexdigest(),
              'source_video_sha256': digest(video), 'code_sha256': digest(__file__)}
    record_path = output/'repair.json'
    save(record_path, result)
    if gate['frame'] is None:
        return result
    inputs = output/'inputs';inputs.mkdir(exist_ok=True)
    frame = gate['frame'];image = Image.fromarray(read_original_frame(video, frame))
    if image.size != (before.shape[2], before.shape[1]):
        raise ValueError('Source video and mask dimensions differ')
    image.save(inputs/'target.png')
    ref = Image.open(reference).convert('RGB');ref.save(inputs/'reference.png')
    result.update(frame=frame, input_hashes=[digest(inputs/'target.png'), digest(inputs/'reference.png')],
                  reference_annotation='manual sixth object-negative added to original VLM2 reference 3',
                  vlm_config=role_config('vlm2'), status='repair_pending')
    save(record_path, result)
    client = client or VLMClient('vlm3_overmask_repair', role_config('vlm2'))
    feedback = initial_feedback
    candidate = None
    yy, xx = np.where(objects[frame])
    object_hint = ''
    if len(xx):
        box = np.rint(np.array([xx.min(), yy.min(), xx.max()+1, yy.max()+1])
                      / [image.width, image.height, image.width, image.height] * 1000).astype(int).tolist()
        object_hint = ('\nThe existing task-object mask gives this approximate visible-object '
                       f'bounding box in Image 1, normalized 0–1000: {box}. '
                       'Use it to identify the named object in RGB, not as gripper material. '
                       'Choose the object negative deep inside its visible surface; choose positives outside the object.')
    for number in range(1, max_attempts+1):
        attempt = {'number': number, 'frame': frame}
        result['attempts'].append(attempt)
        try:
            raw = client.ask([image, ref], POINT_PROMPT.format(target_object=object_input['target_object'])+object_hint+feedback,
                             system_prompt=SYSTEM_PROMPT)
            attempt['vlm_call'] = raw
            if raw['image_sha256'] != result['input_hashes']:
                raise ValueError('VLM3 request images differ from saved exact inputs')
            seed = parse_points(raw, image.width, image.height, objects[frame]);seed['frame'] = frame
            attempt['seed'] = seed
            draw_points(image, seed['points'], seed['labels'], output/'points.png')
            save(record_path, result)
            candidate, metrics = propagate(video, frame, seed['points'], seed['labels'])
            post = audit_gate(candidate, objects, threshold, include_area_guard=include_area_guard)
            attempt.update(sam=metrics, post_gate=post)
            np.savez_compressed(output/'masks.npz', masks=candidate)
            policy = crop_policy(post, metrics)
            attempt['accepted'] = policy['accepted']
            if policy['accepted']:
                result.update(**policy, selected_attempt=number,
                              output_masks=str((output/'masks.npz').resolve()), output_masks_sha256=digest(output/'masks.npz'),
                              post_gate=post)
                break
            feedback = ('\nYour previous six-point candidate failed validation: '
                        +json.dumps({'membership': metrics['membership'], 'empty_frames': metrics['empty_frames'],
                                     'remaining_overlap_frames': post['overlap_frames'], 'remaining_oversized_frames': post['oversized_frames']})
                        +'. Reconsider all six points on the SAME Image 1; use broad gripper interiors and a deep object exclusion. Return the same JSON schema.')
            attempt['error'] = 'SAM candidate failed membership, nonempty-mask, or post-gate validation'
        except Exception as exc:
            attempt['error'] = str(exc)
            if getattr(client, 'records', None) and 'vlm_call' not in attempt:
                attempt['vlm_call'] = client.records[-1]
            feedback = '\nPrevious output was rejected: '+str(exc)+'. Correct the points on the SAME Image 1; return the requested six-point JSON.'
            traceback.print_exc()
        save(record_path, result)
    if not result['accepted']:
        result['status'] = 'repair_failed'
    if candidate is not None:
        compare_video(video, before, candidate, objects, output/'comparison.mp4', canonical_case(case))
    save(record_path, result)
    return result


def repair_persistent_run(work, cases, object_root, *, threshold=.95):
    """Attach optional results without altering VLM1/VLM2 records or original masks."""
    work = Path(work);record = json.loads((work/'provenance.json').read_text())
    for case in cases:
        r = record['results'][case]
        if r['status'] not in {'completed_checks', 'no_confirmed_deformation', 'no_naive_surge'}:
            continue
        if 'vlm3' in r:
            _sync_persistent_case(work, case, r, object_root)
            continue
        if r['status'] == 'completed_checks':
            source, expected_sha = Path(r['mask_source']), r['masks_sha256']
            input_policy = 'persistent_refined_link7'
        else:
            # The old pipeline retains its naive mask when VLM1 finds no
            # deformation. VLM3 must not depend on that diagnosis to run.
            source = work/'naive'/case/'masks.npz'
            if not source.exists():
                source = work/case/'naive'/case/'masks.npz'
            expected_sha = r['naive_masks_sha256']
            input_policy = 'old_pipeline_naive_fallback'
        if digest(source) != expected_sha or digest(r['video']) != r['source_sha256']:
            raise ValueError('Persistent source or masks changed before VLM3')
        with np.load(source, allow_pickle=False) as a:
            before = a['masks'].astype(bool)
        objects, obj = load_objects(object_root, case, r['source_sha256'], before.shape)
        repaired = repair_case(case, r['video'], before, objects, obj, work/'vlm3'/canonical_case(case), threshold=threshold)
        repaired['link7_input'] = {'path': str(source.resolve()), 'sha256': expected_sha,
                                  'mask_policy': input_policy, 'video_sha256': r['source_sha256']}
        save(work/'vlm3'/canonical_case(case)/'repair.json', repaired)
        r['vlm3'] = {k: repaired[k] for k in ('status', 'accepted', 'frame', 'output_masks', 'output_masks_sha256', 'crop_excluded_frames', 'crop_eligible_frames') if k in repaired}
        r['vlm3']['record'] = str((work/'vlm3'/canonical_case(case)/'repair.json').resolve())
        save(work/'provenance.json', record)
        _sync_persistent_case(work, case, r, object_root)
    reviews = [json.loads(Path(record['results'][case]['vlm3']['record']).read_text())
               for case in cases if 'vlm3' in record['results'][case]]
    if reviews:
        save(work/'vlm3/gate_audit.json', [{'case': r['case'], 'gate': r['gate']} for r in reviews])
        export_review(reviews, work/'vlm3')


def _sync_persistent_case(work, case, result, object_root):
    """Accepted VLM3 output is the default mask for detection, viewer and crops."""
    if result.get('vlm3', {}).get('accepted') is not True:
        return
    from object.workflows.mask_sync import sync_case
    folder = Path(object_root)/'cases'/canonical_case(case)
    detection_path = folder/'occlusion/detection.json'
    if not detection_path.exists():
        raise FileNotFoundError('Persistent VLM3 synchronization needs the existing named gripper archive in occlusion/detection.json')
    base = Path(json.loads(detection_path.read_text())['inputs']['gripper']['path'])
    previous = work/'vlm3/downstream/gripper/cases'/folder.name/'v1_cotracker3/segmentation.json'
    if previous.exists():
        base = Path(json.loads(previous.read_text())['base_segmentation'])
    sync_case(case=folder, repair_record=Path(result['vlm3']['record']), base_segmentation=base,
              gripper_root=work/'vlm3/downstream/gripper', crop_root=work/'vlm3/crops')


def export_review(records, output):
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Persistent masks and optional VLM3 repair</title><style>body{font:16px system-ui;background:#f5f6f2;color:#253530;max-width:1400px;margin:24px auto;padding:0 20px}section{padding:24px 0;border-bottom:1px solid #c6d1bf}video{width:100%;background:#111}.images{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#286953}@media(max-width:800px){.images{grid-template-columns:1fr}}</style><body><h1>Persistent masks and optional VLM3 repair</h1><p>Old persistent masking retained. VLM3 uses its first frame with &gt;95% object coverage, or the existing 25% image-area failure, and asks the VLM2 backend for 3 gripper positives + arm, wrist, and object negatives. Magenta: link7. Yellow outline: task object. A useful repair can be accepted with frame exclusions: skip remaining overmask frames when selecting crops.</p>']
    rerun_path = Path(output)/'metadata/lvp_rerun/review.json'
    reruns = json.loads(rerun_path.read_text()) if rerun_path.exists() else {}
    latest_cases = [r['case'] for r in records if (Path(output)/r['case']/'vlm3_fresh/repair.json').is_file()]
    if (Path(output)/'crops/selection_gallery.html').is_file() and not latest_cases:
        parts.append('<p><a href="crops/selection_gallery.html"><strong>Open selected crop pairs and original-frame pictures</strong></a></p>')
    if latest_cases:
        links = ' · '.join(f'<a href="#{html.escape(name)}">{html.escape(name)}</a>' for name in sorted(latest_cases))
        parts.append(f'<p><strong>New VLM3 results:</strong> {links}</p>')
        if (Path(output)/'crops/selection_gallery.html').is_file():
            parts.append('<p><a href="crops/selection_gallery.html"><strong>Open the new selected crop pairs and original frames</strong></a></p>')
    for r in sorted(records, key=lambda x:(int(x['case'].rsplit('_',1)[1]),x['case'])):
        name = r['case'];folder=Path(output)/name
        parts.append(f'<section id="{html.escape(name)}"><h2>{html.escape(name)}</h2>')
        sync_path = Path(output)/'downstream/gripper/cases'/name/'downstream_sync.json'
        if sync_path.is_file():
            synced = json.loads(sync_path.read_text())
            replay = Path(synced['replay'])
            if replay.is_file():
                href = html.escape(os.path.relpath(replay, Path(output)), quote=True)
                parts.append(f'<p><a href="{href}"><strong>Open synchronized occlusion replay</strong></a></p>')
            parts.append(f'<p>Selected source frames: {html.escape(str(synced["selected_frames"]))}. Occlusion intervals: {html.escape(str(synced["occlusion"]["flagged_intervals"]))}.</p>')

        latest_path = folder/'vlm3_fresh/repair.json'
        if latest_path.is_file():
            latest = json.loads(latest_path.read_text())
            role = f'{name}/vlm3_fresh'
            parts.append(f'<h3>New VLM3 run on fresh VLM2 masks</h3><p><strong>{html.escape(latest["status"])}</strong> · gate frame {latest["gate"]["frame"]} · {html.escape(latest.get("vlm_config",{}).get("model",""))}. Input: fresh native VLM2 mask, with Luna positive-point fallback.</p>')
            if latest.get('post_gate'):
                parts.append(f'<p>Frames above 95% object coverage: {len(latest["gate"]["overlap_frames"])} before → {len(latest["post_gate"]["overlap_frames"])} after. Crop exclusions: {html.escape(str(latest.get("crop_excluded_frames",[])))}.</p>')
            if (folder/'vlm3_fresh/comparison.mp4').is_file():
                parts.append(f'<video controls preload="metadata" playsinline src="{role}/comparison.mp4"></video><p><a href="{role}/comparison.mp4">▶ Fresh VLM2 mask versus new VLM3 mask</a></p>')
            if (folder/'vlm3_fresh/points.png').is_file():
                parts.append(f'<details><summary>New VLM3 points and exact input</summary><div class="images"><figure><img src="{role}/inputs/target.png"><figcaption>Exact VLM3 input · frame {latest["frame"]}</figcaption></figure><figure><img src="{role}/points.png"><figcaption>New VLM3 output · green positives, red negatives</figcaption></figure></div></details>')
            parts.append(f'<p><a href="{role}/repair.json">Actual new VLM3 requests and validation</a></p>')
            crop_folder = Path(output)/'crops'/name
            if (crop_folder/'selection.json').is_file():
                selection = json.loads((crop_folder/'selection.json').read_text())
                manifest = json.loads((crop_folder/'manifest.json').read_text())
                parts.append(f'<h3>Available object pixels and crops</h3><p>{manifest["mapped_frames"]}/{manifest["frame_count"]} valid mappings · {selection["selected_count"]} selected crop pairs · frame 0 reference: {html.escape(manifest["frame0_reference_quality"])}.</p>')
                if (Path(output)/'crops/selection_gallery.html').is_file():
                    parts.append('<p><a href="crops/selection_gallery.html">Open paired crops and original frames</a></p>')
                parts.append(f'<p><a href="crops/{name}/selection.json">Selection details</a></p>')
                for row in selection['selected_frames']:
                    covered = manifest['frames'][row['frame']].get('link7_object_covered_fraction',0)
                    if row['reason'] == 'immediate_post_occlusion' and covered > .90:
                        parts.append(f'<p>Mandatory post-occlusion frame {row["frame"]} retains only {row["available_area"]} pixels, with {covered:.1%} link7 overlap. It remains selected because it is below the current 95% gate and passes the existing crop-mapping checks.</p>')
                if (crop_folder/'available_pixels.mp4').is_file():
                    parts.append(f'<video controls preload="metadata" playsinline src="crops/{name}/available_pixels.mp4"></video><p><a href="crops/{name}/available_pixels.mp4">▶ Original frames and available object pixels</a></p>')
            parts.append('<details><summary>Earlier VLM2 investigation and original-mask VLM3 test</summary>')
        parts.append(f'<p><strong>{html.escape(r["status"])}</strong> · gate frame {r["gate"]["frame"]} · {html.escape(", ".join(r["gate"]["reasons"]))}</p>')
        if name in reruns:
            fresh = reruns[name]
            parts.append(f'<h3>Fresh initial masking → Qwen VLM1 → VLM2 (Gemini with Luna fallback)</h3><p>{html.escape(fresh["summary"])}</p>')
            if name == 'LVP_ROBOWM_0010':
                parts.append('<p>The exact same four images and prompts previously produced a Gemini seed with zero &gt;95% overlap frames. Current Gemini-only retries returned null at both recorded token budgets. <a href="metadata/lvp_rerun/input_comparison.json">Actual identical-input comparison</a> · <a href="metadata/lvp_rerun/same_input_success/points.png">Earlier Gemini points on this same frame</a></p>')
            parts.append(f'<p><a href="{name}/initial_masking.mp4">▶ Fresh initial mask (before VLM2)</a> · <a href="metadata/lvp_rerun/gemini_retry/{fresh["internal_case"]}/target.png">Exact Gemini-only retry input</a> · <a href="metadata/lvp_rerun/gemini_retry/{fresh["internal_case"]}/record.json">Gemini-only response</a> · <a href="metadata/lvp_rerun/budget800/{fresh["internal_case"]}/record.json">800-token control</a></p>')
            if (folder/'persistent_rerun.mp4').exists():
                parts.append(f'<video controls preload="metadata" playsinline src="{name}/persistent_rerun.mp4"></video><p><a href="{name}/persistent_rerun.mp4">▶ Original versus fresh persistent masks</a></p>')
            if (folder/'persistent_rerun_points.png').exists():
                parts.append(f'<div class="images"><figure><img src="metadata/lvp_original/{name}/points.png"><figcaption>Original VLM2 points · green: GPT-6-Luna fallback; red: Gemini3.8Flash</figcaption></figure><figure><img src="{name}/persistent_rerun_points.png"><figcaption>Fresh VLM2 points · frame {fresh["frame"]} · green: GPT-6-Luna fallback; red: Gemini3.8Flash</figcaption></figure></div><p><a href="{name}/inputs/persistent_rerun_target.png">Exact uncropped VLM2 target</a> · <a href="metadata/lvp_rerun/work/provenance.json">Actual Qwen/Gemini/Luna requests and responses</a></p>')
            parts.append('<h3>Optional VLM3 correction from the earlier test</h3>')
        if r.get('crop_excluded_frames'):
            parts.append(f'<p><strong>Skip these frames for cropping:</strong> {html.escape(str(r["crop_excluded_frames"]))}. Remaining frames must still pass the normal crop mapping and occlusion checks.</p>')
        candidates = [a for a in r['attempts'] if 'post_gate' in a]
        if candidates:
            post = candidates[-1]['post_gate']
            parts.append(f'<p>Frames above 95% object coverage: {len(r["gate"]["overlap_frames"])} before → {len(post["overlap_frames"])} after. Oversized-mask failures: {len(r["gate"]["oversized_frames"])} before → {len(post["oversized_frames"])} after.</p>')
            if not r['accepted']:
                parts.append('<p>Candidate rejected; original masks remain selected.</p>')
        if (folder/'comparison.mp4').exists():
            parts.append(f'<video controls preload="metadata" playsinline src="{name}/comparison.mp4"></video><p><a href="{name}/comparison.mp4">▶ Open before/after MP4</a></p>')
        if (folder/'inputs/target.png').exists():
            parts.append(f'<div class="images"><figure><img src="{name}/inputs/target.png"><figcaption>Exact VLM3 input · frame {r["frame"]}</figcaption></figure>')
            if (folder/'points.png').exists():parts.append(f'<figure><img src="{name}/points.png"><figcaption>VLM3 output · green positives, red negatives</figcaption></figure>')
            parts.append('</div>')
        parts.append(f'<details><summary>Actual requests, responses, and validation</summary><p><a href="{name}/repair.json">Full record</a> · <a href="{name}/inputs/reference.png">Exact six-point reference supplied to VLM3</a></p>')
        for a in r['attempts']:
            parts.append(f'<h3>Attempt {a["number"]}</h3><pre>{html.escape(a.get("vlm_call",{}).get("answer",a.get("error","")))}</pre>')
        parts.append('</details>')
        if latest_path.is_file():
            parts.append('</details>')
        parts.append('</section>')
    parts.append('<p><a href="gate_audit.json">Gate audit</a></p></body></html>')
    (Path(output)/'index.html').write_text('\n'.join(parts)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit','run'])
    parser.add_argument('--object-root', type=Path, required=True)
    parser.add_argument('--gripper-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--threshold', type=float, default=.95)
    parser.add_argument('--overlap-only', action='store_true')
    parser.add_argument('--max-attempts', type=int, default=2)
    args = parser.parse_args()
    if args.max_attempts < 1 or args.max_attempts > 3:
        parser.error('Use one to three bounded VLM3 attempts')
    load_env_file((_workspace_root() / '.env.vlm'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    cases = args.cases or sorted(p.name for p in (args.object_root/'cases').iterdir() if (p/'masking/segmentation.npz').exists())
    audits, results = [], []
    for case in cases:
        video, before, objects, obj, inputs = load_existing_case(args.object_root,args.gripper_root,case)
        gate = audit_gate(before,objects,args.threshold,include_area_guard=not args.overlap_only)
        audits.append({'case':canonical_case(case),'gate':gate,'link7_input':inputs,'object_input':obj})
        save(args.output_root/'gate_audit.json',audits)
        print(case,'gate frame',gate['frame'],'reasons',gate['reasons'],flush=True)
        if args.stage == 'run':
            output = args.output_root/canonical_case(case)
            if (output/'repair.json').exists():
                raise FileExistsError('Preserve this run; use a new output root for another experiment')
            r = repair_case(case,video,before,objects,obj,output,threshold=args.threshold,
                            include_area_guard=not args.overlap_only,max_attempts=args.max_attempts)
            r['link7_input'] = inputs;save(output/'repair.json',r)
            if r['status'] in {'accepted', 'accepted_with_frame_exclusions', 'not_triggered', 'repair_failed'}:
                from object.workflows.mask_sync import sync_case
                sync_case(case=args.object_root/'cases'/canonical_case(case),
                          repair_record=output/'repair.json', base_segmentation=Path(inputs['path']),
                          gripper_root=args.output_root/'downstream/gripper', crop_root=args.output_root/'crops')
            results.append(r);export_review(results,args.output_root)
            print('VLM3_RESULT',case,r['status'],flush=True)
    save(args.output_root/'summary.json',{'cases':len(cases),'triggered':sum(a['gate']['frame'] is not None for a in audits),
         'accepted':sum(r['accepted'] for r in results),'failed':sum(r['status']=='repair_failed' for r in results),
         'results':[{k:r[k] for k in ('case','status','accepted')} for r in results]})
    print('VLM3_COMPLETE',flush=True)


if __name__ == '__main__':
    main()
