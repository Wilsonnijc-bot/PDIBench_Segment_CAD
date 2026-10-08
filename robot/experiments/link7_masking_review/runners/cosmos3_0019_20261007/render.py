"""Export the coordinator's selected gripper and object masks without inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from infrastructure.deformation_detect.coordinator import digest, read, write
from infrastructure.shared.contracts.segmentation_archive import load_multi_object_segmentation
from robot.preprocessing.link7_persistent.naive_sam3 import render as native_render


def verify_video(path, frames, fps):
    cap = cv2.VideoCapture(str(path))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    count = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if image is None or image.size == 0:
            raise ValueError('Unreadable replay frame')
        count += 1
    cap.release()
    if count != frames or abs(actual_fps - fps) > 1e-3:
        raise ValueError(f'Replay mismatch: {path}: {count} frames at {actual_fps} fps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    config = read(args.manifest)
    output = Path(config['output'])
    state = read(output / 'run.json')
    case = config['cases'][0]
    record = state['cases'][case['id']]
    if state['status'] != 'complete' or record['status'] != 'complete':
        raise ValueError('Coordinator has not completed the requested masking stages')
    stages = record['stages']
    joined = Path(stages['mask_join']['directory'])
    object_folder = Path(stages['object_masks']['directory']) / 'masking'
    video = Path(case['video'])
    gripper_archive = load_multi_object_segmentation(joined / 'segmentation.npz', video)
    object_archive = load_multi_object_segmentation(object_folder / 'segmentation.npz', video)
    gripper = gripper_archive.object_masks[:, gripper_archive.object_names.index('link7')]
    objects = object_archive.object_masks[:, object_archive.object_names.index('task_object')]
    if gripper.shape != objects.shape:
        raise ValueError('Selected gripper/object grids differ')
    frames, height, width = gripper.shape
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != frames or fps <= 0:
        raise ValueError('Source and mask frame counts differ')
    review = output / 'replay'
    review.mkdir(exist_ok=False)
    ffmpeg = config['resources']['ffmpeg']
    native_render(video, gripper, review / 'gripper.mp4', ffmpeg)
    native_render(video, objects, review / 'object.mp4', ffmpeg)
    raw = review / 'combined-raw.mp4'
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height))
    if not writer.isOpened():
        raise RuntimeError('Combined replay writer failed')
    try:
        for index, (grip, obj) in enumerate(zip(gripper, objects)):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Source ended before final mask')
            overlay = frame.copy()
            overlay[grip] = (0.55 * overlay[grip] + 0.45 * np.array([200, 70, 235])).astype(np.uint8)
            overlay[obj] = (0.55 * overlay[obj] + 0.45 * np.array([235, 210, 40])).astype(np.uint8)
            for mask, color in ((grip, (200, 70, 235)), (obj, (235, 210, 40))):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 2)
            image = np.concatenate((frame, overlay), axis=1)
            cv2.rectangle(image, (0, 0), (width * 2, 35), (22, 25, 29), -1)
            cv2.putText(image, f'Original / frame {index + 1}', (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(image, 'Gripper', (width + 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 70, 235), 2, cv2.LINE_AA)
            cv2.putText(image, 'Object: banana', (width + 115, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 210, 40), 2, cv2.LINE_AA)
            writer.write(image)
        if cap.read()[0]:
            raise ValueError('Source has additional frames')
    finally:
        writer.release()
        cap.release()
    subprocess.run([ffmpeg, '-nostdin', '-v', 'error', '-i', str(raw), '-c:v', 'libx264',
                    '-threads', '8', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(review / 'combined.mp4')], check=True)
    raw.unlink()
    for name in ('combined.mp4', 'gripper.mp4', 'object.mp4'):
        verify_video(review / name, frames, fps)
    provenance = review / 'provenance'
    provenance.mkdir()
    for source, name in ((args.manifest, 'manifest.json'), (output / 'run.json', 'run.json'),
                         (joined / 'selection.json', 'selection.json'), (joined / 'vlm3/repair.json', 'vlm3.json'),
                         (object_folder / 'grounding.json', 'object-grounding.json')):
        shutil.copyfile(source, provenance / name)
    shutil.copyfile(joined / 'segmentation.npz', provenance / 'gripper.npz')
    shutil.copyfile(object_folder / 'segmentation.npz', provenance / 'object.npz')
    persistent = Path(stages['link7_masks']['directory']) / 'work/provenance.json'
    shutil.copyfile(persistent, provenance / 'persistent-masking.json')
    metadata = {'case': case['id'], 'seed': 101, 'frames': frames, 'fps': fps, 'source_hw': [height, width],
                'source_sha256': digest(video), 'selection': read(joined / 'selection.json'),
                'target_object': read(object_folder / 'grounding.json')['target_object'],
                'gripper_empty_frames': np.flatnonzero(~gripper.any(axis=(1, 2))).tolist(),
                'object_empty_frames': np.flatnonzero(~objects.any(axis=(1, 2))).tolist(),
                'hashes': {str(path.relative_to(review)): digest(path) for path in sorted(review.rglob('*')) if path.is_file()}}
    write(review / 'metadata.json', metadata)
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cosmos3 0019 · gripper and object masking</title>
<style>body{margin:0;background:#14171b;color:#e8eaed;font:16px/1.5 system-ui}main{max-width:1440px;margin:auto;padding:24px}h1{font-size:26px;margin:0 0 8px}p{color:#afb7c1}video{display:block;width:100%;background:#080a0d;margin:18px 0}button,select{font:inherit;padding:8px;border:1px solid #59616d;border-radius:5px;background:#242a32;color:inherit}input{flex:1;min-width:120px}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}a{color:#aad5ff}.gripper{color:#eb46c8}.object{color:#28d2eb}</style>
<main><h1>Cosmos3 0019 · seed101</h1><p>Persistent gripper masking + object masking · <span class="gripper">Gripper</span> · <span class="object">Banana</span></p>
<select id="view" aria-label="Replay view"><option value="combined">Gripper + object</option><option value="gripper">Gripper only</option><option value="object">Object only</option></select>
<video id="video" src="combined.mp4" controls playsinline preload="metadata"></video>
<div class="controls"><button id="prev">Previous frame</button><button id="next">Next frame</button><input id="frame" type="range" min="0" max="FRAME_MAX" value="0" aria-label="Source frame"><span id="label">Frame 1 / FRAME_COUNT</span></div>
<p><a href="combined.mp4" download>Download combined replay</a> · <a href="gripper.mp4" download>Gripper replay</a> · <a href="object.mp4" download>Object replay</a> · <a href="metadata.json">Provenance</a></p></main>
<script>const v=document.getElementById('video'),s=document.getElementById('frame'),l=document.getElementById('label'),fps=FPS_VALUE,n=FRAME_COUNT;let f=0;function show(){f=Math.min(n-1,Math.floor(v.currentTime*fps));s.value=f;l.textContent=`Frame ${f+1} / ${n}`}function seek(i){v.pause();f=Math.max(0,Math.min(n-1,i));v.currentTime=(f+.5)/fps;show()}s.oninput=()=>seek(+s.value);document.getElementById('prev').onclick=()=>seek(f-1);document.getElementById('next').onclick=()=>seek(f+1);v.ontimeupdate=show;v.onseeked=show;document.getElementById('view').onchange=e=>{const t=v.currentTime;v.src=e.target.value+'.mp4';v.onloadedmetadata=()=>{v.currentTime=t;show()}};document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'){e.preventDefault();seek(f+1)}if(e.key==='ArrowLeft'){e.preventDefault();seek(f-1)}});</script></html>'''
    (review / 'index.html').write_text(html.replace('FRAME_MAX', str(frames - 1)).replace('FRAME_COUNT', str(frames)).replace('FPS_VALUE', str(fps)))
    print(json.dumps({'status': 'complete', 'review': str(review), 'frames': frames, 'fps': fps, 'selection': metadata['selection']['policy']}), flush=True)


if __name__ == '__main__':
    main()
