"""Create browser-compatible display copies; preserve original scored videos."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,profile,pix_fmt:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], text=True))


def prepare(root):
    report = root / 'report'
    metadata = root / 'metadata'
    backup = metadata / 'report_before_playback_fix'
    backup.mkdir(exist_ok=True)
    for name in ('data.js', 'index.html'):
        if (report / name).exists() and not (backup / name).exists():
            shutil.copy2(report / name, backup / name)
    legacy_data = (report / 'data.js').exists()
    if legacy_data:
        data = json.loads((report / 'data.js').read_text().removeprefix('window.RIGIDITY_DATA=').removesuffix(';'))
    else:
        manifest = json.loads((root / 'manifest.json').read_text())
        data = dict(videos=[dict(case=e['video_id']) for e in manifest['entries']])
    (report / 'playback').mkdir(exist_ok=True)

    def convert(video):
        source = report / 'videos' / (video['case'] + '.mp4')
        target = report / 'playback' / source.name
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                        '-map', '0:v:0', '-an', '-c:v', 'libx264', '-profile:v', 'high',
                        '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'veryfast',
                        '-threads', '2', '-fps_mode', 'passthrough', '-movflags', '+faststart',
                        str(target)], check=True)
        before, after = probe(source), probe(target)
        a, b = before['streams'][0], after['streams'][0]
        for key in ('width', 'height', 'avg_frame_rate', 'nb_frames'):
            if a[key] != b[key]:
                raise ValueError(f'{source.name}: display copy changed {key}')
        times = lambda p: [float(f['best_effort_timestamp_time']) for f in p['frames']]
        ta, tb = times(before), times(after)
        if len(ta) != len(tb) or any(abs(x-y) > 1e-6 for x,y in zip(ta,tb)):
            raise ValueError(f'{source.name}: display timestamps changed')
        video['video'] = 'playback/' + source.name
        video['original_video'] = 'videos/' + source.name
        return dict(case=video['case'], source_sha256=sha(source), playback_sha256=sha(target),
                    source_stream=a, playback_stream=b, frames=len(ta), timestamps_verified=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(convert, data['videos']))
    (metadata / 'playback_provenance.json').write_text(json.dumps(dict(
        purpose='Browser display only; source videos, raw scientific arrays and scores unchanged',
        encoding='H264 High, 8-bit yuv420p, CRF18; original frame count/dimensions/fps/PTS verified',
        files=receipts), indent=2) + '\n')
    if legacy_data:
        (report / 'data.js').write_text('window.RIGIDITY_DATA=' + json.dumps(data, separators=(',', ':'), allow_nan=False) + ';')
    print('PLAYBACK_COPIES_VERIFIED', len(receipts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    prepare(parser.parse_args().root)
