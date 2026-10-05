"""Create lossless, independently seekable playback copies for saved replays.

Original source.mp4 files remain untouched. Decoded video-frame checksums must
match before a replay is pointed at its new playback copy.
"""
from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/prepare_smooth_replay_video.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess


def frame_hashes(path: Path) -> list[str]:
    result = subprocess.run(
        ['ffmpeg', '-v', 'error', '-threads', '2', '-i', str(path),
         '-map', '0:v:0', '-f', 'framemd5', '-'],
        check=True, capture_output=True, text=True,
    )
    return [line.rsplit(',', 1)[-1].strip() for line in result.stdout.splitlines()
            if line and not line.startswith('#')]


def prepare(page: Path) -> str:
    source = page.with_name('source.mp4')
    target = page.with_name('source.scrub.mp4')
    temporary = page.with_name('source.scrub.tmp.mp4')
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-threads', '2', '-i', str(source),
         '-map', '0:v:0', '-map', '0:a?', '-c:v', 'libx264', '-threads', '2',
         '-preset', 'fast', '-crf', '0', '-g', '1', '-c:a', 'copy',
         '-movflags', '+faststart', str(temporary)], check=True,
    )
    if frame_hashes(source) != frame_hashes(temporary):
        temporary.unlink()
        raise ValueError(f'Decoded frames differ: {source}')
    temporary.replace(target)
    html = page.read_text()
    html = html.replace('src="source.mp4"', 'src="source.scrub.mp4"')
    page.write_text(html)
    return f'{page.parent.parent.name}: verified identical decoded frames'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    pages = sorted(args.root.glob('cases/*/replay/task_object_exact-group.html'))
    if not pages:
        parser.error('No task-object replays found')
    with ThreadPoolExecutor(max_workers=3) as executor:
        for result in executor.map(prepare, pages):
            print(result, flush=True)
    print(f'Updated {len(pages)} replay playback copies.', flush=True)


if __name__ == '__main__':
    main()
