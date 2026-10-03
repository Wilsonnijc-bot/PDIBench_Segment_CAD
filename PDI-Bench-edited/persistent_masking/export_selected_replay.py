"""Export compact review artifacts from a completed local or global run."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from persistent_masking.frame_lineage import read_original_frame


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(work, destination, *, include_naive=False):
    """Build the review out of sight, then publish it only when complete."""
    if destination.exists():
        raise FileExistsError(f'Preserve existing export: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{destination.name}.building-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'review'
        _export_into(work, staged, include_naive=include_naive)
        if destination.exists():
            raise FileExistsError(f'Preserve existing export: {destination}')
        staged.rename(destination)


def _export_into(work, destination, *, include_naive):
    record = json.loads((work / 'provenance.json').read_text())
    if record['level'] not in {'local', 'global'}:
        raise ValueError('This exporter requires a labeled local or global run')
    destination.mkdir(parents=True)
    vlm1_model = record['config']['vlm1']['model']
    model = record['config']['vlm2']['model']
    input_prefix = 'qwen' if 'qwen' in vlm1_model.lower() else 'vlm1'
    lines = [f'# {Path(vlm1_model).name}/{model} masking review', '',
             'Each completed case has the first three actual VLM1 candidate crops,',
             f'the full source frame VLM1 first classified as deformed, {model}’s',
             'five-point preview on the selected seed frame, and one refined masking replay.',
             'If the first candidate was unusable, VLM2 may have fallen back to a later',
             'VLM1-confirmed deformation frame; the manifest records both frames.',
             f'These are {record["level"]} test outputs pending visual review, not promoted results.', '']
    if include_naive:
        lines.extend(['`naive_masking.mp4` is copied unchanged from this run’s existing',
                      'frame-zero naive SAM3 path, without a VLM refinement call.', ''])
    manifest = {'level': record['level'], 'cases': record['cases'], 'results': {}}
    for case in record['cases']:
        result = record['results'][case]
        status = result['status']
        entry = manifest['results'][case] = {'status': status, 'hashes': {}}
        case_dir = destination / case
        if include_naive:
            naive = work / 'naive' / case / 'naive_masking.mp4'
            if naive.is_file() and naive.stat().st_size:
                expected = result.get('naive', {}).get('mp4_sha256')
                if expected and digest(naive) != expected:
                    raise ValueError(f'{case}: naive replay hash mismatch')
                case_dir.mkdir()
                target = case_dir / 'naive_masking.mp4'
                shutil.copyfile(naive, target)
                entry['hashes'][target.name] = digest(target)
            else:
                if status == 'completed_checks':
                    raise FileNotFoundError(f'{case}: requested naive replay missing: {naive}')
                entry['naive_replay'] = 'unavailable'
        if status != 'completed_checks':
            lines.append(f'- `{case}`: {status}; no completed refined replay exported.'
                         + (' Naive replay included.' if entry['hashes'] else ''))
            continue
        calls = result['calls']
        if len(calls) < 3:
            raise ValueError(f'{case}: fewer than three VLM1 input frames')
        selected = result['earliest_deformed_frame']
        seed = result['seed']
        if seed['frame'] != seed['earliest_deformed_frame'] + 1:
            raise ValueError(f'{case}: reseeding frame is not chosen deformed frame + 1')
        case_dir.mkdir(exist_ok=True)
        hashes = entry['hashes']
        for number, call in enumerate(calls[:3], 1):
            source = Path(call['crop'])
            if digest(source) != call['sha256']:
                raise ValueError(f'{case}: VLM1 crop hash mismatch')
            target = case_dir / f'{input_prefix}_input_{number}.png'
            shutil.copyfile(source, target)
            hashes[target.name] = digest(target)
        selected_path = case_dir / 'first_deformed_frame.png'
        Image.fromarray(read_original_frame(result['video'], selected)).save(selected_path)
        hashes[selected_path.name] = digest(selected_path)
        for name in ('points.png', 'masking.mp4'):
            source = work / 'artifacts' / case / name
            if not source.is_file() or source.stat().st_size == 0:
                raise FileNotFoundError(source)
            target = case_dir / name
            shutil.copyfile(source, target)
            hashes[name] = digest(target)
        entry.update(vlm1_input_frames=[call['frame'] for call in calls[:3]],
                     first_deformed_frame=selected,
                     selected_deformed_frame=seed['earliest_deformed_frame'],
                     fallback_rank=seed.get('fallback_rank', 0),
                     points_frame=seed['frame'],
                     hashes=hashes)
        lines.append(f'- [`{case}`]({case}/): VLM1 inputs '
                     f'{entry["vlm1_input_frames"]}; first deformed frame {selected}; '
                     f'VLM2 points on frame {seed["frame"]}'
                     f'{" (fallback from first deformed frame)" if entry["fallback_rank"] else ""}; '
                     'masking replay included.')
    provenance = destination / 'provenance'
    provenance.mkdir()
    shutil.copyfile(work / 'provenance.json', provenance / 'run.json')
    (provenance / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (destination / 'README.md').write_text('\n'.join(lines) + '\n')
    if (work / 'vlm3/index.html').is_file():
        shutil.copytree(work / 'vlm3', destination / 'vlm3')
        with (destination / 'README.md').open('a') as handle:
            handle.write('\n[Optional VLM3 object-overmask repair review](vlm3/index.html)\n')
    print(json.dumps({case: item['status'] for case, item in manifest['results'].items()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--include-naive', action='store_true',
                        help='Copy the existing naive SAM3 replay beside each refined replay')
    args = parser.parse_args()
    export(args.work.resolve(), args.destination.resolve(), include_naive=args.include_naive)


if __name__ == '__main__':
    main()
