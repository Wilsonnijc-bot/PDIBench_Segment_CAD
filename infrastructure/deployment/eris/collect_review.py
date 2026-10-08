"""Package selected terminal-run artifacts without model links or geometry scratch."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def collect(plan, name, output, cases=None):
    assert re.fullmatch(r'[A-Za-z0-9_.-]+', name)
    review = Path('infrastructure/shared/experimental/migration_validation/results')/name
    files = {}
    manifest = {'cases': {}, 'files': {}, 'omitted': ['model/vendor symlinks', 'MegaSAM scratch', 'geometry caches'],
                'note': 'Selected current artifacts and failed-attempt control records; full attempts remain on ERIS.'}

    def add(path, member):
        if path.is_symlink() or not path.is_file():
            return
        files[str(member)] = path

    for group in plan['groups']:
        root = Path(group['output'])
        state = json.loads((root/'run.json').read_text())
        selected = {case: data for case, data in state['cases'].items() if cases is None or case in cases}
        if not selected:
            continue
        if cases is None and state['status'] not in ('complete', 'completed_with_disabled_cases'):
            raise ValueError('Group is not terminal: '+group['id'])
        if any(data['status'] not in ('complete', 'disabled') for data in selected.values()):
            raise ValueError('Selected cases are not terminal: '+group['id'])
        for filename in ('run.json', 'manifest.json', 'preflight.json'):
            add(root/filename, review/'metadata/remote'/group['id']/filename)
        for case, data in selected.items():
            manifest['cases'][case] = {'status': data['status'], 'reason': data.get('reason'), 'stages': {}}
            for stage, entry in data['stages'].items():
                if not entry.get('directory'):
                    continue
                source = Path(entry['directory'])
                owner = 'object' if source.resolve().is_relative_to((root/'cases'/case/'object').resolve()) else 'robot'
                target = Path(owner)/'results'/name/case/stage/source.name
                manifest['cases'][case]['stages'][stage] = {'status': entry['status'], 'remote': str(source), 'local': str(target)}
                for path in source.rglob('*'):
                    rel = path.relative_to(source)
                    if {'mega_sam', 'geometry-cache', '__pycache__'} & set(rel.parts):
                        continue
                    # Preserve exact current native work snapshots, including replay assets.
                    if path.is_symlink() and path.name == 'source.mp4' and path.parent.name == 'replay':
                        video = path.resolve()
                        expected = next(c['video'] for c in json.loads((root/'cases'/case/'manifest.json').read_text())['cases'] if c['id'] == case)
                        if video != Path(expected).resolve():
                            raise ValueError('Unexpected replay video link')
                        files[str(target/rel)] = video
                    else:
                        add(path, target/rel)
            # Old failed attempt controls are enough for local diagnosis; their full work stays remote.
            for entry in data.get('history', []):
                if entry.get('status') not in ('failed', 'disabled', 'interrupted') or not entry.get('directory'):
                    continue
                source = Path(entry['directory'])
                owner = 'object' if source.resolve().is_relative_to((root/'cases'/case/'object').resolve()) else 'robot'
                target = Path(owner)/'results'/name/case/'failed-attempts'/entry['stage']/source.name
                for filename in ('request.json', 'result.json', 'execution.json', 'stage.log'):
                    add(source/filename, target/filename)
            coordination = root/'coordination'/case
            for path in coordination.rglob('*'):
                add(path, review/'metadata/remote'/group['id']/'coordination'/case/path.relative_to(coordination))
    output.parent.mkdir(parents=True, exist_ok=True)
    if cases is not None and set(manifest['cases']) != set(cases):
        raise ValueError('Some requested cases were not found')
    with tarfile.open(output, 'w:gz', compresslevel=1, dereference=False) as archive:
        for member, path in sorted(files.items()):
            manifest['files'][member] = {'sha256': sha(path), 'bytes': path.stat().st_size, 'remote': str(path)}
            with path.open('rb') as stream:
                info = tarfile.TarInfo(member)
                info.size = path.stat().st_size
                info.mode = 0o644
                archive.addfile(info, stream)
        encoded = (json.dumps(manifest, indent=2)+'\n').encode()
        info = tarfile.TarInfo(str(review/'metadata/retrieved-artifacts.json'))
        info.size = len(encoded)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(encoded))
    print(json.dumps({'archive': str(output), 'sha256': sha(output), 'files': len(files), 'bytes': output.stat().st_size}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', help='Collect only these terminal cases while other videos continue')
    args = parser.parse_args()
    collect(json.loads(args.plan.read_text()), args.name, args.output, args.cases)
