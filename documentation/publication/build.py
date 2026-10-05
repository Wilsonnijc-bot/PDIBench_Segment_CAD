"""Verify and materialize the selected, frozen publication without re-running historical patch scripts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def verify():
    release = json.loads((ROOT/'documentation/publication/release.json').read_text())
    failures = []
    for name, expected in release['files'].items():
        path = ROOT/name
        if not path.is_file():failures.append(f'missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected['sha256']:failures.append(f'changed: {name}')
    if failures:raise ValueError('\n'.join(failures))
    return release


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,help='New staging directory; omit to verify only')
    a=p.parse_args();release=verify()
    if a.output:
        if a.output.exists():raise FileExistsError(a.output)
        a.output.mkdir(parents=True)
        for name in release['files']:
            dst=a.output/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dst)
        shutil.copy2(ROOT/'documentation/publication/release.json',a.output/'release.json')
    print(json.dumps({'status':'verified','files':len(release['files']),'output':str(a.output) if a.output else None,
        'mode':'exact-current-release','note':'Future run promotion requires updating the explicit selection and release inventory.'}))


if __name__=='__main__':main()
