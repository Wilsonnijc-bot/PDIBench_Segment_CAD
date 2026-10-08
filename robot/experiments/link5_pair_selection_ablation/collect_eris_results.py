"""Watch the owned CPU report job, collect review artifacts, verify, and open locally."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import time

HOST = 'zy992@eris2n7.research.partners.org'
CONTROL = '/tmp/codex-link5-eris-%C'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlPath=' + CONTROL, HOST]


def remote_python(script):
    return subprocess.check_output(SSH + ['/usr/bin/python3', '-'], input=script, text=True, timeout=60)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def transfer_tar(remote_root, local_root):
    script = '''import tarfile,sys
from pathlib import Path
root=Path(%r)
files=[]
for name in ['rigidity','report','depth_support','shared_inputs']:
 for p in (root/name).rglob('*'):
  if p.is_file() and not p.is_symlink() and (name!='shared_inputs' or p.suffix=='.json'):files.append(p)
for p in (root/'metadata').rglob('*'):
 if p.is_file() and not p.is_symlink() and p.suffix in {'.json','.jsonl','.csv','.log','.exit','.pid','.sh','.py'}:files.append(p)
if (root/'REPORT.md').exists():files.append(root/'REPORT.md')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for p in files:archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
''' % remote_root
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(SSH + ['/usr/bin/python3', '-'], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors)
        process.stdin.write(script.encode()); process.stdin.close()
        destination_root = local_root.resolve()
        count = 0
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                destination = (destination_root / member.name).resolve()
                if destination_root not in destination.parents or not member.isfile():
                    raise ValueError('Unexpected archive path/type: ' + member.name)
                if destination.exists():
                    destination.chmod(destination.stat().st_mode | 0o200)
                archive.extract(member, destination_root)
                count += 1
        if process.wait(timeout=60):
            errors.seek(0)
            raise RuntimeError('Archive transfer failed: ' + errors.read().decode(errors='replace')[-2000:])
    print('ARTIFACTS_TRANSFERRED_BY_TAR', count, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--cpu-job', required=True)
    p.add_argument('--watch', action='store_true')
    a = p.parse_args()
    if not a.cpu_job.isdigit():
        raise ValueError('Numeric owned Slurm job required')
    a.local_root.mkdir(parents=True, exist_ok=True)
    check_script = """import json,subprocess
from pathlib import Path
r=Path(%r);job=%r
p=r/'metadata'/('depth-score-report-'+job)/'job.exit'
state=subprocess.check_output(['sacct','-j',job,'--format=JobIDRaw,State,ExitCode','--noheader','--parsable2'],text=True)
print(json.dumps(dict(exit=p.read_text().strip() if p.exists() else None,state=state)))
""" % (a.remote_root, a.cpu_job)
    while True:
        status = json.loads(remote_python(check_script))
        if status['exit'] is not None:
            break
        terminated = any(x in status['state'] for x in ('CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL'))
        if terminated:
            raise RuntimeError('Report job ended without an exit receipt: ' + status['state'])
        print('WAITING_FOR_DEPTH_FILTERED_REPORT', a.cpu_job, flush=True)
        if not a.watch:
            return
        time.sleep(30)
    shell = shlex.join(['ssh', '-o', 'BatchMode=yes', '-o', 'ControlPath=' + CONTROL])
    source = HOST + ':' + a.remote_root.rstrip('/') + '/'
    commands = [
        ['rsync', '-a', '--stats', '-e', shell, '--include=rigidity/***', '--include=report/***',
         '--include=depth_support/***', '--include=REPORT.md', '--include=shared_inputs/',
         '--include=shared_inputs/*.json', '--exclude=*', source, str(a.local_root) + '/'],
        ['rsync', '-a', '--stats', '-e', shell, '--include=*/', '--include=*.json', '--include=*.jsonl',
         '--include=*.csv', '--include=*.log', '--include=*.exit', '--include=*.pid', '--include=*.sh',
         '--include=*.py', '--exclude=*', source + 'metadata/', str(a.local_root / 'metadata') + '/'],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            if 'rsync: command not found' in result.stderr:
                transfer_tar(a.remote_root, a.local_root)
                break
            raise RuntimeError('Artifact transfer failed: ' + result.stderr[-2000:])
        print(result.stdout[-1400:], flush=True)
    remote_manifest = json.loads(remote_python("from pathlib import Path\nprint((Path(%r)/'manifest.json').read_text())\n" % a.remote_root))
    (a.local_root / 'metadata/remote_manifest.json').write_text(json.dumps(remote_manifest, indent=2))
    inventory_script = """import json,hashlib
from pathlib import Path
r=Path(%r);files=[]
for name in ['rigidity','report','depth_support','shared_inputs']:
 for p in (r/name).rglob('*'):
  if not p.is_file() or (name=='shared_inputs' and p.suffix!='.json'):continue
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  files.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(files))
""" % a.remote_root
    inventory = json.loads(remote_python(inventory_script))
    for row in inventory:
        path = a.local_root / row['path']
        if not path.is_file() or path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
            raise ValueError('Collected artifact checksum mismatch: ' + row['path'])
    rows = json.loads((a.local_root / 'rigidity/summary.json').read_text())
    expected = {(e['video_id'], m) for e in remote_manifest['entries'] for m in ('balanced_v0', 'refine_v1')}
    actual = {(r['case'], r['method']) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError('Final score rows do not exactly cover the manifest/methods')
    successes = sum(r['status'] == 'complete' for r in rows)
    for row in rows:
        if row['status'] == 'complete' and row.get('depth_filter') != 'refined_all_frames':
            raise ValueError('Final table contains a superseded baseline score')
    record = dict(status='complete' if successes == len(rows) and status['exit'] == '0' else 'failed',
                  method_video_scores=successes, expected_method_video_scores=len(rows),
                  files=len(inventory), bytes=sum(r['bytes'] for r in inventory),
                  artifacts=inventory, raw_geometry_location=a.remote_root + '/shared_inputs', published=False)
    (a.local_root / 'collection.json').write_text(json.dumps(record, indent=2))
    execution = a.local_root / 'metadata/execution_status.json'
    current = json.loads(execution.read_text()) if execution.exists() else {}
    current.update(status='collected_awaiting_visual_review', successful_method_video_scores=successes,
                   expected_method_video_scores=len(rows), checksum_verified_collection=True, published=False)
    execution.write_text(json.dumps(current, indent=2))
    viewer = a.local_root.resolve() / 'report/index.html'
    if viewer.exists():
        subprocess.run(['open', str(viewer)], check=True)
    print('ERIS_RESULTS_COLLECTED', successes, len(rows), str(viewer), flush=True)
    if record['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
