"""Hash-verified collection of refined inputs and their SAM/training receipts."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

from .common import FOUR_REFERENCE_IDS, read, sha, write


def collect(host,control_path,remote,local,interpreter,include_evidence=False,include_prepared=False):
    remote=Path(remote);local=Path(local);local.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S',str(control_path),host]
    code=f'''import hashlib,json
from pathlib import Path
root=Path({str(remote)!r})
receipt=json.loads((root/'normal_reference/construction.json').read_text())
assert receipt['normal_video_ids']=={list(FOUR_REFERENCE_IDS)!r}
assert receipt['normal_frame_ids']==[0]*4 and not receipt['later_frames_in_reference']
files=[root/'config.json']
files.extend((root/'normal_reference').rglob('*'))
videos=receipt['normal_video_ids']
if {include_prepared!r}:
    config=json.loads((root/'config.json').read_text())
    videos=[case['id'] for case in config['cases']]
    for video in videos:
        status=json.loads((root/'cases'/video/'status.json').read_text())
        assert status['status']=='complete', 'preparation incomplete: '+video
for video in videos:
    folder=root/'cases'/video
    files.extend(folder/name for name in ['status.json','observations.json','geometry.json'])
    observations=folder/'observations' if {include_prepared!r} else folder/'observations/frame_00000'
    files.extend(observations.rglob('*'))
    files.extend((folder/'masking').rglob('*'))
for pattern in ['staging.json','preflight.json','references-*.log','references-*.exit','preparation-gpu-*.txt','preparation-host-*.txt','reference-preparation-exits.json']:
    files.extend((root/'metadata').glob(pattern))
if {include_evidence!r}:
    files.extend(root/name for name in ['link5_normalization.json'])
    for name in ['alignment','reference_pose','splits','augmentation','normal_training_data','sanity','sanity_robot_structural']:
        files.extend((root/name).rglob('*'))
    for name in ['training','training_robot_structural']:
        folder=root/name
        if (folder/'completion.json').is_file():
            files.extend(p for p in folder.rglob('*') if not p.name.startswith('epoch_'))
    for pattern in ['round-*.log','round-*.exit','round-host-*.txt','round-gpu-*.txt','round-gpu-telemetry-*.csv','refined-continuation-source*.json','pretrained-backbone.json','foundationpose-runtime-preflight.json','prepared_input_qc.json','preparation-expansion.json','training-acceleration-source.json','vlm_guard_costs.json']:
        files.extend((root/'metadata').glob(pattern))
    files.extend((root/'metadata').glob('refined-round-*/progress.json'))
rows=[]
for path in sorted(set(files)):
    if path.is_file() and not path.is_symlink() and 'vendor-source-snapshot' not in path.parts:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
        rows.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(source_root=str(root),files=rows)))
'''
    result=subprocess.run(ssh+[shlex.join([interpreter,'-'])],input=code,text=True,check=True,capture_output=True)
    manifest=json.loads(result.stdout)
    with tempfile.TemporaryDirectory(prefix='link5-reference-transfer-') as temporary:
        archive=Path(temporary)/'reference.tar'
        with archive.open('wb') as stream:
            command=shlex.join(['tar','-C',str(remote),'-cf','-','-T','-'])
            subprocess.run(ssh+[command],input=('\n'.join(r['path'] for r in manifest['files'])+'\n').encode(),stdout=stream,check=True)
        allowed={r['path'] for r in manifest['files']}
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.name not in allowed or not member.isfile() or Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                    raise ValueError('unexpected collection archive member')
                path=local/member.name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(tar.extractfile(member).read())
    for row in manifest['files']:
        path=local/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('transfer identity mismatch: '+row['path'])
    manifest.update(status='verified',sha256_mismatches=0)
    write(local/'metadata/collection.json',manifest)
    print('LINK5_FOUR_FRAME0_COLLECTION_VERIFIED',len(manifest['files']),sum(r['bytes'] for r in manifest['files']),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True);p.add_argument('--control-path',type=Path,required=True)
    p.add_argument('--remote',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--interpreter',default='/PHShome/zy992/Wilson/dependency/env/geometry/bin/python')
    p.add_argument('--include-evidence',action='store_true',help='collect fixed normalization and completed training/validation evidence')
    p.add_argument('--include-prepared',action='store_true',help='collect fresh masks and saved filtered observations for every case; requires completed preparation')
    a=p.parse_args();collect(a.host,a.control_path,a.remote,a.root,a.interpreter,a.include_evidence,a.include_prepared)
