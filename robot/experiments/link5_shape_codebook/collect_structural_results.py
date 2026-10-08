"""Collect selected structural-training evidence with remote/local hash checks."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

from robot.experiments.link5_shape_codebook.common import sha, write


def collect(remote_root, local_root, trials, checkpoint_paths, metadata_names):
    ssh=['ssh','-S','/tmp/link5-eris.sock','-o','BatchMode=yes','zy992@eris2n7.research.partners.org']
    selection=dict(root=str(remote_root),trials=trials,checkpoints=checkpoint_paths,metadata=metadata_names)
    job_ids=sorted({re.search(r'(\d+)$',name).group(1) for name in metadata_names if re.search(r'(\d+)$',name)})
    states={}
    if job_ids:
        result=subprocess.check_output(ssh+[shlex.join(['sacct','-n','-P','-j',','.join(job_ids),'--format=JobID,State,ExitCode'])],text=True)
        for line in result.splitlines():
            fields=line.split('|')
            if fields[0] in job_ids:states[fields[0]]=dict(state=fields[1],exit_code=fields[2])
        if set(states)!=set(job_ids) or any(r['state'].split()[0] not in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY') for r in states.values()):
            raise ValueError('only terminal allocations can be collected; cancelled/failed history remains explicitly labeled')
    selection['slurm_states']=states
    code='''import hashlib,json,sys
from pathlib import Path
selection=json.loads(sys.stdin.readline());root=Path(selection['root']);paths=[]
for trial in selection['trials']:
    folder=root/'structural_localization'/trial
    assert folder.is_dir(), str(folder)
    paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp') and not p.name.startswith('.'))
for name in selection['checkpoints']:
    path=root/'structural_localization'/name
    assert path.is_file(), str(path)
    paths.append(path)
for name in selection['metadata']:
    folder=root/'metadata'/name
    assert folder.is_dir(), str(folder)
    paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.pt','.tmp'))
rows=[]
for path in sorted(set(paths)):
    assert not path.is_symlink(), str(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    rows.append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(dict(source_root=str(root),selection=selection,files=rows)))
'''
    command=ssh+[shlex.join(['python3','-c',code])]
    manifest=json.loads(subprocess.check_output(command,input=json.dumps(selection)+'\n',text=True))
    local_root=Path(local_root);local_root.mkdir(parents=True,exist_ok=True)
    # Immutable source jobs/evaluations are prerequisites. Every byte below is
    # verified after transfer; stale/mutating files cannot count as success.
    with tempfile.TemporaryDirectory(prefix='link5-structural-collection-') as temporary:
        archive=Path(temporary)/'results.tar'
        with archive.open('wb') as out:
            subprocess.run(ssh+[shlex.join(['tar','-C',str(remote_root),'-cf','-','-T','-'])],
                           input=('\n'.join(r['path'] for r in manifest['files'])+'\n').encode(),stdout=out,check=True)
        allowed={r['path'] for r in manifest['files']}
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if member.name not in allowed or not member.isfile() or '..' in Path(member.name).parts or Path(member.name).is_absolute():
                    raise ValueError('unexpected archive member')
                destination=local_root/member.name;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(tar.extractfile(member).read())
    for row in manifest['files']:
        path=local_root/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('collection mismatch: '+row['path'])
    manifest.update(status='verified',sha256_mismatches=0,total_bytes=sum(r['bytes'] for r in manifest['files']))
    write(local_root/'structural_localization/collection.json',manifest)
    print('LINK5_STRUCTURAL_COLLECTION_VERIFIED',len(manifest['files']),manifest['total_bytes'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root',type=Path,required=True);parser.add_argument('--local-root',type=Path,required=True)
    parser.add_argument('--trial',action='append',default=[]);parser.add_argument('--checkpoint',action='append',default=[])
    parser.add_argument('--metadata',action='append',default=[])
    args=parser.parse_args();collect(args.remote_root,args.local_root,args.trial,args.checkpoint,args.metadata)
