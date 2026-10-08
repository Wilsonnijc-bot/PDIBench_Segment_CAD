"""Collect and hash-verify completed CPU reports without GPU/job polling."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from .collect_full_video import main as collect_main


def run():
    workspace=Path(__file__).resolve().parents[3]
    remote_workspace='/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook'
    remote_root=remote_workspace+'/results/link5_shape_codebook'
    remote_python='/PHShome/zy992/Wilson/dependency/env/geometry/bin/python'
    ssh=['ssh','-S','/tmp/link5-eris-ssh','zy992@eris2n7.research.partners.org']
    local=workspace/'results/link5_shape_codebook'
    wait=['env','PYTHONPATH='+remote_workspace,remote_python,'-m',
          'robot.experiments.link5_shape_codebook.postprocess_completed','--wait-only',
          remote_root+'/full_video/metadata/postprocessing.json']
    subprocess.run(ssh+[shlex.join(wait)],check=True)
    import sys
    sys.argv=['collect_full_video','--once']
    collect_main()
    def remote(command):return subprocess.check_output(ssh+[shlex.join(command)],text=True)
    manifest=json.loads(remote(['cat',remote_root+'/metadata/export_manifest.json']))
    missing=[]
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
        return h.hexdigest()
    for row in manifest['files']:
        path=local/row['path']
        if not path.exists() or path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:
            missing.append(row['path'])
    if missing:
        listing=remote_root+'/full_video/metadata/final_collection_files.txt'
        command=['python3','-c','import sys,pathlib;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())',listing]
        subprocess.run(ssh+[shlex.join(command)],input='\n'.join(missing)+'\n',text=True,check=True)
        proc=subprocess.Popen(ssh+[shlex.join(['tar','-cf','-','-C',remote_root,'-T',listing])],stdout=subprocess.PIPE)
        subprocess.run(['tar','-xf','-','-C',str(local)],stdin=proc.stdout,check=True);proc.stdout.close()
        if proc.wait():raise RuntimeError('final artifact transfer failed')
    mismatches=[r['path'] for r in manifest['files'] if digest(local/r['path'])!=r['sha256']]
    if mismatches:raise ValueError('final artifact SHA256 mismatches: '+str(mismatches))
    (local/'metadata/export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    completion=json.loads((local/'full_video/metadata/postprocessing.json').read_text())
    if completion.get('correlations_verified_against_scipy') is not True:
        raise ValueError('independent correlation verification missing')
    m=json.loads((local/'full_video/replay/manifest.js').read_text()[len('window.LINK5_REPLAY='):-2])
    if m['counts']['full_video_scored_frames']!={'paper_original':3310,'robot_structural':3310} or len(m['cases'])!=30:
        raise ValueError('replay does not contain the requested complete cohort')
    broken=[]
    for c in m['cases']:
        for f in c['frames']:
            if not (local/'full_video/replay'/f['packet']).exists():broken.append((c['video_id'],f['frame_id']))
    if broken:raise ValueError('replay packets missing')
    header='## Link5 full-video matched-generator evaluation — 2026-10-05'
    ledger=workspace/'documentation/gpu/experiment_GPU_record.md'
    remote_ledger=remote(['cat',remote_workspace+'/documentation/gpu/experiment_GPU_record.md'])
    updated=remote_ledger.split(header,1)[1].split('\n## ',1)[0]
    current=ledger.read_text();before,after=current.split(header,1)
    suffix='\n## '+after.split('\n## ',1)[1] if '\n## ' in after else ''
    ledger.write_text(before+header+updated+suffix)
    verification=dict(status='verified',files=len(manifest['files']),bytes=manifest['total_bytes'],
       sha256_mismatches=0,replay_videos=30,replay_frames_per_checkpoint=3310,missing_replay_packets=0,
       label_column='AB',correlations_verified_against_scipy=True)
    (local/'full_video/metadata/local_completed_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('LINK5_COMPLETED_REPORT_COLLECTED_AND_VERIFIED',json.dumps(verification),flush=True)


if __name__=='__main__':run()
