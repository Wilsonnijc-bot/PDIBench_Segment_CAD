"""Incremental local collection of the owned full-video run and its replay.

Uses the existing SSH master; no credentials are stored. Mutable native scratch
is excluded. Partial output remains explicitly partial until terminal hashing.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import time


def run(args):
    workspace=Path(args.remote_workspace);remote_root=workspace/'results/link5_shape_codebook'
    config=remote_root/'full_video/config.json';local=Path(args.local_output);local.mkdir(parents=True,exist_ok=True)
    metadata=local/'full_video/metadata';metadata.mkdir(parents=True,exist_ok=True)
    cache_file=metadata/'collection_cache.json';cache=json.loads(cache_file.read_text()) if cache_file.exists() else {}
    ssh=['ssh','-S',args.control_path,args.host]
    def remote(command):
        return subprocess.run(ssh+[command],capture_output=True,text=True,check=True).stdout
    interpreter=str(Path(args.dependency)/'env/geometry/bin/python')
    module='robot.experiments.link5_shape_codebook.'
    while True:
        for name in (('build_full_video_replay','report') if args.include_replays else ('report',)):
            command=shlex.join(['env','PYTHONPATH='+str(workspace),'OMP_NUM_THREADS=2',interpreter,'-m',module+name,'--config',str(config)])
            print(remote(command).strip(),flush=True)
        script='''import pathlib,json,os,time
r=pathlib.Path(ROOT);f=r/"full_video";records=[]
terminal=(f/"completion.json").exists()
for directory,folders,files in os.walk(f):
 p=pathlib.Path(directory)
 folders[:]=[x for x in folders if x not in ("geometry-work","foundationpose-debug","__pycache__") and (INCLUDE_REPLAYS or "replay" not in x)]
 if p.parent==f/"cases" and not (p/"preparation.json").exists() and not (p/"worker.json").exists():
  folders[:]=[x for x in folders if x=="metadata"]
 for name in files:
  item=p/name
  if item.suffix in (".tmp",".pyc"):continue
  if item.suffix==".mp4" and p!=f/"replay/videos":continue
  if name.startswith("collection_") or name.startswith("export_manifest"):continue
  if name.endswith(".log") and not terminal:continue
  if item.name.startswith("anomaly") and item.suffix==".npz":
   score=p/("score_robot_structural.json" if "robot_structural" in name else "score.json")
   if not score.exists():continue
  if item.name=="input.npz" and not (p/"observation.json").exists():continue
  if item.suffix==".csv" and time.time()-item.stat().st_mtime<2:continue
  if item.is_file():
   stat=item.stat();records.append(dict(path=str(item.relative_to(r)),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
for name in ("README.md","RUN_REVIEW.md"):
 item=r/name
 if item.exists():
  stat=item.stat();records.append(dict(path=name,bytes=stat.st_size,mtime_ns=stat.st_mtime_ns))
print(json.dumps(dict(terminal=terminal,files=records)))
'''
        script=script.replace('ROOT',repr(str(remote_root))).replace('INCLUDE_REPLAYS',repr(args.include_replays))
        catalog=json.loads(remote(shlex.join(['python3','-c',script])))
        changed=[x for x in catalog['files'] if cache.get(x['path'])!=[x['bytes'],x['mtime_ns']] or not (local/x['path']).exists()]
        if changed:
            listing='\n'.join(x['path'] for x in changed)+'\n'
            temporary=remote_root/'full_video/metadata/collection_files.txt'
            subprocess.run(ssh+[shlex.join(['python3','-c','import sys,pathlib;pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())',str(temporary)])],input=listing,text=True,check=True)
            command=shlex.join(['tar','-cf','-','-C',str(remote_root),'-T',str(temporary)])
            stream=subprocess.Popen(ssh+[command],stdout=subprocess.PIPE)
            try:subprocess.run(['tar','-xf','-','-C',str(local)],stdin=stream.stdout,check=True)
            finally:stream.stdout.close()
            if stream.wait():raise RuntimeError('remote export tar failed')
            for row in changed:cache[row['path']]=[row['bytes'],row['mtime_ns']]
            cache_file.write_text(json.dumps(cache,separators=(',',':'))+'\n')
        print(datetime.now(timezone.utc).isoformat(),'COLLECTED',len(changed),'files',sum(x['bytes'] for x in changed),'bytes','terminal',catalog['terminal'],flush=True)
        if catalog['terminal']:
            print('LINK5_FULL_VIDEO_COLLECTION_FINISHED',flush=True);return
        if args.once:return
        time.sleep(45)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='zy992@eris2n7.research.partners.org')
    parser.add_argument('--control-path',default='/tmp/link5-eris-ssh')
    parser.add_argument('--remote-workspace',default='/PHShome/zy992/Wilson/deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006')
    parser.add_argument('--dependency',default='/PHShome/zy992/Wilson/dependency')
    parser.add_argument('--local-output',type=Path,default=Path(__file__).resolve().parents[3]/'results/link5_shape_codebook/round_four_frame0')
    parser.add_argument('--include-replays',action='store_true',help='explicitly build and collect optional replay bundles')
    parser.add_argument('--once',action='store_true');run(parser.parse_args())


if __name__=='__main__':main()
