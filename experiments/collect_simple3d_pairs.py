"""Collect terminal pairs, verify bytes and refresh the offline replay during a run."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'PDI-Bench-edited/src')]
from experiments.simple3d_pipeline import write_json
from experiments.run_simple3d_link5_pairs import aggregate
from experiments.verify_simple3d_pairs import verify
from scripts.build_simple3d_pair_replay import build
from scripts.analyze_simple3d_pair_correlation import analyze


def ledger(state,summary,heading):
    path=ROOT/'experiment_GPU_record.md';body=path.read_text()
    start=body.index(heading);end=body.find('\n## ',start+len(heading));end=end if end>=0 else len(body)
    section=body[start:end]
    terminal=state['exit'] is not None and summary['videos_processed']==45
    status=('completed' if state['exit']=='0' else 'failed') if terminal else 'running'
    section=re.sub(r'(?m)^- \*\*Status:\*\*.*$',f'- **Status:** {status}',section,count=1)
    scope=state.get('scope_videos',45)
    result=f"{summary['successful_simple3d_scores']}/{10*scope} successful scores; {summary['terminal_pairs']} terminal pairs; {summary['videos_processed']}/{scope} videos terminal. "
    result+=f"Observed GPU exit {state['exit']}; native artifacts and SHA256 transfers verified." if terminal else 'Batch remains active; results are partial. User permits leaving it active after verified comparisons.'
    section=re.sub(r'(?m)^- \*\*Results:\*\*.*$','- **Results:** '+result,section,count=1)
    temp=path.with_suffix('.pairs.tmp');temp.write_text(body[:start]+section+body[end:]);temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true')
    p.add_argument('--root',type=Path,default=ROOT/'results/simple3d-20261003-run2/link5_pair_visible_anchor')
    a=p.parse_args();root=a.root.resolve()
    remote='/root/autodl-tmp/pdi/simple3d-evaluation/'+str(root.relative_to(ROOT))
    preview_path=root/'metadata/preview.json'
    heading=(json.loads(preview_path.read_text()).get('ledger_heading') if preview_path.exists() else None) or '## Simple3D pair-visible link5 rerun — 2026-10-03'
    ssh=['ssh','-i',str(ROOT/'.tmp/autodl_pdi_ed25519'),'-o','BatchMode=yes','-o','ConnectTimeout=20',
         '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',
         '-o','ControlMaster=auto','-o','ControlPersist=600','-o','ControlPath='+str(ROOT/'.tmp/s3d-ssh'),
         '-p','26211']
    host='root@region-9.autodl.pro'
    def run(code,*args):
        return json.loads(subprocess.check_output([*ssh,host,shlex.join(['python3','-c',code,*args])],text=True))
    def sync(source,dest,*flags):
        dest.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az',*flags,'-e',shlex.join(ssh),f'{host}:{remote}/{source}/',str(dest)+'/'],check=True)
    done_file=root/'metadata/transfer_verification.json'
    done=json.loads(done_file.read_text()) if done_file.exists() else {}
    state_code="""import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);e=r/'metadata/execution/run.exit'
scope=json.loads((r/'metadata/run.json').read_text()).get('expected_videos',45) if (r/'metadata/run.json').exists() else 45
print(json.dumps(dict(pairs=[str(p.parent.relative_to(r)) for p in sorted(r.glob('cases/*/bin_*/comparison.json'))],
 cases=[p.parent.name for p in r.glob('cases/*/comparisons.json')],scope_videos=scope,exit=e.read_text().strip() if e.exists() else None)))"""
    hash_code="""import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);names=json.loads(sys.argv[2]);print(json.dumps({name:{str(p.relative_to(r/name)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/name).rglob('*') if p.is_file()} for name in names}))"""
    last=-1
    while True:
        try:
            state=run(state_code,remote)
            changed=False
            new=[relative for relative in state['pairs'] if relative not in done]
            if new:
                transfer=root/'metadata/transfer_files.tmp'
                transfer.write_text(''.join(relative+'/\n' for relative in new))
                staging=root/'metadata/.transfer_staging'
                staging.mkdir(parents=True,exist_ok=True)
                subprocess.run(['rsync','-azr','--timeout=60','--files-from='+str(transfer),'-e',shlex.join(ssh),
                    f'{host}:{remote}/',str(staging)+'/'],check=True,timeout=180)
                transfer.unlink()
                expected=run(hash_code,remote,json.dumps(new))
                for relative,files in expected.items():
                    for name,digest in files.items():
                        f=staging/relative/name
                        assert f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest()==digest,str(f)
                    target=root/relative
                    if target.exists():shutil.rmtree(target)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    (staging/relative).replace(target)
                    done[relative]=dict(status='SHA256_verified',file_count=len(files))
                changed=True;write_json(done_file,done)
                shutil.rmtree(staging)
            for video in state['cases']:
                case=root/'cases'/video
                sync('cases/'+video,case,'--include=/*.json','--exclude=/*/','--exclude=*')
            sync('metadata',root/'metadata','--include=/*.json','--include=/execution/***','--exclude=*')
            manifest=json.loads((root/'inputs/manifest.json').read_text())
            summary=aggregate(root,manifest)
            audit=verify(root)
            if changed or last!=len(state['pairs']) or not (root/'replay/index.html').exists():
                build(root);last=len(state['pairs'])
            ledger(state,summary,heading)
            write_json(root/'metadata/collection_status.json',dict(remote=state,summary={k:v for k,v in summary.items() if k!='failures'},verified=audit['verified_successful_pairs']))
            print('COLLECTED',len(done),'pairs;',summary['successful_simple3d_scores'],'scores;',summary['videos_processed'],'videos; GPU exit',state['exit'],flush=True)
            if state['exit'] is not None:
                log=(root/'metadata/execution/run.log').read_text()
                assert 'BATCH_FINISHED' in log, 'missing batch completion marker'
                if state['scope_videos']<45:
                    preview=json.loads((root/'metadata/preview.json').read_text())
                    assert set(preview['video_ids'])<=set(state['cases']),'incomplete preview cases'
                    preview_rows=[r for r in json.loads((root/'comparisons.json').read_text()) if r['video_id'] in preview['video_ids']]
                    assert len(preview_rows)==10*len(preview['video_ids']),'incomplete preview pairs'
                    if preview.get('status')=='approved_full_run':
                        preview.update(successful_scores=sum(r['simple3d_status']=='complete' for r in preview_rows),
                            terminal_pairs=len(preview_rows),gpu_preview_exit_status=state['exit'])
                        write_json(root/'metadata/preview.json',preview)
                        build(root)
                        print('PREVIEW_FINISHED_FULL_CAMPAIGN_QUEUED',flush=True)
                        if not a.watch:return 0
                        time.sleep(15)
                        continue
                    preview.update(status='awaiting_user_review',successful_scores=sum(r['simple3d_status']=='complete' for r in preview_rows),
                        terminal_pairs=len(preview_rows),gpu_exit_status=state['exit'],remaining_videos_deferred=45-len(preview['video_ids']))
                    write_json(root/'metadata/preview.json',preview)
                    build(root)
                    body=(ROOT/'experiment_GPU_record.md').read_text()
                    start=body.index(heading)
                    section=body[start:]
                    section=re.sub(r'(?m)^- \*\*Status:\*\*.*$',
                        '- **Status:** three-video preview complete; awaiting user refinement review.',section,count=1)
                    section=re.sub(r'(?m)^- \*\*Results:\*\*.*$',
                        f"- **Results:** User-requested three-video preview terminal: {preview['successful_scores']}/30 scores, 3/3 videos; GPU preview exit {state['exit']}; byte transfers/native evidence verified. Remaining 42 videos held for the user's refinement review; full 45-video experiment is unfinished.",section,count=1)
                    (ROOT/'experiment_GPU_record.md').write_text(body[:start]+section)
                    print('PREVIEW_READY',json.dumps(preview),flush=True)
                    return 0 if state['exit']=='0' else 1
                assert summary['videos_processed']==45 and summary['terminal_pairs']==450,'incomplete batch'
                preview=json.loads((root/'metadata/preview.json').read_text())
                preview.update(status='full_run_complete',remaining_videos_deferred=0)
                write_json(root/'metadata/preview.json',preview)
                correlation=analyze(root)
                build(root)
                if state['exit']=='0':
                    assert len(done)==450 and set(done)==set(state['pairs']), 'incomplete SHA256 collection'
                    assert audit['status']=='passed' and audit['verified_successful_pairs']==summary['successful_simple3d_scores']
                    receipt=dict(status='verified_complete',verified_at=datetime.now(timezone.utc).isoformat(),
                        run_sha256=hashlib.sha256((root/'metadata/run.json').read_bytes()).hexdigest(),
                        input_manifest_sha256=hashlib.sha256((root/'inputs/manifest.json').read_bytes()).hexdigest(),
                        videos_processed=45,terminal_pairs=450,sha256_verified_pairs=len(done),
                        successful_scores=summary['successful_simple3d_scores'],
                        verified_successful_pairs=audit['verified_successful_pairs'],artifact_audit_status=audit['status'],
                        correlation_status=correlation['status'],replay_export_complete=True,
                        local_result_root=str(root))
                    write_json(root/'metadata/completion_verified.json',receipt)
                    # The remote watcher cannot stop the instance until this acknowledgment arrives.
                    upload="""import json,pathlib,sys
r=pathlib.Path(sys.argv[1]);receipt=json.load(sys.stdin)
p=r/'metadata/completion_verified.json';temp=p.with_suffix('.json.upload.tmp')
temp.write_text(json.dumps(receipt,indent=2)+'\\n');temp.replace(p)"""
                    subprocess.run([*ssh,host,shlex.join(['python3','-c',upload,remote])],
                        input=json.dumps(receipt),text=True,check=True)
                    print('FULL_CAMPAIGN_VERIFIED_COMPLETION_ACK_SENT',flush=True)
                return 0 if state['exit']=='0' else 1
            if not a.watch:return 0
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired,OSError,ValueError,AssertionError) as exc:
            print('COLLECTION_RETRY',str(exc),flush=True)
            if not a.watch:return 1
        time.sleep(45)


if __name__=='__main__':
    raise SystemExit(main())
