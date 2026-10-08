"""Background status receipts and final collection; no chat updates or job control."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .collect_eris_results import remote_python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--gpu-job', required=True)
    parser.add_argument('--cpu-job', required=True)
    args = parser.parse_args()
    if not all(j.isdigit() for j in (args.gpu_job, args.cpu_job)):
        raise ValueError('Numeric owned job IDs required')
    metadata = args.local_root / 'metadata'
    metadata.mkdir(parents=True, exist_ok=True)
    script = '''import json,subprocess,time
from pathlib import Path
root=Path(%r);gpu=%r;cpu=%r
def latest(path):
 if not path.exists():return None
 with path.open('rb') as stream:
  stream.seek(0,2);end=stream.tell();size=min(end,262144);stream.seek(end-size)
  lines=stream.read().decode('utf-8',errors='replace').splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except ValueError:continue
 return None
status=subprocess.check_output(['sacct','-j',gpu+','+cpu,'--format=JobIDRaw,State,ExitCode','--noheader','--parsable2'],text=True)
exit_path=root/'metadata'/('depth-score-report-'+cpu)/'job.exit'
score_path=root/'rigidity/summary.json'
scores=json.loads(score_path.read_text()) if score_path.exists() else []
telemetry=latest(root/'metadata'/('telemetry-'+gpu+'.jsonl'))
print(json.dumps(dict(unix_time=time.time(),jobs=status,
 completed_inputs=len(list((root/'shared_inputs').glob('*.json'))),
 completed_depth_support=len([p for p in (root/'depth_support').glob('*.json') if p.stem!='summary']),
 completed_scores=sum(r['status']=='complete' for r in scores),failed_scores=sum(r['status']!='complete' for r in scores),
 gpu_csv=telemetry.get('gpu_csv') if telemetry else None,
 active_videos=telemetry.get('active_videos',[]) if telemetry else [],
 cpu_exit=exit_path.read_text().strip() if exit_path.exists() else None)))
''' % (args.remote_root, args.gpu_job, args.cpu_job)
    errors = 0
    while True:
        try:
            row = json.loads(remote_python(script)); errors = 0
        except Exception as exc:
            errors += 1
            row = dict(unix_time=time.time(), monitor_error=str(exc), consecutive_errors=errors)
        with (metadata / 'passive_progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        (metadata / 'passive_progress_latest.json').write_text(json.dumps(row, indent=2))
        print(json.dumps(row), flush=True)
        if row.get('cpu_exit') is not None:
            command = [sys.executable, '-u', '-m',
                       'robot.experiments.link5_pair_selection_ablation.collect_eris_results',
                       '--remote-root', args.remote_root, '--local-root', str(args.local_root),
                       '--cpu-job', args.cpu_job]
            raise SystemExit(subprocess.run(command).returncode)
        states = row.get('jobs', '')
        cpu_row = next((line for line in states.splitlines() if line.startswith(args.cpu_job + '|')), '')
        if any(state in cpu_row for state in ('CANCELLED', 'FAILED', 'TIMEOUT', 'NODE_FAIL')):
            print('PASSIVE_MONITOR_REPORT_JOB_TERMINATED', cpu_row, flush=True)
            raise SystemExit(1)
        if errors >= 10:
            print('PASSIVE_MONITOR_CONNECTION_UNAVAILABLE', flush=True)
            raise SystemExit(1)
        time.sleep(60)


if __name__ == '__main__':
    main()
