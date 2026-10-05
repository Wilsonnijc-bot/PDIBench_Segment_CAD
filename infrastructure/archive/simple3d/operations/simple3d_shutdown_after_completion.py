"""Stop the AutoDL instance only after this full campaign is collected and audited."""

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'experiments/simple3d_shutdown_after_completion.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completion_ready(root, schedule):
    """Fail closed; the receipt is issued by the local collector after its audit."""
    run = root / 'metadata/run.json'
    if digest(run) != schedule['run_sha256']:
        return False, 'held: run identity changed'
    exit_path = root / 'metadata/execution/run.exit'
    if not exit_path.exists():
        return False, 'waiting for GPU campaign completion'
    if exit_path.read_text().strip() != '0':
        return False, 'held: GPU campaign exited unsuccessfully'
    receipt_path = root / 'metadata/completion_verified.json'
    if not receipt_path.exists():
        return False, 'waiting for verified local collection, replay and correlation'
    receipt = json.loads(receipt_path.read_text())
    expected = schedule['expected_pairs']
    if not (receipt.get('status') == 'verified_complete'
            and receipt.get('run_sha256') == schedule['run_sha256']
            and receipt.get('input_manifest_sha256') == schedule['input_manifest_sha256']
            and receipt.get('videos_processed') == schedule['expected_videos']
            and receipt.get('terminal_pairs') == expected
            and receipt.get('sha256_verified_pairs') == expected
            and receipt.get('successful_scores') == receipt.get('verified_successful_pairs')
            and receipt.get('artifact_audit_status') == 'passed'
            and receipt.get('correlation_status') == 'complete'
            and receipt.get('replay_export_complete') is True):
        return False, 'held: local completion receipt does not match this run'
    rows = [json.loads(p.read_text()) for p in root.glob('cases/*/bin_*/comparison.json')]
    if len(rows) != expected or len({r['video_id'] for r in rows}) != schedule['expected_videos']:
        return False, 'held: remote comparison coverage incomplete'
    if sum(r['simple3d_status'] == 'complete' for r in rows) != receipt['successful_scores']:
        return False, 'held: remote scores disagree with local audit'
    if 'BATCH_FINISHED' not in (root / 'metadata/execution/run.log').read_text():
        return False, 'held: missing native completion marker'
    return True, 'all campaign and local collection checks passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    execution = root / 'metadata/execution'
    schedule = json.loads((execution / 'shutdown_schedule.json').read_text())
    status_path = execution / 'shutdown_status.json'
    previous = None
    while True:
        try:
            ready, reason = completion_ready(root, schedule)
            if ready:
                active = subprocess.run(['tmux', 'has-session', '-t', schedule['campaign_tmux']],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if active.returncode == 0:
                    ready, reason = False, 'waiting for campaign tmux to exit'
                else:
                    gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                         capture_output=True, text=True, check=True)
                    if gpu.stdout.strip():
                        ready, reason = False, 'waiting for other GPU processes to finish'
            state = dict(status='ready' if ready else 'armed', reason=reason,
                         checked_at=datetime.now(timezone.utc).isoformat(),
                         run_sha256=schedule['run_sha256'], watcher_pid=os.getpid())
            write(status_path, state)
            if reason != previous:
                print(state['checked_at'], reason, flush=True)
                previous = reason
            if ready:
                os.sync()
                state.update(status='shutdown_requested', command='/usr/bin/shutdown')
                write(status_path, state)
                os.sync()
                # AutoDL supplies this command as a shell script, sometimes without a shebang.
                result = subprocess.run(['/bin/sh', '-c', '/usr/bin/shutdown'],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode:
                    state.update(status='shutdown_command_failed', returncode=result.returncode)
                    write(status_path, state)
                return result.returncode
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            write(status_path, dict(status='armed', reason='held: ' + str(exc),
                                    checked_at=datetime.now(timezone.utc).isoformat()))
        time.sleep(30)


if __name__ == '__main__':
    raise SystemExit(main())
