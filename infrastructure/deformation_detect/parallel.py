"""Per-video coordinators with process-safe GPU admission in one allocation."""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import html
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from infrastructure.deformation_detect.coordinator import (
    coordinate, identity, locked, now, preflight, read, refuse_live_worker,
    review, scientific_source, termination_as_interrupt, write,
)
from infrastructure.deformation_detect.layout import environment, root


@contextmanager
def gpu_admission(directory, slots):
    """OS locks release on crashes; no stale semaphore state or GPU imports."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    streams = [(directory/f'slot-{i}.lock').open('a') for i in range(slots)]
    acquired = None
    try:
        while acquired is None:
            for stream in streams:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                acquired = stream
                break
            if acquired is None:
                time.sleep(0.1)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
        for stream in streams:
            stream.close()


def coordinate_parallel(config, *, retry_disabled=False, check=preflight,
                        resource_identity=identity, source_identity=None):
    output = Path(config['output'])
    workers = min(config['execution']['workers'], len(config['cases']))
    slots = config['execution']['gpu_slots']
    dedicated = config['execution'].get('gpu_assignment', 'shared') == 'per_video'
    devices = [v.strip() for v in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if v.strip()]
    if dedicated and (len(devices) < workers or len(set(devices)) != len(devices) or any(v in {'-1', 'NoDevFiles'} for v in devices)):
        raise ValueError('per_video requires distinct allocated CUDA_VISIBLE_DEVICES entries for every active video worker')
    available_devices = devices[:workers] if dedicated else []
    active = {}
    with locked(output), termination_as_interrupt():
        if not (output/'run.json').exists() and any(p.name != '.coordinator.lock' for p in output.iterdir()):
            raise ValueError('New run output must be empty')
        state = read(output/'run.json') if (output/'run.json').exists() else {'schema_version': 1, 'run_id': config['run_id'], 'cases': {}}
        if state['run_id'] != config['run_id']:
            raise ValueError('Output belongs to a different run_id')
        if (output/'run.json').exists() and state.get('layout') != 'per_video':
            raise ValueError('Serial output cannot be converted to parallel output; use a fresh run')
        for case in config['cases']:
            receipt = output/'coordination'/case['id']/'execution.json'
            if receipt.is_file():
                refuse_live_worker({'status': 'running', 'directory': str(receipt.parent)})
        reports = check(config) or {}
        resources = {k: resource_identity(v) for k, v in config['resources'].items()}
        source = scientific_source() if source_identity is None else source_identity
        context = output/'coordination/context.json'
        write(context, {'environments': reports, 'resources': resources, 'source': source})
        write(output/'manifest.json', config)
        write(output/'preflight.json', reports)
        state.update(status='running', layout='per_video', execution=config['execution'], updated_at=now())
        for case in config['cases']:
            state['cases'].setdefault(case['id'], {'status': 'pending', 'stages': {}})
            child=output/'cases'/case['id']/'run.json'
            if child.is_file():
                state['cases'][case['id']]=read(child)['cases'].get(case['id'],state['cases'][case['id']])
        deferred=[c for c in config['cases'] if state['cases'][c['id']]['status']=='deferred']
        pending = iter(c for c in config['cases'] if c not in deferred)
        exhausted = False
        failed = {}

        def refresh():
            for case in config['cases']:
                child = output/'cases'/case['id']/'run.json'
                if child.is_file():
                    state['cases'][case['id']] = read(child)['cases'].get(case['id'], state['cases'][case['id']])
            for name, reason in failed.items():
                state['cases'][name].update(status='failed', reason=reason)
            state['deferred_cases']=[c['id'] for c in deferred]
            state['updated_at'] = now()
            write(output/'run.json', state)
            review(output, state)

        try:
            while active or not exhausted or deferred:
                if exhausted and not active and deferred:
                    pending=iter(deferred);deferred=[];exhausted=False
                while len(active) < workers and not exhausted:
                    case = next(pending, None)
                    if case is None:
                        exhausted = True
                        break
                    folder = output/'coordination'/case['id']
                    folder.mkdir(parents=True, exist_ok=True)
                    serial = len(list(folder.glob('coordinator-*.log'))) + 1
                    log = (folder/f'coordinator-{serial:04d}.log').open('w')
                    command = [sys.executable, '-u', '-m', 'infrastructure.deformation_detect.parallel',
                               '--manifest', str(output/'manifest.json'), '--context', str(context), '--case', case['id']]
                    if retry_disabled:
                        command.append('--retry-disabled')
                    child_env = environment()
                    device = available_devices.pop(0) if dedicated else None
                    if device is not None:
                        child_env['CUDA_VISIBLE_DEVICES'] = device
                    process = subprocess.Popen(command, cwd=root(), env=child_env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    receipt = {'pid': process.pid, 'host': socket.gethostname(), 'started_at': now(), 'command': command,
                               'cuda_visible_devices': child_env.get('CUDA_VISIBLE_DEVICES')}
                    write(folder/'execution.json', receipt)
                    active[case['id']] = (process, log, folder, receipt, serial)
                    print(f'{case["id"]}: coordinator started, pid {process.pid}', flush=True)
                for name, (process, log, folder, receipt, serial) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    write(folder/'execution.json', {**receipt, 'exit_code': code, 'finished_at': now()})
                    (folder/f'coordinator-{serial:04d}.exit').write_text(str(code)+'\n')
                    del active[name]
                    if dedicated:
                        available_devices.append(receipt['cuda_visible_devices'])
                    child = output/'cases'/name/'run.json'
                    if code==3 and child.is_file() and read(child)['cases'][name]['status']=='deferred':
                        deferred.append(next(c for c in config['cases'] if c['id']==name))
                    elif code not in (0, 2):
                        failed[name] = f'Coordinator exit {code}; see coordination/{name}/coordinator-{serial:04d}.log'
                        state['cases'][name] = {'status': 'failed', 'stages': {}, 'reason': f'Coordinator exit {code}; see {folder.name} coordinator log'}
                        if child.is_file():
                            state['cases'][name]['stages'] = read(child)['cases'].get(name, {}).get('stages', {})
                refresh()
                if active:
                    time.sleep(1)
        except BaseException:
            for process, *_ in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            for process, log, folder, receipt, serial in active.values():
                # Child SIGTERM invokes the existing coordinator's worker-group cleanup.
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                log.close()
                write(folder/'execution.json', {**receipt, 'exit_code': process.returncode, 'finished_at': now()})
                (folder/f'coordinator-{serial:04d}.exit').write_text(str(process.returncode)+'\n')
            state['status'] = 'interrupted'
            refresh()
            raise
        refresh()
        state['status'] = 'complete' if all(c['status'] == 'complete' for c in state['cases'].values()) else 'completed_with_disabled_cases'
        write(output/'run.json', state)
        review(output, state)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--retry-disabled', action='store_true')
    args = parser.parse_args()
    config = read(args.manifest)
    parent = Path(config['output'])
    config = copy.deepcopy(config)
    config['cases'] = [c for c in config['cases'] if c['id'] == args.case]
    if len(config['cases']) != 1:
        raise ValueError('Case not uniquely present in parent manifest')
    slots = config['execution']['gpu_slots']
    config['execution'] = {'workers': 1, 'gpu_slots': slots, 'gpu_assignment': 'shared'}
    config['output'] = str(parent/'cases'/args.case)
    config['run_id'] += '.'+args.case
    context = read(args.context)
    state = coordinate(config, retry_disabled=args.retry_disabled,
                       check=lambda _: context['environments'], resource_records=context['resources'],
                       source_identity=context['source'], stage_admission=lambda: gpu_admission(parent/'.gpu-slots', slots),defer_timeouts_only=True)
    return 0 if state['status'] == 'complete' else 3 if state['status']=='deferred' else 2


if __name__ == '__main__':
    raise SystemExit(main())
