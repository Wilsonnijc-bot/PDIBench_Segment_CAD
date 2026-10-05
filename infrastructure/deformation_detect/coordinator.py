"""Resumable, artifact-verified robot/object coordination. No model imports here."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import re
import signal
import socket
import threading
import subprocess

from infrastructure.deformation_detect.layout import environment, root

# Topological order. Tracking for the object is independent of mask synchronization.
STAGES = {
    'link7_initial': ('sam3', (), 'robot'),
    'vlm1': ('vlm1', ('link7_initial',), 'robot'),
    'vlm2': ('sam3', ('vlm1',), 'robot'),
    'link7_masks': ('sam3', ('vlm2',), 'robot'),
    'object_masks': ('sam3', (), 'object'),
    'mask_join': ('sam3', ('link7_masks', 'object_masks'), 'robot'),
    'object_tracks': ('geometry', ('object_masks',), 'object'),
    'object_crops': ('sam3', ('mask_join', 'object_masks', 'object_tracks'), 'object'),
    'robot_score': ('geometry', ('mask_join',), 'robot'),
    'object_anomaly': ('anomaly', ('object_crops',), 'object'),
}
def stages_for(config):
    """Link2 is independent of the object/link7 handoff; only robot scoring joins it."""
    stages = dict(STAGES)
    if 'link2' in config.get('robot_links', ['link7']):
        stages = {}
        for name, spec in STAGES.items():
            if name == 'robot_score':
                stages['link2_masks'] = ('sam3', (), 'robot')
                spec = ('geometry', ('mask_join', 'link2_masks'), 'robot')
            stages[name] = spec
    return stages


VLM_STAGES = {'vlm1', 'vlm2', 'object_masks', 'mask_join'}
RESOURCE_STAGES = {
    'sam3_checkpoint': {'link2_masks', 'link7_initial', 'link7_masks', 'object_masks', 'mask_join'},
    'sam3_bpe': {'link2_masks', 'link7_initial', 'link7_masks', 'object_masks', 'mask_join'},
    'dino_directory': {'link7_initial', 'link2_masks'}, 'robot_references': {'link2_masks'}, 'palm_references': {'link7_initial'},
    'vlm2_examples': {'vlm1', 'vlm2'}, 'vlm3_reference': {'mask_join'},
    'qwen_model': {'vlm1'}, 'tracker_checkpoint': {'object_tracks', 'robot_score'},
    'robot_config': {'robot_score'}, 'mega_sam': {'robot_score'},
    'anomaly_checkpoint': {'object_anomaly'}, 'dino_repo': {'object_anomaly'},
    'ffmpeg': {'link7_initial', 'link7_masks', 'mask_join', 'robot_score'},
}

def now(): return datetime.now(timezone.utc).isoformat()

def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def read(path): return json.loads(Path(path).read_text())

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def identity(value):
    if isinstance(value, list): return [identity(x) for x in value]
    p = Path(value)
    if p.is_file(): return {'path': str(p), 'sha256': digest(p)}
    # Directory model/vendor trees use file identities; large checkpoint files are
    # hashed too. Mutable reconstruction scratch is never part of model identity.
    excluded = {'.git', '__pycache__', 'work_space', 'outputs', 'outputs_cvd', 'cache_flow', 'reconstructions'}
    records = {}; visited = set()
    for base, dirs, files in os.walk(p, followlinks=True):
        resolved = str(Path(base).resolve())
        if resolved in visited:
            dirs[:] = []; continue
        visited.add(resolved)
        dirs[:] = sorted(d for d in dirs if d not in excluded)
        for name in sorted(files):
            f = Path(base) / name
            if f.is_file() and not name.endswith(('.pyc', '.log')):
                records[str(f.relative_to(p))] = digest(f)
    if not records: raise ValueError(f'Empty or missing resource: {p}')
    return {'path': str(p), 'files': records}

def load_manifest(path):
    path = Path(path).resolve(); config = read(path)
    allowed = {'schema_version', 'run_id', 'output', 'environments', 'resources', 'cases', 'vlm', 'policy', 'secrets_file', 'robot_links'}
    if set(config) - allowed: raise ValueError(f'Unknown manifest fields: {set(config)-allowed}')
    if config.get('schema_version') != 1: raise ValueError('schema_version must be 1')
    def location(v):
        p = Path(v).expanduser()
        return str(p.absolute() if p.is_absolute() else (path.parent / p).absolute())
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config.get('run_id', '')):
        raise ValueError('Invalid run_id')
    config['output'] = location(config['output'])
    config['environments'] = {k: location(v) for k, v in config['environments'].items()}
    if set(config['environments']) != {'sam3', 'vlm1', 'geometry', 'anomaly'}:
        raise ValueError('Provide sam3, vlm1, geometry, anomaly interpreters')
    links = config.setdefault('robot_links', ['link7'])
    if not isinstance(links, list) or links not in (['link7'], ['link2', 'link7'], ['link7', 'link2']):
        raise ValueError('robot_links must contain link7, optionally with link2')
    config['robot_links'] = sorted(links)
    resources = config['resources']
    required = set(RESOURCE_STAGES) - {'robot_references'}
    if 'link2' in links: required.add('robot_references')
    if not required.issubset(resources) or set(resources) - set(RESOURCE_STAGES):
        raise ValueError('Required resource keys: ' + ', '.join(sorted(required)))
    for key, value in resources.items(): resources[key] = [location(x) for x in value] if isinstance(value, list) else location(value)
    if not isinstance(resources['vlm2_examples'], list) or len(resources['vlm2_examples']) != 3:
        raise ValueError('Exactly three vlm2_examples required')
    if config.get('secrets_file'): config['secrets_file'] = location(config['secrets_file'])
    cases = config['cases']; seen = set()
    if not cases: raise ValueError('Empty case list')
    for case in cases:
        if set(case) - {'id', 'video', 'prompt_file', 'vlm1_reference', 'base_segmentation'}: raise ValueError('Unknown case field')
        name = case['id']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in seen:
            raise ValueError('Case IDs must be safe and unique')
        seen.add(name)
        for key in ('video', 'prompt_file', 'vlm1_reference', 'base_segmentation'):
            if case.get(key): case[key] = location(case[key])
        if not case.get('video') or not case.get('prompt_file'): raise ValueError('Each case requires video and prompt_file')
    policy = {'vlm_attempts': 4, 'vlm_stage_timeout_seconds': 3600, 'stage_timeout_seconds': 14400}
    policy.update(config.get('policy', {}))
    if set(policy) != {'vlm_attempts', 'vlm_stage_timeout_seconds', 'stage_timeout_seconds'}: raise ValueError('Unknown policy key')
    if type(policy['vlm_attempts']) is not int or not 1 <= policy['vlm_attempts'] <= 4: raise ValueError('vlm_attempts must be 1..4')
    if any(type(policy[k]) is not int or policy[k] <= 0 for k in ('vlm_stage_timeout_seconds','stage_timeout_seconds')): raise ValueError('Timeouts must be positive seconds')
    config['policy'] = policy
    # Public model configurations contain environment-variable names, never keys.
    vlm = config.setdefault('vlm', {})
    if set(vlm) - {'vlm1', 'vlm1_fallback', 'vlm2', 'vlm2_malformed_fallback', 'object_models'}: raise ValueError('Unknown VLM role')
    fields = {'backend','model','python','api_style','api_base','api_key_env','reasoning_effort','timeout_seconds','max_tokens','max_completion_tokens','temperature'}
    for role, values in vlm.items():
        if role == 'object_models':
            if not isinstance(values, list) or not values or not all(isinstance(x,str) and x for x in values): raise ValueError('object_models must be nonempty model names')
        elif not isinstance(values, dict) or set(values) - fields: raise ValueError(f'Invalid public VLM settings for {role}')
        else:
            if values.get('python'): values['python'] = location(values['python'])
            if values.get('backend', 'local_gpu' if role.startswith('vlm1') else 'cloud_api') == 'local_gpu' and values.get('model'):
                values['model'] = location(values['model'])
    return config

@contextmanager
def locked(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.coordinator.lock').open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another coordinator owns this run') from None
        try: yield
        finally: fcntl.flock(stream, fcntl.LOCK_UN)

def inventory(folder):
    return {str(p.relative_to(folder)): digest(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name not in {'request.json', 'stage.log'} and '__pycache__' not in p.parts}

def valid(record):
    try:
        return bool(record.get('files')) and all(digest(Path(record['directory']) / p) == h for p,h in record['files'].items())
    except (OSError, ValueError): return False

@contextmanager
def termination_as_interrupt():
    if threading.current_thread() is not threading.main_thread():
        yield; return
    def stop(signum, frame): raise KeyboardInterrupt('Coordinator termination requested')
    previous=signal.signal(signal.SIGTERM,stop)
    try: yield
    finally: signal.signal(signal.SIGTERM,previous)


def refuse_live_worker(entry):
    path=Path(entry.get('directory','.'))/'execution.json'
    if entry.get('status') not in {'running','interrupted'} or not path.is_file():return
    receipt=read(path)
    if receipt.get('finished_at'):return
    if receipt['host']!=socket.gethostname():
        raise RuntimeError('Unfinished worker belongs to another host; verify it has stopped before moving this run')
    try:os.kill(receipt['pid'],0)
    except ProcessLookupError:return
    raise RuntimeError(f'Previous worker PID {receipt["pid"]} is still alive; do not resume concurrently')


def execute(request_path, interpreter, timeout):
    folder=request_path.parent
    with (folder/'stage.log').open('w') as log:
        process=subprocess.Popen([interpreter,'-u','-m','infrastructure.deformation_detect.worker',str(request_path)],
            cwd=root(),env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'pid':process.pid,'host':socket.gethostname(),'started_at':now()}
        write(folder/'execution.json',receipt)
        try:return process.wait(timeout=timeout)
        except BaseException:
            try:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            raise
        finally:
            write(folder/'execution.json',{**receipt,'exit_code':process.returncode,'finished_at':now()})


def preflight(config):
    """Validate all assets/credentials and real worker environments before any case."""
    for p in config['environments'].values():
        if not Path(p).is_file() or not os.access(p, os.X_OK): raise ValueError('Missing interpreter: '+p)
    for value in config['resources'].values():
        for p in value if isinstance(value,list) else [value]:
            if not Path(p).exists(): raise ValueError('Missing resource: '+p)
    if len({Path(p).name for p in config['resources']['vlm2_examples']}) != 3:
        raise ValueError('VLM2 example filenames must be distinct')
    if not list(Path(config['resources']['palm_references']).glob('*.png')):
        raise ValueError('No DINO palm reference PNGs')
    if not (root()/'infrastructure/shared/replay/assets/plotly.min.js').is_file():
        raise ValueError('Missing bundled replay plotly.min.js')
    if config.get('secrets_file') and not Path(config['secrets_file']).is_file(): raise ValueError('Missing secrets_file')
    for case in config['cases']:
        for k in ('video','prompt_file','vlm1_reference','base_segmentation'):
            if case.get(k) and not Path(case[k]).is_file(): raise ValueError(f'Missing {case["id"]} {k}')
        if not Path(case['prompt_file']).read_text().strip(): raise ValueError('Empty object prompt')
    import tempfile
    reports = {}
    with tempfile.TemporaryDirectory(prefix='pdi-preflight-') as temporary:
        request = Path(temporary)/'request.json'; write(request, {'config':config,'stage':'preflight'})
        for env, interpreter in config['environments'].items():
            command = [interpreter, '-m', 'infrastructure.deformation_detect.worker', str(request), '--check-environment', env]
            result = subprocess.run(command, cwd=root(), env=environment(), capture_output=True, text=True, timeout=180)
            if result.returncode: raise RuntimeError(f'{env} preflight failed:\n{result.stderr[-2500:]}')
            reports[env] = json.loads(result.stdout.strip().splitlines()[-1])
    return reports


def stage_key(config, case, stage, inputs, source, resources, environments=None):
    _, deps, _ = stages_for(config)[stage]
    selected = {'video': identity(case['video'])}
    for key, needed in [('prompt_file', {'object_masks'}), ('vlm1_reference', {'vlm1'}), ('base_segmentation', {'mask_join'})]:
        if stage in needed and case.get(key): selected[key] = identity(case[key])
    payload = {'schema':1, 'stage':stage, 'case':case['id'], 'source':source,
               'inputs':selected, 'resources':{k:v for k,v in resources.items() if stage in RESOURCE_STAGES[k]},
               'dependencies':{k:inputs[k]['files'] for k in deps},
               'environment':config['environments'][stages_for(config)[stage][0]],
               'environment_identity':(environments or {}).get(stages_for(config)[stage][0], {})}
    if stage == 'robot_score': payload['robot_links'] = config.get('robot_links', ['link7'])
    if stage in VLM_STAGES: payload.update(vlm=config['vlm'], policy=config['policy'])
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def review(output, state):
    rows=[]
    for name, case in state['cases'].items():
        links=[]
        for stage, label, suffix in [('robot_score','Link2 replay','score/replay/interactive_exact-group/link2_exact-group.html'),('robot_score','Link7 replay','score/replay/interactive_exact-group/link7_exact-group.html'),('object_crops','Occlusion replay',f'cases/{name}/occlusion/replay.html'),('object_anomaly','Scored crop pairs','crops/selection_gallery.html')]:
            r=case['stages'].get(stage,{})
            if case['status']=='complete' and r.get('status')=='complete':
                target=Path(r['directory'])/suffix
                if target.is_file():links.append(f'<a href="{html.escape(os.path.relpath(target,output),quote=True)}">{label}</a>')
        rows.append(f'<tr><td>{html.escape(name)}</td><td>{html.escape(case["status"])}</td><td>{" · ".join(links)}</td><td>{html.escape(case.get("reason",""))}</td></tr>')
    (output/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Pipeline run</title><style>body{font:16px system-ui;margin:40px}td,th{padding:12px;text-align:left}a{color:#246349}</style><h1>Robot and object pipeline</h1><p><a href="run.json">Run status and provenance</a></p><table><tr><th>Case</th><th>Status</th><th>Replay</th><th>Reason</th></tr>'+''.join(rows)+'</table></html>')


def coordinate(config, *, retry_disabled=False, executor=execute, check=preflight, resource_identity=identity, source_identity=None):
    output=Path(config['output'])
    stages=stages_for(config)
    with locked(output), termination_as_interrupt():
        environments = check(config) or {}  # Configuration failures stop the run before cases.
        if not (output/'run.json').exists() and any(p.name != '.coordinator.lock' for p in output.iterdir()):
            raise ValueError('New run output must be empty')
        state=read(output/'run.json') if (output/'run.json').exists() else {'schema_version':1,'run_id':config['run_id'],'cases':{}}
        for previous_case in state['cases'].values():
            for entry in previous_case['stages'].values():refuse_live_worker(entry)
        if state['run_id']!=config['run_id']: raise ValueError('Output belongs to a different run_id')
        resources={k:resource_identity(v) for k,v in config['resources'].items()}
        # Scientific source changes conservatively invalidate the run. Model data and docs do not.
        source=source_identity if source_identity is not None else {
            str(p.relative_to(root())):digest(p) for owner in ('robot','object','infrastructure/deformation_detect','infrastructure/shared/inference')
            for p in (root()/owner).rglob('*') if p.is_file() and p.suffix in {'.py','.html','.js'} and not {'results','tests','__pycache__','archive','experiments','experimental'} & set(p.relative_to(root()).parts)}
        write(output/'manifest.json',config)
        write(output/'preflight.json',environments)
        state.update(status='running', updated_at=now());write(output/'run.json',state)
        for case in config['cases']:
            record=state['cases'].setdefault(case['id'], {'status':'pending','stages':{}})
            record.update(status='running');record.pop('reason',None)
            for inactive in set(record['stages']) - set(stages):
                record.setdefault('history',[]).append(record['stages'].pop(inactive))
            for stage, (env, deps, owner) in stages.items():
                key=stage_key(config,case,stage,record['stages'],source,resources,environments)
                old=record['stages'].get(stage,{})
                if old.get('key')==key and old.get('status')=='complete' and valid(old): continue
                if old.get('key')==key and old.get('status')=='disabled' and not retry_disabled:
                    record.update(status='disabled',reason=f'{stage}: retry budget exhausted; use --retry-disabled explicitly');break
                # Remove dependent current records; keep their attempt directories/history.
                invalid={stage}
                for candidate,(_,parents,_) in stages.items():
                    if invalid.intersection(parents): invalid.add(candidate)
                for name in invalid:
                    if name in record['stages']:
                        record.setdefault('history',[]).append(record['stages'].pop(name))
                max_attempts=config['policy']['vlm_attempts'] if stage in VLM_STAGES else 1
                continuing = old.get('key')==key and old.get('status') in {'running','interrupted','failed'}
                start_attempt = old.get('attempt',0)+1 if continuing and stage in VLM_STAGES else 1
                previous = old.get('directory') if continuing else None
                budget_started = old.get('budget_started_at',now()) if continuing else now()
                entry = dict(old)
                for attempt in range(start_attempt,max_attempts+1):
                    timeout=config['policy']['vlm_stage_timeout_seconds' if stage in VLM_STAGES else 'stage_timeout_seconds']
                    if stage in VLM_STAGES:
                        timeout -= (datetime.now(timezone.utc)-datetime.fromisoformat(budget_started)).total_seconds()
                        if timeout <= 0:
                            entry.update(status='failed',error='VLM stage time budget exhausted');break
                    base=output/owner/case['id']/stage;base.mkdir(parents=True,exist_ok=True)
                    serial=max([int(p.name.split('-')[-1]) for p in base.glob('attempt-*') if p.is_dir()]+[0])+1
                    folder=base/f'attempt-{serial:04d}';folder.mkdir()
                    request={'config':config,'case':case,'stage':stage,'attempt':attempt,'directory':str(folder),
                             'dependencies':{d:record['stages'][d]['directory'] for d in deps},'previous_attempt':previous}
                    write(folder/'request.json',request)
                    entry={'stage':stage,'status':'running','key':key,'attempt':attempt,'directory':str(folder),'started_at':now(),'budget_started_at':budget_started}
                    record['stages'][stage]=entry;write(output/'run.json',state)
                    print(f'{case["id"]}: {stage} attempt {attempt}/{max_attempts}',flush=True)
                    try:
                        code=executor(folder/'request.json',config['environments'][env],timeout)
                        result=read(folder/'result.json') if (folder/'result.json').is_file() else {}
                        if code or result.get('status')!='complete': raise RuntimeError(result.get('error',f'worker exit {code}, no complete result'))
                        entry.update(status='complete',exit_code=code,files=inventory(folder),finished_at=now())
                        if not entry['files']: raise RuntimeError('No output artifacts')
                    except (KeyboardInterrupt, SystemExit):
                        entry.update(status='interrupted',finished_at=now());state['status']='interrupted';write(output/'run.json',state);review(output,state);raise
                    except Exception as exc:
                        entry.update(status='failed',error=str(exc),finished_at=now())
                    write(output/'run.json',state)
                    if entry['status']=='complete': break
                    record.setdefault('history',[]).append(dict(entry));previous=str(folder)
                if entry['status']!='complete':
                    entry.update(status='disabled',error=entry.get('error','VLM attempt budget exhausted'));record['stages'][stage]=entry;record.update(status='disabled',reason=f'{stage}: {entry["error"]}')
                    for next_stage in stages:
                        if next_stage not in record['stages']:record['stages'][next_stage]={'status':'skipped','reason':'case disabled'}
                    break
            else: record['status']='complete'
            write(output/'run.json',state);review(output,state)
        state.update(status='complete' if all(state['cases'][c['id']]['status']=='complete' for c in config['cases']) else 'completed_with_disabled_cases',updated_at=now())
        write(output/'run.json',state);review(output,state)
        return state
