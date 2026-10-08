"""Run registered pipelines, inspect environments, analyze scores, and stage source."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from infrastructure.deformation_detect.layout import environment, interfaces, root


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def stage(output):
    """Stage canonical source owners without compatibility aliases or archived code."""
    repo = root()
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Stage destination must be new: {output}')
    # Walk only source owners, without following directory links to bulk assets.
    files = set()
    excluded = {'results', 'results_v1', 'references', 'vendor', '__pycache__',
                '.git', '.tmp', '.pytest_cache', 'previous_archive', 'pastresults',
                'robot_link2_diagnostic', 'dino_refer_robot_link_first15', 'archive', 'compat', 'website'}
    for folder in ('robot', 'object', 'infrastructure', 'documentation'):
        for base, dirs, names in os.walk(repo/folder):
            dirs[:] = [d for d in dirs if d not in excluded and not d.endswith('.egg-info')]
            for name in names:
                if name.startswith('.env') or name.endswith(('.env', '.pyc', '.log')) or name == '.DS_Store':continue
                files.add(str((Path(base)/name).relative_to(repo)))
    for name in ('deformation_detect.py', 'pyproject.toml', 'README.md', '.gitignore', '.gitmodules'):
        if (repo/name).is_file():files.add(name)
    output.mkdir(parents=True)
    records = {}
    for name in sorted(files):
        path = repo/name
        if not path.is_file():
            continue
        if '/results/' in name or '__pycache__' in name:continue
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target, follow_symlinks=True)
        records[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(output/'source-stage.json', {'schema_version':1, 'files':records,
          'external_requirements':'Initialize the pinned third-party repositories and provide reference assets/checkpoints separately.'})
    print(json.dumps({'output':str(output), 'files':len(records)}))


def main(argv=None):
    p = argparse.ArgumentParser(prog='deformation_detect', description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    coordinator = commands.add_parser('coordinate', help='Run or resume the autonomous link7/object pipeline')
    coordinator.add_argument('--manifest', type=Path, required=True)
    coordinator.add_argument('--plan', action='store_true', help='Show stages without importing GPU libraries or running inference')
    coordinator.add_argument('--preflight-only', action='store_true')
    coordinator.add_argument('--retry-disabled', action='store_true', help='Explicitly grant a new retry budget to disabled stages')
    coordinator.add_argument('--workers', type=int, help='Concurrent per-video coordinators; defaults to manifest execution.workers')
    coordinator.add_argument('--gpu-slots', type=int, help='Maximum concurrent GPU stages across video coordinators')
    coordinator.add_argument('--gpu-assignment', choices=['shared', 'per_video'], help='Share visible GPUs or pin each video to one allocated GPU')
    status = commands.add_parser('run-status', help='Read a coordinator run without launching any jobs')
    status.add_argument('--output', type=Path, required=True)
    commands.add_parser('list', help='List supported main and experimental interfaces')
    run = commands.add_parser('run', help='Forward arguments to an existing scientific entry point')
    run.add_argument('--python', default=sys.executable, help='Interpreter for the required model environment')
    run.add_argument('--record', type=Path, help='New execution-record directory; never overwrites a previous record')
    run.add_argument('interface', choices=tuple(interfaces()))
    run.add_argument('arguments', nargs=argparse.REMAINDER)
    check = commands.add_parser('env-check', help='Inspect a GPU/analysis environment without installing anything')
    check.add_argument('--python', default=sys.executable)
    check.add_argument('--profile', required=True)
    check.add_argument('--cuda', action='store_true')
    analysis = commands.add_parser('analyze', help='Binary AUROC from explicitly joined score/label CSVs')
    analysis.add_argument('--scores', required=True, type=Path)
    analysis.add_argument('--labels', required=True, type=Path)
    analysis.add_argument('--key', default='case')
    analysis.add_argument('--score', default='score')
    analysis.add_argument('--label', default='label')
    analysis.add_argument('--positive', type=float, default=1)
    analysis.add_argument('--negative', type=float, default=0)
    analysis.add_argument('--lower-is-anomalous', action='store_true')
    analysis.add_argument('--output', type=Path)
    staging = commands.add_parser('stage', help='Create a portable snapshot of canonical source owners')
    staging.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == 'run-status':
        print((args.output / 'run.json').read_text())
        return 0
    if args.command == 'coordinate':
        from infrastructure.deformation_detect.coordinator import stages_for, coordinate, load_manifest, preflight
        config = load_manifest(args.manifest)
        for key, value in [('workers', args.workers), ('gpu_slots', args.gpu_slots)]:
            if value is not None:
                if value < 1: p.error(f'{key} must be positive')
                config['execution'][key] = value
        if args.gpu_assignment:
            config['execution']['gpu_assignment'] = args.gpu_assignment
        if args.plan:
            print(json.dumps({'cases': [c['id'] for c in config['cases']], 'output': config['output'],
                'stages': {name: {'environment': env, 'depends_on': deps, 'owner': owner}
                           for name, (env, deps, owner) in stages_for(config).items()}, 'policy': config['policy'], 'execution': config['execution']}, indent=2))
            return 0
        if args.preflight_only:
            preflight(config)
            print('Coordinator preflight passed')
            return 0
        result = coordinate(config, retry_disabled=args.retry_disabled)
        print(json.dumps({'status': result['status'], 'review': str(Path(config['output']) / 'index.html')}))
        return 0 if result['status'] == 'complete' else 2
    if args.command == 'list':
        for name, entry in interfaces().items():
            print(f"{name:24} {entry['category']:10} {entry['environment']:12} {entry['description']}")
        return 0
    if args.command == 'stage':stage(args.output);return 0
    if args.command == 'analyze':
        from infrastructure.deformation_detect.analysis import analyze
        result = analyze(args.scores, args.labels, key=args.key, score=args.score, label=args.label,
                         positive=args.positive, negative=args.negative, lower_is_anomalous=args.lower_is_anomalous)
        result['inputs'] = {role:{'path':str(path.absolute()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                            for role,path in [('scores',args.scores),('labels',args.labels)]}
        if args.output:write(args.output,result)
        print(json.dumps(result,indent=2,allow_nan=False));return 0
    if args.command == 'env-check':
        command = [args.python, str(root()/'documentation/gpu/check_environment.py'), '--profile', args.profile]
        if args.cuda:command.append('--cuda')
        return subprocess.call(command, env=environment())
    entry = interfaces()[args.interface]
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    command = [args.python, '-m', entry['module'], *forwarded]
    record = {'interface':args.interface, 'environment':entry['environment'], 'command':command,
              'cwd':str(Path.cwd()), 'started_at':datetime.now(timezone.utc).isoformat(), 'status':'running',
              'layout_sha256':hashlib.sha256((root()/'documentation/architecture/layout.json').read_bytes()).hexdigest()}
    if args.record:
        args.record.mkdir(parents=True,exist_ok=False)
        write(args.record/'execution.json',record)
    try:
        code = subprocess.call(command, env=environment())
    except BaseException as exc:
        if args.record:write(args.record/'execution.json',{**record,'status':'interrupted','error':type(exc).__name__})
        raise
    if args.record:
        write(args.record/'execution.json',{**record,'status':'complete' if code==0 else 'failed','exit_code':code,
              'finished_at':datetime.now(timezone.utc).isoformat()})
    return code


if __name__ == '__main__':
    raise SystemExit(main())
