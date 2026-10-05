"""Inspect a recorded environment profile without installing packages or contacting a GPU host."""
import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys


def merged(profiles, name):
    spec = dict(profiles[name])
    if 'extends' not in spec:return spec
    parent = merged(profiles, spec['extends'])
    return {**parent, **spec, 'packages':{**parent['packages'], **spec['packages']},
            'modules':list(dict.fromkeys(parent['modules']+spec['modules']))}


def inspect(spec, cuda=False):
    issues, packages = [], {}
    for name, constraint in spec['packages'].items():
        try:
            distribution = name
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                for alternative in spec.get('alternatives', {}).get(name, []):
                    try:
                        installed = importlib.metadata.version(alternative)
                        distribution = alternative
                        break
                    except importlib.metadata.PackageNotFoundError:
                        pass
                else:
                    raise
            packages[name] = {'distribution':distribution,'installed':installed,'required':constraint or 'unversioned'}
            if constraint:
                try:
                    from packaging.specifiers import SpecifierSet
                    if installed not in SpecifierSet(constraint):issues.append(f'{name}: {installed} does not satisfy {constraint}')
                except ImportError:
                    issues.append(f'{name}: packaging is unavailable; cannot validate constraint {constraint}')
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {'installed':None,'required':constraint or 'unversioned'}
            issues.append(f'{name}: missing')
    expected_python = spec.get('python')
    if expected_python and '.'.join(map(str,sys.version_info[:2])) != expected_python:
        issues.append(f'Python: expected {expected_python}')
    modules = {}
    for name in spec.get('modules',[]):
        try:modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):modules[name] = False
        if not modules[name]:issues.append(f'{name}: module unavailable')
    gpu = {'checked':False}
    if cuda:
        try:
            import torch
            gpu = {'checked':True, 'available':torch.cuda.is_available(), 'runtime':torch.version.cuda}
            if not gpu['available']:issues.append('CUDA is unavailable')
            if spec.get('cuda') and gpu['runtime'] != spec['cuda']:issues.append(f"CUDA runtime: expected {spec['cuda']}")
            if gpu['available']:
                gpu['device'] = torch.cuda.get_device_name(0)
                gpu['tensor_smoke_sum'] = float(torch.ones(2, device='cuda').sum().item())
        except (ImportError, RuntimeError) as exc:issues.append(f'CUDA check failed: {exc}')
    return {'status':'passed' if not issues else 'mismatch','python':sys.version,'interpreter':sys.executable,
            'packages':packages,'modules':modules,'gpu':gpu,'issues':issues,'notes':spec.get('notes',[])}


def main():
    profiles = json.loads(Path(__file__).with_name('environments.json').read_text())['profiles']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=profiles, required=True)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    result = inspect(merged(profiles,args.profile),args.cuda)
    print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
