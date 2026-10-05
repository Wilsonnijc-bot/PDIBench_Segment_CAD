"""Repository locations and compatibility-aware source staging."""
from pathlib import Path
import json
import os


def root():
    for parent in Path(__file__).resolve().parents:
        if (parent / 'documentation/architecture/layout.json').is_file():
            return parent
    raise RuntimeError('Use deformation_detect from the source checkout or an editable install')


def manifest():
    return json.loads((root() / 'documentation/architecture/layout.json').read_text())


def environment():
    repo = root()
    env = os.environ.copy()
    inherited = [p for p in env.get('PYTHONPATH', '').split(os.pathsep)
                 if p and '/compat' not in p and '/archive/' not in p]
    env['PYTHONPATH'] = os.pathsep.join([str(repo), *inherited])
    return env


def interfaces():
    return json.loads((root() / 'documentation/architecture/interfaces.json').read_text())
