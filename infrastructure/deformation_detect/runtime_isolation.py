"""Reject archived/foreign workspace imports and retain actual runtime origins."""
import atexit
import json
import os
from pathlib import Path
import sys

from infrastructure.deformation_detect.layout import root


def validate_origin(name, origin):
    if not origin or origin in {'built-in', 'frozen'}:
        return
    path = Path(origin).resolve()
    parts = path.parts
    if any(parts[i:i+2] == ('infrastructure', 'archive') for i in range(len(parts)-1)) or 'PDIbenchpipelinev2' in parts:
        raise ImportError(f'Legacy source import rejected: {name} from {path}')
    if name.split('.')[0] in {'robot', 'object', 'infrastructure', 'deformation_detect'} and not path.is_relative_to(root().resolve()):
        raise ImportError(f'Foreign workspace import rejected: {name} from {path}')


class FreshSourceFinder:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('infrastructure.archive', 'infrastructure.compat')):
            raise ImportError(f'Legacy namespace import rejected: {fullname}')
        for finder in tuple(sys.meta_path):
            if finder is self:
                continue
            method = getattr(finder, 'find_spec', None)
            spec = method(fullname, path, target) if method else None
            if spec is not None:
                validate_origin(fullname, spec.origin)
                for location in spec.submodule_search_locations or ():
                    validate_origin(fullname, location)
                return spec
        return None


def record_origins():
    directory = os.environ.get('PDI_IMPORT_ORIGINS_DIR')
    if not directory:
        return
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = getattr(module, '__file__', None)
        if origin:
            validate_origin(name, origin)
            origins[name] = str(Path(origin).resolve())
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/f'{os.getpid()}.json').write_text(json.dumps({'status': 'passed', 'pid': os.getpid(),
        'argv': sys.argv, 'origins': origins}, indent=2)+'\n')


def install():
    if not any(isinstance(finder, FreshSourceFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, FreshSourceFinder())
        atexit.register(record_origins)
