"""One result root per Simple3D run, with legacy paths retained as provenance."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

from pathlib import Path

ROOT = (_workspace_root())
FOLDERS = {"object": "object", "cad": "link5_cad", "first_frame": "link5_first_frame",
           "inputs": "inputs", "metadata": "metadata", "correlation": "correlation", "replay": "replay"}
LEGACY_PREFIXES = {"object": "simple3d-object-deformation", "cad": "simple3d-link5-cad-reference",
                   "first_frame": "simple3d-link5-firstframe-reference", "inputs": "simple3d-inputs",
                   "metadata": "simple3d-run"}


def run_root(run_id, output_root=None):
    return Path(output_root or (_workspace_root() / 'results')) / f"simple3d-{run_id}"


def result_path(run_id, role, output_root=None, *, existing=False):
    parent = Path(output_root or (_workspace_root() / 'results'))
    canonical = run_root(run_id, parent) / FOLDERS[role]
    if existing and not canonical.exists() and role in LEGACY_PREFIXES:
        legacy = parent / f"{LEGACY_PREFIXES[role]}-{run_id}"
        if legacy.exists():
            return legacy
    return canonical


def resolve_recorded_path(path, run_id, output_root=None):
    """Resolve historical GPU/Mac result paths without changing original records."""
    original = Path(path)
    if original.exists():
        return original
    parent = Path(output_root or (_workspace_root() / 'results'))
    text = str(path)
    if "/results/" in text:
        relative = text.split("/results/", 1)[1]
    elif text.startswith("results/"):
        relative = text[len("results/"):]
    else:
        return original
    for role, prefix in LEGACY_PREFIXES.items():
        name = f"{prefix}-{run_id}"
        if relative == name or relative.startswith(name + "/"):
            tail = relative[len(name):].lstrip("/")
            return result_path(run_id, role, parent, existing=True) / tail
    return parent / relative
