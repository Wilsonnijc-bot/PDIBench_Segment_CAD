"""Inspect module origins and execute small native kernels inside a GPU allocation."""
import argparse
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--profile', choices=['sam3', 'vlm1', 'geometry', 'anomaly'], required=True)
parser.add_argument('--deployment-root', type=Path, required=True)
args = parser.parse_args()

import torch
assert torch.cuda.is_available()
assert torch.cuda.get_device_capability() == (9, 0), 'Expected H200/Hopper allocation'
assert 'H200' in torch.cuda.get_device_name()
assert (torch.ones(2, device='cuda') + 1).sum().item() == 4
devices = []
for index in range(torch.cuda.device_count()):
    assert torch.cuda.get_device_capability(index) == (9, 0)
    assert 'H200' in torch.cuda.get_device_name(index)
    assert (torch.ones(2, device=f'cuda:{index}') + 1).sum().item() == 4
    devices.append({'logical_index': index, 'name': torch.cuda.get_device_name(index)})
modules = {
    'sam3': ['sam3', 'robot.preprocessing.link7_persistent.pipeline', 'object.preprocessing.segmentation.segment'],
    'vlm1': ['transformers', 'robot.preprocessing.link7_persistent.pipeline'],
    'geometry': ['cotracker', 'droid_backends', 'lietorch', 'robot.workflows.score_v1'],
    'anomaly': ['faiss', 'object.scoring.anomalydino'],
}[args.profile]
origins = {}
for name in modules:
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    if not origin.is_relative_to(args.deployment_root.resolve()):
        raise RuntimeError(f'{name} originates outside the fresh deployment: {origin}')
    if {'archive', 'compat'} & set(origin.parts):
        raise RuntimeError(f'{name} originates in an archived package: {origin}')
    origins[name] = str(origin)

kernels = ['torch_tensor']
tools = {}
if args.profile == 'geometry':
    git = shutil.which('git')
    assert git, 'Git is required by scorer provenance'
    tools['git'] = {'path': git, 'version': subprocess.check_output([git, '--version'], text=True).strip()}
    import droid_backends
    import lietorch
    from torch_scatter import scatter_add
    from xformers.ops import memory_efficient_attention
    points = torch.ones((2, 3), device='cuda')
    transformed = lietorch.SE3.Identity(2, device='cuda') * points
    torch.testing.assert_close(transformed, points)
    result = scatter_add(torch.tensor([1., 2.], device='cuda'), torch.tensor([0, 0], device='cuda'))
    assert result.item() == 3
    q = torch.ones((1, 8, 1, 32), device='cuda', dtype=torch.float16)
    assert torch.isfinite(memory_efficient_attention(q, q, q)).all()
    volume = torch.ones((1, 2, 2, 2, 2), device='cuda')
    coords = torch.zeros((1, 2, 2, 2), device='cuda')
    corr, = droid_backends.corr_index_forward(volume, coords, 1)
    assert corr.numel() and torch.isfinite(corr).all()
    torch.cuda.synchronize()
    kernels += ['lietorch_SE3', 'torch_scatter', 'xformers_attention', 'droid_correlation']
print(json.dumps({'profile': args.profile, 'python': sys.version, 'torch': torch.__version__,
    'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(), 'devices': devices, 'origins': origins, 'kernels': kernels, 'tools': tools}))
