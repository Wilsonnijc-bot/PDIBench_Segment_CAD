"""Import the pinned author code; no local implementation of the detector."""
import importlib
import importlib.util
import subprocess
import sys

from .common import ROOT, sha

SHAPE_REVISION = '1b8de121de8e8801dd99b796127026c7dd279a10'
POSE_REVISION = 'a1b694b83e633c2cb6115b9063d940a687759392'


def source_record(directory, revision):
    commit = subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if commit != revision:
        raise ValueError(f'upstream revision mismatch: {directory}: {commit}')
    changed = subprocess.check_output(['git','-C',str(directory),'diff','--name-only','HEAD'],text=True).strip()
    if changed:
        raise ValueError(f'vendored numerical source modified: {changed}')
    return dict(commit=commit, source_sha256={str(p.relative_to(directory)):sha(p) for p in directory.glob('*.py')})


def shape_modules():
    directory = ROOT/'infrastructure/vendor/Shape-Anomaly-Codebook'
    record = source_record(directory, SHAPE_REVISION)
    sys.path.insert(0,str(directory))
    modules = {}
    for name in ('augmentation','network','losses','dataset'):
        module=importlib.import_module(name)
        if directory.resolve() not in __import__('pathlib').Path(module.__file__).resolve().parents:
            raise RuntimeError(f'foreign {name} module shadows the author implementation')
        modules[name]=module
    return modules,record


def build_model(cfg):
    modules,record=shape_modules()
    import torch
    from .backbone import MinkUNetAdapter
    external=MinkUNetAdapter(cfg['backbone'])
    model=modules['network'].HierarchicalAnomalyNet(
        scales=[tuple(s) for s in cfg['patches']['scales']],feature_dim=cfg['model']['feature_dim'],
        attention_heads=cfg['model']['attention_heads'],attention_head_dim=cfg['model']['attention_head_dim'],
        codebook_threshold=cfg['codebook']['threshold'],codebook_hash_grid=cfg['codebook']['hash_grid'],
        codebook_max_entries=cfg['codebook']['max_entries'],external_encoder=external)
    return model,modules,record


def training_module():
    # FCGF also has a root train.py and inserts its source path for its model.
    # Load the exact pinned author entry point rather than a namespace collision.
    directory=ROOT/'infrastructure/vendor/Shape-Anomaly-Codebook'
    source_record(directory,SHAPE_REVISION)
    shape_modules()
    spec=importlib.util.spec_from_file_location('_link5_shape_author_train',directory/'train.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def codebook_state(model):
    # Upstream does not serialize its Python hash-key sets in state_dict.
    return [dict(size=int(book.size.item()), counts=book.counts[:int(book.size.item())].cpu().tolist(),
                 hash_keys=[sorted(keys) for keys in book.hash_keys[:int(book.size.item())]]) for book in model.codebook.books]


def restore_hash_keys(model, records):
    for book,record in zip(model.codebook.books,records):
        for i,keys in enumerate(record['hash_keys']):book.hash_keys[i]=set(keys)
