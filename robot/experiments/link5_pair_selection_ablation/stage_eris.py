"""Stage with system Python; GPU-runtime preflight is performed in Slurm."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    source=args.base/'workspace-link5-shape-codebook-four-frame0-20261006'
    workspace=args.base/args.name
    if workspace.exists():raise RuntimeError('Isolated target already exists')
    shutil.copytree(source,workspace,symlinks=True,ignore=shutil.ignore_patterns('results','__pycache__','*.pyc','.git'))
    print('WORKSPACE_COPIED',flush=True)
    archive=args.base/'metadata/link5-pair-selection-source-20261007.tar.gz'
    with tarfile.open(archive) as ar:ar.extractall(workspace)
    cache=source/'results/link5_shape_codebook'
    source_manifest=cache/'mask_cache_manifest.json'
    manifest=json.loads(source_manifest.read_text())
    expected=json.loads((args.base/'metadata/link5-pair-selection-query-hashes.json').read_text())
    excluded={'LVP_ROBOWM_0010','LVP_ROBOWM_0030','COSMOS3_0021','COSMOS3_0035','COSMOS2.5_0010'}
    entries=[]
    for entry in manifest['entries']:
        case=entry['video_id']
        if case in excluded:continue
        generator,number=case.rsplit('_',1)
        video=args.base/'inputs/videos'/('LVP' if generator=='LVP_ROBOWM' else generator)/(number+'.mp4')
        mask=cache/entry['mask']
        initialization=args.base/'pair-init-frame0-20261007/output'/case/'pairs.json'
        if sha(video)!=entry['source_video_sha256'] or sha(mask)!=entry['mask_sha256']:
            raise ValueError('Source or mask mismatch: '+case)
        if sha(initialization)!=expected[case]:raise ValueError('Local/remote query mismatch: '+case)
        query=json.loads(initialization.read_text())
        for key in ('source_video_sha256','mask_sha256'):
            if query[key]!=entry[key]:raise ValueError('Query provenance mismatch: '+case)
        entries.append(dict(entry,cohort='additional' if case=='COSMOS2.5_0018' else 'selected45'))
    output=workspace/'results/link5_pair_selection_ablation_v1'
    (output/'metadata').mkdir(parents=True)
    result=dict(source_manifest=str(source_manifest),source_manifest_sha256=sha(source_manifest),
        excluded={case:'Excluded by current HANDOFF.md' for case in sorted(excluded)},
        primary_count=sum(e['cohort']=='selected45' for e in entries),additional_count=sum(e['cohort']=='additional' for e in entries),
        entries=entries,methods=['balanced_v0','refine_v1'],gpu_count=1,parallel_video_workers=5)
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    (output/'metadata/source_snapshot.json').write_text(json.dumps({str(p.relative_to(workspace)):sha(p) for p in workspace.rglob('*.py')},indent=2))
    (output/'metadata/staging.json').write_text(json.dumps(dict(workspace=str(workspace),source_workspace=str(source),
        sources=str(args.base/'inputs/videos'),cache=str(cache),queries=str(args.base/'pair-init-frame0-20261007/output'),
        primary_count=result['primary_count'],additional_count=result['additional_count'],snapshot_archive_sha256=sha(archive),
        sha_validation='all included source/mask/query files passed'),indent=2))
    print(json.dumps(dict(workspace=str(workspace),videos=len(entries),total_frames=sum(e['frame_count'] for e in entries),max_frames=max(e['frame_count'] for e in entries))),flush=True)


if __name__=='__main__':main()
