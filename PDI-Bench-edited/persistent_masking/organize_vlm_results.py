"""Extract only executed VLM image inputs and point-placement previews.

Does not delete source directories. Delete only after inspecting its manifest.
"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'qwen_results'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def extract(source,dest,records):
    manifest_path=source/'manifest.json'
    if not manifest_path.exists():return
    manifest=json.loads(manifest_path.read_text())
    diagnoses=json.loads((source/'diagnoses.json').read_text()) if (source/'diagnoses.json').exists() else []
    correction=json.loads((source/'correction.json').read_text()) if (source/'correction.json').exists() else {}
    frames={int(d['frame']) for d in diagnoses if d.get('raw_response')}
    if 'frame' in correction:frames.add(int(correction['frame']))
    files=[('reference.png','inputs/reference.png')]
    if (source/'full_candidate.png').exists():files.append(('full_candidate.png','inputs/full_candidate.png'))
    else:
        for c in manifest.get('calls',[]):
            if int(c['frame']) in frames:files.append((c['crop'],'inputs/'+Path(c['crop']).name))
    preview=next((n for n in ['grounding_points.png','point_preview.png'] if (source/n).exists()),None)
    if preview:files.append((preview,'points.png'))
    copied=[];missing=[]
    for old,new in dict(files).items():
        p=source/old
        if not p.exists() and (source.parent/old).exists():p=source.parent/old
        if not p.exists():missing.append(old);continue
        target=dest/new;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
        digest=sha(p)
        if sha(target)!=digest:raise ValueError(f'Copy verification failed: {target}')
        copied.append(dict(file=str(target.relative_to(DEST)),source=str(p.relative_to(ROOT)),sha256=digest,bytes=target.stat().st_size))
    raw={}
    for name in ['grounding_calls.json','finger_point.json','arm_point.json','correction.json']:
        if (source/name).exists():raw[name]=json.loads((source/name).read_text())
    if copied:
        records.append(dict(case=str(dest.relative_to(DEST)),files=copied,missing_previously_deleted_inputs=missing,
            crop_boxes=manifest.get('calls',[]),reference_box=manifest.get('reference_box'),
            raw_point_provenance=raw,classification_responses=diagnoses))


def main():
    if DEST.exists():raise FileExistsError('qwen_results already exists; inspect before replacing')
    DEST.mkdir();records=[]
    for parent in ['results_palm_cosmos','results_palm_positive','results_palm_generalization']:
        root=ROOT/parent
        if not root.exists():continue
        for case in sorted(root.iterdir()):
            if not case.is_dir() or case.name=='source_videos':continue
            source=case/'recovery'
            extract(source,DEST/case.name,records)
            # Only Qwen-generated point variants, never the analyst-point trial.
            for name in ['arm_negative']:
                if (source/name).exists():
                    trial=source/name
                    extract(trial,DEST/case.name/name,records)
    root=ROOT/'results_palm_recovery/black_edge_trial'
    if root.exists():
        extract(root,DEST/'LVP_0049',records)
        for name in ['finger_negative_trial','lower_boundary_negative_trial']:
            trial=root/name
            if not trial.exists():continue
            dest=DEST/'LVP_0049'/name
            extract(trial,dest,records)
    provenance=DEST/'provenance';provenance.mkdir()
    (provenance/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# Qwen inputs and point placements','',
           'Green = positive; red = negative. These are actual Qwen outputs, including inaccurate placements. No analyst-placed points are presented as Qwen output.','',
           '| Case / call | Point placement | Actual input images |','|---|---|---|']
    for r in records:
        path=r['case'];preview=DEST/path/'points.png'
        lines.append(f'| {path} | '+(f'[Points]({path}/points.png)' if preview.exists() else 'No point output')+f' | [Inputs]({path}/inputs/) |')
    lines+=['','Only input images and point-placement previews are presented. Exact prompts, raw answers, coordinate mappings and hashes are consolidated in `provenance/manifest.json`. Previously deleted inputs are marked missing; they are not reconstructed or invented.']
    (DEST/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cases=len({r['case'].split('/')[0] for r in records}),point_call_views=len(records),
        image_files=len(list(DEST.rglob('*.png'))),bytes=sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file()))))


if __name__=='__main__':main()
