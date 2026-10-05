"""Publish selected arm/wrist runs without exporting internal mask ablations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--dest',type=Path,default=Path('qwen_results'))
    a=p.parse_args(); records=[]
    for source in sorted(a.work.iterdir()):
        if not (source/'grounding.json').exists(): continue
        record=json.loads((source/'grounding.json').read_text())
        record['validation']={}
        for mode in ['positive','arm','both']:
            path=source/f'{mode}_metrics.json'
            if path.exists():record['validation'][mode]=json.loads(path.read_text())
        dest=a.dest/source.name
        if (source/'points.png').exists():
            dest.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/'points.png',dest/'points.png')
            for path in (source/'inputs').glob('*.png'):
                target=dest/'inputs'/('negative_'+path.name);target.parent.mkdir(exist_ok=True)
                shutil.copy2(path,target)
        if (source/'masking.mp4').exists():
            shutil.copy2(source/'masking.mp4',dest/'masking.mp4')
            record['replay_sha256']=hashlib.sha256((dest/'masking.mp4').read_bytes()).hexdigest()
        records.append(record)
    provenance=a.dest/'provenance';provenance.mkdir(exist_ok=True)
    (provenance/'arm_wrist.json').write_text(json.dumps(records,indent=2)+'\n')
    lines=['# Qwen gripper masking','','Three retained Qwen positives; red points exclude the forearm and upper white wrist housing. Replays show source RGB beside the cyan mask.','','| Case | Points | Actual Qwen inputs | Masking | Run status |','|---|---|---|---|---|']
    lookup={r['case']:r for r in records}
    for dest in sorted(a.dest.iterdir()):
        if not dest.is_dir() or dest.name=='provenance':continue
        case=dest.name;record=lookup.get(case,{})
        points=f'[Points]({case}/points.png)' if (dest/'points.png').exists() else 'Unavailable'
        replay=f'[Replay]({case}/masking.mp4)' if (dest/'masking.mp4').exists() else 'Unavailable'
        status=record.get('review',record.get('status','Existing result'))
        lines.append(f'| {case} | {points} | [Inputs]({case}/inputs/) | {replay} | {status} |')
    lines+=['','Prompts, raw responses, coordinates, input hashes and validation are in `provenance/arm_wrist.json`; retained positive provenance is in `provenance/manifest.json`. Missing or failed seeds are not fabricated.']
    (a.dest/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
