"""Select a tested label subset for a separate SAM propagation experiment."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--negative',choices=['none','finger_tip','held_object','all'],required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','correction.json']:
        shutil.copy2(args.source/name,args.output/name)
    calls=json.loads((args.source/'grounding_calls.json').read_text())
    selected=[c for c in calls if c['status']=='ok' and (c['label']==1 or args.negative=='all' or c['target']==args.negative)]
    correction=json.loads((args.output/'correction.json').read_text())
    correction['proposal']['points_xy']=[c['source_xy'] for c in selected]
    correction['proposal']['labels']=[c['label'] for c in selected]
    correction['selected_negative_policy']=args.negative
    (args.output/'correction.json').write_text(json.dumps(correction,indent=2))


if __name__=='__main__':main()
