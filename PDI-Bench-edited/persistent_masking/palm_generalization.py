"""Small, frozen four-case masking experiment using the existing palm stages."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results_palm_generalization'
IDS=['LVP_0011','LVP_0024','LVP_0016']


def run(env,module,*args,log):
    command=[str(ROOT/env/'bin/python'),'-m',module,*map(str,args)]
    with log.open('w') as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True,
                       env={**os.environ,'OMP_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})


def source(case):
    return next(v for v in json.loads((ROOT/'baseline/selection.json').read_text())['videos'] if v['id']==case)


def mask_stage(case,stage):
    from reg2invbaselinecode import v1_mask
    v1_mask.OUT=OUT/case;v1_mask.OUT.mkdir(parents=True,exist_ok=True)
    v1_mask.VIDEO=Path(source(case)['video'])
    v1_mask.ORIGINAL_CASE=ROOT/'results'/case
    v1_mask.PALM_REFERENCES=ROOT/'results_v1/references/by_link/palm'
    getattr(v1_mask,stage)()


def process():
    OUT.mkdir(exist_ok=True)
    (OUT/'selection.json').write_text(json.dumps(dict(cases=[source(c) for c in IDS],
        policy='Two previously selected gripper-positive cases and one normal control; no score-based selection'),indent=2))
    summary=[]
    for case in IDS:
        dest=OUT/case;dest.mkdir(exist_ok=True)
        entry=dict(case=case,status='running')
        try:
            for stage,name in [('initialize','sam_only_masks.npz'),('track','tracks.npz'),('guided','guided_masks.npz')]:
                if not (dest/name).exists():run('env-sam','persistent_masking.palm_generalization',stage,'--case',case,log=dest/f'{stage}.log')
            recovery=dest/'recovery';entry['recovery']=str(recovery)
            run('env-qwen','persistent_masking.palm_recovery','prepare','--video',source(case)['video'],'--masks',dest/'guided_masks.npz','--output',recovery,log=dest/'prepare.log')
            manifest=json.loads((recovery/'manifest.json').read_text());entry['candidate_frames']=[c['frame'] for c in manifest['calls']]
            if not manifest['calls']:
                entry['status']='no_area_event_not_reviewed'
            else:
                run('env-qwen','persistent_masking.palm_recovery','review','--model',ROOT/'models/Qwen3.5-9B','--output',recovery,'--classify-only',log=dest/'review.log')
                diagnoses=json.loads((recovery/'diagnoses.json').read_text())
                entry['diagnoses']={s:sum(d['state']==s for d in diagnoses) for s in ['normal','deformed','unclear']}
                positives=[d for d in diagnoses if d['state']=='deformed' and not d['parse_error']]
                if not positives:entry['status']='no_confirmed_deformation'
                else:
                    run('env-qwen','persistent_masking.palm_grounding','--output',recovery,'--model',ROOT/'models/Qwen3.5-9B','--black-edges','--three-positive-only',log=dest/'grounding.log')
                    proposal=json.loads((recovery/'correction.json').read_text());entry['correction_frame']=proposal['frame']
                    if proposal['status']!='ready_for_sam':entry['status']='positive_localization_failed'
                    else:
                        run('env-qwen','persistent_masking.palm_finger_negative','--base',recovery,'--model',ROOT/'models/Qwen3.5-9B',log=dest/'negative.log')
                        trial=recovery/'lower_boundary_negative_trial'
                        neg=json.loads((trial/'correction.json').read_text())
                        if neg['status']!='ready_for_sam':entry['status']='negative_localization_failed'
                        else:
                            run('env-sam','persistent_masking.palm_seed_ablation','--output',trial,log=dest/'seed.log')
                            seed=json.loads((trial/'seed_ablation.json').read_text());entry['seed_ablation']=seed
                            if not seed['with_negatives']['supplied_prompt_membership_ok']:entry['status']='sam_prompt_membership_failed'
                            else:
                                run('env-sam','persistent_masking.palm_recovery','repair','--output',trial,log=dest/'repair.log')
                                entry['status']='mask_proposal_completed'
        except Exception as e:entry.update(status='failed',error=str(e))
        summary.append(entry);(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(entry,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['all','initialize','track','guided']);p.add_argument('--case',choices=IDS)
    a=p.parse_args();process() if a.stage=='all' else mask_stage(a.case,a.stage)
