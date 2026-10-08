"""Resume COSMOS2.5_0010 after its bare REJECT; reuse the other nine results."""
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from .common import sha, write
from .run_guard_box_retry import export_case, export_index
from robot.preprocessing.link5_refinement import link5_point_guard as guard


def run(source_run, destination):
    if not os.environ.get('VLM2_API_KEY'): raise ValueError('VLM2_API_KEY is required')
    case = 'COSMOS2.5_0010'
    cache = Path('results/link5_shape_codebook/round_four_frame0')
    entries = {e['video_id']:e for e in json.loads((cache/'mask_cache_manifest.json').read_text())['entries']}
    config = json.loads((source_run/'config.json').read_text())
    settings = dict(next(c['settings'] for c in config['cases'] if c['video_id']==case))
    settings['api_key_env']='VLM2_API_KEY'
    previous = json.loads((source_run/'cases'/case/'link5_guard.json').read_text())
    previous_negative = [a for a in previous['attempts'] if a['role']=='link5_negative_guard']
    remaining = 3-len(previous_negative)
    if remaining < 1: raise ValueError('No remaining retries in the three-call budget')
    shutil.copytree(source_run,destination)
    shutil.copy2(Path(guard.__file__),destination/'code/link5_point_guard_n1_retry.py')
    shutil.copy2(Path(__file__),destination/'code/retest_negative_case.py')
    folder=destination/'cases'/case
    image=Image.open(folder/'frame0.png').convert('RGB')
    prior_folder=(cache/entries[case]['mask']).parent
    cached=json.loads((prior_folder/'link5_guard.json').read_text())
    diagnostic=next(d for d in json.loads((prior_folder/'sam3_prompt_diagnostics.json').read_text()) if d['target']=='link5')
    frozen=np.asarray(cached['original_points_xy'],np.int32).copy()
    frozen[:3]=cached['selected_points_xy'][:3]
    write(destination/'review_config.json',dict(source_run=str(source_run.resolve()),
        required_box_points=['N1'],margin_pixels=5,parsing_failures_retry=True,max_total_attempts=3,
        reused_cases=[c['video_id'] for c in config['cases'] if c['video_id']!=case],
        fresh_negative_case=case,remaining_calls=remaining,positive_guard_calls=0,sam_rerun=False))
    print('NEGATIVE_ONLY_RETRY_START',case,'remaining_calls',remaining,flush=True)
    started=time.monotonic()
    error=None
    try:
        _,fresh=guard.review_link5_points(image,diagnostic['sam3_prompt_box_xyxy'],frozen,
            folder/'followup',required=True,vlm_config=settings,negative_only=True,negative_max_attempts=remaining)
    except Exception as exc:
        error=str(exc)
        fresh=json.loads((folder/'followup/link5_guard.json').read_text())
    assert fresh['selected_points_xy'][:3]==cached['selected_points_xy'][:3]
    assert not fresh['positive_guard_called']
    assert all(a['role']=='link5_negative_guard' for a in fresh['attempts'])
    for call in fresh['attempts']:
        if any(call.get(key)!=previous_negative[0].get(key) for key in ('system_prompt','user_prompt','image_sha256')):
            raise ValueError('Follow-up changed the original retry inputs')
    combined=json.loads(json.dumps(fresh))
    for a in combined['attempts']: a['attempt_number']+=len(previous_negative)
    combined['attempts']=previous_negative+combined['attempts']
    combined['reviews']['negative']['attempt_count']=len(combined['attempts'])
    combined.update(prior_guard_sha256=sha(source_run/'cases'/case/'link5_guard.json'),
        followup_guard_sha256=sha(folder/'followup/link5_guard.json'),
        receipt_note='Original attempt followed by negative-only retries; raw followup receipt preserved.')
    write(folder/'link5_guard.json',combined)
    summary=json.loads((destination/'summary.json').read_text())
    rows={r['video_id']:r for r in summary['cases']};row=rows[case]
    row.update(status='complete' if error is None else 'failed',error=error,
        negative_attempts=len(combined['attempts']),followup_elapsed_seconds=round(time.monotonic()-started,3),
        positive_guard_called_in_followup=False,identical_retry_inputs_verified=True)
    plans=[]
    for c in config['cases']:
        video=c['video_id'];target=destination/'cases'/video;old_folder=(cache/entries[video]['mask']).parent
        old=json.loads((old_folder/'link5_guard.json').read_text())
        raw=json.loads((target/'link5_guard.json').read_text())
        diag=next(d for d in json.loads((old_folder/'sam3_prompt_diagnostics.json').read_text()) if d['target']=='link5')
        rows[video]['retain_previous_negatives']=video=='LVP_ROBOWM_0010'
        export_case(target,raw,old,Image.open(target/'frame0.png').convert('RGB'),diag['sam3_prompt_box_xyxy'],c['cohort'],rows[video])
        write(target/'test_result.json',rows[video])
        points=json.loads((target/'negative_only_points.json').read_text())
        assert points['positive_points_xy']==old['selected_points_xy'][:3]
        if video=='LVP_ROBOWM_0010': assert points['negative_points_xy']==old['selected_points_xy'][3:5]
        if video=='LVP_ROBOWM_0030': assert points['selected_attempt']==2
        plans.append(dict(case=video,cohort=c['cohort']))
    summary.update(status='review_complete',negative_box_policy_required_points=['N1'],
        negative_passed_cases=sum(r['negative_test_status']=='PASS' for r in rows.values()),
        negative_failed_cases=sum(r['negative_test_status']=='FAIL' for r in rows.values()),
        previous_negatives_retained_cases=1,positive_guard_calls_in_followup=0,
        negative_parse_failures_retry=True,source_run=str(source_run.resolve()))
    write(destination/'summary.json',summary)
    export_index(destination,plans,list(rows.values()))
    print('NEGATIVE_ONLY_RETRY_FINISHED',case,rows[case]['negative_test_status'],len(combined['attempts']),flush=True)

