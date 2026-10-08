"""CPU-only paid API review of frozen Link5 frame0 proposals; no SAM inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import shutil
import time

import cv2
from PIL import Image

from .common import ROOT, read, sha, write
from robot.preprocessing.link5_refinement.link5_point_guard import guard_config, review_link5_points, REVIEW_SPECS, POSITIVE_REFERENCE_FRAME
from robot.preprocessing.link7_persistent.interface.config import role_config
from robot.preprocessing.link7_persistent.interface.secrets import load_env_file


def successful_masking(root, video):
    masking=root/'cases'/video/'masking'
    folders=([masking] if (masking/'result.json').exists() else [])+sorted(masking.glob('attempt-*'))
    complete=[p for p in folders if (p/'result.json').exists() and read(p/'result.json').get('status')=='complete']
    if len(complete)!=1:raise ValueError('ambiguous original successful mask attempt: '+video)
    return complete[0]


def video_source(root, case):
    choices=[root/'full_video/replay/videos'/(case['id']+'.mp4'),Path(case['video']),
        ROOT/'robot/experiments/link7_tracker_filter_comparison/results/link5-link7-four-way-selected45-20260927/cases'/case['id']/'v1_cotracker3/replay/interactive_exact-group/source.mp4']
    for path in choices:
        if path.is_file() and sha(path)==case['video_sha256']:return path
    raise FileNotFoundError('no hash-matching local source video: '+case['id'])


def run(root, destination, cases=None, workers=4, compare_luna=False):
    config=read(root/'config.json');train=read(root/'training/train20_manifest.json')
    ids=set(cases or [r['video_id'] for r in train]);selected=[c for c in config['cases'] if c['id'] in ids]
    if {c['id'] for c in selected}!=ids:raise ValueError('unknown case')
    key_env='VLM2_API_KEY' if os.environ.get('VLM2_API_KEY') else 'AI302_API_KEY'
    if not os.environ.get(key_env):raise ValueError('set AI302_API_KEY or VLM2_API_KEY')
    settings={}
    for name,role in [('gemini','vlm2'),('luna','vlm2_malformed_fallback')]:
        values=role_config(role);values.update(backend='cloud_api',api_base=config['vlm']['vlm2']['api_base'],
            api_key_env=key_env,model='gemini-3.8-flash' if name=='gemini' else 'gpt-6-luna')
        settings[name]=guard_config(values)
    destination.mkdir(parents=True,exist_ok=True)
    write(destination/'metadata/guard_config.json',dict(settings=settings,guard_source_sha256=sha(ROOT/'robot/preprocessing/link5_refinement/link5_point_guard.py'),
        client_source_sha256=sha(ROOT/'robot/preprocessing/link7_persistent/vlm_client.py'),sam_rerun=False,
        positive_reference=str(POSITIVE_REFERENCE_FRAME),positive_reference_sha256=sha(POSITIVE_REFERENCE_FRAME)))
    comparison={'LVP_ROBOWM_0065','COSMOS3_0065','LVP_ROBOWM_0030','LVP_ROBOWM_0035'}
    def one(case):
        video=case['id'];old=successful_masking(root,video);guard=read(old/'link5_guard.json')
        diagnostics=next(d for d in read(old/'sam3_prompt_diagnostics.json') if d['target']=='link5')
        source=video_source(root,case);cap=cv2.VideoCapture(str(source));ok,bgr=cap.read();cap.release()
        if not ok:raise ValueError('cannot decode frame0')
        image=Image.fromarray(bgr[...,::-1]);folder=destination/'cases'/video/'guard';folder.mkdir(parents=True,exist_ok=True)
        rgb=folder/'inputs/frame0.png';rgb.parent.mkdir(exist_ok=True)
        if not rgb.exists():image.save(rgb)
        runs={}
        def call(name):
            target=folder/name;receipt=target/'link5_guard.json'
            if receipt.exists():
                record=read(receipt)
                cfg=settings[name];limit=cfg.get('max_completion_tokens',cfg.get('max_tokens'))
                matching=record.get('positive_reference_sha256')==sha(POSITIVE_REFERENCE_FRAME) and len(record.get('attempts',[]))==2 and all(
                    a.get('requested_model')==cfg['model'] and a.get('reasoning_effort')==cfg['reasoning_effort'] and
                    a.get('output_token_limit')==limit and a.get('timeout_seconds')==cfg['timeout_seconds'] and
                    a.get('system_prompt')==REVIEW_SPECS[a['role'].removeprefix('link5_').removesuffix('_guard')]['system_prompt']
                    for a in record['attempts'])
                if not record.get('fallback_to_default') and matching:
                    return dict(status='complete',directory=str(target),decision=record['decision'],reused=True)
                history=folder/'metadata/attempt_history.json'
                prior=read(history) if history.exists() else []
                signature=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()[:16]
                archive=folder/'history'/(name+'-'+signature)
                if not archive.exists():shutil.copytree(target,archive)
                prior.append(dict(route=name,receipt=record,archive=str(archive),
                    reason='failed review' if record.get('fallback_to_default') else 'requested settings changed'))
                write(history,prior)
            start=time.monotonic()
            try:
                _,record=review_link5_points(image,diagnostics['sam3_prompt_box_xyxy'],guard['original_points_xy'],target,
                    required=True,vlm_config=settings[name])
                result=dict(status='complete',directory=str(target),decision=record['decision'])
            except Exception as error:
                result=dict(status='failed',directory=str(target),error_type=type(error).__name__,error=str(error))
            result['elapsed_seconds']=round(time.monotonic()-start,3)
            print('LINK5_LOCAL_GUARD',video,name,result['status'],result.get('decision'),flush=True)
            return result
        runs['gemini']=call('gemini');chosen='gemini'
        if runs['gemini']['status']!='complete':runs['luna']=call('luna');chosen='luna'
        if compare_luna and video in comparison and 'luna' not in runs:runs['luna']=call('luna')
        result=dict(video_id=video,source_video=str(source),source_video_sha256=sha(source),frame0_sha256=sha(rgb),
            original_guard_sha256=sha(old/'link5_guard.json'),runs=runs,selected_model=chosen,
            status=runs[chosen]['status'],sam_rerun=False,mask_source='original saved SAM mask; new points have not been segmented')
        if result['status']=='complete':shutil.copy2(folder/chosen/'link5_guard_selected.png',folder/'points.png')
        write(folder/'provenance.json',result)
        return result
    records=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,c):c['id'] for c in selected}
        for future in as_completed(futures):
            try:records.append(future.result())
            except Exception as error:records.append(dict(video_id=futures[future],status='failed',error=str(error)))
            write(destination/'metadata/guard_summary.json',dict(status='running' if len(records)<len(selected) else
                'complete' if all(r['status']=='complete' for r in records) else 'complete_with_failures',
                requested_cases=len(selected),completed_cases=sum(r['status']=='complete' for r in records),
                cases=sorted(records,key=lambda r:r['video_id']),sam_rerun=False))
    print('LINK5_LOCAL_GUARD_FINISHED',len(records),sum(r['status']=='complete' for r in records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--destination',type=Path);parser.add_argument('--cases',nargs='+');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--compare-luna',action='store_true');parser.add_argument('--secrets-file',type=Path)
    args=parser.parse_args()
    if args.secrets_file:load_env_file(args.secrets_file)
    run(args.root,args.destination or args.root/'refinement',args.cases,args.workers,args.compare_luna)
