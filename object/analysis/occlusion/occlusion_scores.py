"""Preserve naive rigidity and add a separately versioned occlusion-filtered mean."""
from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import csv
import io
import json
import math
import re
from pathlib import Path
from object.preprocessing.occlusion.occlusion import atomic, sha256

START='<!-- OCCLUSION_AUDIT_V1_START -->'
END='<!-- OCCLUSION_AUDIT_V1_END -->'


def filtered_score(score, detection):
    history=score['rigidity_history']; rows=detection['frames']
    if len(history)!=len(rows) or any(r['frame']!=i for i,r in enumerate(rows)):
        raise ValueError('Detection frame indices do not match score')
    if len(history)<2 or not all(math.isfinite(x) for x in history):
        raise ValueError('Invalid rigidity history')
    naive=sum(history[1:])/(len(history)-1)
    if not math.isclose(naive,score['rigidity_score'],abs_tol=1e-9,rel_tol=1e-7):
        raise ValueError('Original score does not follow mean-after-reference contract')
    failed=[i for i in range(1,len(history)) if rows[i].get('mask_valid', True) is False]
    excluded=[i for i in range(1,len(history)) if rows[i]['flagged'] or i in failed]
    retained=[i for i in range(1,len(history)) if not rows[i]['flagged'] and i not in failed]
    return dict(schema_version=2,method='task-object-v1-occlusion-filtered-mean-v2',
        detection_method=detection['method'], detection_config=detection['config'],
        mask_quality_method=detection.get('mask_quality_method'),
        naive_rigidity_score=score['rigidity_score'],
        filtered_rigidity_score=sum(history[i] for i in retained)/len(retained) if retained else None,
        reference_frame=0,frame_index_base=0,retained_frames=retained,excluded_frames=excluded,
        retained_count=len(retained),excluded_count=len(excluded),
        failed_mask_frames=failed,failed_mask_count=len(failed),
        status='complete' if retained else 'no_retained_frames',
        filtered_rigidity_history=[None if i==0 or i in excluded else value for i,value in enumerate(history)],
        retained_unassessable_frames=[i for i in retained if rows[i]['status']!='assessed'],
        retained_carried_frames=[i for i in retained if rows[i].get('rigidity_carried',False)],
        policy='Mean of original rigidity_history[1:] excluding occlusion-flagged frames and failed link7-mask frames. Frame 0 is the reference. Other retained carried and unassessable frames keep their original values. Excluded values are null, never zero. Pair selection, baselines, tracks and original frame scores are unchanged.')


def patch_replay(page, detection, result):
    content=page.read_text()
    content=re.sub(re.escape(START)+r'.*?'+re.escape(END),'',content,flags=re.S)
    payload={k:detection[k] for k in ('flagged_intervals','frames')}
    payload['score']=result
    addon=(_workspace_root() / 'object/replay/occlusion/occlusion_indicator.js').read_text()
    block=START+'\n<script>\n'+addon.replace('__OCCLUSION_DATA__',json.dumps(payload,separators=(',',':')).replace('</','<\\/'))+'\n</script>\n'+END
    if '</body>' not in content:raise ValueError('Missing replay body')
    atomic(page,content.replace('</body>',block+'\n</body>'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();summary=[]
    for case in sorted((args.root/'cases').iterdir()):
        if not case.is_dir():continue
        source=case/'score/rigidity.json';det=case/'occlusion/detection.json';page=case/'replay/task_object_exact-group.html'
        original_hash=sha256(source)
        score=json.loads(source.read_text());detection=json.loads(det.read_text())
        result=filtered_score(score,detection)
        result['inputs']={'naive_score_sha256':original_hash,'detection_sha256':sha256(det)}
        atomic(case/'score/rigidity_occlusion_filtered.json',json.dumps(result,indent=2,allow_nan=False)+'\n')
        patch_replay(page,detection,result)
        assert sha256(source)==original_hash
        summary.append(dict(case=case.name,naive=result['naive_rigidity_score'],filtered=result['filtered_rigidity_score'],excluded=result['excluded_count'],retained=result['retained_count'],failed_mask=result['failed_mask_count'],unassessable_retained=len(result['retained_unassessable_frames'])))
    buf=io.StringIO();writer=csv.DictWriter(buf,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    atomic(args.root/'occlusion/rigidity_comparison.csv',buf.getvalue())
    atomic(args.root/'occlusion/rigidity_comparison.json',json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'updated_replays':len(summary),'examples':[r for r in summary if r['case'] in ['COSMOS3_0025','COSMOS3_0056','COSMOS2.5_0065']]}))


if __name__=='__main__':main()
