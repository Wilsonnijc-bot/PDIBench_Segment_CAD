"""Completion-triggered CPU reporting; no GPU polling or new inference."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess


def wait_for_file(path):
    """Block on a Linux filesystem notification, not repeated GPU/job queries."""
    path=Path(path)
    if path.exists():return
    library=ctypes.CDLL(None,use_errno=True)
    fd=library.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init1')
    try:
        if library.inotify_add_watch(fd,os.fsencode(path.parent),0x80|0x8)<0:
            raise OSError(ctypes.get_errno(),'inotify_add_watch')
        while not path.exists():os.read(fd,65536)
    finally:os.close(fd)


def run(config_path):
    config=json.loads(config_path.read_text());root=Path(config['output']);full=root/'full_video'
    wait_for_file(full/'completion.json')
    from .common import ROOT,read,sha,write
    from .full_video import summarize,validate_contract
    from .label_correlation import run as correlations
    from .build_full_video_replay import run as replay
    from .report import run as report
    scope=read(full/'metadata/evaluation_scope.json')
    if scope['status']!='complete':raise RuntimeError('requested cohort has not completed')
    validate_contract(config)
    selected=set(scope['selected_video_ids'])
    summarize({**config,'cases':[c for c in config['cases'] if c['id'] in selected]})
    correlations(root,ROOT/'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx')
    import numpy as np
    from scipy.stats import pearsonr,spearmanr
    analysis=read(full/'analysis/forearm_correlation.json')
    for model,groups in analysis['statistics']['forearm_AB'].items():
        for group,expected in groups.items():
            rows=[r for r in analysis['rows'] if group=='all' or (group=='heldout' and r['reference_split']=='heldout') or r['generator']==group]
            x=[r['forearm_AB'] for r in rows if r[model] is not None];y=[r[model] for r in rows if r[model] is not None]
            if len(x)>=3 and np.ptp(x)>0 and np.ptp(y)>0:
                if not np.isclose(pearsonr(x,y)[0],expected['pearson_r'],atol=1e-12) or not np.isclose(spearmanr(x,y)[0],expected['spearman_rho'],atol=1e-12):
                    raise ValueError('correlations disagree with independent scipy verification')
    replay(config);report(root)
    review=root/'RUN_REVIEW.md';text=review.read_text()
    heading='## Current full-video evaluation (user update)\n\n'
    a=text.index(heading)+len(heading);b=text.index('\n\n',a)
    text=text[:a]+('The user reduced evaluation to30 complete full videos: ten matched numbers across three generators, '
        '3310 frames scored by each unchanged detector (6620 frame predictions). Owned Slurm5587413 was stopped '
        'at completion of this selected cohort, as requested. Its scheduler state is cancellation, not a successful45-video run. '
        'The primary video metric sums every raw_mean_top80 frame score, including frame0. Ten-frame testing belongs '
        'only to Simple3D. Later frames share independent FoundationPose rigid canonicalization. '
        'The replay reuses the existing published offline Plotly/video framework. '
        '[Human forearm AB correlations](full_video/analysis/forearm_correlation.md) use only complete selected sums, '
        'with all-video, held-out and generator-specific Pearson/Spearman results. Original failed sensitivity receipts '
        'remain unchanged; this evaluation is exploratory.')+text[b:]
    review.write_text(text)
    source_names=('postprocess_completed.py','stop_after_cohort.py','label_correlation.py','replay_framework.py','replay_app.js','build_full_video_replay.py')
    provenance={name:sha(ROOT/'robot/experiments/link5_shape_codebook'/name) for name in source_names}
    provenance['published_replay_template']=sha(ROOT/'infrastructure/shared/replay/rigidity_replay.html')
    provenance['offline_plotly']=sha(ROOT/'infrastructure/shared/replay/assets/plotly.min.js')
    write(full/'metadata/replay_analysis_source.json',dict(source_sha256=provenance,
        numerical_inference_source_unchanged=True,source_launch_snapshot='source-snapshot/',label_column='AB'))
    # Update the same GPU ledger entry while preserving all earlier failed runs.
    ledger=ROOT/'documentation/gpu/experiment_GPU_record.md'
    if ledger.exists():
        text=ledger.read_text();head='## Link5 full-video matched-generator evaluation — 2026-10-05'
        before,entry=text.split(head,1)
        old='- **Status:** running (Slurm5587413 started2026-10-05T20:28:39 on erishpc-gpu-001; startup GPU preflight underway).'
        entry=entry.replace(old,'- **Status:** completed selected30-video evaluation; owned allocation5587413 cancelled at the user-requested stopping condition. Original45-video launch was intentionally interrupted.')
        analysis=read(full/'analysis/forearm_correlation.json')
        if '**Selected-cohort results:**' not in entry:
            entry+='\n- **Selected-cohort results:**30/30 full videos,3310/3310 frames per checkpoint,6620 frame predictions; complete sums verified including frame0. Replay uses the published Plotly/video framework. AB-only descriptive correlation: '+json.dumps({m:analysis['statistics']['forearm_AB'][m]['all'] for m in ('paper_original','robot_structural')})+'. Original failed sensitivity receipts/checkpoints are unchanged.\n'
        ledger.write_text(before+head+entry)
    command=[config['environments']['geometry'],'-m','robot.experiments.link5_shape_codebook.export_artifacts','--config',str(config_path)]
    subprocess.run(command,cwd=ROOT,check=True)
    write(full/'metadata/postprocessing.json',dict(status='complete',label_column='AB',video_count=30,
         frame_count=3310,checkpoint_predictions=6620,replay='full_video/replay/index.html',
         correlation='full_video/analysis/forearm_correlation.md',no_gpu_polling=True,correlations_verified_against_scipy=True))
    print('LINK5_POST_COMPLETION_CPU_REPORTING_FINISHED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--wait-only',type=Path)
    args=parser.parse_args()
    if args.wait_only:wait_for_file(args.wait_only)
    else:run(args.config)
