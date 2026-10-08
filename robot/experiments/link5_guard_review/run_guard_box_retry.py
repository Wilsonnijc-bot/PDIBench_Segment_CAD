"""Run the green-box retry guard on ten frozen frame0 inputs, without SAM."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image

from .common import sha, write
from robot.preprocessing.link5_refinement import link5_point_guard as guard_module
from robot.preprocessing.link7_persistent.interface.config import role_config
from robot.preprocessing.link7_persistent.interface.secrets import load_env_file


INCLUDED = ['LVP_ROBOWM_0001', 'COSMOS2.5_0001', 'COSMOS3_0001',
            'LVP_ROBOWM_0005', 'COSMOS2.5_0005']
STYLE = ':root{color-scheme:dark;font:15px/1.6 system-ui;background:#101719;color:#e7f0eb}body{max-width:1320px;padding:28px;margin:auto}h1{font-size:28px}h2{margin-top:32px}h3{font-size:16px;color:#a7f2cc}a{color:#8edfc0}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#182225;border:1px solid #344548;font:14px/1.6 ui-monospace,monospace}table{border-collapse:collapse;width:100%}td,th{padding:9px 14px;border-bottom:1px solid #344548;text-align:left}img{width:100%;height:auto}.images{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}section{border-top:1px solid #344548;margin-top:32px}@media(max-width:760px){.images{grid-template-columns:1fr}body{padding:16px}}'
esc = lambda value: html.escape(str(value))


def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(title) + '</title><style>' + STYLE + '</style><main>' + body + '</main></html>'


def export_case(folder, result, old, image, box, cohort, row):
    negative = [a for a in result['attempts'] if a['role'] == 'link5_negative_guard']
    checks = [guard_module._negative_box_check(a['negative_box_check']['candidate_points_xy'],box,5)
              if 'negative_box_check' in a else None for a in negative]
    passed_index = next((i for i,check in enumerate(checks) if check and check['passed']),None)
    selected_index = passed_index if passed_index is not None else next((i for i in reversed(range(len(checks))) if checks[i]),None)
    candidate = checks[selected_index]['candidate_points_xy'] if selected_index is not None else None
    passed = passed_index is not None
    retained = row.get('retain_previous_negatives',False)
    if retained: candidate = old['selected_points_xy'][3:5]
    # Keep all three cached positives fixed. Raw API receipts remain unchanged.
    points = np.asarray(old['selected_points_xy'], dtype=np.int32).copy()
    if candidate is not None:
        points[3:5] = candidate
        guard_module._annotate(image, box, points, range(len(points))).save(folder / 'points.png')
    guard_module._annotate(image, box, old['selected_points_xy'], range(len(points))).save(folder / 'previous_points.png')
    prior = 'Included · rigidity scoring completed' if cohort == 'included' else 'Excluded · rigidity scoring not run'
    reason = 'Previous negatives retained by user; new proposal not selected' if retained else '' if passed else ('N1 outside box after all retries' if candidate is not None else 'No usable coordinates after retries')
    row.update(previous_rigidity_run=prior,previous_mask_and_guard_run=True,
               negative_test_status='PREVIOUS RETAINED' if retained else 'PASS' if passed else 'FAIL',negative_failure_reason=reason,
               positives_fixed_to_previous=True,selected_negative_attempt=None if retained or selected_index is None else negative[selected_index]['attempt_number'])
    write(folder / 'negative_only_points.json', dict(positive_points_xy=points[:3].tolist(),
        negative_points_xy=candidate,negative_box_passed=guard_module._negative_box_check(candidate,box,5)['passed'] if candidate is not None else False,
        required_negative_points=['N1'],negative_source='unchanged cached negatives' if retained else 'new VLM negative proposal',
        selected_attempt=row['selected_negative_attempt'],positive_source='unchanged cached guard points',
        previous_guard_sha256=row.get('previous_guard_sha256'),sam_rerun=False))
    status_text = reason if retained else 'PASS · N1 box rule satisfied · selected attempt ' + str(row['selected_negative_attempt']) if passed else 'FAIL · ' + reason
    body = '<h1>' + esc(row['video_id']) + ' · negative guard test</h1><p><a href="../../index.html">All ten cases</a> · <a href="negative_only_points.json">Displayed point coordinates</a> · <a href="link5_guard.json">Raw API receipt</a></p><table><tr><th>Previous rigidity run</th><td>' + prior + '</td></tr><tr><th>Current negative review</th><td>' + esc(status_text) + '</td></tr><tr><th>Actual calls / model</th><td>' + str(len(negative)) + ' / ' + esc(row['model']) + '</td></tr></table><p>Only N1 must be within the green box plus 5 pixels; N2 may lie outside. All ten cases had previous masks and guard calls. Excluded cases were not run for rigidity scoring. P1–P3 are unchanged cached positive points. No masks were generated.</p>'
    comparison = '<figcaption>' + ('Previous negatives retained' if retained else 'Selected negative proposal · ' + ('PASS' if passed else 'FAIL · not accepted')) + '</figcaption><img src="points.png">' if candidate is not None else '<figcaption>No negative coordinates generated</figcaption><pre>' + esc(negative[-1].get('answer','No usable response')) + '</pre>'
    body += '<div class="images"><figure><figcaption>Previous cached points</figcaption><img src="previous_points.png"></figure><figure>' + comparison + '</figure></div>'
    coords = ''.join('<tr><td>' + label + '</td><td>' + esc(before) + '</td><td>' + (esc(points[i].tolist()) if i < 3 or candidate is not None else 'No coordinates returned') + '</td><td>' + ('Unchanged' if i < 3 or retained else 'New negative test') + '</td></tr>' for i,(label,before) in enumerate(zip(('P1','P2','P3','N1','N2'),old['selected_points_xy'])))
    body += '<h2>Point prompt coordinates</h2><table><tr><th>Point</th><th>Previous (x,y)</th><th>Displayed (x,y)</th><th>Source</th></tr>' + coords + '</table>'
    for call,check in zip(negative,checks):
        kind = 'negative'
        candidate = call.get('negative_box_check', {}).get('candidate_points_xy')
        body += '<section><h2>' + kind.capitalize() + ' guard · attempt ' + str(call['attempt_number']) + '</h2>'
        if candidate is not None:
            candidate_points = points.copy(); candidate_points[3:5] = candidate
            filename = f'negative_attempt_{call["attempt_number"]:02d}.png'
            guard_module._annotate(image, box, candidate_points, (3,4)).save(folder / filename)
            body += '<p>N1 box check: ' + ('PASS' if check['passed'] else 'FAIL') + ' · N2 outside allowed: ' + str(not check['n2_inside_box']).lower() + (' · selected proposal' if call['attempt_number']==row['selected_negative_attempt'] else '') + '</p><img src="' + filename + '">'
        body += '<details><summary>Exact input images</summary><div class="images">' + ''.join('<figure><figcaption>' + label + '</figcaption><img src="link5_guard_' + kind + '_' + name + '.png"></figure>' for name,label in [('reference','Reference'),('current','Current frame')]) + '</div></details>'
        body += '<h3>Generated response</h3><pre>' + esc(call.get('answer', call.get('response_error', 'No answer'))) + '</pre><details><summary>Exact system and user prompts</summary>' + ''.join('<h3>' + label + '</h3><pre>' + esc(call.get(key,'')) + '</pre>' for key,label in [('system_prompt','System prompt'),('user_prompt','User prompt')]) + '</details></section>'
    (folder / 'index.html').write_text(page(row['video_id'] + ' negative guard test', body))


def export_index(destination, plans, rows):
    lookup = {r['video_id']:r for r in rows}
    options = ''.join('<option value="' + p['case'] + '"' + (' disabled' if p['case'] not in lookup else '') + '>' + p['case'] + ' · previous ' + p['cohort'] + (' · pending' if p['case'] not in lookup else ' · negative ' + lookup[p['case']].get('negative_test_status','pending')) + '</option>' for p in plans)
    complete = sum(r.get('negative_test_status') == 'PASS' for r in rows)
    table = ''
    for p in plans:
        r = lookup.get(p['case'], {})
        table += '<tr><td><a href="cases/' + p['case'] + '/index.html">' + p['case'] + '</a></td><td>' + ('Included · completed' if p['cohort']=='included' else 'Excluded · not run') + '</td><td>' + r.get('negative_test_status','Pending') + '</td><td>' + str(r.get('selected_negative_attempt') or '—') + '</td><td>' + str(r.get('negative_attempts','—')) + '</td><td>' + esc(r.get('negative_failure_reason','')) + '</td></tr>'
    body = '<h1>Ten-case negative VLM guard test</h1><p>' + str(len(rows)) + '/10 reviewed · ' + str(complete) + ' new negative proposals passed the N1 box rule · no SAM or masks</p><p><strong>Previous run</strong> means rigidity scoring: five cases were included and completed; five were excluded and not scored. All ten already had masks and VLM guard calls. <strong>Current rule:</strong> only N1 must lie inside/near the green box; N2 may lie outside. Parsing failures retry the same prompt. Cached P1–P3 remain fixed. LVP 0010 retains its previous negatives by user selection.</p><table><tr><th>Case</th><th>Previous rigidity run</th><th>Current negative review</th><th>Selected attempt</th><th>Calls made</th><th>Note</th></tr>' + table + '</table><p><label>Inspect case <select id="case">' + options + '</select></label></p><p><a id="detail" target="_blank">Open full negative prompt review</a> · <a href="summary.json">Run summary</a></p><iframe id="review" title="Negative VLM point prompt" style="width:100%;height:100vh;border:0"></iframe><script>const s=document.getElementById("case"),f=document.getElementById("review"),a=document.getElementById("detail");function show(){a.href=f.src="cases/"+s.value+"/index.html"}s.onchange=show;const enabled=[...s.options].find(o=>!o.disabled);if(enabled){s.value=enabled.value;show()}</script>'
    (destination / 'index.html').write_text(page('Ten-case negative VLM guard test',body))


def run(root, cache, sources, destination, workers):
    load_env_file(Path('.env.vlm'))
    key_env = 'VLM2_API_KEY' if os.environ.get('VLM2_API_KEY') else 'AI302_API_KEY'
    if not os.environ.get(key_env):
        raise ValueError('No configured VLM API credential')
    manifest = json.loads((root/'manifest.json').read_text())
    entries = {e['video_id']:e for e in json.loads((cache/'mask_cache_manifest.json').read_text())['entries']}
    if not set(INCLUDED) <= {e['video_id'] for e in manifest['entries']}:
        raise ValueError('Included case selection differs from completed run')
    plans = []
    for case in list(manifest['excluded']) + INCLUDED:
        entry = entries[case]; old_dir = (cache/entry['mask']).parent
        old = json.loads((old_dir/'link5_guard.json').read_text())
        if old['selected_points_xy'] != entry['selected_points_xy']:
            raise ValueError('Cached guard point identity mismatch')
        diagnostic = next(d for d in json.loads((old_dir/'sam3_prompt_diagnostics.json').read_text()) if d['target']=='link5')
        source = sources/case/'v1/replay/interactive_exact-group/source.mp4'
        if sha(source) != entry['source_video_sha256'] or sha(cache/entry['mask']) != entry['mask_sha256']:
            raise ValueError('Frozen source identity mismatch: ' + case)
        capture = cv2.VideoCapture(str(source)); ok,bgr = capture.read(); capture.release()
        if not ok: raise ValueError('Cannot decode frame0: ' + case)
        image = Image.fromarray(bgr[...,::-1])
        prior = old['attempts'][0]
        if any(a['requested_model'] != prior['requested_model'] for a in old['attempts']):
            raise ValueError('Ambiguous previous per-case model')
        settings = role_config('vlm2')
        settings.update(backend='cloud_api', api_key_env=key_env, api_base=prior['api_base'],
                        api_style=prior['api_style'], model=prior['requested_model'])
        settings = guard_module.guard_config(settings)
        plans.append(dict(case=case,cohort='excluded' if case in manifest['excluded'] else 'included',
                          image=image,box=diagnostic['sam3_prompt_box_xyxy'],old=old,
                          old_dir=old_dir,source=source,settings=settings))
    destination.mkdir(parents=True,exist_ok=False)
    snapshots = destination/'code'; snapshots.mkdir()
    for path in [Path(__file__), Path(guard_module.__file__),
                 Path('robot/preprocessing/link7_persistent/vlm_client.py')]:
        shutil.copy2(path,snapshots/path.name)
    write(destination/'config.json',dict(started_at=datetime.now(timezone.utc).isoformat(),workers=workers,
        gpu_count=0,sam_rerun=False,mask_rerun=False,negative_box_margin_pixels=5,negative_max_attempts=3,
        cases=[dict(video_id=p['case'],cohort=p['cohort'],settings=p['settings'],
                    source_video_sha256=sha(p['source']),previous_guard_sha256=sha(p['old_dir']/'link5_guard.json')) for p in plans]))
    rows=[]; export_index(destination,plans,rows)
    def one(plan):
        case=plan['case'];folder=destination/'cases'/case;folder.mkdir(parents=True)
        plan['image'].save(folder/'frame0.png')
        started=time.monotonic()
        print('GUARD_BOX_START',case,plan['settings']['model'],flush=True)
        row=dict(video_id=case,cohort=plan['cohort'],model=plan['settings']['model'])
        try:
            frozen=np.asarray(plan['old']['original_points_xy'],dtype=np.int32).copy()
            frozen[:3]=plan['old']['selected_points_xy'][:3]
            _,result=guard_module.review_link5_points(plan['image'],plan['box'],frozen,folder,
                                                     required=True,vlm_config=plan['settings'],negative_only=True)
            row.update(status='complete',decision=result['decision'])
        except Exception as error:
            row.update(status='failed',error_type=type(error).__name__,error=str(error))
            result=json.loads((folder/'link5_guard.json').read_text())
            row['decision']=result['decision']
        negative=[a for a in result['attempts'] if a['role']=='link5_negative_guard']
        row.update(negative_attempts=len(negative),negative_box_passed=any(a.get('accepted') for a in negative),
                   elapsed_seconds=round(time.monotonic()-started,3),sam_rerun=False)
        # The same generated prompt and image hashes must be used on every retry.
        retry_identity=[(a.get('system_prompt'),a.get('user_prompt'),a.get('image_sha256')) for a in negative]
        row['identical_retry_inputs_verified']=all(identity==retry_identity[0] for identity in retry_identity)
        if not row['identical_retry_inputs_verified']: raise ValueError('Retry inputs changed')
        row['original_prompts_unchanged']=all(a.get('system_prompt')==next(old['system_prompt'] for old in plan['old']['attempts'] if old['role']==a['role']) and a.get('user_prompt')==next(old['user_prompt'] for old in plan['old']['attempts'] if old['role']==a['role']) for a in result['attempts'])
        export_case(folder,result,plan['old'],plan['image'],plan['box'],plan['cohort'],row)
        write(folder/'test_result.json',row)
        print('GUARD_BOX_DONE',case,row['status'],'negative_attempts',len(negative),flush=True)
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(one,p):p['case'] for p in plans}
        for future in as_completed(futures):
            try:rows.append(future.result())
            except Exception as error:
                rows.append(dict(video_id=futures[future],status='failed',negative_attempts=0,error_type=type(error).__name__,error=str(error)))
            write(destination/'summary.json',dict(status='running' if len(rows)<10 else 'complete' if all(r['status']=='complete' for r in rows) else 'complete_with_failures',
                requested_cases=10,finished_cases=len(rows),accepted_cases=sum(r['status']=='complete' for r in rows),cases=rows,
                sam_rerun=False,mask_rerun=False))
            export_index(destination,plans,rows)
    print('GUARD_BOX_TEST_FINISHED',len(rows),sum(r['status']=='complete' for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True);parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args();run(args.root,args.cache,args.sources,args.destination,args.workers)
