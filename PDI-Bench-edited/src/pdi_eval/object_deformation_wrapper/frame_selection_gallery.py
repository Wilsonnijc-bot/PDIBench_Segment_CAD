"""Write a local, dependency-free gallery for selected current/reference crops."""
from html import escape
import json
import hashlib
import math
from pathlib import Path

from .occlusion import atomic


def write_gallery(root: Path) -> Path:
    cases = json.loads((root / 'selection_index.json').read_text())
    cases.sort(key=lambda case: (int(case['case'].rsplit('_', 1)[-1])
                                if case['case'].rsplit('_', 1)[-1].isdigit() else float('inf'),
                                case['case']))
    score_path = root / 'anomalydino_scores.json'
    score_data = json.loads(score_path.read_text()) if score_path.is_file() else None
    scores = {(r['video_id'], r['frame']): r for r in score_data['pairs']} if score_data else {}
    excluded = set(score_data['excluded_videos']) if score_data else set()
    # Do not display scores or sums for input crops whose bytes have changed.
    scores = {key: row for key, row in scores.items() if all(
        (root / row[role + '_crop']).is_file()
        and hashlib.sha256((root / row[role + '_crop']).read_bytes()).hexdigest() == row[role + '_sha256']
        for role in ('reference', 'query'))}
    parts = ['''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Object crop review · PDI benchmark</title>
<style>
:root{color-scheme:light;font-family:"Avenir Next",Avenir,"Segoe UI",sans-serif;color:#24312f;background:#f5f6f2}
*{box-sizing:border-box}body{margin:0;padding:32px clamp(16px,3vw,48px)}main{max-width:1600px;margin:auto}
h1{font-size:32px;margin:0 0 10px;font-weight:600;letter-spacing:-.025em}p{line-height:1.6;margin:8px 0}
a{color:#246757;text-underline-offset:3px}a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #a47820;outline-offset:3px}
header{margin-bottom:24px}.muted{color:#55655d;font-size:13px}
.controls{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}label{display:grid;gap:5px;font-size:13px}input,select{font:inherit;background:#fafbf7;border:1px solid #b6c2b8;padding:10px;min-height:42px}input{width:min(340px,80vw)}
.case{padding:24px 0;border-top:1px solid #cbd3cb}.case[hidden]{display:none}.case-head{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin-bottom:16px}.case-head h2{margin:0 0 6px;font-size:21px;font-weight:600;overflow-wrap:anywhere}.reference{width:64px;height:64px;object-fit:contain;background:#e0e5dc}
.video-total{margin-left:auto;display:flex;align-items:baseline;flex-wrap:wrap;gap:8px 14px;font-size:14px;font-variant-numeric:tabular-nums}.video-total strong{font-size:30px;font-weight:700;color:#8b332d;line-height:1.2}.video-total .muted{flex-basis:100%;text-align:right}
.crops{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:14px}figure{margin:0;padding:12px;background:#fafbf7;border:1px solid #d5dbd2;min-width:0}.recovery{border-color:#7d9b87;background:#f0f5ee}.fallback{border-color:#bea96d}
.frame-title{display:flex;justify-content:space-between;gap:8px;font-weight:600;font-size:14px}.frame-title span:last-child{color:#55655d;font-size:12px;font-weight:400}
.scores{display:flex;align-items:baseline;justify-content:space-between;flex-wrap:wrap;gap:4px 8px;border-top:1px solid #d5dbd2;border-bottom:1px solid #d5dbd2;padding:8px 0;margin:10px 0 12px;font-size:13px;font-variant-numeric:tabular-nums}.scores strong{font-size:25px;font-weight:700;line-height:1.2;color:#8b332d}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0}.pair span{font-size:12px}.pair img{width:100%;height:135px;display:block;object-fit:contain;background:repeating-conic-gradient(#d8ddd5 0% 25%,#e8ece5 0% 50%) 50% / 16px 16px;margin-top:5px}
.pair-details,.case-details{font-size:12px;line-height:1.7;color:#55655d;margin-top:8px}summary{cursor:pointer}.case-details{margin:0 0 16px}.empty{background:#ebeee7;padding:16px;font-size:14px}.footer{margin:32px 0;font-size:13px}
@media(max-width:1100px){.crops{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:720px){.crops{grid-template-columns:repeat(2,minmax(0,1fr))}.case-head h2{font-size:18px}.video-total{margin-left:0;flex-basis:100%}.video-total .muted{text-align:left}}@media(max-width:420px){.crops{grid-template-columns:1fr}.pair img{height:180px}h1{font-size:27px}}
</style></head><body><main><header>''']
    parts.append('<h1>Object crop anomaly scores</h1>' if score_data else '<h1>Selected object crops</h1>')
    if score_data:
        scored_videos = len({name for name, _ in scores})
        parts.append(f'<p>{len(scores)} scored pairs · {scored_videos} scored videos · Higher scores mean more anomalous.</p>')
    else:
        parts.append(f'<p>{sum(case["selected_count"] for case in cases)} crop pairs · {len(cases)} videos.</p>')
    parts.append('''</header><div class="controls"><label>Find a video<input id="search" type="search" placeholder="Case name" autocomplete="off"></label><label>Show<select id="filter"><option value="all">All videos</option><option value="scored">Scored videos</option><option value="recovery">With recovery selections</option><option value="fallback">With replacements</option><option value="exceptions">With exceptions</option></select></label></div><p id="visible" class="muted" aria-live="polite"></p>''')
    reasons = {'immediate_post_occlusion': 'Immediately after occlusion',
               'largest_available_area_in_interval': 'Highest available area in interval',
               'empty_interval_fallback': 'Replacement for empty interval'}
    for case in cases:
        name = case['case']
        safe_name = escape(name, quote=True)
        manifest = json.loads((root / name / 'manifest.json').read_text())
        selected = case['selected_frames']
        fallback = any(row['reason'] == 'empty_interval_fallback' for row in selected)
        recovery = bool(case['required_frames'])
        exceptions = bool(case['shortfall'] or case['unsatisfied_recovery_count'] or fallback)
        values = [scores[(name, row['frame'])]['anomaly_score'] for row in selected if (name, row['frame']) in scores]
        parts.append(f'<section class="case" data-name="{safe_name.lower()}" data-scored="{int(bool(values))}" data-recovery="{int(recovery)}" data-fallback="{int(fallback)}" data-exceptions="{int(exceptions)}"><div class="case-head">')
        if (root / name / 'frame0_reference.png').is_file():
            parts.append(f'<a href="{safe_name}/frame0_reference.png" title="Full frame 0 reference"><img class="reference" src="{safe_name}/frame0_reference.png" alt="Full frame 0 object reference" loading="lazy"></a>')
        parts.append(f'<div><h2>{safe_name}</h2><div class="muted">{manifest["frame_count"]} frames · {len(selected)} pairs · <a href="{safe_name}/selection.json">Details</a></div></div>')
        if score_data:
            if values and len(values) == len(selected):
                parts.append(f'<div class="video-total"><span>Summed anomaly score</span><strong>{math.fsum(values):.4f}</strong><span class="muted">Sum of {len(values)} pairs</span></div>')
            else:
                state = 'Excluded' if name in excluded else 'Unavailable' if not selected else 'Incomplete'
                parts.append(f'<div class="video-total muted">Summed anomaly score · {state}</div>')
        parts.append('</div>')
        geometry_path = root / name / 'pair_geometry.json'
        if geometry_path.is_file() and selected:
            geometry = json.loads(geometry_path.read_text())
            if all(row['crop_directory'] in geometry['pairs'] for row in selected):
                parts.append('<p class="muted">Matched crop canvases: transparent borders trimmed, '
                             'same canvas size and display scale per pair. Visible pixels and shapes '
                             'are preserved; frame-0 correspondence remains estimated.</p>')
        if not selected:
            quality = escape(manifest['frame0_reference_quality'].replace('_', ' '))
            parts.append(f'<p class="empty">No usable reference crops: {quality}.</p>')
        notes = []
        final_policy = case.get('final_interval_policy', {})
        if final_policy.get('collapse_detected'):
            parts.append('<p class="empty">Final-interval mean is low. '
                         f'Exclude frames below {final_policy["frame_area_threshold"]:,.1f} available pixels; '
                         f'{len(final_policy["preserved_eligible_frames"])} substantial crop-eligible frames remain. '
                         'Missing slots use the existing earlier-interval fallback.</p>')
        elif final_policy.get('suppressed'):
            parts.append('<p class="empty">Final 20% skipped: mean available object area fell to '
                         f'{final_policy["final_to_first_80_ratio"]:.1%} of the first-80% mean and '
                         f'{final_policy["final_to_preceding_interval_ratio"]:.1%} of the preceding interval. '
                         'Those crop slots use earlier intervals.</p>')
        if fallback:
            notes.append('Missing interval slots use replacement frames.')
        for event in case['occlusion_episodes']:
            if event['status'] in ('successor_not_assessable', 'successor_crop_invalid'):
                notes.append(f'Frame {event["immediate_successor_frame"]}: recovery crop unavailable.')
            elif event['status'] == 'occlusion_reaches_video_end':
                notes.append('Occlusion reaches the video end.')
            elif event['status'] == 'successor_in_suppressed_final_interval':
                notes.append(f'Frame {event["immediate_successor_frame"]}: recovery selection skipped with the final interval.')
            elif event['status'] == 'successor_below_final_interval_area_gate':
                notes.append(f'Frame {event["immediate_successor_frame"]}: recovery crop skipped for low available area.')
        if notes:
            parts.append(f'<details class="case-details"><summary>Selection notes</summary>{escape(" ".join(notes))}</details>')
        parts.append('<div class="crops">')
        for row in selected:
            folder = escape(f'{name}/{row["crop_directory"]}', quote=True)
            css = 'recovery' if row['reason'] == 'immediate_post_occlusion' else 'fallback' if row['reason'] == 'empty_interval_fallback' else ''
            lo, hi = row['percent_range']
            t = row['frame']
            parts.append(f'<figure class="{css}"><div class="frame-title"><span>Frame {t}</span><span>{lo}–{hi}%</span></div>')
            if score_data:
                score = scores.get((name, t))
                if score:
                    parts.append(f'<div class="scores"><span>Anomaly score</span><strong>{score["anomaly_score"]:.4f}</strong></div>')
                else:
                    state = 'Excluded' if name in excluded else 'Unavailable'
                    parts.append(f'<div class="scores muted">Anomaly score · {state}</div>')
            parts.append('<div class="pair">')
            for file, label in (('current_available.png', 'Current'), ('frame0_shape_crop.png', 'Frame 0')):
                parts.append(f'<div><span>{label}</span><a href="{folder}/{file}"><img src="{folder}/{file}" alt="{safe_name}, frame {t}, {label}" loading="lazy"></a></div>')
            flagged = 'Flagged occlusion' if row['occlusion_flagged'] else 'Unflagged'
            covered = manifest['frames'][t].get('link7_object_covered_fraction')
            coverage = f' · link7 overlap {covered:.1%}' if covered is not None else ''
            parts.append(f'</div><p class="muted">{row["available_area"]:,} available pixels{coverage}</p><a href="{folder}/original_frame.png" target="_blank" rel="noopener">Original frame</a><details class="pair-details"><summary>Crop details</summary>Interval rank {row["interval_area_rank"]}<br>{flagged}<br>{reasons[row["reason"]]}<br><a href="{folder}/preview.png">Three-panel preview</a></details></figure>')
        parts.append('</div></section>')
    parts.append('<p class="footer">')
    if score_data:
        parts.append('<a href="anomalydino_scores.json">Scores</a> · <a href="anomalydino_videos.csv">Video totals</a> · ')
    parts.append('<a href="selection_index.json">Selection index</a>')
    if (root / "README.md").is_file():
        parts.append(' · <a href="README.md">Documentation</a>')
    parts.append('''</p></main>
<script>
const search=document.getElementById('search'), filter=document.getElementById('filter');
function update(){let count=0;const term=search.value.trim().toLowerCase();for(const item of document.querySelectorAll('.case')){const match=item.dataset.name.includes(term)&&(filter.value==='all'||item.dataset[filter.value]==='1');item.hidden=!match;if(match)count++;}document.getElementById('visible').textContent=count+' videos shown';}
search.addEventListener('input',update);filter.addEventListener('change',update);update();
</script></body></html>''')
    path = root / 'selection_gallery.html'
    atomic(path, '\n'.join(parts) + '\n')
    return path
