"""Estimate guard costs from saved API usage, including unsuccessful responses.

These are public-price estimates, not invoices. Output usage already includes
reasoning; never add reasoning_tokens a second time. Zero-valued input/output
aliases emitted by 302.AI must not override its prompt/completion counts.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from .common import read, write
from .refine_guard_local import successful_masking

PRICES = {
    'gemini-3.8-flash': dict(input=0.75, output=3.75, cache_read=0.075,
        source='https://302.ai/product/detail/gemini-3.8-flash'),
    'gpt-6-luna': dict(input=0.10, output=0.50, cache_read=0.01,
        source='https://302.ai/product/detail/openai-gpt-6-luna'),
}


def usage_tokens(call):
    usage = call.get('usage') or (call.get('response_diagnostic') or {}).get('usage')
    if not isinstance(usage, dict): return None
    native = (usage.get('billing_usage') or {}).get('gemini_usage_metadata') or {}
    def number(*values):
        present = [v for v in values if v is not None]
        return next((int(v) for v in present if int(v) > 0), int(present[0]) if present else None)
    input_count = number(usage.get('prompt_tokens'), usage.get('input_tokens'), native.get('promptTokenCount'))
    output_count = number(usage.get('completion_tokens'), usage.get('output_tokens'),
        native.get('candidatesTokenCount', 0) + native.get('thoughtsTokenCount', 0) if native else None)
    if input_count is None or output_count is None: return None
    cached = number((usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
        (usage.get('input_tokens_details') or {}).get('cached_tokens'), native.get('cachedContentTokenCount')) or 0
    reasoning = number((usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
        (usage.get('output_tokens_details') or {}).get('reasoning_tokens'), native.get('thoughtsTokenCount')) or 0
    return dict(input_tokens=input_count, output_tokens=output_count,
        cached_input_tokens=min(cached, input_count), reasoning_tokens=reasoning)


def estimate(call):
    tokens=usage_tokens(call); model=call.get('model') or call.get('requested_model')
    if tokens is None or model not in PRICES:return None
    rate=PRICES[model];cached=tokens['cached_input_tokens']
    return ((tokens['input_tokens']-cached)*rate['input']+cached*rate['cache_read']+
        tokens['output_tokens']*rate['output'])/1_000_000


def run(root):
    destination=root/'refinement';ids=[r['video_id'] for r in read(root/'training/train20_manifest.json')]
    calls=[];seen=set()
    def add(video,phase,route,path,record,superseded=False):
        for index,call in enumerate(record.get('attempts', [])):
            key=call.get('id') or hashlib.sha256(json.dumps(call,sort_keys=True).encode()).hexdigest()
            # A receipt can appear in both a history and the selected result.
            if key in seen:continue
            seen.add(key);tokens=usage_tokens(call)
            calls.append(dict(video_id=video,phase=phase,route=route,receipt=str(path),call_index=index,superseded=superseded,
                request_id=call.get('id'),role=call.get('role'),model=call.get('model') or call.get('requested_model'),
                reasoning_effort=call.get('reasoning_effort'),output_token_limit=call.get('output_token_limit'),
                timeout_seconds=call.get('timeout_seconds'),elapsed_seconds=call.get('elapsed_seconds'),
                valid_answer=not any(call.get(k) for k in ['parse_error','response_error','failure']),
                usage_available=tokens is not None,**(tokens or {}),estimated_usd=estimate(call)))
    baseline=[]
    for video in ids:
        path=successful_masking(root,video)/'link5_guard.json';record=read(path)
        start=len(calls);add(video,'previous_selected','selected',path,record)
        baseline.append(dict(video_id=video,model=record['attempts'][0].get('model') or record['attempts'][0]['requested_model'],
            usd=sum(c['estimated_usd'] or 0 for c in calls[start:])))
        for path in sorted((root/'cases'/video/'masking').glob('**/link5_guard.json')):
            if path==successful_masking(root,video)/'link5_guard.json':continue
            add(video,'previous_retry','retry',path,read(path))
    for phase,folder in [('refined_initial',destination),('refined_64k',destination/'validation_64k')]:
        for video in ids:
            guard=folder/'cases'/video/'guard'
            for route in ['gemini','luna']:
                path=guard/route/'link5_guard.json'
                if path.exists():add(video,phase,route,path,read(path))
            for name in ['failed_attempts.json','attempt_history.json']:
                path=guard/'metadata'/name
                if path.exists():
                    for item in read(path):add(video,phase,item['route'],path,item['receipt'],superseded=True)
    def totals(rows):
        known=[c for c in rows if c['estimated_usd'] is not None]
        return dict(calls=len(rows),usage_known_calls=len(known),unknown_charge_calls=len(rows)-len(known),
            estimated_usd=sum(c['estimated_usd'] for c in known),
            input_tokens=sum(c['input_tokens'] for c in known),output_tokens=sum(c['output_tokens'] for c in known),
            reasoning_tokens=sum(c['reasoning_tokens'] for c in known))
    def pairs(phase,route=None):
        grouped=defaultdict(list)
        for call in calls:
            if not call['superseded'] and call['phase']==phase and (route is None or call['route']==route):grouped[call['video_id']].append(call)
        return {video:sum(c['estimated_usd'] for c in rows)
            for video,rows in grouped.items() if len(rows)==2 and all(c['valid_answer'] and c['estimated_usd'] is not None for c in rows)}
    old={r['video_id']:r['usd'] for r in baseline};old_gemini={r['video_id'] for r in baseline if r['model']=='gemini-3.8-flash'}
    comparisons={}
    for phase in ['refined_initial','refined_64k']:
        current=pairs(phase,'gemini');matched=sorted(old_gemini & current.keys())
        before=[old[v] for v in matched];after=[current[v] for v in matched]
        comparisons[phase]=dict(matched_gemini_frames=len(matched),video_ids=matched,
            previous_mean_usd=mean(before) if before else None,refined_mean_usd=mean(after) if after else None,
            previous_median_usd=median(before) if before else None,refined_median_usd=median(after) if after else None,
            mean_ratio=mean(after)/mean(before) if before and mean(before) else None)
    phases={phase:totals([c for c in calls if c['phase']==phase]) for phase in sorted({c['phase'] for c in calls})}
    route_totals={phase:{route:totals([c for c in calls if c['phase']==phase and c['route']==route])
        for route in ['gemini','luna']} for phase in ['refined_initial','refined_64k']}
    # One successful connectivity probe preceded the guard tests.
    probe=dict(prompt_tokens=40,completion_tokens=19,reasoning_tokens=18,
        estimated_usd=(40*.75+19*3.75)/1_000_000,model='gemini-3.8-flash',scope='text-only authenticated connectivity probe')
    report=dict(currency='USD',kind='public-price estimate; invoice charges unavailable',price_checked_date='2026-10-06',
        prices_per_million_tokens=PRICES,reasoning_already_in_output_tokens=True,baseline_frames=20,
        previous_selected_frame_mean_usd=mean(old.values()),previous_selected_frame_median_usd=median(old.values()),
        phases=phases,route_totals=route_totals,matched_comparisons=comparisons,connectivity_probe=probe,
        local_test_known_total_usd=sum(v['estimated_usd'] for k,v in phases.items() if k.startswith('refined'))+probe['estimated_usd'],
        notes=['Reprice both old and new usage at the same current provider prices; historical invoice rates are unknown.',
            'Each complete frame guard uses two calls: negatives and positives.',
            'Previous_selected includes 15 Gemini and 5 Luna frames; matched comparisons control for that model mixture.',
            'Initial refined batch used a 16,384-token requested budget; final configuration is 65,536 with 600/1200-second timeouts.',
            'Luna comparisons requested on difficult cases are extra diagnostic calls, not routine fallback cost.',
            'Calls without usage may have unknown charges; they are excluded from known totals, not asserted to be free.',
            'Failed responses with usage are included. SAM, GPU, MegaSAM and detector costs are outside this VLM-guard comparison.'],
        baseline_frames_detail=baseline,calls=calls)
    write(destination/'metadata/vlm_costs.json',report)
    with (destination/'metadata/vlm_costs.csv').open('w',newline='') as stream:
        fields=sorted({key for c in calls for key in c});writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(calls)
    lines=['# Link5 guard cost comparison','',
        'USD estimates from saved usage and current public 302.AI prices, checked 2026-10-06. These are not invoice charges. Reasoning tokens are already included in output usage. Each frame review has one negative and one positive call.','',
        f"Original successful 20-frame guards: mean **${report['previous_selected_frame_mean_usd']:.5f}** per frame; median **${report['previous_selected_frame_median_usd']:.5f}**. This mixes 15 Gemini frames with 5 Luna frames.",'',
        '| Same-frame Gemini comparison | Frames | Old mean / frame | New mean / frame | New / old |',
        '|---|---:|---:|---:|---:|']
    for phase,c in comparisons.items():
        if c['matched_gemini_frames']:lines.append(f"| {phase} | {c['matched_gemini_frames']} | ${c['previous_mean_usd']:.5f} | ${c['refined_mean_usd']:.5f} | {c['mean_ratio']:.2f}× |")
    lines.extend(['','| Receipt group | Calls | Known estimated cost | Calls with unknown charges |','|---|---:|---:|---:|'])
    for phase,stats in phases.items():lines.append(f"| {phase} | {stats['calls']} | ${stats['estimated_usd']:.5f} | {stats['unknown_charge_calls']} |")
    lines.extend(['',f"All local tests with reported usage, including Luna diagnostics, unsuccessful responses with usage and the connectivity probe: **${report['local_test_known_total_usd']:.5f}**.",'',
        'The final four-case 64K test explicitly requests both Gemini high and Luna xhigh. Its Luna calls are diagnostics rather than fallbacks. The initial 20-case run used Luna for one malformed Gemini result plus four requested comparisons. All model routes and receipt paths are in [the ledger](metadata/vlm_costs.csv).','',
        'The 65,536-token maximum is a ceiling, not a fixed charge. Cost follows actual returned usage. The old Gemini calls already contained reasoning tokens even though reasoning effort was not explicitly configured.','',
        'Rates per million tokens: Gemini input $0.75, output $3.75, cache read $0.075; Luna input $0.10, output $0.50, cache read $0.01 for this small-context workload. Sources: [302.AI Gemini](https://302.ai/product/detail/gemini-3.8-flash), [302.AI Luna](https://302.ai/product/detail/openai-gpt-6-luna).','',
        'No GPU or SAM rerun cost is included. Unknown-usage transport failures are retained and excluded from the known dollar subtotal. Both previous and refined usage are priced at the same rates to isolate usage changes.',''])
    (destination/'COST_COMPARISON.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in report.items() if k in ['phases','matched_comparisons','local_test_known_total_usd']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    run(parser.parse_args().root)
