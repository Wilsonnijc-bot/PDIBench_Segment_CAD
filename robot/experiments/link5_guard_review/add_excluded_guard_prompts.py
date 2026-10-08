"""Show the exact saved VLM guard calls beside the excluded mask replays."""
import argparse
import html
import json
from pathlib import Path
import shutil

from .common import sha, write


def add(root, cache, *, destination_name='excluded_mask_replays'):
    destination = root / destination_name
    manifest = json.loads((destination / 'manifest.json').read_text())
    entries = {r['video_id']: r for r in json.loads((cache / 'mask_cache_manifest.json').read_text())['entries']}
    receipts = []
    esc = lambda s: html.escape(str(s))
    for row in manifest['cases']:
        case = row['video_id']; original = (cache / entries[case]['mask']).parent
        guard_path = original / 'link5_guard.json'; guard = json.loads(guard_path.read_text())
        if guard['selected_points_xy'] != entries[case]['selected_points_xy']:
            raise ValueError('Guard points differ from cached mask')
        folder = destination / case
        shutil.copy2(guard_path, folder / 'vlm_guard.json')
        images = folder / 'guard_images'; images.mkdir(exist_ok=True)
        selected_name = guard['selected_image']
        shutil.copy2(original / selected_name, images / selected_name)
        coords = ''.join('<tr><td>' + name + '</td><td>' + str(before) + '</td><td>' + str(after) + '</td></tr>'
                         for name,before,after in zip(('P1 positive','P2 positive','P3 positive','N1 negative','N2 negative'),
                                                     guard['original_points_xy'],guard['selected_points_xy']))
        calls = []
        for i, call in enumerate(guard['attempts']):
            kind = 'positive' if 'positive' in call['role'] else 'negative'
            names = ['link5_guard_' + kind + '_' + s + '.png' for s in ('reference','current')]
            if [sha(original / name) for name in names] != call['image_sha256']:
                raise ValueError('VLM input image provenance mismatch')
            for name in names:
                shutil.copy2(original / name, images / name)
            pictures = ''.join('<figure><figcaption>Image ' + str(j+1) + ' · ' + label + '</figcaption><a href="guard_images/' + name + '"><img src="guard_images/' + name + '" alt="' + kind + ' ' + label + '"></a></figure>'
                               for j,(name,label) in enumerate(zip(names,('Reference','Current frame'))))
            texts = ''.join('<h3>' + label + '</h3><pre>' + esc(call.get(key,'')) + '</pre>'
                            for key,label in (('system_prompt','System prompt'),('user_prompt','User prompt'),('answer','Saved response')))
            calls.append('<section><h2>' + str(i+1) + '. ' + kind.capitalize() + ' guard</h2><p>'
                         + esc(call.get('model') or call.get('requested_model')) + ' · '
                         + esc(call.get('reasoning_effort','')) + '</p><div class="images">' + pictures + '</div>' + texts + '</section>')
        body = '<header><h1>' + case + ' · VLM guard</h1><p><a href="index.html">Mask replay</a> · <a href="vlm_guard.json" download>Saved guard JSON</a></p></header><p>Saved mask: ' + original.name + '. Guard decision: ' + esc(guard['decision']) + '. Default-point fallback: ' + str(guard['fallback_to_default']).lower() + '.</p><p>A guard REJECT means it corrected proposed points. The mask exclusion is a separate decision.</p><h2>Selected SAM point prompt</h2><p>Coordinates use the full frame shown to the guard. P1–P3 are positive points; N1–N2 are negative points.</p><table><thead><tr><th>Point</th><th>Proposed (x, y)</th><th>Selected (x, y)</th></tr></thead><tbody>' + coords + '</tbody></table><img class="selected" src="guard_images/' + selected_name + '" alt="Selected point prompt">' + ''.join(calls)
        page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + case + ' VLM guard prompts</title><style>:root{color-scheme:dark;font:15px/1.6 system-ui;background:#101719;color:#e7f0eb}body{max-width:1320px;padding:28px;margin:auto}h1{font-size:28px}h2{margin-top:32px}h3{font-size:16px;color:#a7f2cc}a{color:#8edfc0}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:18px;background:#182225;border:1px solid #344548;font:14px/1.6 ui-monospace,monospace}table{border-collapse:collapse;width:100%;max-width:760px}td,th{padding:9px 14px;border-bottom:1px solid #344548;text-align:left}.images{display:grid;grid-template-columns:1fr 1fr;gap:18px}figure{margin:0}figcaption{margin-bottom:8px;color:#b8ccc3}img{width:100%;height:auto}.selected{max-width:1100px;margin-top:20px}section{border-top:1px solid #344548;margin-top:32px}@media(max-width:760px){.images{grid-template-columns:1fr}body{padding:16px}}</style><main>' + body + '</main></html>'
        (folder / 'vlm_guard.html').write_text(page)
        replay_path = folder / 'index.html'; replay = replay_path.read_text()
        if 'href="vlm_guard.html"' not in replay:
            replay = replay.replace('<div class="links">', '<div class="links"><a href="vlm_guard.html" target="_blank" rel="noopener">VLM guard prompts, responses and points</a>')
            replay_path.write_text(replay)
        receipts.append(dict(case=case, calls=len(guard['attempts']), guard_sha256=sha(guard_path),
                             exact_saved_prompts=True, all_input_image_hashes_verified=True, points_match_cached_mask=True))
    write(destination / 'guard_prompt_verification.json', dict(status='complete', cases=receipts, inference_rerun=False))
    landing=destination/'index.html'
    page=landing.read_text()
    if 'id="guard"' not in page:
        page=page.replace('</header>','<a id="guard" target="_blank" rel="noopener" style="color:#8edfc0">VLM guard prompts and points</a></header>',1)
        page=page.replace('function change(){frame.src=select.value+"/index.html"}',
            'function change(){frame.src=select.value+"/index.html";document.getElementById("guard").href=select.value+"/vlm_guard.html"}')
        landing.write_text(page)
    print('EXCLUDED_GUARD_PROMPTS_COMPLETE', len(receipts), sum(r['calls'] for r in receipts), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--destination', default='excluded_mask_replays')
    args=parser.parse_args(); add(args.root,args.cache,destination_name=args.destination)
