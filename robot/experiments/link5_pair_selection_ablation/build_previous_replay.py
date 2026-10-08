"""Adapt the maintained rigidity replay template to existing ablation evidence."""
import argparse
import gzip
import json
from pathlib import Path
import shutil

import numpy as np

from infrastructure.shared.replay.reconstruct_replay import transform_world_to_camera
from .prepare_gpu import sha, write

REPOSITORY = Path(__file__).resolve().parents[3]
METHODS = ('balanced_v0', 'refine_v1')


def build(root):
    report = root / 'report'
    manifest = json.loads((root / 'manifest.json').read_text())
    completion = json.loads((root / 'replay_context/completion.json').read_text())
    if completion['status'] != 'complete' or completion['videos'] != len(manifest['entries']):
        raise ValueError('All replay contexts required')
    template_path = REPOSITORY / 'infrastructure/shared/replay/rigidity_replay.html'
    template = template_path.read_text()
    template = template.replace('src="plotly.min.js"', 'src="../../plotly.min.js"')
    template = template.replace('Whole-robot cloud · source camera view', 'Filtered Link5 cloud · source camera view')
    template = template.replace("`${data.version || 'V1'} Link 7 filter / ${data.tracker_label || 'CoTracker'}${data.mask_label?' / '+data.mask_label:''} / camera-aligned point cloud`", "`${data.version} / all-frame depth support / ${data.tracker_label} / camera-aligned point cloud`")
    template = template.replace("const [height,width]=data.image_hw, focal=data.focal_length;\n  return [focal*point[0]/point[2]+width/2,focal*point[1]/point[2]+height/2];", "const [fx,fy,cx,cy]=data.camera_intrinsic;\n  return [fx*point[0]/point[2]+cx,fy*point[1]/point[2]+cy];")
    template = template.replace("cloudTrace(cloud,'#b9edcc'", "cloudTrace(cloud,data.cloud_colors?.[frame]||'#b9edcc'")
    template = template.replace('The source camera view projects the reconstructed scene and whole robot at the original image scale.', 'The source camera view projects the reconstructed scene and filtered Link5 at the original image scale. Both methods share the same saved depth support on every frame.')
    template = template.replace('When fewer than three pairs are visible', 'When fewer than three pairs have both visibility and depth support')
    template = template.replace('<span id="time"></span>', '<span id="time"></span><label>Go to frame <input id="frame-number" type="number" min="1" value="1" aria-label="Display frame" style="width:70px;background:#1c2525;color:#edf2ed;border:1px solid #688779;padding:6px"></label>')
    template = template.replace('slider.max=n-1;', "slider.max=n-1;document.getElementById('frame-number').max=n;document.getElementById('frame-number').onchange=e=>seek(Number(e.target.value)-1);")
    template = template.replace("document.getElementById('time').textContent=", "if(document.activeElement!==document.getElementById('frame-number'))document.getElementById('frame-number').value=frame+1;\n  document.getElementById('time').textContent=")
    (report / 'context').mkdir(exist_ok=True)
    shutil.copy2(REPOSITORY / 'infrastructure/shared/replay/assets/plotly.min.js', report / 'plotly.min.js')
    rows, pages = [], []
    correlations = json.loads((root / 'analysis/forearm_AB_correlation.json').read_text())
    labels = {r['video_id']: r['forearm_AB'] for r in correlations['rows']}
    playback = {r['case']: r for r in json.loads((root / 'metadata/playback_provenance.json').read_text())['files']}
    for entry in manifest['entries']:
        case = entry['video_id']
        context_path = root / 'replay_context' / (case + '.json.gz')
        context_receipt = json.loads(context_path.with_suffix('.receipt.json').read_text())
        raw_receipt = json.loads((root / 'shared_inputs' / (case + '.json')).read_text())
        depth_receipt = json.loads((root / 'depth_support' / (case + '.json')).read_text())
        if (sha(context_path) != context_receipt['context_sha256']
                or context_receipt['raw_input_sha256'] != raw_receipt['raw_sha256']
                or context_receipt['support_sha256'] != depth_receipt['support_sha256']):
            raise ValueError('Replay context provenance mismatch: ' + case)
        with gzip.open(context_path, 'rt') as stream:
            context = json.load(stream)
        poses = np.asarray(context.pop('camera_poses'))
        intrinsic = np.asarray(context.pop('intrinsic'))
        k = intrinsic.tolist() if intrinsic.shape == (4,) else [intrinsic[0,0], intrinsic[1,1], intrinsic[0,2], intrinsic[1,2]]
        context['camera_intrinsic'] = k
        target_video = report / 'playback' / (case + '.mp4')
        if sha(target_video) != playback[case]['playback_sha256'] or playback[case]['source_sha256'] != entry['source_video_sha256']:
            raise ValueError('Playback provenance mismatch')
        (report / 'context' / (case + '.js')).write_text('window.LINK5_REPLAY_CONTEXT=' + json.dumps(context, separators=(',', ':'), allow_nan=False) + ';')
        directory = report / 'cases' / case
        directory.mkdir(parents=True, exist_ok=True)
        row = dict(case=case, cohort=entry['cohort'], label=labels[case], methods={})
        for method in METHODS:
            folder = root / 'rigidity' / case / method
            evidence = json.loads((folder / 'evidence.json').read_text())
            result = json.loads((folder / 'result.json').read_text())
            with np.load(folder / 'trajectories.npz', allow_pickle=False) as archive:
                world = archive['sampled_world_xyz']; ids = archive['point_ids']
                supported = (archive['raw_visibility'] > .5) & archive['query_depth_support']
                indices = archive['pair_track_indices']; ratios = archive['distance_ratios']; available = archive['pair_available']
                baselines = archive['baseline_distances']
            if len(world) != len(poses) or len(world) != len(context['clouds']):
                raise ValueError('Replay frame coverage mismatch')
            camera = np.stack([transform_world_to_camera(w, p) for w, p in zip(world, poses)])
            lengths = np.linalg.norm(camera[:, indices[:,0]] - camera[:, indices[:,1]], axis=-1)
            # Saved ratios are the scientific evidence. Check the display transform preserves distances.
            if not np.allclose(lengths[available], (ratios * baselines)[available], rtol=2e-6, atol=1e-8):
                raise ValueError('Replay distance transform mismatch')
            points = [[p.tolist() if ok else None for p, ok in zip(frame, visible)] for frame, visible in zip(camera, supported)]
            history = evidence['rigidity_history']; carried = set(evidence['carried_frames'])
            fresh = {r['frame'] for r in evidence['pair_frames']}
            if not np.isclose(np.mean(history[1:]), result['rigidity_score'], rtol=1e-12, atol=1e-12):
                raise ValueError('Saved temporal score mismatch')
            data = dict(object='Link5 · ' + case, version=method, tracker_label='CoTracker3',
                fps=raw_receipt['fps'], score=result['rigidity_score'], history=history,
                observed=[history[t] if t in fresh else None for t in range(len(history))],
                carried_frames=sorted(carried), coverage=len(fresh)/max(1,len(history)-1),
                point_ids=ids.tolist(), points=points,
                windows=[dict(start=0, end=len(history), query_frame=0,
                              selected_pairs=evidence['selected_pairs'], pair_frames=evidence['pair_frames'])],
                evidence_file='../../../rigidity/' + case + '/' + method + '/evidence.json')
            embedded = 'Object.assign({},window.LINK5_REPLAY_CONTEXT,' + json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c') + ')'
            page = template.replace('__REPLAY_DATA__', embedded)
            page = page.replace('<script>\nconst data=', '<script src="../../context/' + case + '.js"></script>\n<script>\nconst data=')
            page = page.replace('src="source.mp4"', 'src="../../playback/' + case + '.mp4" muted')
            destination = directory / (method + '.html')
            destination.write_text(page)
            pages.append(dict(path=str(destination.relative_to(root)), sha256=sha(destination)))
            row['methods'][method] = result['rigidity_score']
        rows.append(row)
    navigation = dict(rows=rows, correlations={p: s['all_labeled'] for p,s in correlations['statistics'].items()})
    (report / 'navigation.js').write_text('window.REPLAY_NAVIGATION=' + json.dumps(navigation, separators=(',', ':'), allow_nan=False) + ';')
    (report / 'index.html').write_text(INDEX)
    # This file supported the rejected canvas viewer; no active viewer uses it.
    (report / 'data.js').unlink(missing_ok=True)
    write(root / 'metadata/replay_style_receipt.json', dict(status='complete', videos=len(rows), pages=len(pages),
          reused_template=str(template_path.relative_to(REPOSITORY)), template_sha256=sha(template_path),
          exporter_sha256=sha(__file__), scientific_scores_modified=False,
          adaptations=['Video and method navigation', 'Exact native intrinsics', 'Saved depth-supported query availability',
                       'Shared sampled CVD context', 'Browser display copies with verified timestamps'],
          removed_view='Current canvas overlay comparison style', artifacts=pages, published=False))
    print('PREVIOUS_STYLE_REPLAY_COMPLETE', len(rows), len(pages), flush=True)


INDEX = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Link5 rigidity replay</title><style>
:root{color-scheme:dark;font:14px ui-sans-serif,system-ui,sans-serif;background:#12181a;color:#edf2ed}*{box-sizing:border-box}body{margin:0}header{padding:14px 24px;border-bottom:1px solid #3c4644;display:flex;align-items:center;gap:18px;flex-wrap:wrap}h1{font-size:20px;margin:0;font-weight:600}select{background:#294036;color:#eff7ec;border:1px solid #688779;padding:7px 10px}a{color:#b9edcc}#scores{color:#c7d4cb;font-variant-numeric:tabular-nums}iframe{width:100%;height:calc(100vh - 125px);min-height:1080px;border:0;display:block}.analysis{padding:10px 24px;border-bottom:1px solid #3c4644;color:#aab9b1;line-height:1.7}label{white-space:nowrap}@media(max-width:760px){header,.analysis{padding-left:14px;padding-right:14px}iframe{min-height:1550px}}
</style></head><body><header><h1>Link5 rigidity replay</h1><label>Video <select id="case"></select></label><label>Method <select id="method"><option value="balanced_v0">Balanced v0</option><option value="refine_v1" selected>Refine v1</option></select></label><span id="scores"></span></header><div class="analysis">82/82 scores complete · all-frame depth support · <a href="../rigidity/paired_scores.csv">Scores CSV</a> · <a href="../analysis/forearm_AB_correlation.md">AB forearm correlations</a> · <span id="label"></span> · Local, not published</div><iframe id="replay" title="Interactive rigidity evidence"></iframe><script src="navigation.js"></script><script>
const D=window.REPLAY_NAVIGATION,$=id=>document.getElementById(id);for(const r of D.rows){let o=document.createElement('option');o.value=r.case;o.textContent=r.case+(r.cohort==='additional'?' · additional':'');$('case').append(o)}function select(){const r=D.rows.find(v=>v.case===$('case').value);$('scores').textContent=`Balanced ${(100*r.methods.balanced_v0).toFixed(2)}% · Refine ${(100*r.methods.refine_v1).toFixed(2)}%`;$('label').textContent='AB forearm: '+(r.label===null?'unlabeled':r.label);$('replay').src='cases/'+r.case+'/'+$('method').value+'.html'}$('case').value='COSMOS3_0010';$('case').onchange=select;$('method').onchange=select;select();
</script></body></html>'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    build(parser.parse_args().root)
