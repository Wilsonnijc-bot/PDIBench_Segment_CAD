"""Adapt the existing published rigidity replay layout for anomaly evidence.

Reuse its offline Plotly asset, page structure/style, synchronized video/cloud,
transport and history layout. No pairwise rigidity measurements are fabricated.
"""
from pathlib import Path


def render(workspace):
    source = Path(workspace)/'infrastructure/shared/replay/rigidity_replay.html'
    page = source.read_text().split('<script>\nconst data=__REPLAY_DATA__;')[0]
    replacements = {
        '<title>Rigidity evidence replay</title>':'<title>Link5 deformation replay</title>',
        'Whole-robot cloud · source camera view':'Observed Link5 cloud',
        '<button id="focus-points" type="button">Zoom to pairs</button>':'<button id="reference" type="button">Show normal frame0</button>',
        '<button id="view-mode" type="button">3D orbit</button>':'<button id="view-mode" type="button">Camera view</button>',
        '<video id="video" src="source.mp4" preload="auto" playsinline></video>':'<div class="video-wrap"><video id="video" preload="metadata" playsinline muted></video><canvas id="overlay"></canvas></div>',
        '<label>Window <select id="window"></select></label>':'',
        'Final rigidity score':'Sum of all frame scores',
        'Rigidity through time · click to seek':'Raw anomaly score through time · click to seek',
        'Selected CoTracker pairs · <span id="used-count"></span> · <span id="median"></span>':'Highest raw anomaly points · click to highlight',
    }
    for old,new in replacements.items():
        if old not in page:
            raise ValueError('published replay template changed: '+old[:60])
        page=page.replace(old,new)
    page=page.replace('<small>Current frame score</small>', '<small id="current-label">Previous run · current frame score</small>')
    a=page.index('<div class="pair-list">');b=page.index('</section>',a)
    page=page[:a]+'<div class="pair-list"><div id="points"></div></div>'+page[b:]
    a=page.index('  <p class="legend"><a id="evidence"');
    page=page[:a]+'''<p class="legend" id="detail"></p>
<details><summary>Method and pose details</summary><p class="legend">Saved scores belong to the previous GPU run. The refined depth preview applies two original-image pixels of mask erosion, then a robust camera-footprint-normalized 32-neighbor density check to the saved MegaSAM samples. No Simple3D cropping, depth smoothing, CAD fitting or depth replacement is applied. Coherent wrong masks or depth can remain. Retained XYZ and the fixed display normalization are unchanged. Later previews may use the saved FoundationPose alignment; pose inference, SAM segmentation and codebook scoring have not been rerun. The existing video score is the sum of raw frame top-80 scores, including frame0. Both detectors previously failed synthetic deformation sensitivity.</p></details>
<p class="legend"><a href="../reference_replay/index.html#guard">Frame0 VLM prompts, masks and exact 20 references</a> · <a href="../../refinement/COST_COMPARISON.md">Guard cost comparison</a> · <a href="../../refinement/metadata/depth_summary.json">All-frame depth-filter receipt</a></p>
<section class="analysis"><h2>Human forearm labels · AB</h2><p id="analysis-status">Correlation is calculated after all 30 selected videos finish.</p><div id="correlation-table"></div><div id="correlation-chart"></div><p class="legend">Label 0 = almost none, 0.5 = moderate, 1 = severe. Correlations use full-video sums. Generator-specific results account for different frame counts without changing the requested score. Training-reference videos and held-out videos are reported separately.</p><a href="../analysis/forearm_correlation.md">Correlation report</a> · <a href="../analysis/forearm_labels_scores.csv">Labels and scores CSV</a></section>
<h2>Matched videos</h2><p class="muted">Select a generator’s score to open that video. Numbers match across generators. Sums include every source frame.</p><div class="table-scroll"><table id="matched"></table></div>
</main><script src="manifest.js"></script><script src="../../refinement/replay/manifest.js"></script><script src="labels.js"></script><script src="app.js"></script></body></html>'''
    page=page.replace('<main>','''<main><div class="selectors">
<label>Video number <select id="number"></select></label>
<label>Generator <select id="generator"><option>LVP_ROBOWM</option><option>COSMOS2.5</option><option>COSMOS3</option></select></label>
<label>Detector <select id="mode"><option value="robot_structural">Robot structural augmentation</option><option value="paper_original">Original augmentation</option></select></label>
<label>Depth support <select id="depth"><option value="refined">New depth filter</option><option value="comparison">Kept + removed points</option><option value="original">Previous scored cloud</option></select></label>
<label>3D coordinates <select id="space"><option value="aligned">Frame0 reference pose</option><option value="camera">Observed camera pose</option></select></label>
<label>Video overlay <select id="image"><option value="depth">New depth support</option><option value="heat">Previous raw anomaly</option><option value="mask">Original Link5 mask</option><option value="rgb">Original RGB</option></select></label>
<label>Points <select id="density"><option value="3000">3,000 + highest 80</option><option value="10000">All scored points</option></select></label>
<button id="refresh">Refresh results</button></div><p id="run-status" class="muted"></p>''',1)
    css='''
  .selectors{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-bottom:15px}.selectors label{display:grid;gap:5px;font-size:12px;color:#acbeb4}
  .video-wrap{position:relative;background:#090e0f;height:520px}.video-wrap video{height:100%}#overlay{position:absolute;pointer-events:none;inset:0;width:100%;height:100%;object-fit:contain}
  .analysis{margin:30px 0;border-top:1px solid #3b4845;padding-top:15px}#correlation-chart{height:330px}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:12px;border-bottom:1px solid #34403d;font-variant-numeric:tabular-nums}td button{background:transparent;border:0;text-align:left;color:#f3c77c;padding:0;font:inherit}td small{display:block;color:#acbeb4}.point{padding:6px 4px;border-bottom:1px solid #34403d;cursor:pointer}.point.active{background:#455047}.muted a,a{color:#b9edcc}summary{cursor:pointer}h2{font-size:19px;font-weight:600}#matched tr.active{background:#294036}
  @media(max-width:1050px){.video-wrap{height:430px}}@media(max-width:640px){main{padding:12px}.transport{flex-wrap:wrap}.summary{gap:18px}.summary strong{font-size:23px}#cloud,.video-wrap{height:350px}.selectors select{max-width:100%}}
'''
    return page.replace('</style>',css+'</style>',1)
