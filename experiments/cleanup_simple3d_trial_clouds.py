"""Remove only the superseded strict-crop trial XYZ and embedded replay clouds."""
import argparse
import hashlib
import json
from pathlib import Path


def cleanup(root):
    root=Path(root).resolve()
    assert root.name=='link5_pair_visible' and root.parent.name=='simple3d-20261003-run2'
    marker=root/'metadata/trial_clouds_removed.json'
    if marker.exists():
        return json.loads(marker.read_text())
    config=json.loads((root/'inputs/manifest.json').read_text())['config']
    assert config['erosion_px']==3 and config.get('visibility_mode','strict_hulls')=='strict_hulls'
    assert (root/'metadata/execution/run.exit').read_text().strip()=='0'
    replacement=root.parent/'link5_pair_visible_anchor'
    current=json.loads((replacement/'inputs/manifest.json').read_text())['config']
    assert current['erosion_px']==2 and current['visibility_mode']=='smaller_view_anchor'
    assert len(list(replacement.glob('cases/*/bin_*/*_cloud.npz')))>=4
    clouds=sorted(root.glob('cases/*/bin_*/*_cloud.npz'))
    scripts=sorted(root.glob('replay/cases/*.js'))
    inventory=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in clouds+scripts]
    report=dict(status='superseded_trial_clouds_removed',cloud_files_removed=len(clouds),
        embedded_replay_files_removed=len(scripts),bytes_removed=sum(x['bytes'] for x in inventory),
        removed_files=inventory,preserved='original masks, selections, score arrays/summaries and cloud-count metadata',
        replacement='link5_pair_visible_anchor/replay/index.html')
    # Record the exact authorized inventory before removing its owned files.
    marker.write_text(json.dumps(report,indent=2)+'\n')
    for p in clouds+scripts:
        p.unlink()
    assert not list(root.glob('cases/*/bin_*/*_cloud.npz')) and not list(root.glob('replay/cases/*.js'))
    target='../../link5_pair_visible_anchor/replay/index.html'
    (root/'replay').mkdir(exist_ok=True)
    (root/'replay/index.html').write_text(f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url={target}"><title>Trial clouds removed</title><a href="{target}">Current point-cloud replay</a>')
    (root/'replay/metadata.json').write_text(json.dumps(dict(status=report['status'],replacement=report['replacement']),indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    result=cleanup(p.parse_args().root)
    print(json.dumps({k:v for k,v in result.items() if k!='removed_files'}))
