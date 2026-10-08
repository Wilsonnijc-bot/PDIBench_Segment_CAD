"""Freeze eligible mask-cache cases without modifying the original manifest."""
import argparse,json,hashlib
from pathlib import Path
EXCLUDED={
 'LVP_ROBOWM_0010':'User rejected mask',
 'LVP_ROBOWM_0030':'User rejected mask',
 'COSMOS3_0021':'User rejected mask',
 'COSMOS3_0035':'User rejected mask',
 'COSMOS2.5_0010':'Excluded by current HANDOFF.md',
}
def build(cache):
    source=cache/'mask_cache_manifest.json';m=json.loads(source.read_text())
    found={e['video_id'] for e in m['entries']}
    assert set(EXCLUDED)<=found
    included=[dict(e,cohort='additional' if e['video_id']=='COSMOS2.5_0018' else 'selected45') for e in m['entries'] if e['video_id'] not in EXCLUDED]
    return dict(source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),excluded=EXCLUDED,primary_count=sum(e['cohort']=='selected45' for e in included),additional_count=sum(e['cohort']=='additional' for e in included),entries=included,methods=['balanced_v0','refine_v1'],gpu_count=1,parallel_video_workers=5)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(build(a.cache),indent=2))
