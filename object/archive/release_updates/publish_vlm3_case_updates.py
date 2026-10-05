#!/usr/bin/env python3
"""Surgically replace requested object crops and occlusion evidence on Pages."""
from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/publish_vlm3_case_updates.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break

import argparse
import base64
import gzip
import hashlib
import html
import json
import re
import shutil
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = _SOURCE_PATH.parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publish(run: Path, cases: list[str]):
    run = run.resolve()
    out = ROOT/'docs'
    metadata = run/'metadata/publication'
    metadata.mkdir(parents=True, exist_ok=True)
    media = metadata/'release-assets'
    media.mkdir(exist_ok=True)
    release = 'https://github.com/Wilsonnijc-bot/PDIBench_Segment_CAD/releases/download/'+run.name+'/'
    manifest_path = out/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    existing_manifest = json.loads(manifest_path.read_text())
    records = manifest['files']
    by_hash = {(v['sha256'], Path(k).suffix): v['url'] for k,v in records.items()}
    registered = set()
    queue = []
    def register(path):
        path = Path(path).resolve()
        key = path.relative_to(ROOT).as_posix()
        if key in registered:
            return records[key]['url']
        raw = path.read_bytes(); digest = sha(raw); suffix = path.suffix
        name = digest+suffix
        url = by_hash.get((digest,suffix))
        if url is None:
            if suffix == '.html':
                url = 'replay.html?asset='+name+'.gz'
                (out/'assets'/(name+'.gz')).write_bytes(gzip.compress(raw,mtime=0))
            elif suffix in {'.mp4','.png','.jpg'}:
                url = release+name
                shutil.copy2(path,media/name)
            else:
                url = 'assets/'+name
                shutil.copy2(path,out/'assets'/name)
            by_hash[digest,suffix] = url
        records[key] = dict(sha256=digest,bytes=len(raw),url=url)
        registered.add(key)
        if suffix == '.html':
            queue.append(path)
        return url

    # Preserve the exact bytes of every unrelated gallery section.
    gallery_path = out/'objects/selection_gallery.html'
    original_gallery = gallery_path.read_text()
    old_page = BeautifulSoup(original_gallery,'html.parser')
    old_sections = {el['data-name']:str(el) for el in old_page.select('section.case')}
    fresh_page = BeautifulSoup((run/'crops/selection_gallery.html').read_text(),'html.parser')
    pack_manifest_path = out/'objects/image-pack-manifest.json'
    packs = json.loads(pack_manifest_path.read_text())
    old_packs = dict(packs['image_packs'])
    replacements = {}
    for case in cases:
        node = fresh_page.select_one('section.case[data-name="'+case.lower()+'"]')
        if node is None:
            raise ValueError('Missing refreshed crop section: '+case)
        # Keep original frames one click away, including older saved galleries.
        for image in node.select('img.original-frame'):
            anchor = image.find_parent('a')
            if anchor is None:
                raise ValueError('Original frame image is missing its link: '+case)
            anchor.clear()
            anchor.string = 'Original frame'
        # Current mask and occlusion reviews are available from the crop row too.
        links = fresh_page.new_tag('p',attrs={'class':'muted'})
        for label, source in [('VLM3 mask comparison',run/'index.html'),
                              ('Synchronized occlusion replay',run/'inputs/object/cases'/case/'occlusion/replay.html')]:
            anchor = fresh_page.new_tag('a',href=register(source)+(('#'+case) if source.name=='index.html' else ''))
            anchor.string = label
            if links.contents:links.append(' · ')
            links.append(anchor)
        node.select_one('.case-head').insert_after(links)
        images = {}
        for el in node.select('[src],[href]'):
            for attr in ('src','href'):
                val = el.get(attr)
                if not val or val.startswith(('http:','https:','#','data:','replay.html?')):
                    continue
                path = run/'crops'/val
                if path.suffix in {'.png','.jpg'}:
                    images[val] = base64.b64encode(path.read_bytes()).decode()
                    el['data-image-path']=val;el['data-image-case']=case
                    el[attr]='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' if attr=='src' else '#'
                else:
                    el[attr]=register(path)
        compressed=gzip.compress(json.dumps(images,separators=(',',':')).encode(),mtime=0)
        name=sha(compressed)+'.json.gz'
        (ROOT/'web-assets/object-image-packs'/name).write_bytes(compressed)
        packs['image_packs'][case]='https://raw.githubusercontent.com/Wilsonnijc-bot/PDIBench_Segment_CAD/main/web-assets/object-image-packs/'+name
        replacements[case.lower()]=str(node)
    def replace_section(match):
        raw=match.group(0);node=BeautifulSoup(raw,'html.parser').section
        return replacements.get(node['data-name'],raw)
    gallery=re.sub(r'<section\b[^>]*class="case"[^>]*>.*?</section>',replace_section,original_gallery,flags=re.S)
    gallery,n=re.subn(r'const imagePacks=\{.*?\};', 'const imagePacks='+json.dumps(packs['image_packs'])+';',gallery,count=1,flags=re.S)
    assert n==1
    # Statistics are recomputed from old unchanged scores plus refreshed cases.
    stats=json.loads((metadata/'OBJECT_ANOMALY_STATISTICS.json').read_text())
    for name in ['OBJECT_ANOMALY_STATISTICS.json','OBJECT_ANOMALY_REPORT.md']:
        shutil.copy2(metadata/name,out/'analysis'/name)
    header=BeautifulSoup(gallery,'html.parser').select_one('section[aria-label="Object anomaly AUROC"]')
    if header:
        new_header=BeautifulSoup(str(header),'html.parser').section
        new_header.h2.string=f'Detection AUROC · {stats["binary_auroc"]:.3f}'
        new_header.find('p').string=(f'AnomalyDINO · sum of ten unrounded pair scores per video. Revised object-deformation labels: v2 workbook column AF. '
            f'{stats["positive_count"]} deformed versus {stats["negative_count"]} non-deformed videos; '
            f'{stats["ambiguous_scored_count"]} moderate label excluded. '
            f'{stats["unscored_count"]} videos have no score; {stats["partial_count"]} partial crop selections are omitted from this ten-pair AUROC.')
        gallery=gallery.replace(str(header),str(new_header))
    score_export=json.loads((metadata/'merged_anomalydino_scores.json').read_text())
    gallery=re.sub(r'<p>\d+ scored pairs · \d+ scored videos · Higher scores mean more anomalous\.</p>',
        f'<p>{len(score_export["pairs"])} scored pairs · {sum(v["status"]=="scored" for v in score_export["videos"])} scored videos · Higher scores mean more anomalous.</p>',gallery)
    current=BeautifulSoup(gallery,'html.parser')
    new_sections={el['data-name']:str(el) for el in current.select('section.case')}
    assert set(old_sections)==set(new_sections)
    for name in old_sections.keys()-replacements.keys():
        assert old_sections[name]==new_sections[name], 'Unrelated gallery case changed: '+name
    for case,url in old_packs.items():
        if case not in cases:assert packs['image_packs'][case]==url
    gallery_path.write_text('\n'.join(line.rstrip() for line in gallery.splitlines())+'\n')
    packs['case_updates']={'run':run.name,'cases':cases,'frame_selection_method':'time-quintiles-with-post-occlusion-v5'}
    pack_manifest_path.write_text(json.dumps(packs,indent=2)+'\n')

    index_path=out/'index.html';index=index_path.read_text()
    main=BeautifulSoup(index,'html.parser');table=main.select_one('#object-occlusion table')
    old_table=str(table)
    untouched_rows={row.get_text(' ',strip=True):str(row) for row in table.select('tbody tr')
                    if not any(case in row.get_text() for case in cases)}
    summary=[]
    for case in cases:
        repair=json.loads((run/case/'repair.json').read_text())
        detection=json.loads((run/'inputs/object/cases'/case/'occlusion/detection.json').read_text())
        selection=json.loads((run/'crops'/case/'selection.json').read_text())
        for row in table.select('tbody tr'):
            if row.find('td') and row.find('td').get_text(strip=True)==case:row.decompose()
        url=register(run/'inputs/object/cases'/case/'occlusion/replay.html')
        intervals=', '.join(f'{a}–{b}' for a,b in detection['flagged_intervals']) or 'None'
        row=BeautifulSoup(f'<tr><td><a href="{html.escape(url)}">{case}</a></td><td>{detection["flagged_frames"]} / {detection["frame_count"]}</td><td>{intervals}</td><td>{len(detection["failed_mask_frames"])}</td></tr>','html.parser').tr
        table.tbody.append(row)
        summary.append(dict(case=case,vlm3_frame=repair['gate']['frame'],vlm3_status=repair['status'],
            accepted=repair['accepted'],selected_frames=[r['frame'] for r in selection['selected_frames']],
            crop_count=selection['selected_count'],occlusion_intervals=detection['flagged_intervals'],
            final_interval_policy=selection['final_interval_policy']))
    for row in table.select('tbody tr'):
        key=row.get_text(' ',strip=True)
        if key in untouched_rows:assert str(row)==untouched_rows[key]
    assert old_table in index
    index=index.replace(old_table,str(table))
    review_url=register(run/'index.html')
    update_html='<section id="vlm3-object-update" class="review-section"><h2>VLM3 mask repairs · seven cases</h2><p>Accepted link7 masks now feed occlusion detection and replay, available-pixel crops, and ten-frame selection together. Original 0/2/2/2/4 quotas and earlier-interval fallback remain. If the final-interval mean falls below half of both the pooled first-80% mean and the preceding interval mean, only individual late frames below half of the first-80% mean are excluded; substantial late recoveries remain.</p>'
    update_html+=f'<p><a href="{review_url}">Exact VLM3 input pictures, output points and masking replays</a> · <a href="#object-reference">Updated crop pictures</a> · <a href="#object-occlusion">Updated occlusion replays</a></p><div class="table-scroll"><table><thead><tr><th>Case</th><th>VLM3 input frame</th><th>Result</th><th>Selected source frames</th></tr></thead><tbody>'
    for row in summary:
        update_html+=f'<tr><td>{row["case"]}</td><td>{row["vlm3_frame"]}</td><td>{html.escape(row["vlm3_status"])}</td><td>{", ".join(map(str,row["selected_frames"]))}</td></tr>'
    update_html+='</tbody></table></div></section>'
    old_update=main.select_one('#vlm3-object-update')
    if old_update:
        index,n=re.subn(r'<section\b(?=[^>]*\bid="vlm3-object-update")[^>]*>.*?</section>',update_html,index,count=1,flags=re.S)
        assert n==1
    else:index=index.replace('<section class="review-section" id="object-reference">',update_html+'<section class="review-section" id="object-reference">') if '<section class="review-section" id="object-reference">' in index else index.replace('<section id="object-reference"',update_html+'<section id="object-reference"')
    assert 'id="vlm3-object-update"' in index
    index=index.replace('src="objects/selection_gallery.html"',f'src="objects/selection_gallery.html?v={run.name}"')
    index_path.write_text(index)

    # Register every relative dependency needed by the exact HTML replay loader.
    processed=set()
    while queue:
        source=queue.pop(0)
        if source in processed:continue
        processed.add(source)
        page=BeautifulSoup(source.read_text(),'html.parser')
        for el in page.select('[src],[href]'):
            for attr in ('src','href'):
                value=el.get(attr)
                if not value or value.startswith(('http:','https:','#','data:')):continue
                path=source.parent/value.split('#',1)[0]
                if path.is_file():register(path)
                else:raise FileNotFoundError(f'Broken replay dependency {source}: {value}')
    # Global score downloads use the merged current dataset, while old cases retain their bytes.
    score_url=register(metadata/'merged_anomalydino_scores.json')
    old_score=existing_manifest['files'].get('results/object-reference-crops-20261001/anomalydino_scores.json')
    if old_score:
        gallery_path.write_text(gallery_path.read_text().replace(old_score['url'],score_url))
    run_prefix=run.relative_to(ROOT).as_posix()+'/'
    for key,value in existing_manifest['files'].items():
        if not key.startswith(run_prefix):
            assert records[key]==value, 'Unrelated manifest entry changed: '+key
    for source_key,new_file in [
        ('results/object-reference-crops-20261001/anomalydino_videos.csv','merged_anomalydino_videos.csv'),
        ('results/object-reference-crops-20261001/selection_index.json','merged_selection_index.json')]:
        previous=existing_manifest['files'].get(source_key)
        if previous:
            gallery_path.write_text(gallery_path.read_text().replace(previous['url'],register(metadata/new_file)))
    manifest.setdefault('case_updates',{})[run.name]=dict(cases=cases,summary=summary,
        unchanged_crop_cases=len(old_sections)-len(cases),release_tag=run.name)
    manifest['source_indexes'][str((run/'crops/selection_gallery.html').relative_to(ROOT))]=sha((run/'crops/selection_gallery.html').read_bytes())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    (metadata/'published_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (metadata/'publication_validation.json').write_text(json.dumps(dict(
        status='passed',unchanged_crop_cases=len(old_sections)-len(cases),
        unchanged_occlusion_rows=len(untouched_rows),registered_files=len(registered),
        release_assets=len(list(media.iterdir())),cases=cases),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--cases',nargs='+',required=True)
    args=p.parse_args();publish(args.run,args.cases)
