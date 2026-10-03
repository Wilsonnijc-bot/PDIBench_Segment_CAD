#!/usr/bin/env python3
"""Publish normalized crop pictures and verified scores; preserve mask replays."""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def publish(run: Path):
    run = run.resolve()
    out = ROOT / 'docs'
    stats = json.loads((run / 'metadata/OBJECT_ANOMALY_STATISTICS.json').read_text())
    comparison = json.loads((run / 'metadata/comparison.json').read_text())
    assert comparison['settings_equal'] and comparison['selected_frames_equal']
    assert comparison['pair_count'] == 380
    before = (out / 'objects/selection_gallery.html').read_text()
    old_page = BeautifulSoup(before, 'html.parser')
    fresh = BeautifulSoup((run / 'crops/selection_gallery.html').read_text(), 'html.parser')
    packs_path = out / 'objects/image-pack-manifest.json'
    packs = json.loads(packs_path.read_text())
    original_packs = dict(packs.get('original_image_packs', packs['image_packs']))
    manifest_path = out / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    old_manifest = json.loads(manifest_path.read_text())
    replacements = {}
    registered = {}
    def register(path):
        path = path.resolve()
        raw = path.read_bytes()
        filename = sha(raw) + path.suffix
        (out / 'assets' / filename).write_bytes(raw)
        url = 'assets/' + filename
        key = path.relative_to(ROOT).as_posix()
        manifest['files'][key] = {'sha256': sha(raw), 'bytes': len(raw), 'url': url}
        registered[key] = manifest['files'][key]
        return url
    for node in fresh.select('section.case'):
        name = node.h2.get_text(strip=True)
        source = old_page.select_one('section.case[data-name="' + name.lower() + '"]')
        assert source is not None
        # Keep current repaired-mask and occlusion replay links exactly as published.
        for paragraph in source.select(':scope > p.muted'):
            if any('replay.html' in link.get('href', '') for link in paragraph.select('a')):
                node.select_one('.case-head').insert_after(BeautifulSoup(str(paragraph), 'html.parser'))
        images = {}
        for element in node.select('[src],[href]'):
            for attribute in ('src', 'href'):
                value = element.get(attribute)
                if not value or value.startswith(('http:', 'https:', '#', 'data:', 'replay.html?')):
                    continue
                path = run / 'crops' / value
                if path.suffix in {'.png', '.jpg'}:
                    # Full original frames reuse their existing public packs and
                    # are fetched only when clicked. Avoid duplicating videos'
                    # unchanged full-frame PNGs in normalized crop packs.
                    if path.name == 'original_frame.png':
                        element['data-image-source'] = 'original'
                    else:
                        images[value] = base64.b64encode(path.read_bytes()).decode()
                    element['data-image-path'] = value
                    element['data-image-case'] = name
                    element[attribute] = ('data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'
                                          if attribute == 'src' else '#')
                else:
                    element[attribute] = register(path)
        if images:
            compressed = gzip.compress(json.dumps(images, separators=(',', ':')).encode(), mtime=0)
            filename = sha(compressed) + '.json.gz'
            (ROOT / 'web-assets/object-image-packs' / filename).write_bytes(compressed)
            packs['image_packs'][name] = ('https://raw.githubusercontent.com/Wilsonnijc-bot/PDIBench_Segment_CAD/main/'
                                         'web-assets/object-image-packs/' + filename)
        replacements[name.lower()] = str(node)
    def replace_section(match):
        name = BeautifulSoup(match.group(0), 'html.parser').section['data-name']
        return replacements[name]
    gallery, count = re.subn(r'<section\b[^>]*class="case"[^>]*>.*?</section>', replace_section, before, flags=re.S)
    assert count == 45
    gallery, count = re.subn(r'const imagePacks=\{.*?\};', 'const imagePacks=' + json.dumps(packs['image_packs']) + ';', gallery, count=1, flags=re.S)
    assert count == 1
    loader = '''const originalImagePacks=ORIGINAL_PACKS;
const imageJobs=new Map();
async function loadImages(caseName,original=false){
 const key=caseName+(original?'@original':'');
 if(imageJobs.has(key))return imageJobs.get(key);
 const packUrl=(original?originalImagePacks:imagePacks)[caseName];
 const job=(async()=>{
  const r=await fetch(packUrl);if(!r.ok)throw Error('Image pack download failed');
  const compressed=await r.arrayBuffer();
  const digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',compressed))).map(b=>b.toString(16).padStart(2,'0')).join('');
  if(!packUrl.endsWith(digest+'.json.gz'))throw Error('Image pack integrity check failed');
  const data=await new Response(new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'))).json();
  for(const el of document.querySelectorAll('[data-image-case]'))if(el.dataset.imageCase===caseName&&(el.dataset.imageSource==='original')===original){
   const raw=atob(data[el.dataset.imagePath]);const bytes=Uint8Array.from(raw,c=>c.charCodeAt(0));
   const url=URL.createObjectURL(new Blob([bytes],{type:el.dataset.imagePath.endsWith('.jpg')?'image/jpeg':'image/png'}));
   el.setAttribute(el.tagName==='IMG'?'src':'href',url);
  }
 })();imageJobs.set(key,job);
 try{return await job;}catch(err){imageJobs.delete(key);throw err;}
}
const imageObserver=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){loadImages(e.target.dataset.imageCase,e.target.dataset.imageSource==='original').catch(err=>{e.target.alt=err.message});imageObserver.unobserve(e.target);}},{rootMargin:'600px'});
document.querySelectorAll('img[data-image-case]').forEach(el=>imageObserver.observe(el));
document.addEventListener('click',async event=>{
 const link=event.target.closest('a[data-image-path]');
 if(link&&link.getAttribute('href')==='#'){
  event.preventDefault();const popup=window.open('','_blank');
  try{await loadImages(link.dataset.imageCase,link.dataset.imageSource==='original');if(popup){popup.opener=null;popup.location.href=link.href;}else{link.click();}}
  catch(err){if(popup)popup.close();link.title=err.message;}
 }
});'''.replace('ORIGINAL_PACKS', json.dumps(original_packs))
    gallery, count = re.subn(r'(?:const originalImagePacks=\{.*?\};\s*)?const imageJobs=new Map\(\);.*?</script>', lambda match: loader+'\n</script>', gallery, count=1, flags=re.S)
    assert count == 1
    old_summary = BeautifulSoup(gallery, 'html.parser').select_one('section[aria-label="Object anomaly AUROC"]')
    summary = BeautifulSoup(str(old_summary), 'html.parser').section
    summary.h2.string = f'Detection AUROC · {stats["binary_auroc"]:.3f}'
    summary.find('p').string = (f'AnomalyDINO · sum of ten unrounded pair scores per video, using matched crop canvases. '
                               f'Revised object-deformation labels: v2 workbook column AF. '
                               f'{stats["positive_count"]} deformed versus {stats["negative_count"]} non-deformed videos; '
                               f'{stats["ambiguous_scored_count"]} moderate label excluded. '
                               f'{stats["unscored_count"]} videos have no score.')
    for previous_note in summary.select('a[href="analysis/OBJECT_CROP_NORMALIZATION.html"]'):
        previous_note.find_parent('p').decompose()
    note = BeautifulSoup('<p>Matched crop canvases preserve visible pixels and shapes while removing unequal transparent margins. '
                         f'Previous AUROC {comparison["baseline"]["binary_auroc"]:.3f} → current {stats["binary_auroc"]:.3f}. '
                         '<a href="analysis/OBJECT_CROP_NORMALIZATION.html" target="_blank">Before/after score comparison</a></p>', 'html.parser')
    summary.append(note.p)
    assert gallery.count(str(old_summary)) == 1
    gallery = gallery.replace(str(old_summary), str(summary), 1)
    # Replace download links while retaining the existing image loader and layout.
    for filename, label in (('anomalydino_scores.json', 'Scores'),
                            ('anomalydino_videos.csv', 'Video totals'),
                            ('selection_index.json', 'Selection index')):
        url = register(run / 'crops' / filename)
        gallery, count = re.subn(r'<a href="[^"]+">' + label + r'</a>', f'<a href="{url}">{label}</a>', gallery, count=1)
        assert count == 1
    documentation_url = register(run / 'crops/README.md')
    gallery, count = re.subn(r'<a href="[^"]+">Documentation</a>',
                            f'<a href="{documentation_url}">Documentation</a>', gallery, count=1)
    assert count == 1
    result_page = BeautifulSoup(gallery, 'html.parser')
    assert not result_page.select('img.original-frame')
    assert len(result_page.select('a[data-image-path$="/original_frame.png"]')) == 440
    for old, new in zip(old_page.select('section.case'), result_page.select('section.case')):
        assert old['data-name'] == new['data-name']
        assert [r.get_text() for r in old.select('.frame-title')] == [r.get_text() for r in new.select('.frame-title')]
    (out / 'objects/selection_gallery.html').write_text(gallery)
    packs['source_gallery'] = (run / 'crops/selection_gallery.html').relative_to(ROOT).as_posix()
    packs['crop_geometry'] = comparison['method']
    packs['original_image_packs'] = original_packs
    packs_path.write_text(json.dumps(packs, indent=2) + '\n')
    analysis = out / 'analysis'
    for filename in ('OBJECT_ANOMALY_REPORT.md', 'OBJECT_ANOMALY_STATISTICS.json'):
        shutil.copy2(run / 'metadata' / filename, analysis / filename)
    shutil.copy2(run / 'metadata/comparison.json', analysis / 'OBJECT_CROP_NORMALIZATION.json')
    report = (run / 'index.html').read_text()
    report = report.replace('crops/selection_gallery.html', '../objects/selection_gallery.html?v=' + run.name)
    report = report.replace('metadata/OBJECT_ANOMALY_REPORT.md', 'OBJECT_ANOMALY_REPORT.md')
    report = report.replace('metadata/comparison.json', 'OBJECT_CROP_NORMALIZATION.json')
    (analysis / 'OBJECT_CROP_NORMALIZATION.html').write_text(report)
    index_path = out / 'index.html'
    index = index_path.read_text()
    index, count = re.subn(r'src="objects/selection_gallery.html(?:\?[^\"]*)?"',
                          f'src="objects/selection_gallery.html?v={run.name}"', index, count=1)
    assert count == 1
    heading = '<h2>Object reference crops and anomaly scores</h2>'
    assert index.count(heading) == 1
    note = '<p>Current crops use matched canvases with exact visible pixels preserved. <a href="analysis/OBJECT_CROP_NORMALIZATION.html">Compare AnomalyDINO before and after normalization</a>.</p>'
    index = index.replace(note, '')
    index = index.replace(heading, heading + note)
    # Mask/occlusion/video outputs and every earlier manifest entry stay intact.
    old_occlusion = BeautifulSoup(index_path.read_text(), 'html.parser').select_one('#object-occlusion')
    assert str(old_occlusion) == str(BeautifulSoup(index, 'html.parser').select_one('#object-occlusion'))
    index_path.write_text(index)
    for key, value in old_manifest['files'].items():
        assert manifest['files'][key] == value
    manifest['source_indexes'][(run / 'crops/selection_gallery.html').relative_to(ROOT).as_posix()] = sha((run / 'crops/selection_gallery.html').read_bytes())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    writeout = {'case_count': 45, 'normalized_pairs': 440, 'scored_pairs': 380,
                'frame_titles_unchanged': True, 'occlusion_section_unchanged': True,
                'prior_manifest_entries_unchanged': True, 'registered': registered,
                'image_packs': packs['image_packs'], 'original_image_packs': original_packs}
    (run / 'metadata/publication.json').write_text(json.dumps(writeout, indent=2) + '\n')
    print(json.dumps({k: v for k, v in writeout.items() if k not in {'registered', 'image_packs', 'original_image_packs'}}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    publish(parser.parse_args().run)
