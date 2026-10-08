"""Read unchanged human labels and correlate complete full-video sums.

Only the user-selected AB forearm column is used. Average ranks handle ties; no missing
scores are imputed and no duration normalization changes the requested score.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from robot.preprocessing.selection.matched_selection import DATASETS


def workbook_rows(path):
    """Read numeric/cached values, inline strings and Excel shared strings."""
    namespace={'x':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with ZipFile(path) as archive:
        strings=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            table=ET.fromstring(archive.read('xl/sharedStrings.xml'))
            strings=[''.join(t.text or '' for t in item.findall('.//x:t',namespace)) for item in table.findall('x:si',namespace)]
        sheet=ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
    def cells(row):
        result={}
        for cell in row.findall('x:c',namespace):
            column=re.match(r'[A-Z]+',cell.attrib['r']).group()
            value=cell.find('x:v',namespace)
            if cell.attrib.get('t')=='s':v=strings[int(value.text)]
            elif cell.attrib.get('t')=='inlineStr':v=''.join(t.text or '' for t in cell.findall('.//x:t',namespace))
            else:v=value.text if value is not None else None
            result[column]=v
        return result
    rows=sheet.findall('.//x:sheetData/x:row',namespace)
    headers=cells(rows[0])
    if headers.get('AB')!='Forearm deformation' or headers.get('C')!='Dataset' or headers.get('A')!='Matched number':
        raise ValueError('expected human forearm annotation at physical column AB')
    for row in rows[1:]:
        values=cells(row)
        if values.get('C') is None:continue
        yield int(row.attrib['r']),{headers[k]:v for k,v in values.items() if k in headers}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2 + 1
        start = end
    return result


def pearson(x, y):
    if len(x) < 3:
        return None
    mx, my = sum(x)/len(x), sum(y)/len(y)
    dx, dy = [v-mx for v in x], [v-my for v in y]
    scale = math.sqrt(sum(v*v for v in dx)*sum(v*v for v in dy))
    return sum(a*b for a,b in zip(dx,dy))/scale if scale else None


def statistics(rows, label, mode):
    pairs = [(r[label], r[mode]) for r in rows if isinstance(r[label], (int,float)) and r[mode] is not None]
    x, y = [p[0] for p in pairs], [p[1] for p in pairs]
    return dict(n=len(pairs), selected_n=len(rows), pearson_r=pearson(x,y),
                spearman_rho=pearson(ranks(x),ranks(y)), label_counts=dict(Counter(map(str,x))))


def run(root, workbook):
    full = root/'full_video'
    config = json.loads((full/'config.json').read_text())
    completion_path=full/'completion.json'
    if not completion_path.exists():
        raise RuntimeError('user requested AB correlation only after the selected evaluation finishes')
    scope_path = full/'metadata/evaluation_scope.json'
    scope = json.loads(scope_path.read_text()) if scope_path.exists() else None
    if scope and scope['status']!='complete':
        raise RuntimeError('selected full-video cohort has not finished')
    selected = scope['selected_video_ids'] if scope else [c['id'] for c in config['cases']]
    labels = {}
    for row, cells in workbook_rows(workbook):
        dataset = DATASETS[cells['Dataset']]
        number = str(int(cells['Matched number'])).zfill(4)
        video = dataset+'_'+number
        if video in labels:
            raise ValueError('duplicate workbook video')
        def value(name):
            v = cells[name]
            return None if v in (None,'') else float(v)
        labels[video] = dict(workbook_row=row, forearm_AB=value('Forearm deformation'))
    if set(labels) != {c['id'] for c in config['cases']}:
        raise ValueError('workbook and frozen 45-video launch do not match')
    split = json.loads((root/'splits/train20_test25.json').read_text())
    train = {r['video_id'] for r in split['train']}
    modes = ('paper_original','robot_structural')
    rows, hashes = [], {}
    for video in selected:
        row = dict(video_id=video, generator=video.rsplit('_',1)[0], number=video.rsplit('_',1)[1],
                   reference_split='train' if video in train else 'heldout', **labels[video])
        for mode in modes:
            path = full/'cases'/video/f'video_{mode}.json'
            summary = json.loads(path.read_text()) if path.exists() else None
            row[mode] = None
            row[mode+'_frames'] = summary['scored_frame_count'] if summary else 0
            if summary:
                hashes[str(path.relative_to(root))] = sha(path)
                row['expected_frames'] = summary['expected_frame_count']
                if summary['sum_is_complete']:
                    frames = json.loads((path.parent/f'frames_{mode}.json').read_text())
                    if sorted(r['frame_id'] for r in frames) != list(range(summary['expected_frame_count'])):
                        raise ValueError('full video contains missing or duplicate frames')
                    measured = sum(r['raw_mean_top80'] for r in frames if r['status']=='complete')
                    if not math.isclose(measured,summary['sum_all_frame_raw_mean_top80'],abs_tol=1e-12,rel_tol=1e-12):
                        raise ValueError('video sum differs from saved frame scores')
                    row[mode] = summary['sum_all_frame_raw_mean_top80']
        rows.append(row)
    groups = {'all':rows, 'heldout':[r for r in rows if r['reference_split']=='heldout']}
    groups.update({g:[r for r in rows if r['generator']==g] for g in DATASETS.values()})
    results = {label:{mode:{group:statistics(items,label,mode) for group,items in groups.items()} for mode in modes}
               for label in ('forearm_AB',)}
    payload = dict(status='complete' if all(r[m] is not None for r in rows for m in modes) else 'partial',
        score='sum_all_frame_raw_mean_top80', include_frame0=True, selected_videos=len(rows),
        label_source=dict(path=str(workbook),sha256=sha(workbook),sheet='Selected 45',
             columns={'forearm_AB':'AB2:AB46'}),
        column_interpretation={'forearm_AB':'Recorded numeric labels 0, 0.5, 1; established primary forearm annotation.'},
        statistics=results, rows=rows, video_summary_sha256=hashes,
        notes=['Only complete video sums enter correlations; no zero imputation.',
             'Pearson uses numeric values as recorded. Spearman uses average ranks for tied labels and scores.',
             'Unnormalized sums depend on frame count; generator-specific correlations are reported separately.',
             'All-video analysis includes videos supplying training frame0; heldout analysis is separate.',
             'Videos share ten matched tasks. Correlations are descriptive; no independent-sample significance claim.',
             'Both checkpoints previously failed synthetic deformation sensitivity; results remain exploratory.'])
    destination = full/'analysis'; destination.mkdir(exist_ok=True)
    (destination/'forearm_correlation.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (destination/'forearm_labels_scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    lines=['# Link5 anomaly sums versus human forearm labels','',
        f'Status: **{payload["status"]}**. Selected cohort: {len(rows)} full videos.',
        'Score: sum of every frame’s raw top-80 point anomaly score, including frame0.','',
        '| Label column | Detector | Group | Complete videos | Pearson r | Spearman rho |',
        '|---|---|---|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.6f}'
    for label, modes_result in results.items():
        for mode, grouped in modes_result.items():
            for group, result in grouped.items():
                lines.append(f'| {label} | {mode} | {group} | {result["n"]}/{result["selected_n"]} | {fmt(result["pearson_r"])} | {fmt(result["spearman_rho"])} |')
    lines += ['','Only workbook column AB is used, as explicitly requested by the user.',
              '',*payload['notes'],'','[Exact results](forearm_correlation.json) · [Case-level labels and scores](forearm_labels_scores.csv)','']
    (destination/'forearm_correlation.md').write_text('\n'.join(lines))
    replay=full/'replay';replay.mkdir(exist_ok=True)
    temp=replay/'labels.tmp';temp.write_text('window.LINK5_LABELS='+json.dumps(payload,separators=(',',':'),allow_nan=False)+';\n');temp.replace(replay/'labels.js')
    print('LINK5_FOREARM_CORRELATIONS',payload['status'],{m:results['forearm_AB'][m]['all'] for m in modes},flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--workbook',type=Path,required=True)
    args=parser.parse_args();run(args.root,args.workbook)
