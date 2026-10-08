"""Correlate completed rigidity scores with physical workbook column AB."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from robot.experiments.link5_shape_codebook.label_correlation import workbook_rows, sha, pearson, ranks
from robot.preprocessing.selection.matched_selection import DATASETS


def run(root, workbook):
    manifest = json.loads((root / 'manifest.json').read_text())
    scores = json.loads((root / 'rigidity/summary.json').read_text())
    methods = ('balanced_v0', 'refine_v1')
    by_key = {(r['case'], r['method']): r for r in scores}
    if len(by_key) != len(scores):
        raise ValueError('Duplicate score rows')
    labels = {}
    for row, cells in workbook_rows(workbook):
        video = DATASETS[cells['Dataset']] + '_' + str(int(cells['Matched number'])).zfill(4)
        value = float(cells['Forearm deformation'])
        if value not in (0, .5, 1) or video in labels:
            raise ValueError('Unexpected or duplicate AB label')
        labels[video] = dict(workbook_row=row, forearm_AB=value)
    rows = []
    for entry in manifest['entries']:
        video = entry['video_id']
        item = dict(video_id=video, generator=video.rsplit('_', 1)[0], cohort=entry['cohort'],
                    **labels.get(video, dict(workbook_row=None, forearm_AB=None)))
        for method in methods:
            score = by_key[(video, method)]
            if score['status'] != 'complete' or score['depth_filter'] != 'refined_all_frames':
                raise ValueError('Expected complete all-frame-filtered scores')
            item[method] = score['rigidity_score']
        rows.append(item)
    policies = {
        'exact_0_1': lambda v: v if v in (0, 1) else None,
        'half_as_1': lambda v: int(v > 0),
        'half_as_0': lambda v: int(v == 1),
        'recorded_0_half_1': lambda v: v,
    }
    groups = {'all_labeled': rows}
    groups.update({g: [r for r in rows if r['generator'] == g] for g in DATASETS.values()})
    results = {}
    for name, policy in policies.items():
        results[name] = {}
        for group, items in groups.items():
            valid = [(r, policy(r['forearm_AB'])) for r in items if r['forearm_AB'] is not None]
            valid = [(r, v) for r, v in valid if v is not None]
            x = [v for _, v in valid]
            measured = {}
            for method in methods:
                y = [r[method] for r, _ in valid]
                pr, sr = pearson(x, y), pearson(ranks(x), ranks(y))
                if pr is not None:
                    if not np.isclose(pr, pearsonr(x, y).statistic, atol=1e-12):
                        raise ValueError('Independent Pearson check failed')
                    if not np.isclose(sr, spearmanr(x, y).statistic, atol=1e-12):
                        raise ValueError('Independent Spearman check failed')
                measured[method] = dict(n=len(x), label_counts=dict(Counter(map(str, x))),
                                        pearson_r=pr, spearman_rho=sr)
                auc = None
                if set(x) == {0, 1}:
                    positive = [score for label, score in zip(x, y) if label == 1]
                    negative = [score for label, score in zip(x, y) if label == 0]
                    ranked = ranks(y)
                    n_positive, n_negative = len(positive), len(negative)
                    auc = (sum(rank for label, rank in zip(x, ranked) if label == 1)
                           - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
                    independently = sum(float(a > b) + .5 * float(a == b)
                                        for a in positive for b in negative) / (n_positive * n_negative)
                    if not np.isclose(auc, independently, atol=1e-12):
                        raise ValueError('Independent AUROC check failed')
                measured[method]['auroc'] = auc
            results[name][group] = measured
    payload = dict(status='complete', score='frame0-excluded temporal mean of rigidity MAD/median',
        label_source=dict(path=str(workbook.resolve()), sha256=sha(workbook), sheet='Selected 45', column='AB2:AB46'),
        score_source_sha256=sha(root / 'rigidity/summary.json'), labeled_scored_videos=sum(r['forearm_AB'] is not None for r in rows),
        unmatched_scored_videos=[r['video_id'] for r in rows if r['forearm_AB'] is None],
        excluded_workbook_videos=sorted(set(labels)-{r['video_id'] for r in rows}),
        policies=dict(exact_0_1='Exclude 0.5', half_as_1='0.5 -> 1', half_as_0='0.5 -> 0', recorded_0_half_1='Retain recorded 0, 0.5, 1'),
        statistics=results, rows=rows,
        notes=['AB only; no labels used in pair selection or score construction.',
               'Additional COSMOS2.5_0018 has no workbook row and is excluded from correlation.',
               'Spearman uses average ranks for ties. Binary Pearson is point-biserial correlation.',
               'AUROC uses higher rigidity as the positive score, average ranks for ties, and is undefined for the ordinal policy.',
               'Descriptive correlations on selected videos; matched tasks across generators are not independent observations.'])
    folder = root / 'analysis'
    folder.mkdir(exist_ok=True)
    (folder / 'forearm_AB_correlation.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    with (folder / 'forearm_AB_labels_scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Rigidity correlation with AB forearm labels', '',
             '40 scored videos match the unchanged workbook. COSMOS2.5_0018 has no label.', '',
             '| AB policy | Group | Method | n | Pearson r | Spearman rho | AUROC |',
             '|---|---|---|---:|---:|---:|---:|']
    fmt = lambda v: 'undefined' if v is None else f'{v:.6f}'
    for policy, grouped in results.items():
        for group, measured in grouped.items():
            for method, stats in measured.items():
                lines.append(f"| {policy} | {group} | {method} | {stats['n']} | {fmt(stats['pearson_r'])} | {fmt(stats['spearman_rho'])} | {fmt(stats['auroc'])} |")
    lines += ['', *payload['notes'], '', '[Exact results](forearm_AB_correlation.json) · [Labels and scores](forearm_AB_labels_scores.csv)', '']
    (folder / 'forearm_AB_correlation.md').write_text('\n'.join(lines))
    print(json.dumps({p: v['all_labeled'] for p, v in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--workbook', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.workbook)
