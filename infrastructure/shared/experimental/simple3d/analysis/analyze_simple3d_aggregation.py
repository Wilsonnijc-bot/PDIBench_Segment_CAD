"""Compare top-80/160/320 aggregation using saved native Simple3D point scores."""

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = (_workspace_root())
sys.path.insert(0, str(ROOT))
from infrastructure.shared.experimental.simple3d.analysis.analyze_simple3d_correlation import read_labels, metric, sha256, write_csv, write_json

KS = (80, 160, 320)


def analyze(root):
    root = Path(root).resolve()
    records = json.loads((root/'comparisons.json').read_text())
    labels = read_labels((_workspace_root() / 'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx'),
                         (_workspace_root() / 'results/link5-only-selected45-updated-mask-20260929/selection.json'))
    assert len(records) == 450 and all(r['simple3d_status'] == 'complete' for r in records)
    rows = []
    grouped = defaultdict(list)
    hashes = {}
    for r in records:
        pair = root/'cases'/r['video_id']/f"bin_{r['temporal_bin']:02d}"
        path = pair/'point_scores.npy'
        scores = np.load(path, allow_pickle=False)
        assert scores.ndim == 1 and len(scores) >= max(KS) and np.isfinite(scores).all()
        ranked = np.sort(scores.astype(np.float64))[::-1]
        means = {k: float(ranked[:k].mean()) for k in KS}
        assert np.isclose(means[80], r['scalar_anomaly_score'], rtol=2e-6, atol=1e-6)
        assert means[80] >= means[160] >= means[320]
        means[80] = r['scalar_anomaly_score']  # Preserve the upstream scalar bit-for-bit.
        row = dict(video_id=r['video_id'], temporal_bin=r['temporal_bin'],
                   reference_frame_id=r['reference_frame_id'], test_frame_id=r['test_frame_id'],
                   sampled_point_count=len(scores), official_top80_score=r['scalar_anomaly_score'],
                   **{f'top{k}_mean': means[k] for k in KS})
        rows.append(row)
        grouped[r['video_id']].append(row)
        hashes[str(path.relative_to(root))] = sha256(path)
    assert set(grouped) == set(labels) and all(len(x) == 10 for x in grouped.values())
    videos, metrics = [], []
    for k in KS:
        cohort = []
        for video, frames in grouped.items():
            values = [f[f'top{k}_mean'] for f in frames]
            lab = labels[video]
            v = dict(video_id=video, dataset=lab['dataset'], human_label=lab['AB'],
                     workbook_cell='AB'+str(lab['workbook_row']), top_k=k,
                     mean_score=float(np.mean(values)), max_score=float(max(values)),
                     successful_comparisons=10, coverage_status='complete')
            cohort.append(v)
            videos.append(v)
        for aggregation in ('mean_score', 'max_score'):
            for policy in ('ordinal', 'binary'):
                metrics.append(dict(top_k=k, **metric('link5_pair_visible', cohort,
                                   aggregation=aggregation, labels=policy)))
        for dataset in sorted({v['dataset'] for v in cohort}):
            metrics.append(dict(top_k=k, **metric('link5_pair_visible', cohort, dataset=dataset)))
    out = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
               videos=45, frames=450, top_k=list(KS),
               official_aggregation=80, alternative_aggregations=[160, 320],
               method='Mean of the k largest saved native interpolated/smoothed point scores; video mean of ten frame means.',
               labels='Selected 45!AB; ordinal 0/0.5/1; binary excludes 0.5.',
               caveat='Post-hoc aggregation sensitivity on the same labeled cohort; no labels used to change descriptors, clouds, ranks or settings. Not independent validation.',
               source_comparisons_sha256=sha256(root/'comparisons.json'),
               point_score_sha256=hashes, metrics=metrics)
    write_csv(root/'metadata/aggregation_frames.csv', rows)
    write_csv(root/'metadata/aggregation_videos.csv', videos)
    write_csv(root/'metadata/aggregation_metrics.csv', metrics)
    write_json(root/'metadata/aggregation_analysis.json', out)
    report(root, out, rows, videos)
    return out


def report(root, analysis, frames, videos):
    summary = json.loads((root/'metadata/summary.json').read_text())
    diagnostics = json.loads((root/'metadata/analysis_diagnostics.json').read_text())
    baseline = json.loads((root.parent/'correlation/summary.json').read_text())
    def m(k, policy='ordinal', aggregation='mean_score'):
        return next(x for x in analysis['metrics'] if x['top_k']==k and x['dataset']=='all'
                    and x['aggregation']==aggregation and x['label_policy']==policy)
    def bm(experiment, policy='ordinal'):
        return next(x for x in baseline['metrics'] if x['experiment']==experiment and x['dataset']=='all'
                    and x['subset']=='all_scored' and x['aggregation']=='mean_score' and x['label_policy']==policy)
    lines = ['# Simple3D evaluation and aggregation analysis', '',
             'Completed 2026-10-04. All **45/45 Link5 videos and 450/450 pair-specific comparisons** succeeded. '
             'All 450 pair directories were downloaded with SHA256 verification; native cloud/score auditing passed, '
             'the GPU job exited 0 with `BATCH_FINISHED`, and the final replay contains 900 clouds and 450 anomaly maps. '
             'No traceback or CUDA out-of-memory error appears in the execution log.', '',
             '[Open the final interactive replay](link5_pair_visible_anchor/replay/index.html). '
             'Choose a video and temporal bin; inspect original, common-visible, eroded, rejected-depth and final-valid masks, '
             'RGB inputs and both rotatable point clouds. Red marks the selected top-k contributors. '
             'Continuous heat colors and raw score/source-pixel hover remain available.', '',
             '## What the two displayed scores mean', '',
             'For a test cloud, the authors compute local FPFH/MSND/LFSA descriptors, compare them with the '
             'pair-specific normal coreset using nearest-prototype distances, then apply their interpolation and '
             'spatial score smoothing. The resulting continuous score belongs to each sampled test point. '
             '**Higher means less similar local geometry to this reference**, including reconstruction artifacts and '
             'visibility errors; it is not a calibrated deformation probability or physical displacement.', '',
             'With point scores sorted descending as `s[1] >= ... >= s[N]`, the official frame score is '
             '`S80 = (s[1] + ... + s[80]) / 80`. The video mean is the arithmetic mean of its ten frame scores; '
             'every selected frame has equal weight. Each frame uses its own cropped reference and rebuilt coreset.', '']
    example = [x for x in frames if x['video_id']=='COSMOS3_0001' and x['temporal_bin']==0][0]
    lines += [f"The displayed example is **COSMOS3_0001, test frame {example['test_frame_id']}**: "
              f"official top-80 frame mean **{example['top80_mean']:.3f}**, and ten-frame video mean **"
              f"{next(v['mean_score'] for v in videos if v['video_id']=='COSMOS3_0001' and v['top_k']==80):.3f}**. "
              'The video mean is higher because other selected frames score higher.', '',
              '## Top-160 and top-320 sensitivity', '',
              'These alternatives reuse the exact same saved point scores, ranks, reference memories and geometry. '
              'Only the final scalar averaging window changes. **Top-80 remains the official author aggregation; '
              'top-160 and top-320 are our post-hoc alternatives.** No MegaSaM or Simple3D rerun is needed.', '',
              '| Aggregation | Example frame | Example video mean | Pearson r (45 videos) | Spearman rho (45) | Binary AUROC (35) |',
              '|---|---:|---:|---:|---:|---:|']
    for k in KS:
        v = next(v for v in videos if v['video_id']=='COSMOS3_0001' and v['top_k']==k)
        lines.append(f"| Top {k} {'(official)' if k==80 else '(alternative)'} | {example[f'top{k}_mean']:.3f} | "
                     f"{v['mean_score']:.3f} | {m(k)['pearson_r']:.3f} | {m(k)['spearman_rho']:.3f} | {m(k,'binary')['binary_auroc']:.3f} |")
    lines += ['', 'On this cohort, **all three choices have the same binary AUROC (0.717)**. '
              'Top-160 and top-320 do not demonstrate improved discrimination; Pearson correlation falls '
              'slightly and Spearman correlation changes little. Keep top-80 as the official primary result '
              'and inspect the alternatives as aggregation sensitivity.', '',
              'Larger k necessarily lowers or preserves each frame mean, because it includes less extreme '
              'points. It may reduce sensitivity to a tiny patch of bad depth, but it can dilute a small real '
              'deformation. It does not remove noisy points or change the point anomaly map. On an 8192-point '
              'cloud, 80/160/320 cover about 0.98%/1.95%/3.91% of points; smaller clouds cover larger fractions.', '',
              'The label is `selected_45_matched_videos_styledv2.xlsx`, sheet `Selected 45`, column **AB: Forearm '
              'deformation**. All 45 videos are included in ordinal correlations: 13 label-0, 10 label-0.5, '
              '22 label-1. Binary AUROC excludes the ten label-0.5 videos. The statistical unit is the video, '
              'not the 450 dependent frames. These are exploratory comparisons on this cohort, not validation '
              'of a label-selected winning k. No score direction, descriptor or preprocessing was tuned to labels.', '',
              '## Current preprocessing and measured retention', '',
              'The existing Link5 masks, saved tracks and frozen 45-video mapping are reused. The base image is '
              'frame 0 or first valid observation. Ten tests come from ten bins over the later 90% of the video, '
              'choosing the largest usable raw Link5 mask in each bin. The smaller visible mask, adjusted for '
              'image scale, anchors the 2D similarity mapping into the other original mask. Track hulls do not '
              'crop the masks; count/inlier-fraction/hull-quality vetoes are disabled at the user’s request. '
              'This is image mask mapping, with no 3D registration.', '',
              'Pair masks receive **2 px erosion**. A fresh masked, cropped eleven-image MegaSaM bootstrap '
              'sequence supplies depth for a conservative 5x5 local median/MAD spike test. Threshold is '
              '`max(5% of local median, 8 × 1.4826 × local MAD)` with an isolated-neighbor condition. Cleaned '
              'inputs enter a second fresh MegaSaM sequence, whose depth is checked again before masked '
              'back-projection. Both stages report CVD for all 450 pairs. There is no global depth smoothing. '
              'Clouds use isolated-outlier removal, per-cloud author centering/unit-radius normalization, '
              '0.005 normalized voxel sampling and a seeded 8192-point cap. A new author 5% coreset is built '
              'for each pair.', '',
              '| View | Mean common-mask / original area | Mean eroded-mask / original area | Minimum eroded retention |',
              '|---|---:|---:|---:|']
    for role in ('reference','test'):
        d=diagnostics['mask_retention'][role]
        lines.append(f"| {role.title()} | {100*d['common']['mean']:.1f}% | {100*d['eroded']['mean']:.1f}% | {100*d['eroded']['min']:.1f}% |")
    lines += ['', 'Cropping remains substantial in some pairs: test masks retain 69.3% of original pixels on '
              'average after erosion, and as little as 33.7%. These are source-image mask areas; native depth '
              'grids have different resolution and their counts should not be directly divided by source areas.', '',
              'The spike filter removed **52 native-grid pixels across 40/900 bootstrap views**, and **22 '
              'pixels across 15/900 final views**. It is extremely conservative and does not establish that '
              'reconstruction noise has been eliminated. Final sampled reference clouds contain 8084–8192 '
              'points; tests contain 2636–8192, with median 8192. Fixed k therefore covers different fractions '
              'of different clouds.', '',
              '239 of the 450 pairs would have failed the old track-quality gate and now have actual scores. '
              'All pairs have computable mask maps; median RANSAC inlier fraction is 58.9%, minimum 22.5%. '
              'The mapping anchor is the reference in 421 pairs and the test in 29. Full scoring coverage is '
              'confirmed, but correspondence accuracy is not thereby proven.', '',
              '## Score distribution and remaining confounds', '',
              f"Official frame scores: mean **{summary['score_distribution']['mean']:.3f}**, median "
              f"**{summary['score_distribution']['median']:.3f}**, standard deviation "
              f"**{summary['score_distribution']['std']:.3f}**, range "
              f"**{summary['score_distribution']['min']:.3f}–{summary['score_distribution']['max']:.3f}**.", '',
              '| Generator | Videos | Mean of video means | Labels 0 / 0.5 / 1 | Within-generator Spearman rho (top 80) |',
              '|---|---:|---:|---|---:|']
    for dataset,d in diagnostics['dataset_scores'].items():
        mm=next(x for x in analysis['metrics'] if x['top_k']==80 and x['dataset']==dataset)
        counts=d['labels']
        lines.append(f"| {dataset} | 15 | {d['distribution']['mean']:.3f} | "
                     f"{counts.get('0.0',0)} / {counts.get('0.5',0)} / {counts.get('1.0',0)} | {mm['spearman_rho']:.3f} |")
    lines += ['', 'COSMOS2.5 has higher scores and mostly label-1 videos, while within-generator correlations '
              'are weak. The pooled association may partly reflect generator/domain differences. The 45 videos '
              'also represent 15 matched tasks across three generators, so ordinary p-values ignore that '
              'dependence and multiple comparisons. For top-80 ordinal video means, nominal Pearson p is '
              f"{m(80)['pearson_p']:.4f} and Spearman p is {m(80)['spearman_p']:.4f}.", '',
              'Across pairs, frame score and the smaller common-mask retention fraction have descriptive '
              f"Spearman rho **{diagnostics['score_vs_minimum_common_fraction']['spearman_rho']:.3f}**. "
              'Lower retained area tends to accompany higher scores; this does not establish causation, and '
              'frames within a video are dependent.', '',
              'Useful disagreement cases to inspect: label-0 **LVP_ROBOWM_0065** has video mean **296.379**, '
              'whereas label-1 **COSMOS3_0005** has **54.737**, **COSMOS3_0021** has **68.492**, and '
              '**LVP_ROBOWM_0060** has **77.168**. These are label/score disagreements, not evidence assigning '
              'a particular cause. MegaSaM scattered geometry, partial surfaces, viewpoint and self-occlusion, '
              'mask errors and uncertain extrapolated correspondence remain plausible confounds to inspect. '
              'A global 2D similarity does not guarantee identical visible 3D surfaces. The assumed-normal '
              'first observed frame can itself contain deformation.', '',
              '## Earlier experiments retained in this result folder', '',
              '| Experiment | Successful scores / attempted | Videos reconstructed | Pearson r (video mean) | Spearman rho | Binary AUROC |',
              '|---|---:|---:|---:|---:|---:|']
    for exp,count,vid in [('object','317 / 369','41 / 45'),('link5_cad','450 / 450','45 / 45'),('link5_first_frame','405 / 405','45 / 45')]:
        om,b= bm(exp),bm(exp,'binary')
        lines.append(f"| {exp} | {count} | {vid} | {om['pearson_r']:.3f} | {om['spearman_rho']:.3f} | {b['binary_auroc']:.3f} |")
    lines += [f"| Revised pair-specific first frame (top 80) | 450 / 450 | 45 / 45 | {m(80)['pearson_r']:.3f} | "
              f"{m(80)['spearman_rho']:.3f} | {m(80,'binary')['binary_auroc']:.3f} |", '',
              'The earlier object experiment uses **AF: Object deformation (0/1)2**: four videos lack existing '
              'usable selected crops; all 41 eligible sequences reconstruct with CVD, but 52 comparisons '
              'lack the 128 valid/unique points required by native neighborhoods. Only 37 videos have usable '
              'scores (32 complete, five partial); object ordinal n=37 and binary n=36. Link5 baseline '
              'ordinal n=45 and binary n=35.', '',
              'Earlier CAD and first-frame baselines share their exact observed clouds. CAD restricted to the '
              'same nine later frames gives binary AUROC **0.486**, versus **0.717** for first-frame reference. '
              'CAD score distributions are narrower (frame mean 116.745, SD 20.835) than first-frame scores '
              '(119.648, SD 38.334). Full CAD versus partial reconstructed surfaces, unit-radius scale '
              'normalization and reconstruction noise create a plausible domain gap; no registration fixes '
              'were introduced.', '',
              'The revised top-80 run has the **same binary AUROC (0.717)** as the earlier first-frame '
              'baseline, with weaker ordinal correlation (Pearson 0.316 versus 0.361; Spearman 0.320 versus '
              '0.421). This rerun improves inspectability and removes the old coverage veto, but does not '
              'demonstrate improved label discrimination. Frame selection, cropping and pair-specific '
              'reconstruction also changed together, so this is not an isolated ablation.', '',
              '## Files and reproduction', '',
              '- [450 official comparisons](link5_pair_visible_anchor/comparisons.csv)',
              '- [450 frame scores under top-80/160/320](link5_pair_visible_anchor/metadata/aggregation_frames.csv)',
              '- [Video aggregates and AB labels](link5_pair_visible_anchor/metadata/aggregation_videos.csv)',
              '- [Correlation/AUROC metrics](link5_pair_visible_anchor/metadata/aggregation_metrics.csv)',
              '- [Aggregation provenance and point-score hashes](link5_pair_visible_anchor/metadata/aggregation_analysis.json)',
              '- [Preprocessing diagnostics](link5_pair_visible_anchor/metadata/analysis_diagnostics.json)',
              '- [Completed transfer/audit receipt](link5_pair_visible_anchor/metadata/completion_verified.json)',
              '- [Earlier object/CAD correlation analysis](correlation/summary.json)', '',
              '```bash', 'python scripts/analyze_simple3d_aggregation.py',
              'python scripts/build_simple3d_pair_replay.py', '```', '',
              'The canonical result root remains `results/simple3d-20261003-run2/`. Official comparison '
              'JSON, native point scores and prototypes are preserved. Alternative scalar summaries live '
              'in the existing metadata folder. The scheduled AutoDL stop ran after verified local '
              'collection; the provider console showed instance `6eb54ead64-3baf4a31` stopped on 2026-10-04. '
              'All further aggregation and replay work is local CPU work.', '']
    (root.parent/'ANALYSIS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=(_workspace_root() / 'results/simple3d-20261003-run2/link5_pair_visible_anchor'))
    result = analyze(p.parse_args().root)
    print(json.dumps(dict(status=result['status'], frames=result['frames'], videos=result['videos'],
                          metrics=[m for m in result['metrics'] if m['dataset']=='all' and m['aggregation']=='mean_score']), indent=2))
