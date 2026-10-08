# Pair-selection ablation restored — October 8, 2026

The user reinstated this as a retained research experiment. Its code, handoff,
scores, evidence and replay belong here and must survive workspace cleanup.

## What was recovered

- The ERIS experiment snapshot, including both selectors, baseline and all-frame
  depth-supported scorers, preparation, capacity runners, tests and provenance.
- Later local replay, label-analysis and collection code recovered from the original
  development records. The active replay uses the previous Plotly template.
- **860 ERIS files / 1,012,172,456 bytes**, verified individually by SHA-256.
  These include **541 scientific evidence files**, 41 videos, depth-support arrays,
  replay context, trajectories, per-frame scores, graphs and allocation records.
- **82 complete scores: 40 primary videos + one additional video, both methods.**
  AB-label correlations reproduce the recorded results: exact-binary AUROC
  **0.708333 balanced / 0.779167 refine** on 32 labeled videos.

The eight preparation/selection/scoring modules match the ERIS source byte for
byte. Original ERIS code and restored local code are also preserved as compressed
snapshots with file hashes under [provenance/recovery](provenance/recovery).

## Where the evidence lives

Local outputs: [results/20261007](results/20261007). The repository-root
`results/link5_pair_selection_ablation_v1` is a compatibility link.

- [Experiment report](results/20261007/REPORT.md)
- [Interactive replay](results/20261007/report/index.html)
- [Paired scores](results/20261007/rigidity/paired_scores.csv)
- [AB-label analysis](results/20261007/analysis/forearm_AB_correlation.md)
- [Restoration receipt](results/20261007/metadata/restoration.json)

The original remote workspace remains unchanged:
`/PHShome/zy992/Wilson/deformationdetection/workspace-link5-pair-selection-20261007-v2`.
Full raw MegaSAM/CoTracker inputs and native scratch geometry remain there;
their checksums and provenance were recovered locally. They were not rerun.

Guard-only reviews remain owned by `../link5_guard_review`; links preserve the
old included/excluded-mask and negative-review locations within this experiment.

## Small updates needed for the restored workspace

The depth-filter adapter imports shared preprocessing from its new location.
The historical selector-parity test follows the relocated September evidence.
Preview navigation follows the actual five exclusions and selects an included
video. The handoff identifies the completed all-frame-filtered run separately
from its original planning recipe. Scientific selection and scoring are unchanged.

The 12 experiment tests pass. Browser copies preserve all original frames,
dimensions, frame rates and timestamps. Compact score/analysis records are
included by Git's file rules; larger evidence remains preserved locally and on ERIS.
No GPU experiment, model inference or paid VLM call was launched for restoration.
