# Latest link2, link5, and link7 results

Current sources for the selected-45 review:

| Link | Latest result used | Method | AUROC |
| --- | --- | --- | ---: |
| Link2 · upper arm | [selected-45-v1](results/selected-45-v1/review.html) | V1 + CoTracker3 | 0.941 |
| Link5 · forearm | [updated-mask selected-45](results/link5-only-selected45-updated-mask-20260929/index.html) | V1 + CoTracker3, updated SAM3 mask | 0.538 |
| Link7 · gripper | [four-way selected-45](results/link5-link7-four-way-selected45-20260927/index.html) | V1 + CoTracker3, measured cases only | 0.845 |
| Link7 · gripper | Same four-way run | V2 + TAPIP3D | 0.842 |

**Analysis:** [Link2](results/selected-45-v1/DEFORMATION_LABEL_ANALYSIS.md), [Link5](results/link5-only-selected45-updated-mask-20260929/LINK5_FOREARM_AUROC.md), [Link7](results/link5-link7-four-way-selected45-20260927/rigidity-vs-deformation-labels.md). AUROC compares severe label 1 with none label 0, excludes moderate 0.5, and uses higher reported ε rigidity as the positive score.

**Labels:** Link2 uses `selected_45_matched_videos_styled.xlsx` column AD; link5 uses `selected_45_matched_videos_styledv2.xlsx` column AB (forearm deformation); link7 uses v2 column AC. Link5 has 45 scored cases; its full-precision replay scores yield 0.542 instead of the reported-score AUROC 0.538.

**Link7 availability:** V1 has 38 scored cases and seven unavailable cases. Assigning all seven unavailable cases the highest score gives **0.890**, an imputation sensitivity result. V2 scores all 45. The review uses four-way V1 replays; the older `selected-45-v1` link7 result (0.848 on 37 scored cases) is a separate run.

**SAM tuning:** [link5-joint-prompt-20260929](results/link5-joint-prompt-20260929/SAM_TUNING_RESULTS.md) contains prompt-tuning experiments, not rigidity scores. Published link5 prompt replays come from the updated-mask run's guard records and diagnostics; link7 persistent-mask examples come from `selected-45-v1/persistent_work/`.

[Published grouped videos and interactive point-cloud replays](https://wilsonnijc-bot.github.io/PDIBench_Segment_CAD/).
