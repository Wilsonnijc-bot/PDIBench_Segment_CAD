# Link5 SAM prompt tuning trials

**Artifact category: SAM tuning results, not a deformation-scoring result set.**

This folder preserves the frame-zero experiments used to tune the VLM guard's positive/negative SAM3 prompts: prompt variants, Gemini/GPT decisions, point previews, masks, and diagnostics. It contains no complete link5 rigidity-score run; `sam3_score` is segmentation confidence, not deformation rigidity.

The latest scored link5 run is [link5-only-selected45-updated-mask-20260929](../link5-only-selected45-updated-mask-20260929/index.html), with its [forearm AUROC analysis](../link5-only-selected45-updated-mask-20260929/LINK5_FOREARM_AUROC.md).

The published representative link5 prompt replays use that scored run's `cases/*/base_generation/link5_guard.json` and `sam3_prompt_diagnostics.json`, rather than these earlier tuning trials. Original tuning artifacts remain at their existing paths.
