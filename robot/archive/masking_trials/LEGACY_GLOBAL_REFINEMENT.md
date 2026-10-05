# Global VLM refinement

`persistent_masking.global_vlm_refine` selects `--provider openrouter --model google/gemini-3.8-flash` or `--provider qwen --model /path/to/Qwen3.5-9B`. OpenRouter reads `OPENROUTER_API_KEY` from the environment and does not load Qwen on the GPU. SAM uses the existing GPU environment/checkpoint.

This is a **global experimental policy**, not an accepted canonical replacement. It reuses the declared source run's seed frames, exact reference/candidate crops, and negative coordinates, but requeries all three positives in every case. The white prompt targets the center of white material within the mixed-color gripper. Both dark points may lie on deformed black gripper material/fingers; the second call includes the first point's normalized crop coordinates. Points less than five source pixels apart are rejected.

At 50% and 75%, the VLM sees reference, clean crop and cyan mask crop. If correction is requested, three new positive calls must pass validation before a fresh SAM session replaces masks from the checkpoint forward. Earlier frames retain their prior masks. A missing or ambiguous correction is recorded, never represented as a successful fix. This sparse review schedule does not guarantee uninterrupted tracking between checkpoints.

Run from the remote repository with its existing `env-sam/bin/python`:

```sh
export OPENROUTER_API_KEY='<provided through your environment>'
env-sam/bin/python -m persistent_masking.global_vlm_refine \
  --source lasteset_qwen_results/experiments/global/global_combined_lower_dark_20260912 \
  --work gemini_global_work_NEW_RUN_NAME \
  --provider openrouter --model google/gemini-3.8-flash \
  --ffmpeg /path/to/ffmpeg
```

Use a new work directory for each attempt. Raw masks/intermediate metadata stay in remote work. Export only exact model inputs, `points.png`, final `masking.mp4`, and consolidated run provenance into `lasteset_qwen_results/experiments/global/<run>/`. Point previews show the latest applied correction, or seed points when none applied; frame and coordinates are in provenance. Final replays use source RGB beside cyan masks, H.264 Main, yuv420p, faststart. Before promotion, review every replay, membership checks and recorded failures; nonempty masks and valid coordinates alone do not establish anatomical accuracy.

The 2026-09-12 attempt was blocked by OpenRouter HTTP 403 before any generated points or SAM inference. Resolve the provider restriction before a new attempt. No model substitution was made.
