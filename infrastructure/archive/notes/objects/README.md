# Task-object rigidity wrapper

This folder is a separate entry point for the 45 selected videos. It reads each
`Generation Prompt`, asks GPT-6 Luna (high reasoning) to identify the object the
robot directly manipulates, gives SAM3 one positive box and one positive point,
and reports only the active V1 3D pairwise rigidity score. A failed Luna
grounding or frame-zero mask gets at most one Gemini 3.8 Flash attempt.

The wrapper retains source hashes, exact VLM responses, a prompt preview,
full-video SAM3 mask replay, track archive, rigidity history, and a status for
every case. A score requires five frame-zero CoTracker anchors and three
nondegenerate 3D pairs. Occluded mask frames are recorded but do not cause an
automatic coverage failure. No binary classification threshold is applied.
Each completed case also gets an interactive rigidity replay through the
existing `pdi_eval.utils.rigidity_replay` exporter. The replay shows the source
video, task-object mask and tracks, 3D cloud, pair evidence, and score history.

On the prepared GPU host, with `PYTHONPATH=src:.`, run:

```bash
env/pdi-bench/bin/python -m pdi_eval.object_deformation_wrapper prepare \
  --workbook /path/selected_45_matched_videos_styled.xlsx \
  --selection /path/selected-45-v1/selection.json \
  --output-root /path/object-deformation-selected45

env/pdi-bench/bin/python -m pdi_eval.object_deformation_wrapper run \
  --output-root /path/object-deformation-selected45 \
  --video-root /root/autodl-tmp/motionsmoothness-robot/videos \
  --sam-python /root/autodl-tmp/pdi/env/sam3/bin/python \
  --sam3-checkpoint /root/autodl-tmp/pdi/models/sam3/sam3.pt \
  --sam3-bpe /root/autodl-tmp/pdi/models/sam3/bpe_simple_vocab_16e6.txt.gz \
  --tracker-checkpoint /root/autodl-tmp/pdi/models/tracker/scaled_offline.pth \
  --gpu-lock /path/selected-45-v1/gpu.lock
```

The existing project `.env.vlm` supplies `VLM2_API_KEY`; it is never copied into
the output. `--case DATASET_0001` runs a single case. A later `run` resumes
pending cases and skips final statuses, so a failed case never gets extra VLM
attempts implicitly. `status` refreshes `summary.json` and `scores.csv`.

For the prepared AutoDL host, run `python3
PDI-Bench-edited/src/pdi_eval/object_deformation_wrapper/control_gpu.py
--workers 2` from the workspace root. The two workers stage inputs and collect
verified outputs concurrently. GPU model calls use the existing shared lock to
stay within A100 memory. Results are retrieved to
`results/object-deformation-selected45-20260929/`, with a batch `index.html`
linking the per-case interactive replays. Case geometry caches are removed from
the GPU only after their local copies pass SHA-256 verification.

## Gripper occlusion audit

An area-only guard rejects table-sized link7 masks before occlusion analysis:
`link7 mask pixels / (image height * image width) >= 0.25` fails that frame.
The cutoff is configurable with `--max-gripper-area-fraction`. The shared rule
is in `pdi_eval.perception.mask_quality` and is recorded as `link7-frame-area-v1`.
Detection rows record `mask_valid`, `mask_failure_reason`, `gripper_area`, and
`gripper_area_fraction`; failed rows have status `failed_link7_mask_area`.
They supply neither a reference nor occlusion evidence and reset both episode
states. Passing this size check does not verify segmentation identity.

Filtered-score method `task-object-v1-occlusion-filtered-mean-v2` excludes failed
mask frames as well as occlusion catches, with null values and a null final
score if no frames remain. Existing naive scores remain available as historical
baselines. Replays display **FAILED MASK** separately from an occlusion catch.
The reference-visible-pixel exporter also skips failed frames and records their
validity. With the default cutoff, all 49 frames of the saved LVP010 and LVP015
masks fail; the other 43 saved cases have no oversized link7 frames.

See [the exact detection logic and frame-162 diagnosis](OCCLUSION_DETECTION.md)
for the intended problem, all decision gates, and verified evidence from the
three example replays.
The [case comparison and refinement](OCCLUSION_CASE_REVIEW.md) records successful
detections, missed episodes, acceptable-contact baselines, and the review that
led to v2.

Run the lightweight post-processing audit after object scoring:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.occlusion \
  --object-root results/object-deformation-selected45-20260929 \
  --gripper-root results/link5-link7-four-way-selected45-20260927
```

This reads object channel 0, link7 channel 5, and the existing object CoTracker
tracks. It writes `cases/<case>/occlusion/{detection.json,frames.csv}` and one
`occlusion/README.md` index under the object result root. It does not recompute
models or alter scores. Requires NumPy and OpenCV.

The original branch requires at least 20% of the expected silhouette to disappear under
link7, link7 to explain at least 60% of missing pixels, and at least 20% of
anchor object tracks to fall under link7. Once established, the episode can
continue during mask regrowth while gripper contact and affected tracks persist.
Isolated one-frame runs are rejected.

Method `gripper-occlusion-v2` adds an independent severity-supported branch:
replacement >= 20%, loss explanation >= 80%, and either track overlap >= 10%
or replacement >= 50% establish an episode. Continue while replacement >= 20%
and loss explanation >= 80%, with the same assessability requirements but no
mandatory track overlap. Keep runs of at least three frames. Filter each branch
independently, then take their union; the new branch cannot activate original
continuation. This preserves every original catch without backfilling onset or
bridging gaps. Direct segmentation overlap cannot start either branch.

`legacy_flagged` and `severity_flagged` expose the surviving branches; their OR
is `flagged`. `onset` and `continued` retain their original-branch meanings;
`severity_onset` and `severity_continued` expose the new branch. Thresholds are
CLI options; every output records its configuration, method and source hash.

These are reviewable risk flags, not causal labels. The expected silhouette uses
translation only. Segmentation leakage, rotation, complete disappearance, or
tracker drift can cause false positives or missed frames. Consult the generated
index for the three requested example checks and known limitations.

After detection, preserve the naive score and calculate the separate filtered
mean, then annotate all existing interactive replays:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.occlusion_scores \
  --root results/object-deformation-selected45-20260929
```

This writes `score/rigidity_occlusion_filtered.json` per case and
`occlusion/rigidity_comparison.{csv,json}`. Caught frames and failed mask frames are excluded from
the mean after frame 0. Retained carried values and unassessable frames are not
changed. Empty retained sets produce null, not zero. The replay patch is
idempotent and leaves original score files, embedded history, and pair evidence
unchanged. Reapply it after regenerating the original replays.

Optional separate example catch viewers:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.occlusion_replay \
  --root results/object-deformation-selected45-20260929
```

Open `occlusion/replay/index.html` through the existing results server. By
default the exporter refreshes all existing catch viewers plus the three
original examples. `--cases` selects specific exports while retaining other
viewers in the index. Pages show the exact saved flags.
