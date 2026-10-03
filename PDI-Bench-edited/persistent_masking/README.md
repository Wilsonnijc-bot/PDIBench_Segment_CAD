# Persistent masking

This package owns the project's persistent mask-generation and mask-repair
workflows. It is intentionally separate from the Reg2Inv baseline in
`baseline/`.

The current global pipeline is:

1. create a naive frame-zero SAM3 mask and propagate it;
2. detect suspicious temporal mask-area changes;
3. use VLM1 to diagnose deformation timing;
4. use VLM2 to place positive and negative correction points;
5. reseed SAM3 and propagate the corrected mask;
6. validate lineage and export point previews and masking replays.

For a new run, use the [configuration and usage interface](interface/README.md).
Edit one file, [`interface/config.py`](interface/config.py), then use
`./persistent_masking/pmask check`, `run` (optionally `--detach`), and `status`.
The interface handles the naive path, the refined pipeline, and the compact
review export.

Lower-level entry points, for reproducibility and custom experiments:

- `python -m persistent_masking.naive_sam3`
- `python -m persistent_masking.pipeline`
- `persistent_masking/run_pipeline.sh`
- `persistent_masking/run_pipeline_sequential.sh`
- `persistent_masking/run_case_experiment.sh` (one-case local trial; exports the same six review files as `sep22 run/gemini/Cosmos3_0048` into a new destination)

For a new one-case trial on the GPU, set `FFMPEG_BINARY` and run
`persistent_masking/run_case_experiment.sh Cosmos3_0048 trial_name /absolute/new/review_dir`.
The work/provenance stays under `lasteset_qwen_results/experiments/local/trial_name/Cosmos3_0048/`.
The review directory must be new; the script never overwrites the September 22 Gemini results.

The small [`vlm_interface/`](vlm_interface/) folder remains the human-readable
developer reference for exact prompts and the three default images. Each run
also records its latest VLM2 target frame and actual prompt in provenance.

Environment requirements and the pinned VLM1 download utility are in `setup/`.

The `palm_*`, `gripper_*`, and `refine_positive_points.py` modules are retained
mask-repair experiments and diagnostics. `v1_mask.py` is here because it owns
SAM/CoTracker mask construction, although the Reg2Inv v1 inference and scoring
code remains in `baseline/`.

`global_vlm_refine.py` and the `palm_*`/`gripper_*` scripts are retained legacy
experiments. Some retain explicit historical model names and their original CLI
arguments for reproducibility; they are not the configurable VLM1/VLM2 runner.

See `PIPELINE_SPEC.md` for the pipeline contract.

Optional VLM3 object-overmask repair runs **after** the original VLM1/VLM2 flow.
It does not change surge windows, Qwen diagnoses, VLM2 frame selection, or the
original five-point seed. It uses the existing object mask to find the earliest
frame with `count(link7 & task_object) / count(task_object) > 0.95` (strict);
empty object masks do not qualify. The existing `link7-frame-area-v1` failure
(`link7 / image >= 0.25`) is a separate trigger for table-sized masks. VLM3 uses
the exact gate frame, with no +1 shift, and the same backend/config as VLM2.
It asks for three gripper positives and three negatives: arm, wrist, and the
named task object. The six-point reference is in `vlm3_interface/`; its added
object exclusion is a manual example, not a model prediction.

To evaluate already saved PDI masks independently:

```sh
PYTHONPATH=.:src python -m persistent_masking.vlm3_overmask audit \
  --object-root /path/to/object-results --gripper-root /path/to/four-way-results \
  --output-root /path/to/review/metadata/audit45
PYTHONPATH=.:src python -m persistent_masking.vlm3_overmask run \
  --object-root /path/to/object-results --gripper-root /path/to/four-way-results \
  --output-root /path/to/new-review \
  --cases COSMOS2.5_0005 COSMOS2.5_0010 COSMOS2.5_0065
```

The default is at most two six-point attempts on the same frame, using the
existing SAM3 predictor and bidirectional propagation. Candidates require all
six point memberships to match, no empty frames, and at least one crop-eligible
frame. Remaining gate hits exclude those frames from cropping; they do not
reject the entire video (`accepted_with_frame_exclusions`). The crop selector
also checks the saved occlusion audit's direct object/link7 overlap and skips
strictly >95% frames, including an unusable immediate recovery successor.
This detects these mask failure patterns, not all segmentation errors or true
physical occlusion. Failed candidates do not replace old masks. The review
contains exact request inputs, point previews, actual responses/metrics, and a
before/after MP4; accepted masks are in each case's `masks.npz`.

For a new native persistent run, add `--enable-vlm3 --object-mask-root /path/to/object-results`
to `persistent_masking.pipeline continue`; or run its explicit `repair_overmask`
stage after validation with `--object-mask-root`. The everyday interface has
`VLM3_ENABLED` (default false) and `VLM3_OBJECT_MASK_ROOT`. This opt-in integration
attaches a separate `vlm3` record, preserving original diagnoses and seed/masks.
It also checks the old naive fallback when VLM1 finds no deformation, so that
Qwen's diagnosis cannot suppress VLM3. Failed original runs are not reused.
When merging a **new** refined archive, `mask_merge` uses
only an accepted, hash-verified VLM3 candidate. Existing scores, archives, crops,
and diagnoses are never overwritten by this experiment.
