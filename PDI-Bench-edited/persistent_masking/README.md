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
