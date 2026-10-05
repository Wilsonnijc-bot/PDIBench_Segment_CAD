# Runtime errors: ten-video V1/V2 persistent-mask experiment

Experiment: `v1-v2-persistent-10-20260923` on the AutoDL GPU, launched through
`python -m pdi_eval.experiment resume` in tmux `pdi-ten-two`.

## Environment / runner

### Qwen virtual environment was bypassed

- **Observed:** Seven cases failed in `persistent_masking.pipeline select_frames`
  with `ModuleNotFoundError: No module named 'transformers'`.
- **Cause:** `ExperimentSpec.location()` used `Path.resolve()` on the configured
  `env/qwen/bin/python`. That executable is a symlink to the SAM environment's
  Python binary. Resolving it changed the invoked virtual environment from
  `env/qwen` to `env/sam3`.
- **Fix:** Preserve the configured executable path with `Path.absolute()` in
  `src/pdi_eval/experiment/spec.py`.
- **Verification:** The loaded remote spec now reports
  `/root/autodl-tmp/pdi/env/qwen/bin/python` for both Qwen and SAM execution;
  invoking that path imports `transformers` and `sam3`.

## Pipeline code: persistent masking

### SAM3 first point prompt lacked a cached frame entry

- **Observed:** Three LVP cases reached `segment` but failed with
  `AssertionError: No cached outputs found. Ensure normal propagation has run first
  to populate the cache.` The validator recorded `failed_sam`.
- **Cause:** The installed SAM3 code builds a first-object mask from point
  prompts, then its `_build_tracker_output()` requires a cached entry for that
  frame. A new session has no such entry before its first propagation.
- **Fix:** `persistent_masking/v1_mask.py` initializes an empty frame-cache entry
  before assembling a point-prompt response. The generated object mask still
  comes from SAM3's point-prompt path.
- **Verification:** `LVP_ROBOWM_0042` and `LVP_ROBOWM_0049` completed the
  resumed end-to-end run, including V1/V2 scoring and replays.

## Environment / runner: resume

### Failed masking stages were not retried by experiment resume

- **Observed:** On the first resume, the three `failed_sam` cases immediately
  failed again. Stage commands exited successfully because each stage skipped
  cases whose status was already `failed_sam`; validation read the old failure.
- **Cause:** The experiment runner retried failed cases, but the persistent
  masking pipeline retained its own failed state and only executes `*_pending`
  stages.
- **Fix:** The experiment runner now resets `failed_sam`, `failed_vlm2`, and
  `failed_validate` to their corresponding pending stage before retrying that
  case. Existing stage artifacts and source provenance stay in place.
- **Verification:** The previously `failed_sam` cases `LVP_ROBOWM_0042` and
  `LVP_ROBOWM_0049` were retried and completed.

## Environment / runner: MegaSAM model access

### UniDepth checkpoint was unavailable from the GPU's default Hugging Face endpoint

- **Observed:** `LVP_ROBOWM_0016` completed masking with
  `no_confirmed_deformation` and entered V1, then MegaSAM returned all-zero
  point maps. Its UniDepth subprocess had failed to load
  `lpiccinelli/unidepth-v2-vitl14` because `huggingface.co` timed out and
  refused connections. The later missing-`config` TypeError was a consequence
  of that failed download.
- **Cause:** The pinned UniDepth model files were absent from the GPU cache,
  and the default model host was unreachable from this instance.
- **Fix:** Download the exact pinned revision from the reachable
  `https://hf-mirror.com` endpoint into the Hugging Face cache, then set
  `HF_ENDPOINT` for the experiment process. This retains the same revision
  and model weights; only the download host changes.
- **Verification:** The pinned `config.json` and 1.45 GB
  `pytorch_model.bin` downloaded successfully. `LVP_ROBOWM_0042` and
  `LVP_ROBOWM_0049` subsequently completed V1/V2 scoring and replays.

## Environment / runner: disk capacity

### GPU data volume filled during final COSMOS3 cases

- **Observed:** At 21:30:36 UTC, the runner stopped with
  `OSError: [Errno 28] No space left on device` while opening a case log.
  Seven cases were complete; two COSMOS3 cases were in progress and one had
  not yet been retried. `/root/autodl-tmp` was at 50 GB used out of 50 GB.
- **Cause:** The data volume held the Qwen model, both Python environments,
  source videos, experiment outputs, and a 3.3 GB pip HTTP download cache.
- **Fix:** Removed only `/root/autodl-tmp/pdi/cache/pip/http-v2`, which is
  disposable package-download cache; installed packages, model weights,
  source videos, and experiment artifacts were retained. This freed 3.3 GB.
- **Verification:** `df -h /root/autodl-tmp` reports 3.3 GB available.
  Final case completion is pending.

### MegaSAM intermediates continued consuming the data volume

- **Observed:** During the resumed COSMOS3 scoring, free space fell to 2.0 GB.
  MegaSAM's `work_space`, `reconstructions`, and `cache_flow` contained large
  per-case intermediates even for the seven completed videos.
- **Fix:** Moved only those three intermediate directories for the seven
  completed cases to `/root/pdi_completed_intermediate` on the separate root
  volume, then replaced their original paths with symlinks. No source videos,
  model weights, metrics, or replay outputs were removed.
- **Verification:** The data volume had 4.7 GB free after the move; the
  symlinked intermediate files remain accessible at their original paths.

### Motion smoothness material removed after user request

- **Action:** Removed the motion smoothness code, logs, and generated artifacts
  from `/root/autodl-tmp/motionsmoothness-robot`, plus the separate
  `/root/autodl-tmp/motion-smoothness-smoke` directory. Preserved the entire
  `/root/autodl-tmp/motionsmoothness-robot/videos` source directory.
- **Verification:** The motion smoothness directory now contains only `videos`;
  its 197 source video files remain present. The data volume has 4.5 GB free.

### Qwen weights moved to the system disk after user request

- **Action:** Copied `/root/autodl-tmp/models/Qwen3.5-9B` to
  `/root/models/Qwen3.5-9B`, verified the copy with a checksum comparison,
  then replaced the original configured path with a symlink. The fourth shard
  already pointed to `/root/models/Qwen3.5-9B-overflow` and still does.
- **Verification:** `AutoConfig.from_pretrained(..., local_files_only=True)`
  loads through the original path; all four weight shards resolve. The system
  disk has 4.0 GB free and the data volume has 17 GB free after the old copy
  was removed.

## Current run

The first attempt ended with 0 complete and 10 failed. After the fixes and
resumes above, the same frozen ten cases finished with **10 complete, 0
failed**, and runner `EXIT_STATUS=0`. V1 and V2 metrics and replays were
verified for all ten. Exact selected-pair interactive bundles were produced
for both versions and copied to the local result root. The user stopped the
Luna monitor after completion.
