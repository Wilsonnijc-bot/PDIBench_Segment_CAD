# PDI V1 experiment interface

For autonomous link7 plus object runs, optionally including link2, use the [resumable coordinator](../../documentation/pipeline/COORDINATOR.md). `coordination.py` adapts explicit video/reference inputs to the existing persistent-masking and V1 scoring functions. The experiment interfaces below remain available for historical workflows.

The native pipeline is `pdi_eval.v1`. This package owns frozen selection, input verification, persistent masking, V1 scoring, status, and replay exports. Run commands from `PDI-Bench-edited/` with `PYTHONPATH=src:.`.

## Ten-video persistent-mask experiment

The active GPU profile is `configs/experiment_v1_persistent_10.json`. Its default run root is `/root/autodl-tmp/pdi/experiments/v1-persistent-10`; set `PDI_RUN_ROOT` to override it. The earlier `v1-v2-persistent-10-20260923` folder is a historical result and must remain untouched.

Stage the verified workbook and source manifest under the new run root's `input/` directory. Freeze the same ten sample IDs with:

```bash
PYTHONPATH=src:. python -m pdi_eval.experiment select \
  --workbook "$PDI_RUN_ROOT/input/deformationhumanlabel.xlsx" \
  --source-manifest "$PDI_RUN_ROOT/input/full-corpus-manifest.json" \
  --cases configs/v1_replay_10.json \
  --output-root "$PDI_RUN_ROOT"
```

For each selected case, stage `cases/<sample>/base_segmentation.npz` and `base_segmentation_source.json` from the verified prior inputs. The provenance must match the selected video and mask SHA-256 hashes. The runner never creates a missing base segmentation silently.

```bash
PYTHONPATH=src:. python -m pdi_eval.experiment check --spec configs/experiment_v1_persistent_10.json
PYTHONPATH=src:. python -m pdi_eval.experiment run --spec configs/experiment_v1_persistent_10.json
PYTHONPATH=src:. python -m pdi_eval.experiment resume --spec configs/experiment_v1_persistent_10.json
PYTHONPATH=src:. python -m pdi_eval.experiment status --spec configs/experiment_v1_persistent_10.json
PYTHONPATH=src:. python -m pdi_eval.experiment export --spec configs/experiment_v1_persistent_10.json
PYTHONPATH=src:. python -m pdi_eval.experiment replay-pairs --spec configs/experiment_v1_persistent_10.json
```

`check` verifies source videos, base masks, workbook identity, and model assets without inference. `run` requires no case status files; `resume` reuses completed masking and scoring artifacts. Two case workers may overlap API and file work, while one file lock serializes GPU model calls. Each case runs persistent masking, then V1 scoring and its MP4 and interactive pair replays. A validated refinement replaces only `link7`; a labeled negative with no confirmed deformation uses the base archive. Invalid masking or provenance fails the case.

Outputs are under `cases/<sample>/v1/`, with a shared geometry cache beside that directory. The top-level index links V1 metrics and both replays. `replay-pairs` regenerates interactive pages from saved V1 artifacts and verifies their rigidity histories against the scored metrics. For detached execution, launch `scripts/record_v1_replay_two.sh` under tmux; it calls `scripts/run_v1_replay_two.sh` and records an exit status.

## Individual scoring and V1 batch

```bash
PYTHONPATH=src python -m pdi_eval.experiment score \
  --config configs/default.yaml \
  --input /path/video.mp4 --segmentation-npz /path/masks.npz \
  --output-dir /path/result --geometry-cache-dir /path/cache \
  --tracker-checkpoint /path/scaled_offline.pth --tracking-mode exact-group
```

The scorer has no version flag. Other supported commands are `video-manifest`, `batch-v1`, `export-v1-batch`, `merge-mask`, and `migrate-checkpoint`; each accepts `--help`. The older V1 batch retains its original manifest and staging format. It does not add persistent masking unless invoked through the ten-case `run` or `resume` flow.
