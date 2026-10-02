# PDI-Bench Native Multi-Object Pipeline

The active V1 tracking contract is in [D_PIPELINE_SPEC.md](D_PIPELINE_SPEC.md).
The earlier A/B/C/D experiment proposal is archived at
[archive/verification/ABCD_PIPELINE_VERIFICATION_PLAN.md](archive/verification/ABCD_PIPELINE_VERIFICATION_PLAN.md).
The only editable Python pipeline is `PDI-Bench-edited/src/pdi_eval/v1/`, with
`pdi_eval.v1.evaluate(...)` as its short Python interface.

> [!WARNING]
> **UNVERIFIED DEVELOPMENT PIPELINE.** The edited seven-link pipeline has not
> yet passed the A/B/C/D GPU verification protocol. Its metrics, grades, and
> performance results must not be treated as validated.

Clone this whole project, including its pristine comparison baseline and
third-party dependencies, with:

```bash
git clone --recurse-submodules \
  https://github.com/Wilsonnijc-bot/PDIBench_Segment_CAD.git
```

This repository evaluates the rigid Franka FER links in one video-level
PDI run. DINOv2-guided SAM3 segments the active links, MegaSAM reconstructs the
full video once, and PDI reports scale, trajectory, rigidity, and perspective
metrics separately for each link in the same world-coordinate frame.

All editable model and metric code is native to `PDI-Bench-edited/`. The
upstream comparison checkout is pinned as the read-only `PDI-Bench-original/`
submodule. The former
`scripts/adapters/` layer and retired manual, RobotSeg, DINO, and SAM2 launch
paths have been removed.


## Robot-Deformation Evaluation Decision

For the robot-deformation study, the validation metric is the per-link
**rigidity component only**. Scale,
trajectory, the combined PDI score, and PDI grades are not used as deformation
targets. They measure other geometric consistency properties and can be
dominated by mask or depth artifacts that are not deformation.

The decision recorded on September 6, 2026 is:

- Human target: `Robot deformation (0/1)`.
- Upper arm rigidity: mean of valid `link2` and `link3` rigidity.
- Forearm rigidity: mean of valid `link4` and `link5` rigidity.
- Gripper rigidity: valid `link7` rigidity.
- Whole-robot rigidity: equal-weight mean of upper arm, forearm, and gripper.
- `link6` is excluded because no supplied component label maps to it.
- Failed links and insufficient-evidence rigidity values are missing data, not
  maximum deformation.
- In the historical `outputs/metrics.csv`, `rigidity_component == 1.0` is the
  insufficient-evidence sentinel and is excluded from the primary analysis.
- Direct-depth and full-SAM filters are reported only as sensitivity analyses;
  they are not the primary rigidity validity rule.
- Raw rigidity values remain unchanged in CSV and JSON artifacts. The HTML
  report displays `100 * raw rigidity` as **rigidity deviation (%)** so values
  such as `0.027` read as `2.70%`. This is a relative pair-distance dispersion
  index, not a probability or confidence score.
- The report's low, middle, and high labels are descriptive tertiles of the
  primary valid whole-robot cohort. They are not universal deformation
  thresholds and do not affect correlation or AUROC.


On the completed LVP_ROBOWM + COSMOS3 label cohort, the primary
sentinel-excluded rigidity result is `n = 45`, Pearson `r = 0.293`, Spearman
`rho = 0.372`, and AUROC `0.769`. Component AUROC is `0.536` for upper arm,
`0.467` for forearm, and `0.756` for gripper. The whole-robot association is
therefore driven mainly by gripper rigidity and must not be described as broad
robot-link deformation detection.

The active scorer is the V1 multi-object pipeline. It reports each link's
rigidity alongside the other PDI components. For deformation analysis, use the
per-link rigidity component under the evidence rules above; the combined PDI
score is not a deformation label.

```bash
cd PDI-Bench-edited
PYTHONPATH=src python -m pdi_eval.experiment score \
  --config configs/default.yaml \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --output-dir /path/output \
  --geometry-cache-dir /path/megasam-cache \
  --tracker-checkpoint checkpoints/tracker/scaled_offline.pth \
  --tracking-mode exact-group
```

The [ten-case V1 experiment](PDI-Bench-edited/docs/v1_replay_batch.md) retains
persistent masking, two workers, resume, MP4 replay, and interactive scored-pair
replay. Prior comparison results remain historical records; do not combine
scores from different methods in one correlation cohort.

## Dataflow

```text
DINOv2 link reference images
        |
        v
DINOv2-guided SAM3 once
        |
        +--> object_masks[T,N,H,W]
        +--> union mask for background exclusion only
        |
        +--> MegaSAM once on the full video
        |      camera poses + intrinsics + world pointmaps
        |
        +--> one deterministic CoTracker query manifest
               |                         |
               v                         v
          joint-query               exact-group
          one joint update          shared backbone,
                                    isolated updates
               |                         |
               +------------+------------+
                            v
                 per-link PDI metrics and
                 exact-vs-joint comparison
```

The articulated union is never scored for rigidity. Each rigidity call receives
only one link's mask, tracks, and visibility.

## DINOv2 Reference Pipeline

Reference groups are discovered under:

```text
robot_link_first15/by_link/
  link_1/
    001_*.png ... 015_*.png
  ...
  link_7/
    001_*.png ... 015_*.png
```

`contact_sheet_15.png` files are excluded. The videos default to
`.tmp/COSMOS_Videos/`, and both paths can still be overridden with
`PDI_REFERENCE_DIR` and `PDI_INPUT_VIDEO`.

`link1` is retired from this automatic branch. Its reference directory may
remain present, but it is ignored; active outputs preserve the canonical names
and IDs `link2` through `link7`.

Install the pinned DINOv2 model on the GPU, then run DINOv2 localization and
SAM3 box-prompted video segmentation:

```bash
bash scripts/prepare_dinov2_gpu.sh
PDI_VIDEO_NAME=0000.mp4 \
bash scripts/run_dinov2_sam3_video.sh
```

Each active link is localized independently by DINOv2 and passed to an isolated SAM3
box session, so prompts cannot reset or merge another link's identity. The
launcher combines link-specific text with the unchanged DINOv2 boxes for
`link4`, `link5`, and `link7`; the other links remain visual-box-only prompts.
The launcher writes `dinov2_boxes.json`, a box preview, dense similarity
heatmaps, `sam3_prompt_diagnostics.json`, the canonical multi-object
`segmentation.npz`, and `segmentation.json` under `results/dinov2-sam3/`.
The run fails if any active link is tracked on less than 80% of the video by default;
override this only with `PDI_MINIMUM_TRACKED_FRACTION`.

For a prepared 40 GB GPU, process videos in deterministic pairs. Shared code
and model preparation runs once, each video gets an isolated remote work
directory, and logs are kept separately under
`results/dinov2-sam3/batch-logs/`:

```bash
bash scripts/run_dinov2_sam3_batch.sh \
  0000.mp4 0001.mp4 0002.mp4 0003.mp4
```

The launcher starts at most two GPU jobs at a time. An odd final video runs by
itself. Set `PDI_RUN_VARIANT` to keep differently configured batches separate.

Install SAM3 and download its checkpoint from the pinned ModelScope revision:

```bash
bash scripts/install_sam3_gpu.sh
export PDI_SAM3_CHECKPOINT=/root/autodl-tmp/pdi/models/sam3/sam3.pt
export PDI_SAM3_BPE=/root/autodl-tmp/pdi/models/sam3/bpe_simple_vocab_16e6.txt.gz
```

This path does not require Hugging Face authentication. The installer verifies
the checkpoint and ModelScope tokenizer merges before making them available to
the pipeline.

## Native Benchmark CLI

Inside a prepared PDI environment:

```bash
cd PDI-Bench-edited
PYTHONPATH=src python -m pdi_eval.experiment score \
  --config configs/default.yaml \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --output-dir /path/output \
  --geometry-cache-dir /path/megasam-cache \
  --tracker-checkpoint /path/scaled_offline.pth \
  --tracking-mode both
```

## Outputs

The scorer writes results to the requested `--output-dir`:

```text
/path/output/
|-- metrics.json
|-- timing.json
|-- manifest.json
|-- run_config.yaml
|-- cotracker_joint-query.npz
|-- cotracker_exact-group.npz
|-- console.log
`-- replay/
    |-- combined_joint-query.mp4
    |-- combined_exact-group.mp4
    `-- interactive_exact-group/index.html
```

`metrics.json` contains seven reports under each requested mode and, when both
modes run, exact-minus-joint deltas for every PDI component plus grade changes.
`timing.json` separates shared work, query preparation, model time, metric time,
forward counts, and peak GPU memory.

MegaSAM geometry is stored once in a versioned, content-addressed GPU cache.
Results reference that cache identity instead of copying seven pointmap archives.

## Tests

```bash
PYTHONPATH=PDI-Bench-edited/src python3 -m unittest discover \
  -s PDI-Bench-edited/tests -v
```

The lightweight local environment can run CAD and replay tests. CoTracker mode
tests require the PDI PyTorch environment and verify that exact-group performs
isolated updates while invoking the video backbone only once.

See [SHARED_MULTI_OBJECT_PDI_DESIGN.md](SHARED_MULTI_OBJECT_PDI_DESIGN.md) for
the design analysis and [AGENT.md](AGENT.md) for implementation constraints.

## AnomalyDINO object crop pairs

The isolated [AnomalyDINO scorer](PDI-Bench-edited/src/pdi_eval/anomaly_scoring/README.md)
provides a reusable one-shot reference/query API, reference-feature caching,
JSON/JSONL batch inference, and a real-model smoke command. It uses the official
implementation pinned through `PDI-Bench-edited/third_party/AnomalyDINO`; it does
not invoke the segmentation, depth, tracking, or rigidity pipeline.
