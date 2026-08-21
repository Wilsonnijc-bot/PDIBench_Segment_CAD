# PDI-Bench Native Multi-Object Pipeline

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

This repository evaluates the seven rigid Franka FER links in one video-level
PDI run. CAD-guided SAM3 segments every link once, MegaSAM reconstructs the
full video once, and PDI reports scale, trajectory, rigidity, and perspective
metrics separately for each link in the same world-coordinate frame.

All editable model and metric code is native to `PDI-Bench-edited/`. The
upstream comparison checkout is pinned as the read-only `PDI-Bench-original/`
submodule. The former
`scripts/adapters/` layer and retired manual, RobotSeg, DINO, and SAM2 launch
paths have been removed.

## Dataflow

```text
link1.dae ... link7.dae
        |
        v
CAD-guided SAM3 once
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

## CAD Inputs

The benchmark includes `link1.dae` through `link7.dae` from Franka's
[`franka_description`](https://github.com/frankarobotics/franka_description/tree/main/meshes/robots/fer/visual),
pinned at commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9`.

Files and expected hashes are stored under:

```text
PDI-Bench-edited/assets/cad/franka_fer/
PDI-Bench-edited/configs/sam3-cad-franka.yaml
```

SAM3 does not consume Collada directly. The native CAD module renders
deterministic multiview silhouettes and matches SAM3 proposals to unique links.

## Run From The Mac

Run both CoTracker modes for a direct metric and speed comparison:

```bash
PDI_SKIP_SAM3_INSTALL=1 \
PDI_VIDEO_NAME=0000.mp4 \
PDI_TRACKING_MODE=both \
bash scripts/run.sh
```

Run one mode only:

```bash
PDI_TRACKING_MODE=joint-query bash scripts/run.sh
PDI_TRACKING_MODE=exact-group bash scripts/run.sh
```

Generate only the SAM3/CAD archive:

```bash
PDI_SAM3_SEGMENT_ONLY=1 bash scripts/run.sh
```

## Separate DINOv2 Reference Pipeline

The reference-conditioned pipeline is independent of the manual/CAD SAM3
launcher. Its reference groups are discovered under:

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
PYTHONPATH=src python evaluation/run_multi_object.py \
  --config configs/default.yaml \
  --input /path/video.mp4 \
  --segmentation-npz /path/segmentation.npz \
  --output-dir /path/output \
  --geometry-cache-dir /path/megasam-cache \
  --tracker-checkpoint /path/scaled_offline.pth \
  --tracking-mode both
```

## Outputs

Mac-local results are written to:

```text
results/sam3-cad-multi/<video>/<manifest-hash>/
|-- metrics.json
|-- timing.json
|-- manifest.json
|-- run_config.yaml
|-- cotracker_joint-query.npz
|-- cotracker_exact-group.npz
|-- console.log
`-- replay/
    |-- combined_joint-query.mp4
    `-- combined_exact-group.mp4
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
