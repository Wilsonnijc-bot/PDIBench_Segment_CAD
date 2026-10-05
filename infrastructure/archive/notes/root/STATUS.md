# PDI Benchmark Pipeline and GPU Status

Updated: 2026-08-21 15:45 CST (UTC+08:00)

## Current implementation

The active, unverified pipeline is now native to `PDI-Bench-edited/`. The
pristine upstream comparison baseline is kept in `PDI-Bench-original/`. The former
`scripts/adapters/` layer and the disabled manual, RobotSeg, DINO, and SAM2
launch paths have been removed.

Implemented locally:

- pinned Franka FER `link1.dae` through `link7.dae` CAD assets;
- one CAD-guided SAM3 pass producing `object_masks[T,N,H,W]`;
- one versioned, content-addressed MegaSAM geometry reconstruction per video;
- per-link camera depth derived from shared world pointmaps and camera poses;
- one union-excluding shared background query set;
- `joint-query` CoTracker mode;
- `exact-group` CoTracker mode with one replayed video-backbone feature pass;
- independent scale, trajectory, rigidity, VP, and PDI reports per link;
- direct exact-minus-joint metric deltas, grade changes, timing, and peak memory;
- numeric offset-based track archives for both modes;
- combined reconstruction replay per requested mode;
- one Mac launcher and one native benchmark CLI.

## Current GPU

The active machine was queried directly on 2026-08-21 at 15:44 CST.

| Item | Current value |
| --- | --- |
| Host | `36.140.33.200:46110` |
| Container | `autodl-container-4c754cbcba-9294adba` |
| GPU | NVIDIA A100-PCIE-40GB |
| Driver | 580.105.08 |
| VRAM | 40,960 MiB total; 0 MiB used; 40,442 MiB free after the diagnostic |
| GPU utilization | 0% after the diagnostic |
| GPU temperature | 30 C before the diagnostic |
| Data filesystem | 50 GB total; 31 GB used; 20 GB available (62% used) |
| GPU project root | `/root/autodl-tmp/pdi` |

Environment sizes and runtime versions:

| Environment | Size | PyTorch | CUDA runtime | Purpose |
| --- | ---: | --- | --- | --- |
| `/root/autodl-tmp/pdi/env/sam3` | 7.6 GB | 2.10.0+cu128 | 12.8 | SAM3 0.1.4 segmentation |
| `/root/autodl-tmp/pdi/env/pdi-bench` | 7.3 GB | 2.1.0+cu118 | 11.8 | PDI, MegaSAM, CoTracker, Depth Anything |

The model directory is 4.6 GB. Important pinned SAM3 assets:

| Asset | SHA-256 |
| --- | --- |
| `models/sam3/sam3.pt` | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| `models/sam3/bpe_simple_vocab_16e6.txt.gz` | `1205dae6dae721092a9df0da6e215a80a185fbd9be8e511f7891fa73ad5b28ab` |

SAM3 was downloaded from the pinned ModelScope revision. MegaSAM, UniDepth,
Depth Anything, RAFT, and CoTracker assets are installed. The pristine original
PDI-Bench import and all 18 GPU tests passed before segmentation testing.
`scripts/bootstrap_gpu.sh` does not install SAM2; this project uses SAM3 only.

## Verification state

Completed:

- Python static compilation of native multi-object modules and CLI;
- nine CAD matching and reconstruction replay tests passing;
- PyTorch-dependent joint/exact tests added, including proof that exact-group
  invokes the underlying video feature network once across isolated groups;
- the first five input videos (`0000.mp4` through `0004.mp4`) are staged locally
  and on the GPU under `runs/cosmos-2.5/videos/`;
- SAM3 model loading, single-prompt segmentation, and a seven-query first-frame
  diagnostic have run on the A100.

The current production SAM3/CAD implementation is not valid for the seven-link
benchmark yet. Its single text prompt returned one whole-arm proposal, which CAD
post-ranking mislabeled as `link5`. No PDI, MegaSAM, Depth Anything, CoTracker,
or A/B/C/D benchmark inference has been accepted from that mask.

The first-frame seven-query diagnostic for `0000.mp4` returned seven non-empty
masks in 5.83 seconds after model load. It uses one loaded SAM3 model and seven
independent `white robotic arm link` plus positive-box queries, followed by
per-link CAD silhouette matching:

| Link | SAM3 score | Area (px) | CAD similarity | Mask inside box |
| --- | ---: | ---: | ---: | ---: |
| link1 | 0.875 | 11,969 | 0.380 | 95.6% |
| link2 | 0.914 | 19,620 | 0.518 | 100.0% |
| link3 | 0.887 | 34,212 | 0.694 | 99.0% |
| link4 | 0.914 | 30,179 | 0.710 | 98.3% |
| link5 | 0.863 | 23,181 | 0.091 | 99.7% |
| link6 | 0.855 | 27,860 | 0.503 | 95.6% |
| link7 | 0.902 | 11,022 | 0.667 | 100.0% |

Maximum pairwise mask IoU is 0.0503. Duplicate overlap is 5,774 pixels, or 3.79%
of the seven-mask union. These statistics show good region separation, but the
result remains diagnostic-only: `link5` has poor CAD agreement, the link boxes
are pose-specific, and the current `link7` mask includes the gripper rather than
isolating only the requested `link7.dae` wrist geometry.

Diagnostic artifacts are under
`results/cosmos-2.5/0000/seven-link-masking-performance/`. The source metrics are
in `summary.json`; `seven-mask-overlay.jpg` shows the combined masks, and
`seven-link-cad-comparison.jpg` places each mask beside its closest CAD render.

Required next gate: derive link localization from calibrated CAD projection or
robot kinematics, isolate link7 from the hand/gripper, resolve link5, then verify
the propagated masks on multiple frames before starting A/B/C/D.

## Planned benchmark command

```bash
PDI_SKIP_SAM3_INSTALL=1 \
PDI_VIDEO_NAME=0000.mp4 \
PDI_TRACKING_MODE=both \
bash scripts/run.sh
```

Expected Mac-local result:

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
