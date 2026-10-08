# Wilson server inventory

**Server:** `zy992@eris2n7.research.partners.org`  
**Folder:** `/PHShome/zy992/Wilson` (capital **W**)  
**Resolved folder:** `/PHShome_actual/z/zy992/Wilson`  
**Inspected:** 2026-10-06; recursive scan began at 2026-10-06T20:06:43.560185+00:00.  
**Scope:** files, symlinks, code, installed package metadata, dependency declarations, models, inputs, and experiment receipts beneath this folder. Inventory documents were generated after this snapshot and are not included in its counts.

## At a glance

The folder contains one deformation-detection project in **four workspaces**, **nine environment/build prefixes**, **45 input videos with 45 matching prompts**, pretrained models, reference assets, and the latest Link5 shape-codebook experiment.

The latest experiment prepared all 45 videos and trained two models. **Both models failed deformation-sensitivity validation. Full-video scoring and interactive replay export did not run.** The saved checkpoints are completed training artifacts, not validated detectors.

There are **303,377 regular-file paths** and **12,149 symlinks**. Regular-file lengths sum to **72.21 GiB**. Per-path allocated blocks sum to **98.60 GiB**; these totals do not deduplicate hardlinks and are not a quota measurement. Directory sizes below are sums of regular-file lengths, excluding symlink targets. The scan reported **0 filesystem errors**.

At inspection, `squeue -u zy992` returned no jobs. The SSH endpoint landed on `erishpc-login-001`; Linux kernel `5.14.0-687.47.1.el9_8.x86_64`. No environment installation, model inference, training, or Slurm submission was performed for this inventory.

## Folder map

```text
Wilson/
├── deformationdetection/
│   ├── inputs/                         45 videos + 45 prompts
│   ├── metadata/                       manifests, logs, hashes, cleanup receipts
│   ├── workspace/                      original staged robot/object pipeline
│   ├── workspace-timeout-policy-20261005/
│   ├── workspace-link5-20261005/
│   ├── workspace-link5-shape-codebook-four-frame0-20261006/
│   │   └── results/link5_shape_codebook/ latest preserved inputs + training + validation
│   └── results/                        currently empty
└── dependency/
    ├── env/                            isolated runtimes and build tools
    ├── source/                         third-party source and native builds
    ├── models/                         pretrained weights, tokenizer, CAD
    ├── references/                     robot-link, guard, palm, VLM references
    ├── cache/                          Conda, CUDA packages, Hugging Face assets
    ├── metadata/                       preparation logs and package snapshots
    ├── bin/                            ffmpeg symlink and Git wrapper
    └── private/vlm.env                  private VLM configuration; contents omitted
```

| Directory | Regular files | File lengths |
|---|---:|---:|
| `deformationdetection` | 8,752 | 7.02 GiB |
| `deformationdetection/inputs` | 90 | 746.76 MiB |
| `deformationdetection/metadata` | 425 | 52.60 MiB |
| `dependency/env` | 214,693 | 29.05 GiB |
| `dependency/source` | 17,386 | 1.87 GiB |
| `dependency/models` | 59 | 22.02 GiB |
| `dependency/references` | 144 | 12.38 MiB |
| `dependency/cache` | 62,287 | 12.24 GiB |

## Code and workspaces

Paths in this report are relative to `Wilson/` unless explicitly absolute.

| Workspace | Purpose | Python source files¹ | Regular files | File lengths |
|---|---|---:|---:|---:|
| `deformationdetection/workspace` | Original staged robot/object coordinator | 181 | 507 | 22.55 MiB |
| `deformationdetection/workspace-timeout-policy-20261005` | Separate timeout-policy snapshot | 183 | 276 | 9.33 MiB |
| `deformationdetection/workspace-link5-20261005` | Separate Link5-ready coordinator snapshot | 184 | 281 | 9.44 MiB |
| `deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006` | Latest four-frame0 Link5 experiment; includes saved run data | 239 | 7,173 | 6.20 GiB |

¹ Counts exclude results, bytecode, vendor and third-party trees. Source hashes show the snapshots differ; they must not be treated as interchangeable copies. The newest workspace has no root `README.md`, `pyproject.toml`, or `deformation_detect.py`; its experiment sources and result README are present. The other three have staged packaging files and `source-stage.json`.

### Main project ownership

| Code path inside a workspace | Responsibility |
|---|---|
| `robot/preprocessing/` | SAM3/DINOv2 robot-link masks, Link5 refinement, persistent Link7 masking |
| `robot/workflows/` | Geometry, tracking, scoring and coordination |
| `robot/scoring/` | Robot rigidity metrics |
| `robot/replay/` | Robot review/replay generation |
| `robot/experiments/` | Retained Link5 Simple3D, Link7 tracking/depth experiments; newest snapshot adds shape codebook |
| `object/preprocessing/` | Grounding, segmentation, occlusion and visible crop pairs |
| `object/scoring/anomalydino/` | Object anomaly scoring |
| `object/analysis/`, `object/replay/` | Object score analysis and occlusion replay |
| `object/experiments/` | Retained object rigidity and Simple3D branches |
| `infrastructure/deformation_detect/` | CLI, resumable coordinator, workers, parallelism and import isolation |
| `infrastructure/shared/` | Shared model adapters, geometry, tracking, replay and experimental helpers |
| `documentation/` | Architecture, dependency profiles, pipeline contracts, labels and reports |

Robot processing uses segmentation → geometry/tracking → rigidity scores. Object processing uses grounding → masks/tracks → occlusion/available pixels → crop pairs → AnomalyDINO. The main object path does not require object MegaSAM/rigidity; that remains a separate experiment.

### Entry points

In the original three staged workspaces, `deformation_detect.py` forwards to `infrastructure.deformation_detect.__main__`. The CLI exposes `list`, `run`, `coordinate`, `run-status`, `env-check`, `analyze`, and `stage`. Its project package is `deformation-detect-workspace==0.3.0`, Python `>=3.10`, with setuptools `>=68`. **`pyproject.toml` declares an empty runtime dependency list**; model dependencies are provisioned separately.

The registered interfaces in the original workspace are:

| Interface | Environment profile | Python module |
|---|---|---|
| `links.mask` | `sam3` | `robot.preprocessing.segmentation.sam3_dinov2_segment` |
| `links.persistent` | `sam3` | `robot.preprocessing.link7_persistent.pipeline` |
| `links.run` | `geometry` | `robot.workflows` |
| `links.score` | `geometry` | `robot.workflows.score_v1` |
| `links.replay` | `analysis` | `infrastructure.shared.replay.rigidity_replay` |
| `objects.mask` | `sam3` | `object.preprocessing.segmentation.segment` |
| `objects.occlusion` | `analysis` | `object.preprocessing.occlusion.occlusion` |
| `objects.crops` | `analysis` | `object.preprocessing.crop_pairs.reference_visible_pixels` |
| `objects.select` | `analysis` | `object.preprocessing.crop_pairs.frame_selection` |
| `objects.score` | `anomalydino` | `object.scoring.anomalydino` |
| `objects.replay` | `analysis` | `object.replay.occlusion.occlusion_replay` |
| `objects.rigidity` | `geometry` | `object.experiments.rigidity.workflows` |
| `links.simple3d` | `simple3d` | `robot.experiments.link5_simple3d.run_simple3d_link5` |
| `links.simple3d-pairs` | `simple3d` | `robot.experiments.link5_simple3d.run_simple3d_link5_pairs` |
| `objects.simple3d` | `simple3d` | `object.experiments.simple3d.run_simple3d_object_deformation` |
| `simple3d.prepare` | `analysis` | `infrastructure.shared.experimental.simple3d.prepare_simple3d_inputs` |
| `simple3d.verify` | `analysis` | `infrastructure.shared.experimental.simple3d.verify_simple3d_outputs` |
| `simple3d.collect` | `analysis` | `infrastructure.shared.experimental.simple3d.collect_simple3d_results` |
| `simple3d.analyze` | `analysis` | `infrastructure.shared.experimental.simple3d.analysis.analyze_simple3d_correlation` |

The labels `analysis` and `simple3d` are declared interface profiles; there are no separate prefixes with those names in `dependency/env/`. Their code is present, but this inspection did not establish a complete runnable environment for every retained branch.

The newest experiment is under `robot/experiments/link5_shape_codebook/`: preparation (`prepare.py`, `prepare_batch.py`, `refresh_masks.py`, `depth_filter.py`), reference/pose construction (`construct_reference.py`, `pose.py`, `audit_alignment.py`), training (`train.py`, `train_stage.py`, `training_loader.py`, `robot_structural.py`), validation (`sanity.py`, `structural_validation.py`), scoring (`score.py`, `full_video.py`), reports/replay/collection, and contract tests. Presence of scoring/replay code does not mean those stages ran.

## Installed dependencies

Environment interpreters are `dependency/env/<name>/bin/python`. Versions below come from live distribution metadata in each interpreter's search path, including inherited base packages; Torch CUDA labels come from its installed `version.py`. GPU libraries were not imported or exercised.

| Environment | Python | Torch / CUDA build | NumPy | Key visible packages | Base relationship |
|---|---|---|---|---|---|
| `geometry` | 3.10.21 | 2.1.0+cu118 | 1.26.4 | Geometry, CoTracker, MegaSAM; torchvision 0.16.0+cu118; transformers 4.41.2; xformers 0.0.22.post7+cu118; torch-scatter 2.1.2+pt21cu118 | Conda base |
| `sam3` | 3.12.14 | 2.10.0+cu128 | 2.2.6 | SAM3 0.1.4; torchvision 0.25.0+cu128; transformers 4.41.2; OpenCV headless 4.12.0.88 | Conda base |
| `qwen` | 3.12.14 | 2.10.0+cu128 | 2.2.6 | transformers 5.5.0; accelerate 1.15.0; local Qwen3.5-9B | Venv inherits sam3; overrides packages |
| `anomalydino` | 3.10.21 | 2.1.0+cu118 | 1.26.4 | FAISS CPU 1.8.0.post1; DINO/analysis packages inherited | Venv inherits geometry |
| `foundationpose` | 3.11.16 | 2.8.0+cu128 | 2.4.6 | nvdiffrast 0.4.0; warp-lang 1.18.0; Open3D 0.19.0; trimesh 5.1.1 | Separate Conda base |
| `shape-codebook` | 3.10.21 | 2.1.0+cu118 | 1.26.4 | MinkowskiEngine 0.5.4; Open3D 0.19.0; NumPy override 1.26.4 | Venv inherits geometry |
| `download` | 3.12.14 | — | — | Model acquisition/Hugging Face tooling; no Torch metadata found | Conda utility environment |

`compiler` and `shape-build` are toolchain prefixes without a `bin/python` interpreter. Both contain GCC/G++ **11.4.0**, binutils **2.40**, and a Linux sysroot **2.34**; `shape-build` also contains OpenBLAS **0.3.34**. These are the remaining two of the nine prefixes.

The three venvs have `include-system-site-packages = true`: deleting or changing `geometry` affects `anomalydino` and `shape-codebook`; changing `sam3` affects `qwen`. Qwen's transformer metadata overrides the base 4.41.2 installation with 5.5.0. The full package appendix records inherited origins and shadowed distribution metadata.

### Native and system dependencies

- MegaSAM/DROID/lietorch native modules and Torch-scatter belong to the Torch 2.1/CUDA 11.8 geometry stack. `dependency/metadata/native-imports.txt` records a prior successful DROID/lietorch import; this inventory did not rerun that GPU check.
- Shape-codebook uses the compiled MinkowskiEngine backend and FCGF features. FoundationPose has its own CUDA/rendering stack and source/build tree.
- `dependency/bin/ffmpeg` points into SAM3's `imageio_ffmpeg` binary, named `ffmpeg-linux-x86_64-v7.0.2`.
- `dependency/bin/git` loads the cluster module `git/2.45.1-GCCcore-13.3.0` and executes `/apps/software/git/2.45.1-GCCcore-13.3.0/bin/git`. This wrapper isolates Git's module dependencies from Torch workers.
- Slurm, Lmod and host libraries are cluster services outside `Wilson`; this report does not inventory the entire host OS.
- Cloud VLM calls require network services and private configuration. `dependency/private/vlm.env` exists with mode `0600`; its contents were not read or copied.

The repository's `documentation/gpu/DEPENDENCIES.md` is a historical requirements summary. The installed versions above take precedence when describing this server. Full observed package versions and all 12 collected upstream requirement/environment files are linked at the end.

## Third-party code

| Path under `dependency/source/` | Purpose | Observed Git HEAD |
|---|---|---|
| `infrastructure` | Canonical vendor copies: MegaSAM and AnomalyDINO | `No .git directory observed` |
| `mega_sam` | Separate MegaSAM native/base source copy | `No .git directory observed` |
| `cotracker` | CoTracker source and package build | `No .git directory observed` |
| `dinov2` | DINOv2 source | `No .git directory observed` |
| `FoundationPose` | Pose estimation, rendering, CUDA code and weights | `a1b694b83e633c2cb6115b9063d940a687759392` |
| `Shape-Anomaly-Codebook` | Upstream shape-codebook training/inference | `1b8de121de8e8801dd99b796127026c7dd279a10` |
| `MinkowskiEngine` | Sparse convolution C++/CUDA source and build | `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228` |
| `FCGF` | Pretrained sparse geometric feature backbone | `499813ea35f57d885cbc3458cd74fc57dae860c7` |
| `NVTX` | NVIDIA profiling markers/headers | `e170594ac7cf1dac584da473d4ca9301087090c1` |

Git HEAD identifies the checkout revision, not a guarantee of an unmodified working tree. For copied source without `.git`, historical dependency documentation records CoTracker `82e02e8029753ad4ef13cf06be7f4fc5facdda4d` and MegaSAM DINOv2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`; those are recorded provenance, not newly verified source equivalence.

Workspaces link `infrastructure/vendor/mega_sam` and `infrastructure/vendor/AnomalyDINO` to `dependency/source/infrastructure/vendor/`. The latest workspace additionally links `third_party/FoundationPose` and `third_party/Shape-Anomaly-Codebook` to their dependency source folders. Result geometry work directories also link back to MegaSAM. Copying a workspace alone will not carry all dependencies with it.

## Models, checkpoints and CAD

| Asset path | Function | File lengths |
|---|---|---:|
| `dependency/models/sam3_checkpoint/sam3.pt` | SAM3 segmentation | 3.21 GiB |
| `dependency/models/sam3_bpe/bpe_simple_vocab_16e6.txt.gz` | SAM3 tokenizer vocabulary | 207.25 KiB |
| `dependency/models/qwen_model` | Qwen/Qwen3.5-9B: four safetensors shards, tokenizer/configs | 18.00 GiB |
| `dependency/models/dino_directory` | DINOv2 model.safetensors + config.json | 330.30 MiB |
| `dependency/models/tracker_checkpoint/scaled_offline.pth` | CoTracker | 97.17 MiB |
| `dependency/models/anomaly_checkpoint/anomalydino_vitb14_cached.pth` | AnomalyDINO cached features/weights | 330.33 MiB |
| `dependency/models/franka_fer/link5.dae` | Franka Link5 CAD mesh | 1.37 MiB |
| `dependency/models/shape_codebook/fcgf-3dmatch-32feat.pth` | Official pretrained 32-D FCGF backbone | 66.89 MiB |
| `dependency/source/infrastructure/vendor/mega_sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth` | Depth-Anything weights stored inside source tree | 1.25 GiB |
| `dependency/source/infrastructure/vendor/mega_sam/checkpoints/megasam_final.pth` | MegaSAM weights stored inside source tree | 19.85 MiB |
| `dependency/source/infrastructure/vendor/mega_sam/cvd_opt/raft-things.pth` | RAFT weights stored inside source tree | 20.13 MiB |
| `dependency/source/FoundationPose/weights/2023-10-28-18-33-37/model_best.pth` | FoundationPose checkpoint | 65.06 MiB |
| `dependency/source/FoundationPose/weights/2024-01-11-20-02-45/model_best.pth` | FoundationPose checkpoint | 181.42 MiB |

`dependency/cache/huggingface/` also contains the UniDepth-v2-ViTL14 snapshot; it is a model asset despite living under `cache`. The download receipt records revision `1d0d3c52f60b5164629d279bb9a7546458e6dcc4`. Qwen's receipt records revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Exact cache files and symlink targets appear in the metadata JSON.

The latest training receipt identifies a frozen FCGF `ResUNetBN2C`, 32-D output, voxel size `0.0078125`, with pretrained SHA256 `a23cfd3d33b5aef2bd2518a51d16acb5a78e554ea89c0d32f052f99b86178e32`. It explicitly records `paper_exact: false`: the backbone substitutes official FCGF for the paper's MinkUNet34C.

## Input and reference data

`deformationdetection/inputs/videos/` contains **45 MP4 files**, totaling 746.74 MiB: 15 each under `COSMOS2.5/`, `COSMOS3/`, and `LVP/`. The paired prompt IDs use `COSMOS2.5`, `COSMOS3`, and `LVP_ROBOWM` prefixes.

Each source has these 15 case numbers: `0001`, `0005`, `0010`, `0015`, `0021`, `0025`, `0030`, `0035`, `0040`, `0044`, `0046`, `0054`, `0056`, `0060`, `0065`. Prompts use a source prefix plus the numeric filename. For example:

| Video path under `inputs/videos/` | Prompt path under `inputs/prompts/` |
|---|---|
| `COSMOS2.5/0001.mp4` | `COSMOS2.5_0001.txt` |
| `COSMOS3/0001.mp4` | `COSMOS3_0001.txt` |
| `LVP/0001.mp4` | `LVP_ROBOWM_0001.txt` |

All 45 source-prefixed video IDs have a matching prompt.

| Reference directory | Contents | Files | File lengths |
|---|---|---:|---:|
| `dependency/references/dino_refer_robot_link_first15` | Canonical/by-link images and debugging assets | 125 | 8.82 MiB |
| `dependency/references/vlm2` | VLM2 visual references | 3 | 3.02 MiB |
| `dependency/references/link5_guard` | Link5 guard reference image | 1 | 370.59 KiB |
| `dependency/references/palm` | Palm reference assets | 15 | 177.14 KiB |

Labels and experiment configuration also appear in staged `documentation/data/` and `robot/configs/` trees; exact present files are listed in the file appendix. References linked from a snapshot remain shared dependencies, not private copies.

## Latest experiment data and status

**Run root:** `deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006/results/link5_shape_codebook`.

All 45 videos have fresh guarded masks and MegaSAM/CVD observations. The QC receipt confirms **495 filtered clouds**, 11 selected observations per video (frame0 plus ten context frames), minimum **3,325 points**, finite clouds and preserved original depth. These are selected frames, not full-frame video scoring.

The normal reference uses only frame0 of `COSMOS3_0046`, `COSMOS2.5_0044`, `LVP_ROBOWM_0044`, and `COSMOS3_0056`: **34,280 pooled points**. Later frames do not enter reference training.

| Directory relative to run root | Meaning | Files | File lengths |
|---|---|---:|---:|
| `cases/` | Per-video masks, observations, geometry-work, preparation evidence | 6,142 | 4.88 GiB |
| `normal_reference/` | Four frame0 references; construction.json, pooled_frame0.npz/.ply | 14 | 11.60 MiB |
| `normal_training_data/` | Prepared normal training samples | 4 | 402.22 KiB |
| `splits/` | Reference/training split receipts | 2 | 148.15 KiB |
| `alignment/` | Shared alignment artifacts | 5 | 833.41 KiB |
| `reference_pose/` | Reference pose estimates | 4 | 284.50 KiB |
| `augmentation/` | Augmentation diagnostics | 16 | 3.78 MiB |
| `training/` | Original-augmentation training, checkpoints, losses and receipts | 34 | 626.65 MiB |
| `training_robot_structural/` | Structural-augmentation training, checkpoints, losses and receipts | 37 | 623.45 MiB |
| `sanity/` | Original model validation and test clouds | 50 | 10.00 MiB |
| `sanity_robot_structural/` | Structural model validation and test clouds | 48 | 9.85 MiB |
| `metadata/` | Slurm logs, telemetry, stage receipts, QC and collection evidence | 112 | 39.64 MiB |

The run currently contains **6,471 regular files** totaling **6.18 GiB**, including intermediate geometry-work and checkpoint history. The run README's 4,263-file/1,196,659,661-byte figure describes a collected subset, not the complete current server tree.

| Variant | Training | Final checkpoint | Validation |
|---|---|---|---|
| Original augmentations | Completed: 1,500 epochs, 6,000 optimizer batches | `training/link5/final.pt` | Failed deformation sensitivity |
| Robot structural augmentations | Completed: 1,500 epochs, 6,000 optimizer batches | `training_robot_structural/link5/final.pt` | Failed deformation sensitivity |

Both training folders also retain `best.pt`, checkpoints every 100 epochs through 1,500, codebook files, losses, configuration and completion receipts. The validation requires a deformation/normal score ratio of 1.20; recorded ratios stayed near 1.0. Pose alignment and rigid-invariance checks passed. The reason for the nearly constant response remains unresolved in the saved report.

Recorded execution used one H200 on `erishpc-gpu-001`, allocation `5621305`, with peak memory **69,905 / 143,771 MiB**. That is historical telemetry, not a measurement from this inspection. Allocation ended with exit 1 after the validation gates failed.

## Historical records, caches and missing outputs

- `deformationdetection/metadata/` retains coordinator manifests, group A/B logs, source-stage hashes, import-origin audits, transfer archives, and cleanup receipts. Its earlier `final-outcome.json` records nine complete cases and one disabled Link7 case from a different evaluation; it does not establish that those outputs still exist.
- `deformationdetection/results/` is empty. The preserved current outputs are inside the newest workspace's run root above.
- Cleanup receipts dated 2026-10-06 record deletion of previous Link5 outputs and obsolete review archives. Historical reports and paths can refer to those deleted artifacts.
- `dependency/cache/` still has `conda-geometry`, `conda-link5`, `conda-models`, `conda-sam3`, `cuda-packages`, and `huggingface`. Its total regular-file lengths are **12.24 GiB**. Conda caches may share hardlinks with environments; the Hugging Face cache includes required model files.
- Apple `._*` sidecars and quarantined copies remain in parts of the source/metadata trees; they are included in file counts, not treated as executable source.
- All symlinks outside environment/cache trees resolved at inspection. Absolute links depend on the current server layout.

## Complete supporting inventories

| File | Coverage |
|---|---|
| [PACKAGES.md](../../../infrastructure/deployment/eris/results/inventory-20261006/PACKAGES.md) | Every observed visible Python distribution by environment, origin, shadowed metadata, and every Conda package/build |
| [DEPENDENCY_DECLARATIONS.md](../../../infrastructure/deployment/eris/results/inventory-20261006/DEPENDENCY_DECLARATIONS.md) | All 12 collected third-party requirements/environment declarations; distinct from installed versions |
| [CODE.md](../../../infrastructure/deployment/eris/results/inventory-20261006/CODE.md) | Every first-party Python source path and SHA256 for all four workspaces; static import roots and snapshot differences |
| [FILES.md](../../../infrastructure/deployment/eris/results/inventory-20261006/FILES.md) | Every enumerated regular file in project/source/models/references/bin/metadata, plus their symlinks; grouped by directory |
| [inventory.json](../../../infrastructure/deployment/eris/results/inventory-20261006/inventory.json) | Recursive directory counts/sizes, symlinks including caches/environments, selected file metadata, and scan errors |
| [package_metadata.json](../../../infrastructure/deployment/eris/results/inventory-20261006/package_metadata.json) | Observed distribution versions, origins and declared transitive requirements, plus Conda package dependencies |

Installed individual package files and package-cache entries are summarized by directory and package rather than expanded into the file appendix. Private configuration contents, the login password, and unrelated home/data folders are excluded. This is an inventory, not a fresh GPU compatibility or model-accuracy test; source and data were inspected without changing experiment artifacts.
