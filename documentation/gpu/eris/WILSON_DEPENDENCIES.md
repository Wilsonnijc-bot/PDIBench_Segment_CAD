# Wilson — models and third-party dependencies

Server: `zy992@eris2n7.research.partners.org` · inspected **2026-10-06**.  
All paths below are relative to `/PHShome/zy992/Wilson/dependency/`.

## Models and third-party code

| Dependency | Purpose | Code location | Model / asset location |
|---|---|---|---|
| **SAM3 0.1.4** | Robot/object segmentation | `env/sam3/lib/python3.12/site-packages/sam3/` | `models/sam3_checkpoint/sam3.pt`; vocabulary: `models/sam3_bpe/` |
| **CoTracker3** (`cotracker==3.0`) | Point tracking and occlusion evidence | `source/cotracker/`; installed in `env/geometry/` | `models/tracker_checkpoint/scaled_offline.pth` |
| **DINOv2** | Visual features and mask matching | `source/dinov2/` | `models/dino_directory/model.safetensors` |
| **AnomalyDINO** | Object anomaly scoring | `source/infrastructure/vendor/AnomalyDINO/` | `models/anomaly_checkpoint/anomalydino_vitb14_cached.pth` |
| **Qwen3.5-9B** | Local vision-language model | Transformers in `env/qwen/` | `models/qwen_model/`: four safetensors shards, tokenizer and configs |
| **MegaSAM** | Camera motion, depth and geometry | `source/infrastructure/vendor/mega_sam/`; separate native source in `source/mega_sam/base/` | `source/infrastructure/vendor/mega_sam/checkpoints/megasam_final.pth` |
| **Depth-Anything ViT-L/14** | Monocular depth within MegaSAM | `source/infrastructure/vendor/mega_sam/Depth-Anything/` | Its `checkpoints/depth_anything_vitl14.pth` |
| **UniDepth v2 ViT-L/14** | Depth estimation within MegaSAM | `source/infrastructure/vendor/mega_sam/UniDepth/` | `cache/huggingface/hub/models--lpiccinelli--unidepth-v2-vitl14/` |
| **RAFT** | Optical flow for geometry refinement | MegaSAM `cvd_opt/` | `source/infrastructure/vendor/mega_sam/cvd_opt/raft-things.pth` |
| **DROID backend / lietorch** | Native geometry and Lie-group operations | MegaSAM native/base source; compiled modules in `env/geometry/` | No separate model asset identified |
| **FoundationPose** | Link5 pose estimation | `source/FoundationPose/` | Its `weights/2023-10-28-18-33-37/model_best.pth` and `weights/2024-01-11-20-02-45/model_best.pth` |
| **Shape-Anomaly-Codebook** | Shape-codebook training and inference | `source/Shape-Anomaly-Codebook/` | Trained checkpoints live in the experiment result folders¹ |
| **FCGF** | Frozen 32-D geometric feature backbone | `source/FCGF/` | `models/shape_codebook/fcgf-3dmatch-32feat.pth` |
| **MinkowskiEngine 0.5.4** | Sparse convolution backend for FCGF | `source/MinkowskiEngine/`; installed in `env/shape-codebook/` | Compiled extension; no separate checkpoint |
| **NVTX** | Profiling markers and headers | `source/NVTX/` | No checkpoint |
| **Franka Link5 CAD** | Robot geometry for pose/alignment | Asset only | `models/franka_fer/link5.dae` |

¹ Result root, relative to `Wilson/`: `deformationdetection/workspace-link5-shape-codebook-four-frame0-20261006/results/link5_shape_codebook/`. Checkpoints are `training/link5/final.pt` and `training_robot_structural/link5/final.pt`. Both trainings completed; both failed deformation-sensitivity validation. The backbone is official FCGF `ResUNetBN2C`, substituted for the paper's MinkUNet34C.

## Runtime dependencies

Interpreters: `env/<environment>/bin/python`. These are observed installed versions, including inherited packages.

| Environment | Python | PyTorch / CUDA build | NumPy | Main supporting dependencies |
|---|---|---|---|---|
| `sam3` | 3.12.14 | 2.10.0 / CUDA 12.8 | 2.2.6 | torchvision 0.25.0; transformers 4.41.2; OpenCV headless 4.12.0.88 |
| `geometry` | 3.10.21 | 2.1.0 / CUDA 11.8 | 1.26.4 | torchvision 0.16.0; transformers 4.41.2; xformers 0.0.22.post7+cu118; torch-scatter 2.1.2+pt21cu118; Open3D 0.19.0 |
| `qwen` | 3.12.14 | Inherits `sam3` | 2.2.6 | transformers 5.5.0; accelerate 1.15.0 |
| `anomalydino` | 3.10.21 | Inherits `geometry` | 1.26.4 | FAISS CPU 1.8.0.post1; scientific/image-processing packages |
| `foundationpose` | 3.11.16 | 2.8.0 / CUDA 12.8 | 2.4.6 | torchvision 0.23.0; nvdiffrast 0.4.0; warp-lang 1.18.0; Open3D 0.19.0; trimesh 5.1.1 |
| `shape-codebook` | 3.10.21 | Inherits `geometry` | 1.26.4 | MinkowskiEngine 0.5.4; FCGF; Open3D 0.19.0 |

- **Shared Python libraries:** Pillow, OpenCV, SciPy, scikit-learn, matplotlib, imageio/imageio-ffmpeg, PyYAML, Hugging Face Hub, safetensors, timm, einops, tqdm, requests, rich and openpyxl. Versions vary by environment.
- **Acquisition/build tools:** `env/download/` (Python 3.12.14); `env/compiler/` and `env/shape-build/` (GCC/G++ 11.4.0, binutils 2.40); OpenBLAS 0.3.34 in `shape-build`; Ninja and CUDA libraries in the relevant runtimes.
- **Server tools:** Slurm and Lmod; `bin/git` wraps Git 2.45.1; `bin/ffmpeg` links to the SAM3 environment's ffmpeg 7.0.2 binary.
- **External VLM services:** private configuration is in `private/vlm.env`; credential values are omitted.

`qwen` inherits `sam3`; `anomalydino` and `shape-codebook` inherit `geometry`. Preserve these base environments. Workspace vendor/third-party directories link to shared `dependency/source/` folders.

For every installed package and exact source revision, see [the full inventory](WILSON_SERVER_INVENTORY.md), [package versions](../../../infrastructure/deployment/eris/results/inventory-20261006/PACKAGES.md), and [upstream dependency declarations](../../../infrastructure/deployment/eris/results/inventory-20261006/DEPENDENCY_DECLARATIONS.md). GPU compatibility was not retested during the inventory.
