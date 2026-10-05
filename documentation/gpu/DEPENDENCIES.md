# GPU and analysis environments

Extracted from the repository's installation scripts, wrappers and requirement files. These are recorded requirements, not a newly tested GPU lockfile. Unpinned or unavailable versions remain explicitly unknown.

| Profile | Python / Torch / CUDA recorded | Main dependencies |
|---|---|---|
| Geometry + CoTracker | Python 3.10; Torch 2.1.0, torchvision 0.16.0, torchaudio 2.1.0; CUDA 11.8 | NumPy 1.26.4, transformers 4.41.2, timm 0.9.16, einops 0.8.0, xformers 0.0.22.post7+cu118, torch-scatter, MegaSAM/droid/lietorch, CoTracker |
| SAM3 + DINOv2 masks | Python 3.12; Torch 2.10.0; CUDA 12.8 | SAM3 0.1.4, NumPy 2.2.6, OpenCV headless 4.12.0.88, transformers 4.41.2; torchvision version unpinned in original installer |
| AnomalyDINO | Python/Torch/CUDA pins not specified by reviewed sources | NumPy 1.26.4, OpenCV <4.12, torchvision, CPU FAISS, scipy, scikit-learn, Pillow, tifffile, pandas, matplotlib |
| Simple3D | Extends the CUDA 11.8 geometry environment | Pointnet2 ops, knn_cuda, Open3D, tifffile 2025.5.10, trimesh 5.1.1, pycollada 0.9.3, ninja |
| Local VLM | Separate Qwen environment; exact pins not recovered | Current configuration points VLM1 to Qwen3.5-9B; cloud roles retain their configured APIs |
| TAPIP3D | Separate interpreter/repository/checkpoint; exact pins not recovered | Torch, NumPy, OpenCV and official TAPIP3D source environment |
| Analysis/replay | CPU environment | NumPy, scipy, Pillow, OpenCV (regular or headless), matplotlib, openpyxl, BeautifulSoup, PyYAML, rich; pytest for validation |

The standard-library CSV analysis and orchestration interface does not require the GPU packages. Interactive replay export may import shared scientific packages; its existing CPU tests were run in the temporary test environment.

Machine-readable constraints, source references and unresolved versions are in `environments.json`. `requirements/` contains separate extracted package lists. Keep these environments separate; combining the NumPy/Torch/CUDA profiles would change the recorded runtime assumptions.

Compiled dependencies and assets:

- CoTracker source: `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.
- MegaSAM DINOv2 source: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- DINOv2 mask model: `facebook/dinov2-base`, revision `f9e44c814b77203eaa57a6bdbbd535f21ede1415`.
- SAM3 checkpoint revision: `96f3e1b404ba14f2cfac60ee6ae87c269a7b7923`; original checkpoint and tokenizer hashes remain in the archived installer.
- Pointnet2 source: `b5ceb6d9ca0467ea34beb81023f96ee82228f626`.
- knn_cuda source: `8b21dbfc86988f56588a5ed8ca3cc354122c5656`.
- Geometry extensions require a compatible CUDA compiler, C++ build tools and matching Torch headers. The historical Simple3D build targeted compute capability 8.0; configure this for the actual GPU.
- Simple3D/Open3D may need `libegl1`; video generation needs ffmpeg. Provide existing model checkpoints, SAM3 tokenizer assets and reference images separately.

```bash
python -m pdibench env-check --profile geometry --python /path/to/geometry/bin/python --cuda
python -m pdibench env-check --profile sam3 --python /path/to/sam3/bin/python --cuda
python -m pdibench env-check --profile anomalydino --python /path/to/anomalydino/bin/python --cuda
```

The checker reports installed versions, missing modules, Python/CUDA mismatches and optional CUDA tensor availability. It does not install packages, download checkpoints, connect to a host, or claim that unknown pins reproduce a historical run. `infrastructure/shared/experimental/simple3d/preflight_simple3d.py` retains the more specific native-operation validation for a prepared Simple3D experiment.
