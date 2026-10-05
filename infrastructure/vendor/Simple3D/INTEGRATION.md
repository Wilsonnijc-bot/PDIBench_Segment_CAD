# PDI direct evaluation fork

Upstream: https://github.com/hustCYQ/MiniShift-Simple3D
Hosted fork: https://github.com/Wilsonnijc-bot/MiniShift-Simple3D
Pinned upstream commit: `5951e097ed5f6c52c1e058490f8c64f9b6323501`.

The adapter is `experiments/simple3d_adapter.py`. It calls `FPFHFeatures.collect_features`,
`run_coreset`, and `predict` directly. FPFH, MSND (including the upstream repeated
second-scale concatenation when `num_MSND=2`), LFSA, native FPS/KNN, 5% greedy coreset,
nearest-prototype distances, interpolation, native spatial score smoothing, and
MiniShift top-80 score aggregation remain the authors' implementation.

One compatibility patch in `feature_extractors/features.py` moves `s_map` onto
`idx.device` before indexing. The original code creates CPU scores and CUDA KNN
indices, which PyTorch 2.1 rejects. This changes tensor placement only. Source hashes
are recorded in each run. No training, fine-tuning, descriptor substitution,
registration, or additional anomaly model is used.

The unavailable upstream `unlimblue/KNN_CUDA` dependency is installed from its preserved
fork https://github.com/willxxy/knn_cuda at
`8b21dbfc86988f56588a5ed8ca3cc354122c5656`. Its original CUDA kernels/API are used.
PointNet2 comes from https://github.com/erikwijmans/Pointnet2_PyTorch at
`b5ceb6d9ca0467ea34beb81023f96ee82228f626`.
