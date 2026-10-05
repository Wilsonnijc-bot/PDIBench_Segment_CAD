# AnomalyDINO crop-pair scorer

`AnomalyDINOScorer` compares one normal reference image with a query image. The
backbone loads once per scorer. This module does not call the video pipeline.

```python
from object.scoring.anomalydino import AnomalyDINOScorer

scorer = AnomalyDINOScorer(model_name='dinov2_vitb14', device='cuda:0')
scorer.precompute_reference('reference.png')  # optional
result = scorer.score(reference_image='reference.png', query_image='query.png')
print(result['anomaly_score'])
# Optional: result = scorer.score('reference.png', 'query.png', return_map=True)
# Returns anomaly_score, patch_anomaly_map, and full-resolution anomaly_map.
```

Paths, PIL images, and uint8 RGB/RGBA NumPy arrays are supported. RGB channel
order is required. Transparent pixels are composited on black before upstream
preprocessing; alpha is not a feature mask. This is necessary for the project's
PNG crops, whose hidden RGB pixels retain source-scene content.

Native selected object-pair exports use `shared-canvas-native-pixels-v1`: trim
transparent exterior margins and center both crops on identically sized canvases.
No visible pixels are resampled and no contours or native pixel areas are changed.
This gives both images the same shorter-edge resize factor; it does not force
equal object sizes or improve the estimated frame-zero correspondence. The exact
gallery PNGs are the model inputs. `prepare_selected()` checks each pair's recorded
canvas dimensions and PNG hashes in `pair_geometry.json` before scoring. Historical
exports without this metadata remain explicitly labeled `legacy-independent-crops`.
Changing crop geometry requires new scores; the controlled comparison is under
`results/paired-crop-normalization-20261003` and the project GPU ledger.

## Installation and upstream provenance

```sh
git submodule update --init infrastructure/vendor/AnomalyDINO
python -m pip install -r object/scoring/anomalydino/requirements.txt
export PYTHONPATH="$PWD"
```

Official repository: <https://github.com/dammsi/AnomalyDINO>.
Fork: <https://github.com/Wilsonnijc-bot/AnomalyDINO/tree/integration/pdi-pairwise>.
Official starting commit: `b9d1c2648e3a5247437d4d953d907a8f3d994457`.
Pinned integration commit: `b2fef560b8b310835261abe2d8a0357823a58f40`.

The fork refactors the original `src.detection.run_anomaly_detection` inference
blocks into `extract_reference`, `build_knn_index`, `query_distances`, and
`score_query`. The original evaluation entry point also calls those blocks.
The project reuses `src.backbones.get_model` / `DINOv2Wrapper.prepare_image` /
`DINOv2Wrapper.extract_features`, `src.utils.augment_image`, the unchanged
`src.post_eval.mean_top1p`, and optional `src.utils.dists2map`.
Feature extraction and anomaly formulas are not reimplemented in this module.

The runtime requirements use the same upstream packages, with its supported
`faiss-cpu` alternative, NumPy 1.26.4 (a patch release of the requested 1.26
series), and OpenCV below 4.12 to preserve NumPy 1.x compatibility. The upstream
`argparse` requirement is omitted because it is part of Python's standard library.
The existing GPU environments are preserved; this run uses a separate virtual
environment inheriting the instance's existing PyTorch installation.

## Exact method

The reusable default backbone is upstream's `dinov2_vits14`. The October 2 crop
experiment explicitly uses **`dinov2_vitb14`**, the instance's existing
`facebook/dinov2-base` weights at revision
`f9e44c814b77203eaa57a6bdbbd535f21ede1415`.

Preprocessing is the official DINOv2 wrapper: bicubic resize with antialiasing
to a **448-pixel short edge**, preserving aspect ratio; ImageNet normalization
(mean 0.485/0.456/0.406, standard deviation 0.229/0.224/0.225); crop the bottom
and right remainder so dimensions are multiples of the 14-pixel patch size.
Features are the last normalized intermediate-layer patch tokens returned by
`get_intermediate_layers`; the class token is excluded.

The default crop configuration follows upstream's custom-data
`agnostic_no_mask` procedure: PCA masking off, reference rotations at
0/45/90/135/180/225/270/315 degrees using the official `augment_image` function.
These are augmentations of **one** supplied reference, not additional shots.
Set `rotation=False` or `--no-rotation` to use its single orientation.
No training, fine-tuning, or cross-video memory bank is used.

For each query patch, the official FAISS 1NN search computes squared Euclidean
distance between L2-normalized features, then divides it by two. This equals
`1 - cosine_similarity`. Let `N` be the number of query-grid patches and
`k = floor(0.01 * N)`. The scalar is the mean of the largest `k` raw patch
distances; when `k == 0`, it is their maximum. The unchanged official
`mean_top1p` performs this aggregation. The scalar is calculated **before**
resizing/smoothing the map. Optional dense maps use linear resize to the query
image dimensions followed by Gaussian smoothing with `sigma=4`.
If PCA masking is enabled, masked query positions are zero-filled, matching
upstream's returned evaluation score (the denominator remains the full grid).

Each reference's processed RGB dimensions and bytes are SHA-256 hashed. The
scorer caches its own FAISS index, including normalized reference features and
any GPU resource owner. Identical content from a path, PIL image, or array
reuses that bank. The default 16-entry LRU bounds memory; eviction or explicit
`clear_reference_cache()` requires recomputation next time. No bank concatenates
different references. Shape-specific frame-zero crops are distinct references
when their RGB content differs, even within the same video.

## Pair and selected-video runners

A JSON list or JSONL file may contain arbitrary paths, without any core filename
convention:

```json
[{"video_id": "example", "reference_crop": "reference.png", "query_crop": "query.png"}]
```

```sh
python -m object.scoring.anomalydino run \
  --pairs pairs.json --input-root /path/to/images --output-root /path/to/results \
  --model-name dinov2_vitb14
```

Outputs: `pairs.jsonl` (one score per supplied pair), `summary.json` (settings,
counts, timing, provenance, reference-cache statistics), and `videos.csv`
(descriptive mean/maximum of supplied pair scores, not an official video metric).
There are no labels, rigidity scores, or AUROC calculations.

The current crop gallery uses ten frames with quotas [0, 2, 2, 2, 4] and
350 scored pairs from 35 videos. Its latest run is
`results/anomalydino-object-crops-10frames-20261002`; the original five-frame
experiment remains preserved separately.

For this project's frozen selected crops:

```sh
python -m object.scoring.anomalydino prepare \
  --crop-root results/object-reference-crops-20261001 \
  --output results/anomalydino-object-crops-10frames-20261002/metadata/pair_manifest.json

python -m object.scoring.anomalydino run \
  --pairs results/anomalydino-object-crops-10frames-20261002/metadata/pair_manifest.json \
  --input-root results/object-reference-crops-20261001 \
  --output-root results/anomalydino-object-crops-10frames-20261002 \
  --model-name dinov2_vitb14
```

The thin pairing layer uses each selected `frame0_shape_crop.png` and its
`current_available.png`; it does not select new frames or alter the crops.
Default exclusions are all three `0001` videos, `COSMOS2.5_0005`,
`COSMOS2.5_0010`, and `COSMOS2.5_0065`. All 45 videos remain in the manifest:
6 excluded, 4 unavailable, 35 scored with ten pairs each (350 total).
Unavailable videos are `COSMOS2.5_0021`, `LVP_ROBOWM_0010`,
`LVP_ROBOWM_0015`, and `LVP_ROBOWM_0060`. Their missing references are not scored.
Input hashes are checked before inference, and finished JSON outputs are atomic.

## Existing GPU checkpoint and validation

`--dino-repo /path/to/official/dinov2 --checkpoint /path/to/weights.pth` loads
local official DINOv2 code and weights without downloading another model.
The `convert_checkpoint` command reverses Hugging Face's official parameter
renaming/QKV split for the observed base model. `safetensors` and `transformers`
are required only for this optional conversion/validation step; they were
already installed on the supplied instance. It strictly loads all 175 tensors
and checks actual patch features against the cached HF model.

```sh
python -m object.scoring.anomalydino smoke \
  --reference reference.png --query query.png --model-name dinov2_vitb14 \
  --dino-repo /path/to/official/dinov2 --checkpoint /path/to/converted.pth

python -m unittest discover \
  -s object/tests -p test_anomalydino.py -v
```

The smoke test checks self/query finite float scores, dense map shapes, and
reference-cache reuse, without assuming the query score must exceed the self
score. The five local tests also compare to the untouched official evaluation
routine for four rotation/masking configurations. The Git-history parity test
requires the initialized upstream submodule.

Method choices and deviations are explicit: the instance's base backbone
instead of upstream's small default; official custom-data no-mask/rotation
configuration; black alpha compositing before inference; CPU FAISS; optional
local checkpoint loading and weight-format conversion. The upstream anomaly
calculation, feature extraction, and scalar aggregation are preserved.

## Changed files

- Added native package files: `__init__.py`, `anomalydino.py` (API/cache),
  `runner.py` (pairing/output), `__main__.py` (CLI/smoke),
  `convert_checkpoint.py` (existing-weight conversion), `requirements.txt`,
  and this `README.md`.
- Added `PDI-Bench-edited/tests/test_anomalydino.py` and
  `scripts/run_anomalydino_gpu.sh`.
- Added the pinned `PDI-Bench-edited/third_party/AnomalyDINO` submodule;
  updated `.gitmodules`, root `README.md`, and `experiment_GPU_record.md`.
- In the fork only: refactored `src/detection.py`; added optional local loading
  in `src/backbones.py`; changed `src/post_eval.py`'s import to package-relative;
  added `INTEGRATION.md`. All other upstream files remain unchanged.

To attach a completed run directly to the existing crop gallery and selections:

```sh
python -m object.replay.annotate_crops \
  --crop-root results/object-reference-crops-20261001 \
  --results-root results/anomalydino-object-crops-10frames-20261002
```

`annotate_crops.py` validates pair hashes and selection relationships, attaches
the original anomaly scores and the sum of scored pairs for each video, and
regenerates the existing gallery. `frame_selection_gallery.py` displays the
pair scores and video sums prominently, and checks crop hashes on regeneration.
Summed video scores use unrounded pair values and are a descriptive aggregation,
not an additional upstream anomaly formula. Excluded and unavailable videos
have null totals. Similarity is omitted from the current crop export.
