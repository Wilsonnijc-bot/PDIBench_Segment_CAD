# Direct Simple3D evaluation

Uses the vendored authors' Simple3D implementation. See
[`third_party/Simple3D/INTEGRATION.md`](../third_party/Simple3D/INTEGRATION.md)
for the upstream revision and the single tensor-device compatibility patch.

## Prepare existing inputs

From the repository root, in an environment containing NumPy, OpenCV and PyTorch:

```bash
python experiments/prepare_simple3d_inputs.py --run-id NEW
```

Object frames come verbatim from the supplied AnomalyDINO pair manifest and crop
selection files. Every available selected sequence is included, even if its prior
AnomalyDINO scores were excluded; that prior status is recorded. Inputs without usable
selected frames remain recorded as unavailable. The normal
reference is the **first selected frame**, not video frame zero or the AnomalyDINO
shape-mapped RGB reference. The exact saved `available_pure_object` masks are used.

Link5 uses the existing frozen 45-video selection, native `contracts.video_path`
mapping, and saved updated `base_segmentation.npz` masks. Native `frame_measurements`
defines valid nonempty masks (height > 0, more than 10 pixels); select ten uniformly
distributed valid mask frames. No new video-selection system, segmentation, VLM,
or occlusion detector is run. Truncation is recorded, not silently discarded.

Prepared archives contain only selected masks and their original frame IDs. Source
hashes and paths remain in the manifest. Stage them to the GPU, plus the original
`link5.dae` and its `LICENSE.franka_description`, and the new code/official fork.
The GPU must have the existing PDI package, MegaSaM implementation and checkpoints.
Stage the current `PDI-Bench-edited/src/pdi_eval` source into the experiment checkout;
an older GPU source checkout may lack current saved-mask/crop helpers. Its
`third_party/mega_sam` may point to the existing MegaSaM installation. Additional
native extensions can be installed with `install_simple3d_gpu_dependencies.sh`.

## Run independently

```bash
python experiments/run_simple3d_object_deformation.py --inputs results/simple3d-NEW/inputs --run-id NEW --workers 2
python experiments/run_simple3d_link5.py --inputs results/simple3d-NEW/inputs --run-id NEW --reference both --cad assets/link5.dae --workers 2
# Or score modes independently with the same run ID, sharing identical fresh clouds:
python experiments/run_simple3d_link5.py --inputs results/simple3d-NEW/inputs --run-id NEW --reference cad --cad assets/link5.dae
python experiments/run_simple3d_link5.py --inputs results/simple3d-NEW/inputs --run-id NEW --reference first_frame
```

Each new run ID creates new experiment directories. Existing output directories
require explicit `--resume`. Resume checks input and Simple3D source/settings hashes.
`--cases ID ...` runs a pilot/subset without changing the complete input manifest.
`--summarize-only` rebuilds aggregate CSV/JSON from completed per-video artifacts.

For the hosted instance, use `bash experiments/launch_simple3d_gpu.sh` after preflight.
It launches two parallel video workers in tmux, records durable logs and exit status,
and runs object and both link5 modes concurrently (one video worker per workload
when the total worker count is two). A run may be left active after verified cases;
running summaries are partial and include pending IDs. It never labels a detached
process as complete.

Validate paths, current runtime helper imports, all video/mask hashes and the actual
method before launch:

```bash
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.0 python experiments/preflight_simple3d.py --inputs results/simple3d-NEW/inputs --cad assets/link5.dae --output results/simple3d-NEW/metadata/execution/preflight.json
```

After completed cases exist, check actual native back-projection, mask membership,
scalar aggregation and shared cloud paths with `verify_simple3d_outputs.py`. Retrieve
results from the Mac using `python experiments/collect_simple3d_results.py --run-id NEW`;
add `--watch` to keep collecting until the remote exit marker appears. Stable case
and geometry folders are verified against remote SHA-256 hashes after copying.
`finalize_simple3d.py --run-id NEW --watch` can run in another tmux session: it
waits for the batch exit, audits every cloud (including centroids of small failed
clouds), verifies all successful native score artifacts, and writes the combined
`simple3d-NEW/metadata/summary.json` and cloud-audit CSV/JSON. The collector waits for this
finalization marker in watch mode. The current run's collector also updates its
existing GPU-ledger entry when actual terminal status is available.

## Geometry and preprocessing

Selected full RGB frames are decoded from the hash-verified original video and
encoded together into one ten-frame sequence. Full scene context stays available
to MegaSaM; saved masks filter XYZ after native back-projection. Each invocation
uses a unique scene name and `MegaSamWrapper.infer_shared(cache_dir=None)`. No old
geometry cache or object cloud is read. Both link5 reference modes use the same
run-specific `link5_cad/observed/` directory, guarded by a per-video file lock.

The existing MegaSaM wrapper runs Depth-Anything, UniDepth, DROID, RAFT and CVD.
If its existing CVD fallback is triggered, raw DROID geometry is explicitly labeled
`raw_DROID_CVD_unavailable`. All-zero fallback geometry fails. Save native depth,
poses, intrinsics, selected sequence, and masked clouds; delete only unique owned
MegaSaM scratch after saving to limit disk consumption.

Remove nonfinite XYZ, zero points, nonpositive camera depth, and isolated outliers
(Open3D 20 neighbors, four standard deviations). Apply the official `pc_normalize`
utility independently to every observed and CAD cloud: centroid subtraction and
unit maximum radius, no rotation. Select one real point per 0.005 normalized voxel,
then seeded uniform sampling without replacement up to 8192 points. Keep raw XYZ,
normalized XYZ, RGB and native pixel indices. Fewer than 128 real voxel points fail
instead of synthesizing points. Official FPS may repeat **centers** on small clouds;
its 4096 feature centers and 1024 score-smoothing centers remain unchanged.

The original CAD mesh is loaded with scene transforms and uniformly sampled by
surface area at 100,000 points, then processed identically. CAD has no MegaSaM status.
No mesh optimization, non-rigid/rigid registration, learned registration or ICP.

Per-cloud unit-radius normalization removes global size anomalies and can amplify
partial-mask shape differences. It provides compatible numerical units, not metric
CAD calibration. Fixed voxel spacing/cap does not eliminate the density and visible
surface domain gap between complete CAD and partial video geometry. Those are
limitations of this direct evaluation, to expose before adding methods.

## Outputs

One root: `results/simple3d-RUN_ID/`, containing `inputs/`, `object/`,
`link5_cad/`, `link5_first_frame/`, `metadata/`, `correlation/` and `replay/`.
Each experiment has aggregate `comparisons.csv/json`,
`summary.json`, `run.json`, reference prototypes, per-pair comparison JSON and raw
point-score NPY. Geometry metadata records original/valid/final counts, centroid,
scale, source hashes, native reconstruction status and failures. RGB crops retain
alpha masks and are accompanied by full binary masks and original decoded frames.

`simple3d-RUN_ID/link5_cad/combined_comparison.csv/json` joins all 450 selected
frame rows; first observed frames have no first-frame-reference score by design.
Representative first/middle/last comparisons from one video per dataset get static
3D reconstruction/anomaly PNGs with raw score color bars and scalar score titles.
The CAD root includes its sampled cloud visualization. Larger anomaly scores mean
larger distances from normal prototypes; no calibration or deformation threshold
is claimed. Distributions, valid counts and native CVD status should be inspected
alongside mask errors, viewpoint change, occlusion and reconstruction noise.

## Label correlation and interactive replay

After collecting a complete run, calculate video-level Pearson/Spearman
correlations against the selected-45 workbook (requires NumPy, SciPy and openpyxl):

```bash
python scripts/analyze_simple3d_correlation.py --run-id NEW
```

The workbook's `Selected 45!AB` is the link5/forearm label and `AF` is the object
label. Mean successful comparison score is the primary video aggregation. Outputs
include maximum-score and complete-video sensitivity, binary AUROC excluding 0.5
labels, coverage, matched video rows and source hashes. Missing scores are never
imputed. CAD sensitivity also restricts scoring to the same nine later frames used
by the first-frame reference.

Build the offline replay from saved artifacts (requires NumPy; no inference):

```bash
python scripts/build_simple3d_replay.py --run-id NEW
```

Open `results/simple3d-NEW/replay/index.html` directly in a browser. Keep its
`cases/` scripts, `plotly.min.js`, and sibling experiment folders in place; image and
raw-evidence links are relative. The replay uses one shared observed cloud per
link5 frame and switches its saved CAD/first-frame point score array. All sampled
points are exported in their original order with exact float32 XYZ and scores.
No visualization resampling or new anomaly inference is performed. The default
red points are the 80 highest-scoring contributors to the official MiniShift
scalar. A fixed-range continuous heatmap and original RGB mode are also available.
Reference clouds, source images/crops, mask overlays, point/pixel inspection,
frame history, failures and CSV downloads make each saved measurement inspectable.

Historical GPU paths remain in the raw result records as execution provenance.
`experiments/simple3d_layout.py` resolves those recorded paths to the consolidated
local folders. Readers also accept the previous layout for older remote runs; new
producers and collectors write only the single run root.
