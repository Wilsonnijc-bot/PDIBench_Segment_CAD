# PDI Benchmark Pipeline Status

Updated: 2026-08-20 CST

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

## GPU environment

Last known configured compute environment:

- GPU: NVIDIA RTX 4090
- PyTorch: 2.1.0+cu118
- CUDA: 11.8
- CoTracker checkpoint: `models/tracker/scaled_offline.pth`
- MegaSAM/Depth Anything/RAFT assets: installed under the PDI GPU root

`scripts/bootstrap_gpu.sh` no longer installs or validates SAM2. SAM3 uses its
separate pinned environment from `scripts/install_sam3_gpu.sh`.

## Verification state

Completed locally:

- Python static compilation of native multi-object modules and CLI;
- nine CAD matching and reconstruction replay tests passing;
- PyTorch-dependent joint/exact tests added, including proof that exact-group
  invokes the underlying video feature network once across isolated groups.

The current local shell does not have PyTorch, so the CoTracker test module is
skipped locally. It must run without a skip in the configured PDI GPU environment.

No GPU inference has been run for this new native multi-object implementation.
The first GPU validation must run `PDI_TRACKING_MODE=both`, retain cold- and
warm-cache timings, and compare all seven per-link reports before either mode is
accepted as the default.

## Active command

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
