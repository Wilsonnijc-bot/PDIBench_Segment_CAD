# Persistent mask add-on

`persistent_masking/` is a copy of the source package at
`/Users/nijiachen/Downloads/hierachicaldeformation/persistent_masking`
(September 23, 2026), excluding Python bytecode caches. Its refinement
algorithm has not been changed here.

The integration boundary is the package's completed full-video output:

```text
<persistent-work>/provenance.json
<persistent-work>/sam/<case>/seed_masks.npz   # masks[T,H,W]
```

After the package marks a case `completed_checks`, merge its mask into a new
canonical PDI segmentation archive:

```bash
PYTHONPATH=src python -m pdi_eval.experiment merge-mask \
  --video /path/to/source.mp4 \
  --base-segmentation /path/to/base-segmentation.npz \
  --persistent-work /path/to/persistent-work \
  --case Cosmos25_0003 \
  --output-npz /path/to/refined-segmentation.npz
```

The adapter checks source-video and mask hashes, frame count, dimensions, and
successful refinement status. It replaces `link7` across the full video,
preserves the other named link masks and IDs, and updates the union-mask fields.
It writes a small provenance JSON beside the output archive. Supply that new
archive as `--segmentation-npz` to the V1 scorer when ready.

The copied masking package retains its original model and asset path
assumptions. Running the mask-generation stages inside this checkout requires
the corresponding GPU models and source-project support modules. This adapter
only consumes a completed, validated mask run. Replay changes are a separate
step.
