# Architecture and migration record

This migration establishes physical ownership and direct owner imports while preserving scientific behavior. `layout.json` records each original Git path (`old`), current canonical path (`new`), compatibility alias (`compat`), category, and pre-migration source hash. Historical aliases are archived under `infrastructure/archive/compat/`; they are not runtime dependencies. See [the direct-import audit](DIRECT_IMPORT_MIGRATION.md).

The top level separates `robot/`, `object/`, shared runtime `infrastructure/`, and supporting `documentation/`. Publication tooling, GPU requirements, architecture records, labels, and reference images live under `documentation/`. Existing public `docs/` and `web-assets/` paths remain stable. The root `results/` directory preserves paths embedded in historical artifacts.

## Pipeline boundaries

- **Robot links:** link2–link7 segmentation, link5 guard/refinement, link7 persistent masks, orchestration of shared MegaSAM/CoTracker, rigidity/PDI scoring and interactive replay.
- **Objects:** grounding and object masks, saved object-track consumers, occlusion, visible-pixel mapping, crop selection/normalization, AnomalyDINO scoring and replay/gallery export.
- **Shared:** actual reused model adapters, numerical routines, archive contracts, reference localization and replay infrastructure. Similar-looking DINO implementations retain separate identities until an intentional equivalence review.
- **Experiments:** link5 Simple3D, link7 TAPIP3D, link7 depth filtering and tracker/filter comparisons, object rigidity and object Simple3D. Shared Simple3D code has one implementation; per-track outputs are physically pipeline-owned.

The robot scoring functions were extracted verbatim from `v1/pipeline.py` into `robot/scoring/metrics.py`. The old module re-exports these functions; `pdi_eval.robot_scoring` also exposes them directly. Geometry/tracking orchestration remains in `workflows/pipeline.py`, and its public command continues to run the existing sequence. This migration does not introduce a new intermediate serialization format or alter numerical formulas.

## Deferred link7/object redesign

The [current coordination guide](../link7_object_masking_occlusion_cropping.md) records the agreed ordering: independent link7/object masks, both existing VLM3 gates, final mask selection, then robot geometry/tracking and object occlusion/crops before their respective scorers. It identifies the existing artifact-based adapter dependencies and proposes their separation. This design is documented; the autonomous coordinator is not yet implemented.

There is deliberately no new `integrations/link7_object` implementation yet. `vlm3_overmask.py` remains with persistent masking and `mask_sync.py` remains with the object workflow. Existing imports and control flow are preserved. VLM1/VLM2 decisions, VLM3 gates, frame selection, accepted/rejected masks, crop exclusions, downstream refresh order and default behavior are unchanged. Their locations describe ownership, not a change to their interaction contract.

The coordinator runs object CoTracker directly for occlusion; it does not run object geometry or rigidity. The main object track's scorer is AnomalyDINO. Do not mistake the retained object rigidity command for the main scorer.

## Direct imports and deployment

Canonical folders are now the Python namespaces: `robot`, `object`, and `infrastructure`. The coordinator and worker remain under `infrastructure.pdibench`. `python -m pdibench` is only a CLI shortcut. Source modules, subprocess `-m` commands, public package exports, model roots, reference defaults, replay assets and source fingerprints use the canonical layout.

Use a full checkout, or `python -m pdibench stage --output NEW_DIRECTORY`. Staging excludes archives and compatibility packages. Vendor checkouts, reference images, model weights and run data remain separately provisioned assets. Use the coordinator's explicit manifest paths for hosted model environments.

Editable installation exposes the canonical packages and CLI. The source checkout is still required for manifests and assets. Supported entry points appear in `python -m pdibench list`; use those or `python -m <canonical.module>`, rather than archived shell runners. The import regression tests and a staged-copy check run without the old compatibility tree. Fresh GPU acceptance after this import migration has not been run.

## Results

Existing run contents and serialized provenance are immutable through this migration. Complete runs moved to pipeline-owned roots, with `results/<run-id>` aliases retaining original logical paths. Relative filesystem symlinks were retargeted to the same logical destination. No experiment scores, crop PNGs or replay HTML were rewritten.

The mixed `simple3d-20261003-run2` batch keeps its common inputs, metadata and comparison/replay outputs under `infrastructure/shared/experimental/simple3d/results/`. Its object outputs live under the object Simple3D experiment; all four link5 variants live under the link5 Simple3D experiment. Both expose links to common batch provenance.

The older `object-deformation-selected45-20260929` run remains intact under object results: current preprocessing consumes its masks/tracks, even though its score is the retained rigidity experiment. Views from the rigidity experiment point to that same run rather than duplicating it.

New outputs should use the owning pipeline/experiment's `results/<run-id>` root through explicit output arguments or configuration. Historical artifacts retain their recorded paths; maintained defaults now use canonical source and asset paths. Source hash changes caused by this migration are recorded honestly; no old code hashes are forged and no hard-coded compatibility allowlists are expanded.

## Keep versus archive

`layout.json` is the file-level ownership ledger. `infrastructure/archive/INDEX.md` describes why historical tools were archived. Reusable adapters and active dependencies were kept even when their old names sound experimental, including `palm_recovery`, `gripper_negatives` and the generation wrapper. Historical setup scripts remain evidence for the extracted GPU profiles.

Archived scripts are not in the supported interface registry. Historical aliases are retained only inside the archive for inspection. Archiving does not authorize running a historical script against new or published outputs.

## Verification

`python documentation/architecture/verify.py` checks all mapped source/result aliases and compares Python syntax trees against the pre-migration commit, allowing only the source-path bootstrap and the documented verbatim scoring extraction. It also checks shell path-only adaptations and unchanged non-Python resources.

`verification.json` records the completed full artifact check: 55,687 files, 72,307,688,632 bytes and 179 pre-existing symlinks. The local per-file baseline is `.tmp/migration/artifact-baseline.json`. Tests cover the existing scientific contracts plus the new interface's missing-data, duplicate-ID, run-record and alias behavior. `validation.json` summarizes final checks.

The current website remains byte-identical. `documentation/publication/build.py` verifies or copies the frozen bundle recorded by `documentation/publication/release.json`. Legacy HTML assemblers remain available for provenance, but are not the current publication command because they originally relied on subsequent case-specific patches.
