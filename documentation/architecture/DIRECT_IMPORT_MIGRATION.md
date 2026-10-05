# Direct imports: audit and migration

The maintained Python namespaces are now `robot`, `object`, and `infrastructure`. The coordinator, worker, run registry and deployment helpers remain in `infrastructure/pdibench/`. `python -m pdibench` is only a CLI shortcut; maintained modules import `infrastructure.pdibench` directly.

## What was checked first

The [pre-migration inventory](audits/vendor-import-inventory.json) scans 606 Python/shell files across maintained code, historical scripts and populated vendor sources. It records 4,119 import statements and 281 import/deployment path references. Three Python-2-only scripts inside nested vendor dependencies could not be parsed by Python 3; they are listed in the inventory and were not modified.

The initial vendor-only proposal was rolled back before this migration. It missed launchers that replace `PYTHONPATH`: a new dependency on a workspace helper can appear to work under an editable installation while failing in a separate deployment. The direct and nested MegaSAM checkouts also have different initialization states: the original PDI reference checkout contains the populated base/Eigen/lietorch tree, while the direct MegaSAM base remains uninitialized locally. Those pinned vendor trees have been preserved.

## Final import boundaries

| Former namespace | Direct owner |
|---|---|
| `pdi_eval.experiment`, `pdi_eval.v1.pipeline` | `robot.workflows`, with selection, analysis and experiment helpers in their owning robot folders |
| `persistent_masking` | `robot.preprocessing.link7_persistent`; replay exporters in `robot.replay.persistent` |
| `pdi_eval.object_deformation_wrapper` | `object.preprocessing`, `object.workflows`, `object.replay`; rigidity stays in `object.experiments.rigidity` |
| `pdi_eval.anomaly_scoring` | `object.scoring.anomalydino` |
| Shared geometry, tracking, contracts, metrics and replay helpers | `infrastructure.shared.*` |
| `generation.link_crop_wrapper` | `infrastructure.shared.inference.generation.link_crop_wrapper` |
| Flat `experiments` and `scripts` imports | Concrete robot/object experiment modules, shared experiment modules, or publication builders |
| `pdibench.*` library imports | `infrastructure.pdibench.*` |

The [module map](audits/direct-import-module-map.json) records individual destinations. It is audit evidence, not a runtime alias resolver. Imports, public package exports, dynamic imports, test patch targets, subprocess module arguments and launcher search paths were migrated together.

The old tree is under `infrastructure/archive/compat/`, outside runtime search paths and deployment snapshots. A code-only [before-migration snapshot](../../infrastructure/archive/before-direct-imports.tar.gz) preserves the immediate prior implementation. Fixed-case shell runners and the redundant VLM configuration and volume-audit re-exports were archived; the reusable `pmask` CLI now invokes the canonical module.

## Resource and result paths

- Vendor integrations use `infrastructure/vendor/`. MegaSAM's explicit per-attempt override remains supported. No vendor source, pin or nested dependency was changed.
- Replay Plotly assets live in `infrastructure/shared/replay/assets/`.
- Reference images and camera metadata live under `documentation/data/`; model/environment defaults use `infrastructure/models/` and `infrastructure/environments/`. Hosted runs should continue to provide explicit manifest paths.
- Root `results/` is a directory of per-run aliases to pipeline-owned results. It no longer points through the compatibility tree. Existing historical alias spellings remain available.
- Provenance now records canonical source files. Source fingerprints change honestly; old output files and recorded hashes are not rewritten. A resumable run may recompute stages after detecting these source changes.

The [resource expression map](audits/direct-import-resource-map.json) records the mechanical rebasing pass. Final reviewed changes also include formatted VLM-reference filenames, environment defaults, the GPU ledger path and source-fingerprint traversal.

## Logic preservation and validation

The [function review](audits/direct-import-function-review.json) compares the immediate before/after code: 796 function bodies were identical as ASTs before normalizing imports and resource locations. Normalized comparison narrows the remaining production-function changes to launch/search paths, resource defaults, provenance, staging and CLI descriptions. The [manual review diff](audits/direct-import-manual-review.diff) records that review. The rigidity, occlusion, crop-selection and anomaly arithmetic was not redesigned.

Validation after removing the runtime compatibility tree:

- **214 tests and 19 subtests passed.** Previous depth-filter and TAPIP3D tests remain collected. Three object-rigidity contract tests formerly outside the configured collection are now included, as are two direct-import regression checks.
- **11 main CLI entry points** passed `--help` from outside the checkout: [results](audits/direct-import-cli-checks.json).
- A **staged workspace excludes archives/compatibility packages** and runs the coordinator plan with Python site packages disabled. Four main CLI help checks also pass from outside the staged checkout: [results](audits/direct-import-stage-check.json).
- Structural verification resolves **391 historical source aliases and 18 result aliases**. Its optional `--historical-semantics` comparison targets the older Git baseline and can report intentional later changes; structural success alone is not a numerical-parity claim.
- The **714-file published release** passed its existing hash verification.

No fresh GPU inference or new numerical reproduction run was performed for this import migration. Earlier GPU/crop-parity evidence predates it. The rented GPU deployment was not modified. A full deployment should stage the whole canonical workspace, provide pinned vendor/model environments and explicit manifest resources, run preflight, and then repeat the two-video acceptance run before claiming GPU end-to-end equivalence.
