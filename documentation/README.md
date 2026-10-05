# Workspace documentation and supporting data

- [Latest link results](analysis/LATEST_LINK_RESULTS.md): selected-45 link2/link5/link7 benchmark sources, AUROC, labels and availability.
- [Resumable coordinator](pipeline/COORDINATOR.md): permanent autonomous link7/object execution with optional link2, manifest, retry budgets, case disabling and resume rules.
- [Link7/object coordination](link7_object_masking_occlusion_cropping.md): current masking interaction, exact downstream rules, and the proposed autonomous preprocessing handoff before scoring.
- [publication/](publication/README.md): selected current release, publication builders, and frozen-site verification.
- [website/](website/README.md): published website pages, replays, galleries, and analysis.
- [gpu/](gpu/DEPENDENCIES.md): environment requirements, reusable environment checks, and the existing GPU experiment ledger.
- [Direct-import migration](architecture/DIRECT_IMPORT_MIGRATION.md): import audit, archived compatibility tree, validation and deployment limits.
- [architecture/](architecture/README.md): pipeline boundaries, file ownership, migration checks, and validation records.
- `data/labels/`: the shared evaluation workbooks.
- `data/references/`: robot reference images used by preprocessing.
- [THIRD_PARTY.md](THIRD_PARTY.md): exact locations and revisions of forks and upstream checkouts.

Experiment results stay with `robot/` or `object/`, including their experiment branches. The existing published website lives in `documentation/website/`, with image packs in root `web-assets/` so public URLs remain stable.
