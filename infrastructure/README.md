# Supporting infrastructure

The main scientific tracks live in [robot](../robot/) and [object](../object/). This folder contains their intentionally shared code and the tools needed to run, review, and publish them.

- `shared/`: inference adapters, geometry, numerical scoring, artifact contracts, interactive replay, and shared Simple3D support.
- `deformation_detect/`: run registry, execution recording, CSV analysis, and portable source staging.
- `vendor/`: upstream implementations; Git submodule names and pinned revisions are preserved.
- `tests/`: common interface tests.
- `archive/`: historical setup, diagnostics, and operational trials. Pipeline-specific archives live in their owning pipeline.

Runtime imports use `robot`, `object`, and `infrastructure` directly. The old namespace tree is archived under `archive/compat/` and excluded from runtime paths and deployment snapshots. The root `results/` directory holds per-run aliases to pipeline-owned outputs. The coordinator remains in `deformation_detect/`.

Publication tooling, GPU requirements and checks, architecture records, labels, and reference data live under [documentation](../documentation/README.md). The [third-party index](../documentation/THIRD_PARTY.md) lists the forks and upstream checkouts in `vendor/`.
