# Third-party forks and source checkouts

These are the sources present in this workspace, checked locally on 2026-10-04. The three direct submodules retain their original revisions. No upstream source was modified during the move.

| Component | Location from repository root | Source and local status |
|---|---|---|
| AnomalyDINO | `infrastructure/vendor/AnomalyDINO/` | Fork: `Wilsonnijc-bot/AnomalyDINO`; branch `integration/pdi-pairwise`; upstream remote `dammsi/AnomalyDINO`; clean submodule at `b2fef560b8b310835261abe2d8a0357823a58f40` |
| MegaSAM | `infrastructure/vendor/mega_sam/` | Upstream `mega-sam/mega-sam`; clean submodule at `a27b4e633c5cc0828a62ed943ef9f6505705fd3f` |
| Original PDI-Bench | `infrastructure/vendor/PDI-Bench-original/` | Upstream `AnteaWu/PDI-Bench`; clean submodule at `2433323453574abb3c41fe83ca19b968a54ce75a` |
| Simple3D | `infrastructure/vendor/Simple3D/` | Vendored source tracked by this repository; its README identifies `hustCYQ/MiniShift-Simple3D`. No independent Git metadata is present, so an upstream revision is not inferred. |
| Edited PDI-Bench implementation | `robot/`, `object/`, and `infrastructure/shared/` | This workspace's adapted implementation, previously under `PDI-Bench-edited/`; preserved import namespaces and resource aliases now live under `infrastructure/compat/PDI-Bench-edited/`. It is not an additional standalone Git submodule. |

The original PDI-Bench submodule also contains a populated dependency tree, preserved in place:

```text
infrastructure/vendor/PDI-Bench-original/third_party/mega_sam/
  base/
    thirdparty/eigen/
    thirdparty/lietorch/
      eigen/
```

Nested revisions: MegaSAM `a27b4e633c5cc0828a62ed943ef9f6505705fd3f`, base `ee9ac6af512c09ed48264c6933eeeb66d31a0dc9`, Eigen `3d4ba855e014987cad86d62a8dff533492255695`, lietorch `0fa9ce8ffca86d985eca9e189a99690d6f3d4df6`, and lietorch's Eigen `824272cde8ca2541e8b67b0887f5ded92b128d1f`.

The direct `infrastructure/vendor/mega_sam/base` submodule is recorded at that same base revision but is not initialized in this local checkout. The populated nested copy above is distinct; this migration does not merge or deduplicate third-party working trees.

SAM3, CoTracker, TAPIP3D, and native extension sources are also referenced by the GPU environments and experiment wrappers. They are not additional standalone checkouts present in this workspace. Their available pins and provisioning requirements are recorded in [gpu/DEPENDENCIES.md](gpu/DEPENDENCIES.md).

Use `git submodule status --recursive` to inspect populated and uninitialized submodules. `.gitmodules` records the relocated paths; historical submodule names remain unchanged.

Maintained adapters now import directly from their owning workspace modules and locate vendor checkouts under `infrastructure/vendor/`. MegaSAM still honors the explicit `PDI_MEGA_SAM_ROOT` per-attempt override. AnomalyDINO retains its isolated upstream module namespace, and Simple3D retains the upstream import path its unmodified code requires. The old `third_party` and original-PDI aliases are confined to `infrastructure/archive/compat/`; runtime and staging exclude them. The original PDI checkout and nested build dependencies were not deleted or repinned.
