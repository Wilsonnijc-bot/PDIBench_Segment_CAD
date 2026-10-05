# Robot links

Main targets are link2–link7. Preprocessing owns segmentation, link5 refinement and the active persistent link7 implementation. Shared adapters provide MegaSAM geometry and CoTracker tracking; `workflows/` coordinates the existing stages. `scoring/metrics.py` contains the unchanged metric calculation, and `replay/` owns pipeline-specific review generation while `infrastructure/shared/replay/` owns reusable interactive viewers.

Retained experiments: link5 Simple3D; link7 TAPIP3D, depth filter v2 and tracker/filter comparisons; masking-trial results. Historical palm trial code and obsolete fixed-batch launchers are under `robot/archive/`.

Use `python -m pdibench list` and the `links.*` interfaces. Existing Python namespace: `pdi_eval`, plus `persistent_masking`. Current run selection remains in `documentation/publication/release.json`. Existing result replays can still be served through the original `results/<run-id>` URLs.

Link7/VLM3 behavior has not been redesigned in this migration.
