# Objects

Main path: grounding → object masks → CoTracker for occlusion → selected link7-mask handoff → occlusion/available pixels → selected normalized crop pairs → AnomalyDINO → replay/gallery. No object MegaSAM or rigidity is needed. The preprocessing files retain their existing functions and serialized formats.

Occlusion is an active object feature with these ownership boundaries:

| Stage | Location |
|---|---|
| Detection and availability | `preprocessing/occlusion/` |
| Occlusion analysis and score aggregation | `analysis/occlusion/` |
| Interactive synchronized replay and indicator | `replay/occlusion/` |
| Scientific contract tests | `tests/test_object_occlusion*.py` |

The occlusion-filtered rigidity helper remains reusable by the retained rigidity experiment. Its shared use does not make the occlusion feature experimental or obsolete. Historical batch/release scripts alone live in `archive/`.

`scoring/anomalydino/` is the main scoring implementation. `experiments/rigidity/` retains the earlier object geometry/rigidity workflow, and `experiments/simple3d/` owns the Simple3D branch. Old object rigidity results also contain masks and tracks consumed by the main track, so that mixed historical run remains intact under `results/`.

`workflows/coordination.py` supplies the object stages of the [permanent resumable coordinator](../documentation/pipeline/COORDINATOR.md), without requiring old rigidity/occlusion outputs. `workflows/mask_sync.py` remains the historical artifact-based handoff for existing experiment entry points.

Use the `objects.*` interfaces listed by `python -m pdibench list`. Python imports use `object.preprocessing`, `object.scoring.anomalydino`, and `object.replay` directly. Current publication includes 42 scored videos and 420 pairs; see `documentation/publication/release.json`.
