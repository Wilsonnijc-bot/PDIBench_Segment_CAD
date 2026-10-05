# Robot and object deformation detection

Two pipelines own their code, experiments, and results: **robot** and **object**. Shared implementation and operational tools live under **infrastructure**.

```text
robot/
  preprocessing/    Six-link masks; link5 refinement; persistent link7 masking
  scoring/          Robot rigidity metrics
  workflows/        Existing geometry, tracking, and scoring orchestration
  replay/           Robot replay generation and review
  experiments/      Link5 Simple3D; link7 TAPIP3D/depth filtering; comparisons
  results/          Main robot run outputs
  archive/          Historical robot trials and fixed-batch runners
object/
  preprocessing/    VLM grounding, masks, occlusion, and crop pairs
  scoring/          AnomalyDINO
  workflows/        Existing link7 mask synchronization
  replay/           Object galleries and interactive occlusion replay
  analysis/         Object score and occlusion analysis
  experiments/      Object rigidity and Simple3D
  results/          Main object run outputs
  archive/          Historical object batch and release scripts
infrastructure/
  shared/           Reused inference, geometry, scoring, and replay components
  pdibench/         Common run, analysis, and source-staging interface
  vendor/           Third-party implementations and pinned submodules
  tests/            Workspace interface tests
  archive/          Historical setup and diagnostic tools
documentation/
  publication/      Current website selection and publication tooling
  gpu/              Dependency profiles, environment checks, and GPU ledger
  architecture/     Ownership inventory and migration verification
  data/             Shared labels and robot reference images
docs/               Published GitHub Pages site
web-assets/        Published site assets
results/            Per-run aliases to pipeline-owned results
```

Each pipeline also owns its tests; robot owns its configurations. Shared labels and robot reference images live under `documentation/data/`. Experiment outputs belong under the corresponding `experiments/<name>/results/` directory.

**Occlusion is an active part of the object pipeline.** Its detection, analysis, tests, and interactive replay belong to `object/`. The existing link7/object interaction, including VLM1/VLM2 and optional VLM3 behavior, is preserved; redesign remains deferred.

The root `results/` directory preserves paths stored inside existing experiment artifacts. Actual runs live under their owning pipeline. `docs/` and `web-assets/` retain their published URLs and content.

## Run and analyze

Use Python 3.10 or newer from this checkout. `pip install --no-deps -e .` optionally installs the `pdibench` command; it does not install GPU dependencies.

```bash
python -m pdibench list
python -m pdibench run links.run -- --help
python -m pdibench run objects.occlusion -- --help
python -m pdibench run objects.score -- --help
python -m pdibench run objects.replay -- --help
python -m pdibench env-check --profile geometry --python /path/to/geometry/bin/python --cuda
```

Select a model environment and record an invocation with explicit output arguments:

```bash
python -m pdibench run --python /path/to/anomalydino/bin/python \
  --record object/results/my-run/execution objects.score -- --help
python -m pdibench analyze --scores scores.csv --labels labels.csv \
  --key case --score epsilon_rigidity --label deformation --output analysis.json
```

Replace `--help` with the existing command's arguments. New outputs should use `robot/results/`, `object/results/`, or the owning experiment's results directory. Frozen historical defaults and configurations retain their recorded paths. CSV analysis reports missing values and excludes moderate labels; pipeline-specific analysis retains its existing aggregation policies.

Code imports directly from `robot`, `object`, and `infrastructure`. The coordinator lives in `infrastructure/pdibench/`; `python -m pdibench` remains a CLI shortcut. The former compatibility namespaces are archived under `infrastructure/archive/compat/` and excluded from runtime and source staging.

## Context and verification

- [Robot pipeline](robot/README.md) and [object pipeline, including occlusion](object/README.md)
- [Latest robot results](docs/analysis/LATEST_LINK_RESULTS.md) and [object results](docs/analysis/OBJECT_ANOMALY_REPORT.md)
- [Existing link7/object interaction](documentation/link7_object_masking_occlusion_cropping.md)
- [Third-party forks and upstream checkouts](documentation/THIRD_PARTY.md)
- [GPU dependency profiles](documentation/gpu/DEPENDENCIES.md)
- [Architecture and ownership](documentation/architecture/README.md)
- [Archive selection](infrastructure/archive/INDEX.md)

```bash
python documentation/architecture/verify.py
python documentation/publication/build.py
python -m pdibench stage --output .tmp/source-copy
python -m pytest -q
```

Source staging copies canonical modules and excludes compatibility code and archives; provision third-party repositories, reference assets, checkpoints, and results separately. Publication tooling verifies or copies the selected current bundle. Future run promotion requires updating its explicit selection and inventory.
