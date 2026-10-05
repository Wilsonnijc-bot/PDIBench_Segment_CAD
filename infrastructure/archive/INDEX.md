# Historical tools and trials

These files remain available for provenance and reproducibility. They are excluded from the supported `deformation_detect` interface registry. No scientific results were deleted.

| Archive | Why archived | Retained general replacement |
|---|---|---|
| `robot/archive/runners/` | Fixed selected-45 batches, particular remote hosts, four-way release/fetch jobs and manual filter synchronization | `deformation_detect run links.run`, explicit experiment configs and portable source staging |
| `robot/archive/benchmark_runners/` | Fixed two-video, selected-45 and batch launch wrappers | Registered module entry points with explicit arguments and execution records |
| `robot/archive/masking_trials/` | Palm seed/point ablations, proposal iteration, previews, old global refinements and historical case selections | Current persistent-masking implementation and retained masking experiment results |
| `object/archive/runners/` | Host-specific controller for the older object rigidity batch | Retained `objects.rigidity` experiment interface |
| `object/archive/release_updates/` | Seven-case repair publications, cohort additions and one-time normalization/update scripts | Current object preprocessing/scoring components; frozen publication selection and verifier |
| `diagnostics/` | First-frame probes, prequery checks, old UI refreshes and cleanup utilities | Existing pipeline tests and reusable replay exporters |
| `gpu/setup/` | Provider/mirror-specific installation and checkout scripts, with mixed environment assumptions | `documentation/gpu/environments.json`, separate requirement profiles and `deformation_detect env-check` |
| `simple3d/operations/` | Trial cleanup, continuation/requeue, worker resizing, shutdown monitoring and historical extension installer | Shared Simple3D runner, preparation, verification, collection and analysis interfaces |
| `compatibility/` | Existing thin import aliases, not alternative scientific implementations | Maintained only for historical API compatibility |
| `notes/root/` | Pre-organization overview and older root-level specifications/status notes | Root README, architecture inventory and user-selected recent analysis docs |
| `results/`, `previous_archive/` | Already historical local output collections | Preserved locally; ignored by Git |

Active shared helpers were not archived based on naming. In particular, `palm_recovery.py`, `gripper_negatives.py`, generation localization/prompt utilities, VLM configuration and mask synchronization still serve the main pipelines.

The exact old-to-new mapping and original source hashes are in `documentation/architecture/layout.json`. Old paths generally remain compatibility links; edit only the canonical target. Historical scripts can still contain host-specific actions and should not be treated as a generic new-run interface.
