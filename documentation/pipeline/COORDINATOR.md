# Resumable robot/object coordinator

`python -m deformation_detect coordinate` replaces the validation-only `masking_driver.py`, `downstream_driver.py`, and `case_tasks.py`. The coordinator runs the existing numerical implementations in their own environments, with explicit inputs and immutable stage attempts. It does not publish the results website or run object geometry/rigidity.

Copy [coordinator.example.json](coordinator.example.json) and fill in the installed environment, model, reference, video and output paths. Paths are relative to the manifest file unless absolute. Credentials belong in environment variables or the external `secrets_file`, never in this manifest. Environment variable names may be configured in public VLM role settings; their values are not serialized.

```bash
python -m deformation_detect coordinate --manifest run.json --plan
python -m deformation_detect coordinate --manifest run.json --preflight-only
python -m deformation_detect coordinate --manifest run.json
python -m deformation_detect run-status --output /path/to/runs/link7-object-run
```

Run the same command again to resume. A disabled case remains disabled with unchanged inputs; `--retry-disabled` explicitly grants it a new attempt budget. Changed inputs automatically invalidate the affected stage and its dependents. A run exits 0 when every case completes, 2 when some cases are disabled. A preflight/configuration failure stops the run before inference.

Use the existing job supervisor or a unique tmux session for a remote GPU run. The coordinator itself writes durable `run.json`, per-attempt logs, and `index.html`. It needs no per-video shell script. It is a foreground process: stopping the terminal without a supervisor can interrupt it. Keyboard interruption or SIGTERM terminates the active worker process group and records the interrupted stage. If the coordinator is killed without cleanup, resume refuses to start while the recorded worker is still alive. An unfinished worker on another host requires resolving its ownership first. Never run two coordinators against the same output; a filesystem lock enforces this.

## Stage boundaries

1. Robot initial DINO/SAM3 masks.
2. VLM1 frame diagnoses, followed by VLM2 point selection.
3. Persistent SAM propagation and native validation.
4. Object grounding and SAM masks.
5. Joint mask selection: either object overlap >95% or link7/image area ≥25% triggers VLM3. One committed selected mask feeds both downstream branches.
6. Object CoTracker, existing occlusion detector, available-pixel crop mapping and ten-pair selection.
7. Optional link2 DINOv2-guided SAM3 masking, then shared native MegaSAM + V1/CoTracker rigidity and interactive replay for the requested robot links.
8. Object AnomalyDINO and scored crop gallery.

The DAG records dependencies even though the initial release executes sequentially on one GPU. Object CoTracker has no dependency on link7 masks. Occlusion/crops require the selected link7 mask. Numerical occlusion, tracking, crop selection, anomaly and rigidity functions remain shared with the existing interfaces.

The owner adapters are [robot/workflows/coordination.py](../../robot/workflows/coordination.py) and [object/workflows/coordination.py](../../object/workflows/coordination.py); execution/state lives under [infrastructure/deformation_detect/](../../infrastructure/deformation_detect/coordinator.py). Legacy experiment entry points remain available.

## Inputs and portability

Case IDs are arbitrary safe names. No historical dataset directory aliases or previous score/occlusion artifacts are required. Supply the original generation prompt for object grounding. An optional `vlm1_reference` supplies the nondeformed reference; without it, the existing pipeline uses the initial mask's frame-zero crop.

Set `"robot_links": ["link2", "link7"]` to include link2 (enabled in the example manifest). Existing manifests that omit this field retain link7-only behavior. Link2 uses the maintained DINOv2-guided SAM3 single-target segmenter with reference spatial priors, 0.10 box padding and the native 0.80 tracked-frame threshold. Supply `resources.robot_references` as the existing reference root containing `by_link/link2` through `by_link/link7`, or the `by_link` directory itself. Preflight validates those groups; SAM propagation targets only link2. No persistent masking or VLM1/VLM2/VLM3 calls are added for link2.

`link2_masks` owns its mask, native diagnostics and provenance. Only `robot_score` depends on it: that adapter combines the fresh link2 mask with the selected link7 archive in its own attempt directory, reuses one native MegaSAM reconstruction, and requests both named rigidity scores/replays. Object occlusion/cropping continues to consume the original joint link7/object handoff. Changing link2 references or corrupting its output invalidates link2 masking and robot scoring, not the object branch or persistent link7 masking. Link2 receives the normal non-VLM failure policy (one attempt, then case disabled until an explicit retry). GPU tasks remain sequential, with no additional user-managed runner.

The combined robot mask changes background exclusion and geometry context compared with a link7-only archive. Numerical equivalence to a link7-only run is therefore not implied. The native algorithms are unchanged.

An optional case `base_segmentation` preserves the other robot channels and replaces only the named `link7` mask. Supply it to reproduce the six-link benchmark context. Without it, the joint handoff contains a real single-link archive; when enabled, link2 is added only to the robot scoring input. No dummy channels are introduced. Background tracking/geometry context can consequently differ from a six-link run. Cropping and replay resolve link7 by name.

Preflight checks input readability, all required resource paths, reference images, replay assets, credentials, actual module imports, CUDA operations and key MegaSAM checkpoints. Install/pin the existing third-party forks and model environments first using the [GPU requirements](../gpu/DEPENDENCIES.md). Deployment installation is deliberately separate from case execution. Model/resource contents and environment versions are included in resume identity. MegaSAM writable scratch is isolated per scoring attempt and removed after successful scoring; required geometry and replay artifacts are retained.

## Bounded VLM attempts and failure handling

The default is one initial attempt plus three further attempts per VLM stage. This is a stage-level budget, not three retries multiplied by hidden HTTP retries: the coordinator sets native cloud transport attempts to one. The default whole-VLM-stage time budget is 3600 seconds across its attempts; other stages have a 14400-second execution timeout. Both are configurable. Interrupted VLM attempts consume the recorded budget; explicit `--retry-disabled` is required to reopen an exhausted budget.

- **VLM1:** retry failed/unparseable frame responses, retaining successful diagnoses. Valid no-deformation results proceed with the initial mask. An invalid response or a sequence with no assessable frames cannot masquerade as no deformation. An optional `vlm1_fallback` public role configuration switches subsequent attempts; otherwise the same configured role is retried.
- **VLM2:** an initial logical attempt comprises positive and negative point requests. Subsequent attempts use the existing alternate model route. The old nested one-call alternate is disabled for coordinator execution. Thus four logical attempts can make at most eight point requests, not four requests total. Each attempt retains exact input images, responses and its failure evidence.
- **Object grounding/SAM:** use the configured `object_models` sequence, selecting one model per attempt. The default is Luna, then Gemini for remaining attempts. Failed SAM grounding also advances this stage's bounded attempt sequence.
- **VLM3:** one repair candidate per attempt, using the primary role first and its alternate afterward. Preserve the same gate frame, prompts, membership checks and crop-exclusion policy. Validation feedback is carried to the next attempt.

A required VLM3 repair that exhausts its budget disables the case; it never silently falls back to the suspect core mask. Accepted repairs with allowed frame exclusions remain usable. Mask/scoring/crop failures disable the case rather than fabricating results or aborting unrelated cases. Non-VLM stages are not automatically repeated four times. Completed numerical results can still contain quality flags such as a failed native scale-jump audit; those remain visible in metrics and are not rewritten into passes.

## Resume and output contract

The output contains a single review index, run state, sanitized manifest, environment preflight record, and `robot/<case>/<stage>/attempt-NNNN` and `object/<case>/<stage>/attempt-NNNN` trees. Stages record content hashes of outputs, inputs, dependencies, model resources and implementation sources. Corrupt or missing artifacts are recomputed; changed object prompts do not rerun initial link7 masking. Implementation changes conservatively invalidate all stages. Completed predecessor artifacts are immutable: native persistent work and annotated crop selections are copied into the next stage's owned attempt before modification.

Failures and superseded attempts stay available for diagnosis. This initial implementation favors traceability over storage deduplication; persistent stage snapshots duplicate some masks/images. It does not automatically delete old attempts. Output directories must be new/empty or belong to the same run ID. Deployment/model errors fail preflight rather than spending every video's retry budget.

The temporary historical-parity scripts are still validation evidence, not runtime dependencies. The earlier two-video GPU results predate this coordinator. CPU tests, four-environment GPU-host preflight, and [exact saved-artifact adapter parity](validation/saved-artifact-parity.json) validate integration. Both cases reproduced all detector rows, crop selections, and 40 crop PNGs on the original GPU environment; a fresh GPU end-to-end acceptance run is still needed before claiming this coordinator itself has completed a full model run.

Current [integration verification](validation/integration-checks.json): 214 tests and 19 subtests pass. The saved-artifact comparison runs the maintained worker adapter on both earlier videos without invoking fresh models. No new full GPU acceptance run is claimed.

The link2 extension has CPU scheduler, manifest and archive-merge coverage; fresh link2 model inference has not yet been run through this coordinator. Earlier GPU preflight and crop-parity evidence apply to the link7/object implementation before this extension.

Runtime imports now use `robot`, `object`, and `infrastructure` directly. The coordinator remains `infrastructure.deformation_detect`; the root `deformation_detect` command is a CLI shortcut. See the [direct-import migration audit](../architecture/DIRECT_IMPORT_MIGRATION.md). Old compatibility namespaces are archived and excluded from source staging. Fresh GPU acceptance has not been repeated after this migration.
