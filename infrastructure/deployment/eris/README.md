# ERIS deployment

Preparation runs separately from inference. The scripts install isolated model runtimes and verify checkpoint/source hashes; `launch_coordinator.sh` runs the maintained `deformation_detect coordinate` command inside a Slurm GPU allocation. It records runtime import origins, native CUDA checks, a durable job log and an independent exit file. Additional coordinator options, including `--retry-disabled`, can follow its four directory/manifest arguments. Retry preserves old attempts.

Use one H200 with independent end-to-end video coordinators. Five concurrent videos are the baseline; increase concurrency only when measured memory and throughput support it. GPU allocation is separate from worker count:

```json
"execution": {"workers": 5, "gpu_slots": 5, "gpu_assignment": "shared"}
```

Request one GPU with `--gres=gpu:nvidia_h200:1`. Choose CPU, host memory and concurrency from measured stage capacity. Each video keeps its own state, outputs and scratch and runs its full pipeline. `launch_coordinator.sh` delegates to that coordinator; no stage scheduling redesign is required.

The October 5 acceptance recipe used two jobs with five H200s per job and `per_video` assignment. That is historical evidence, not the current default. `batch_status.py` reads and merges group state; it does not launch inference. Case-specific masking launchers now belong to `robot/experiments/link7_masking_review/`.

Curated server documentation: [dependencies](../../../documentation/gpu/eris/WILSON_DEPENDENCIES.md) and [inventory summary](../../../documentation/gpu/eris/WILSON_SERVER_INVENTORY.md). The exhaustive October 6 snapshot is a local ignored artifact under `results/inventory-20261006/`.

When transferring a staged working tree or reference assets from macOS, set `COPYFILE_DISABLE=1` for the tar producer. Otherwise tar can generate `._*` AppleDouble files beside PNG references. Those files are metadata, not images; `verify_assets.py` rejects them. Preserve any failed attempt before quarantining unintended sidecars. Do not copy historical masks, tracks, geometry, crops or scores into fresh inference inputs.

The October 5 network diagnosis found that a 5.5 MB VLM upload stalled on `api.302ai.cn`, while the identical request completed through the provider's [documented overseas endpoint](https://doc.302.ai/188441510e0), `https://api.302.ai/v1`, in 78 seconds. The recovery manifests override `vlm.vlm2.api_base` and `vlm.vlm2_malformed_fallback.api_base`; VLM3 uses the same role. The first endpoint recovery retained all timeouts. A later identical Luna diagnostic returned successfully after494 seconds, so the user-requested additional retry sets only `vlm.vlm2_malformed_fallback.timeout_seconds` to600. Models, image bytes, prompts, four logical attempts and the3600-second VLM-stage budget remain unchanged. Validate actual full-image requests before changing a route. Credentials remain outside the workspace and public manifests.

VLM3 mask synchronization precedes geometry and cropping. It triggers when Link7 covers more than95% of the object mask or occupies at least25% of the image. The reconstruction scale-jump audit runs later and does not trigger VLM3. Inspect quality flags and mask anatomy even when a coordinator reports completion.

`collect_review.py` exports selected terminal artifacts to their robot/object result owners, with transfer hashes and one review metadata directory. It omits model links and geometry scratch, preserves failed-attempt control records, and never performs inference. `--cases` permits exporting completed cases while other cases continue.

See the project [experiment ledger](../../../documentation/gpu/experiment_GPU_record.md) for preparation failures, repairs, source provenance, job IDs and final results.
