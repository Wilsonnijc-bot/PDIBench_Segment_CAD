# Single-H200 scheduling and acceleration

User instruction: **one GPU handles multiple videos concurrently**.

Use **one H200** and maximize useful GPU memory usage and measured throughput.
**Five parallel videos is a starting point, not a fixed limit.** Apply this rule
to preparation, training and inference.

- Share each GPU between multiple independent video workers. Do not reserve one
  GPU for each video or use a single sequential video lane per GPU as the default.
- Configure video-worker concurrency separately from GPU count. Assign several
  workers to each GPU, with isolated per-video outputs and scratch directories.
- Choose the concurrent worker limit from the actual stage's measured GPU memory
  and runtime. The previous full-video H200 evaluation used five concurrent video
  workers per GPU; validate that capacity for SAM/MegaSAM preparation before
  applying it to that stage.
- Record GPUs, total video workers, and concurrent workers per GPU before launch.
  Do not increase GPU count solely because there are more videos.
- Measure peak GPU memory, GPU utilization, CPU/I/O pressure and completed work
  per unit time. Increase useful concurrency while capacity remains; retain
  changes only when throughput improves. Keep headroom for measured peaks.
  Allocating unused buffers merely to fill memory does not count as utilization.
- Run independent detector variants concurrently on the same H200 when they fit,
  with separate processes, output directories, optimizers and random states.
  Use locks and completion checks so the pipeline does not train a variant twice.
- Once the normal reference is frozen, overlap independent test-video preparation
  with training when measured memory permits. Test data must never change or enter
  the training reference. Use immutable per-allocation launcher snapshots; never
  edit a shell script that is still running.
- Investigate training bottlenecks, including repeated DataLoader startup,
  host/device transfers and synchronization. Preserve normal references, losses,
  optimizer, batch size, epochs, sampling and codebook updates for execution-only
  changes. Record changes affecting numerical behavior explicitly.
- A request to stop an experiment also cancels its queued dependent jobs. Preserve
  completed outputs and do not restart work until the user requests continuation.

The current Link5 normal reference uses only frame0 from COSMOS3_0046,
COSMOS2.5_0044, LVP_ROBOWM_0044, and COSMOS3_0056. Later frames must never enter
normal-reference training. Apply the refined depth filter to both reference and
test observations, and keep Link5 VLM changes scoped to Link5.

Cleanup selection, 2026-10-06: retain the latest Link5 VLM guard outputs and
full-video SAM mask tracks for the 45 selected videos and COSMOS2.5_0018.
Retain their source identities, configuration, reference prompts and exact guide
selection. Generated geometry, observed clouds (including frame0), normalization,
trained checkpoints, training reports, synthetic examples and replay exports were
authorized for deletion. Regeneration must reuse the validated mask cache without
new VLM/SAM calls; MegaSAM geometry must be rebuilt. No new GPU job is authorized
by the cleanup itself. Preserve source videos, shared models and unrelated work.

When exporting the Link5 replay, keep the existing canonical HTML focused on
only the four-frame0 normal reference, observed test-cloud anomaly display and
one checkpoint download. The user explicitly removed prompt/mask tabs, synthetic
examples, training tables, model variants and long explanations from this view.
Keep exact provenance outside the viewing path. Do not restore the removed
diagnostic tabs on a later export. Preserve raw scores and label failed validation
plainly; do not describe an unvalidated candidate as SOTA.
