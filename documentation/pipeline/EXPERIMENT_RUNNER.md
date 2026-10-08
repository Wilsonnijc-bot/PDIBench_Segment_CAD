# Parallel experiments and deferred VLM recovery

Updated 2026-10-05. The maintained coordinator now implements timeout deferral and expected low-track outcomes. A separate experiment-level Slurm controller remains a proposal. No new GPU inference was launched to validate these changes.

## Implemented coordinator behavior

Use `infrastructure/deformation_detect/coordinator.py`, with one child coordinator per video. `execution.workers` controls simultaneous video processes and `gpu_slots` bounds model stages. In a multi-GPU allocation, `gpu_assignment: per_video` pins one allocated GPU to each active video. The maintained ERIS launcher runs multiple end-to-end video coordinators on one H200 with shared GPU assignment. Start with five concurrent videos and adjust only from measured memory and throughput. GPU allocation and video-worker concurrency are separate settings. The earlier two-job/five-GPU acceptance recipe is historical.

| Outcome | Action |
|---|---|
| Robot cloud request has no response before its timeout | Persist typed `vlm_timeout` evidence; defer the video; continue other cases; retry at the tail with the same model |
| Gemini response is null, malformed, or its points/candidate fail the existing checks | Switch the next bounded logical attempt to the authorized Luna fallback; preserve rejection evidence |
| Luna request | Allow at least 600 seconds per request, bounded by remaining active stage time |
| Link2 or Link7 has insufficient CoTracker tracks | Complete the measurement with an unavailable/null rigidity score; retain track counts; continue object anomaly scoring |
| Complete score with a scale-jump warning | Retain score and diagnostic warning; no automatic mask repair or score-driven retry |
| Other request/infrastructure/SAM/scoring errors | Preserve existing retry/disable rules; do not relabel unknown failures as timeouts |

Typed no-response timeout handling and semantic VLM fallback are robot-side policies. Object grounding retains its existing model sequence and error handling; object tracking remains the occlusion tracker.

Serial coordination uses a durable FIFO tail queue. Parallel coordination releases the timed-out child (exit3) and GPU, schedules remaining original cases, then waits for **all original children in that allocation** to finish before a deferred wave starts. One batch need not wait for another independent Slurm allocation. A repeatedly timed-out stage gets at most four logical attempts total, including deferred launches. Queue waiting is excluded from the 3600-second active VLM-stage budget; execution time and interrupted attempts still consume it. All attempts, rejection evidence, logs and independent exits remain available. Exhaustion still requires an explicit `--retry-disabled` to reopen a budget.

VLM3 retains its pre-geometry gate: object coverage >95% OR Link7 image area >=25%. Its accepted-mask membership, object exclusion and crop-exclusion rules are unchanged. A later scale-jump audit measures abrupt changes in reconstructed foreground median depth; it does not trigger VLM3. Timeout feedback is not presented to VLM3 as a rejected point candidate.

## Proposed outer Slurm controller

A global queue after **both** main batches needs an outer scheduler controller. It has not been implemented or deployed. Keep it under `infrastructure/deformation_detect/`, with a Slurm adapter under `infrastructure/deployment/eris/`, and invoke the existing coordinator for scientific work.

1. Freeze the current working-tree snapshot, input/checkpoint identities and public runtime configuration. Preflight dependencies before spending case budgets.
2. Split N cases into stable groups of `ceil(N/2)` and `floor(N/2)`. Keep group size separate from worker capacity: 45 videos can use groups of 23/22 with five active workers per group.
3. Persist submission intents, unique scheduler tokens and returned IDs. Submit each batch through the existing launcher with independent logs/exits.
4. Use a small CPU `afterany:A:B` controller to reconcile both terminal batch states. `afterok` would skip recovery when a batch has disabled cases. The controller does not occupy a GPU while waiting.
5. For a global first-failure tail queue, add an explicit defer-only batch mode that stops after original cases and leaves deferred state. The controller then resumes only deferred cases with the same consumed budgets. Do not use `--retry-disabled` for ordinary deferrals or silently grant a fresh budget.
6. Submit reduced retry allocations under the same output ownership; omit empty groups. Persist round decisions, scheduler receipts and state atomically under a controller lock. Reconcile a saved submission token before submitting again after interruption.
7. Stop on completion or configured experiment deadline/exhaustion. Build one combined status/replay index and retain unavailable measurements and quality flags separately from runtime failure.

The currently implemented per-batch tail queue is sufficient to progress without close monitoring inside already submitted jobs. A read-only five-minute observer can report state; it does not perform retries. The proposed outer controller additionally automates submission/resubmission across allocations and provides the requested global barrier. Provider-wide circuit breakers, cross-job request admission and HTTP Retry-After policy are also future work.

## Verification and provenance

CPU tests cover robot timeout classification, primary-model retention, Luna duration, semantic routing, preserved attempts/budgets, serial queue order, real child processes finishing unrelated cases before recovery, bounded exhaustion, per-video GPU assignment and expected low-track outcomes continuing object scoring. Paid VLM requests and GPU inference are not used by these tests. A new GPU acceptance is still needed for the changed policy.

The earlier ten-video H200 run produced nine full pipelines and a tenth Link2 score/crop set but no object anomaly score after low Link7 tracks. Its native outputs and source inventory remain historical evidence. The new policy accepts its tracking measurement as an expected outcome; it does not fabricate the object score that the old execution skipped or establish numerical reproduction fidelity.
