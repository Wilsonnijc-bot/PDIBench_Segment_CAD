# VLM1–VLM2–SAM3 masking pipeline specification

## Purpose

Track a deforming robot gripper through a generated video while preserving explicit frame lineage. The pipeline separates naive mask-area event detection, VLM1 deformation diagnosis, VLM2 point placement, and SAM3 propagation.

## Configuration and request inspection

`persistent_masking/vlm_interface/config.py` is the only model/backend configuration
interface for the active pipeline. VLM1 and VLM2 can independently use a local
GPU model or an OpenAI-compatible cloud API. Cloud credentials remain in the
environment and are referenced by variable name from the configuration.

Before every VLM2 request, the pipeline writes `vlm2_interface/README.md` and an
`images/` subfolder. The Markdown shows the exact system prompt, user task,
model/backend, selected reseeding frame, and three annotated reference images.
The case copy is retained and the run-root copy always represents the latest
attempted VLM2 request, including a request that later fails.

Each case has two distinct important frames:

- **Initial VLM1 frame:** the first frame selected from the surge-centered diagnosis window and sent to VLM1 for deformation classification. This frame must be shown in the replay artifacts.
- **VLM2 reseeding frame:** the original-video frame immediately after the earliest frame that VLM1 diagnosed as deformed. This is the only frame used for VLM2 point placement and SAM3 reseeding.

These frames must never be silently substituted for one another.

## Stage 1 — Naive SAM3 masking

Run one naive SAM3 initialization on the original video. Propagate the selected object through the complete video without VLM1 or VLM2 point prompts.

Record:

- source video and frame count;
- naive SAM3 object ID;
- mask for every source frame;
- mask area for every source frame;
- naive masking replay, when requested for diagnostics.

The naive mask is an event detector and baseline. It is not the final mask.

## Stage 2 — Detect mask-area surges

Compute the naive mask area `A[t]` for every frame. Detect each declared surge transition from `A[t-1]` to `A[t]` using one frozen threshold policy.

For each event, record:

```text
transition: [t-1, t]
event_frame: t
surge_frame: t-1
area_before: A[t-1]
area_after: A[t]
```

`surge_frame` is the earlier side of the transition. The event detector must not be confused with the later VLM2 reseeding frame.

## Stage 3 — Select VLM1 diagnosis frames

For every surge event, select the declared diagnosis window around the event. Preserve the exact original-video frame numbers and crops.

The VLM1 diagnosis input consists of:

- one reference crop from a known non-deformed frame;
- one candidate crop for each selected diagnosis frame;
- the segment-specific deformation prompt.

The initial VLM1 frame is the first candidate frame selected for diagnosis in the accepted run. It is an inspection artifact and must be exported even when VLM1 classifies it as normal.

## Stage 4 — VLM1 deformation diagnosis

Call VLM1 once per selected candidate frame. Each call receives the same non-deformed reference and exactly one candidate frame.

Each response must contain:

```json
{
  "frame": 123,
  "state": "deformed" | "normal" | "unclear",
  "probability": 0.0,
  "evidence": "visible geometric evidence"
}
```

Do not merge multiple candidate frames into one VLM1 call. Preserve the raw response and the parsed result in compact provenance.

The **ordered deformed diagnosis frames** are:

```text
deformed_frames = sorted(frame for diagnosis if state == "deformed")
earliest_deformed = deformed_frames[0]
```

If no frame is diagnosed as deformed, the case ends as `no_confirmed_deformation`; do not reseed SAM3 and do not fabricate VLM2 points.

## Stage 5 — Select the SAM3 reseeding frame

Return to the original source video. Set:

```text
reseeding_frame = selected_deformed_frame + 1
```

Normally `selected_deformed_frame` is `deformed_frames[0]`. If VLM2 returns
`{"positive_points": null}` (or otherwise fails the exact three-positive
schema), retry the positive VLM2 call on the next ordered VLM1-deformed frame:
`deformed_frames[1]`, then `[2]`, and so on. Each retry must read a fresh
original-video frame, rebuild the four-image VLM2 input, and record its rank.
Stop at the first valid three-positive response. If all confirmed deformation
frames fail, record `failed_vlm2` and do not run SAM3. The negative call is made
on the final selected frame; a valid negative response from a failed earlier
attempt is provenance only and is never silently mixed with another frame.

The reseeding frame must be read directly from the original video, not from a VLM1 crop, a surge screenshot, or an already rendered replay.

If `reseeding_frame` is outside the source video, record `reseeding_frame_unavailable` and stop the case without point placement.

Export both frame artifacts separately:

- `vlm1_initial_frame.png`: the initial VLM1 diagnosis frame;
- `vlm2_reseed_frame.png`: the original-video frame immediately after the selected VLM1-deformed frame.

## Stage 6 — VLM2 SAM point placement

Use the VLM2 backend and model declared in `vlm_interface/config.py`. A cloud VLM2 can use
an OpenAI-compatible Responses or Chat Completions endpoint; a local VLM2 uses
the configured GPU model directory.

VLM2 receives:

1. `vlm2_reseed_frame.png`, exactly as read from the original source video;
2. three cropped ideal point-placement examples.

The non-deformed reference is used by VLM1 only and is not sent to VLM2.

### Positive call

Make one call that returns exactly three positive points. The prompt must state directly:

- place all three points strictly inside visible white or dark gripper material;
- spatially cover the deformed gripper;
- follow the green positive pattern in the examples;
- never place a positive point on the forearm, upper wrist, background, or held object;
- return only normalized coordinates relative to the VLM2 reseeding frame.

Required schema:

```json
{"positive_points":[[x1,y1],[x2,y2],[x3,y3]]}
```

### Negative call

Make one call that returns exactly two negative points. The prompt must state directly and in order:

1. place the first negative in the elbow-side half of the **forearm**, inside the black-rimmed link with white inset;
2. place the second negative on the upper white surface of the wrist end-cap near its dark top cap, just inside the **upper edge of the wrist**;
3. keep both negatives spatially distinct;
4. never place a negative on the gripper, lower fingers, held object, cup, banana, or background;
5. use the red negative pattern in the examples;
6. return only normalized coordinates relative to the VLM2 reseeding frame.

Required schema:

```json
{"negative_points":[[arm_x,arm_y],[wrist_x,wrist_y]]}
```

If either required negative location is not visible, record the call as invalid and do not reseed SAM3 with incomplete points. When positive fallback changes the selected frame, rerun the negative call on that same final frame.

## Stage 7 — SAM3 reseeding and propagation

Start a new SAM3 session on the original source video. Add all five VLM2 points at `reseeding_frame`:

```text
three positive labels: 1, 1, 1
two negative labels: 0, 0
```

Propagate the reseeded object through the complete source video according to the frozen direction policy. Save the final mask sequence and render the full masking replay with source RGB beside the cyan mask.

The SAM3 prompt frame and VLM2 point-placement frame must be exactly the same `reseeding_frame`.

## Required user-facing artifacts per case

Each completed case must contain:

```text
<case>/
  vlm1_initial_frame.png       # initial frame given to VLM1
  vlm2_reseed_frame.png        # original frame immediately after earliest VLM1-deformed frame
  points.png                   # VLM2 points drawn on vlm2_reseed_frame.png
  vlm2_interface/README.md     # exact latest prompt with embedded image links
  vlm2_interface/images/      # target frame plus three reference PNGs
  masking.mp4                  # complete source-RGB + cyan-mask replay
```

`points.png` must show:

- green: the three VLM2 positives;
- red: the wrist-edge and forearm negatives;
- the underlying RGB image from `vlm2_reseed_frame.png`.

Do not draw points on a candidate crop while labeling them as source-frame coordinates. Convert normalized VLM2 coordinates to source pixels using the exact reseeding-frame dimensions.

## Validation

Before a case is considered complete:

- verify `vlm1_initial_frame.png` frame number and hash;
- verify `vlm2_reseed_frame.png` is exactly `selected_deformed_frame + 1` from the original video;
- verify all five points are inside source-frame bounds;
- verify three positive labels and two negative labels;
- verify negative role order: forearm, then wrist edge;
- verify positive/negative points are not duplicates;
- verify the reseeding frame in SAM3 equals the VLM2 frame;
- verify every source frame has a propagated mask or an explicitly recorded unavailable status;
- verify the replay frame count equals the source video frame count;
- verify replay encoding is H.264 Main, `yuv420p`, with fast-start metadata.

## Output and cleanup contract

Keep only the user-requested PNG frames, point preview, and masking replay in each case viewing directory. Keep prompts, raw model answers, frame numbers, crop boxes, hashes, model configuration, and validation under one compact run provenance file outside the case viewing folders. Do not export surge contact sheets, duplicate frame grids, alternate replays, or intermediate masks unless specifically requested.

A case with no VLM1-confirmed deformation, unavailable reseeding frame, invalid VLM2 response after exhausting the ordered fallback frames, or incomplete replay is a recorded failure and must not replace an earlier accepted result. Point-membership mismatch is diagnostic metadata and does not by itself invalidate an otherwise complete replay.

## Implementation guard

`persistent_masking/frame_lineage.py` is the frame-lineage guard used by the corrected runner. It refuses empty VLM1 diagnoses, refuses out-of-range reseeding frames, reads both exported frames from the original video, and asserts that VLM2 and SAM3 use the identical `selected_deformed_frame + 1` frame.

The executable VLM2 handoff is `persistent_masking/vlm2_sam_prompting.py`. It accepts VLM1 diagnosis records, resolves the original-video selected deformation frame, retries positive-null VLM2 responses through later VLM1-deformed frames, sends the final frame plus the three examples to both VLM2 calls, validates exact point counts, draws `points.png` on the reseeding frame, and records the SAM3 frame invariant.


## Frozen execution retained from 2026-09-13

- Cases: Cosmos25_0023, Cosmos25_0053, LVP_0049, Cosmos3_0048.
- Fresh naive SAM3 initialization at frame zero from the existing DINO reference-box policy.
- Detect all upward log-area-ratio crossings >= 0.25 (approximately 28.4% growth), separated by at least round(0.5 * FPS) frames. Declines alone are not surges.
- Diagnose frames event_frame-4 through event_frame+5, clipped to video bounds, deduplicated and sorted.
- The configured VLM1 retains the palm visibility/deformation prompt and existing nondeformed reference crop, one reference/candidate pair per call. It performs diagnosis only.
- VLM2 uses the full original frame at selected_deformed_frame+1 and exactly three complete annotated examples. No nondeformed reference, previous mask overlay or crop is sent to VLM2. It makes one positive call, then ordered positive-null retries on later VLM1-deformed frames when needed, followed by one negative call on the final selected frame; no 50%/75% review.
- Corrected SAM3 uses one five-point prompt at that exact frame and bidirectional propagation over the complete video.
- Remote provenance records failures and hashes; local case folders contain PNG/MP4 only. `vlm1_initial_frame.png` is the exact first input crop, with the source-frame number stated in the README.
- Executable: `python -m persistent_masking.pipeline continue`; the preceding fresh naive stage must have completed for all declared cases.
