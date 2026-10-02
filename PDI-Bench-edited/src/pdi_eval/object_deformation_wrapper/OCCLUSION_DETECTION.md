# Gripper occlusion: current rule and the original frame-162 diagnosis

## Current implementation: v2

The current v2 audit first applies `link7-frame-area-v1`: a link7 mask covering
25% or more of the source image fails that frame. This cutoff is configurable
with `--max-gripper-area-fraction`. Failed rows have `mask_valid=false` and
`status=failed_link7_mask_area`. They cannot supply references or replacement
evidence, and they reset both episodes. The filtered mean (method v2) excludes
failed rows independently of occlusion catches; all-excluded cases have null
scores. The saved LVP010 and LVP015 tabletop masks fail in all 49 frames. The
older LVP015 30–44 catch described in the case review is superseded by this
mask failure, rather than accepted as physical-occlusion evidence.

The approved refinement is implemented as `gripper-occlusion-v2`. It preserves
the v1 reference-mask construction, track membership, assessability checks,
onset, continuation, and two-frame filtering described below as the **original
branch**. Its final flags are now stored as `legacy_flagged`.

An independent new branch uses the same per-frame measurements:

```text
base = assessable AND occluded_fraction >= 20% AND explained_fraction >= 80%
severity_onset = base AND (track_fraction >= 10% OR occluded_fraction >= 50%)
severity_candidate = severity_onset OR (previous_severity_candidate AND base)
severity_flagged = severity_candidate belongs to a run of >=3 consecutive frames
flagged = legacy_flagged OR severity_flagged
```

Any failed base resets the new branch. Filtering is independent for each branch;
new-branch state never activates original-branch continuation. No frames before
onset are backfilled. Each surviving three-frame run includes its first two
frames. `candidate` is the OR of both unfiltered candidates. Existing `onset`
and `continued` fields retain their original meanings; `severity_onset` and
`severity_continued` expose the new branch, with continuation recorded only when
onset does not also qualify. `legacy_candidate` and `severity_candidate` retain
the separate unfiltered states.

The new defaults live in `Config`; `--min-occluded-fraction` controls the shared
replacement floor, while `--severity-min-explained-fraction`,
`--severity-min-track-fraction`, and `--severity-min-occluded-fraction` control
the new branch's 80%, 10%, and 50% thresholds. Method, configuration, input hashes,
and detector-source SHA-256 are saved in `detection.json`.

Frame 162 in COSMOS3_0025 is now caught by the added branch. Its mask and track
measurements below have not changed. The [case review](OCCLUSION_CASE_REVIEW.md)
records the refinement and its comparisons.

## Historical v1 diagnosis

The remainder of this document describes the v1 result before this update;
statements about a missed frame or a retained filtered score refer to that
historical result, not current v2 outputs.

Verified on 2026-09-30 against `results/object-deformation-selected45-20260929`. Implementation: [occlusion.py](occlusion.py), method `gripper-occlusion-v1`. All frame indices here are **zero based**, matching the occlusion replay. Its frame 162 is frame 163 in the original rigidity replay counter.

**The detector should identify substantial loss of visible object evidence caused by the gripper, so that unreliable rigidity measurements do not masquerade as object deformation. The current implementation misses COSMOS3_0025 frame 162 because it makes current track–gripper overlap mandatory, even when the mask evidence strongly supports occlusion.**

## The problem being solved

A rigid object can retain its shape while a gripper finger hides much of it. The visible object mask shrinks, tracking evidence becomes unreliable, and the rigidity score can rise without actual deformation. We want a per-frame risk flag for this situation: a substantial part of the expected object silhouette disappears specifically where link7 is located.

Minor contact should remain acceptable. A small hidden fraction, especially with little effect on tracked object evidence, should not automatically exclude a frame. Equally, shrinking masks alone are insufficient: rotation, perspective, actual deformation, and segmentation errors can all change a silhouette. Gripper overlap alone is also insufficient because link7 segmentation can leak onto visible object pixels.

The practical question is: **How much expected object evidence is missing, how much of that loss is spatially explained by the gripper, and what evidence supports tracking risk?** These observations support an occlusion-risk decision; they cannot prove that occlusion caused a particular score surge.

## Inputs and alignment

Paths below are relative to `results/`:

| Evidence | File and selection |
|---|---|
| Current object mask `O[t]` | `object-deformation-selected45-20260929/cases/<case>/masking/segmentation.npz` → `object_masks[:, 0]`, named `task_object` |
| Current gripper mask `G[t]` | `link5-link7-four-way-selected45-20260927/cases/<case>/v1_cotracker3/segmentation.npz` → `object_masks[:, 5]`, named `link7` |
| Object tracks and visibility | Object case `score/cotracker_exact-group.npz` → the `task_object` slice selected by `object_names` and `object_offsets` |

The detector uses **both segmentation NPZs plus the saved tracks**. It checks mask shapes, track/frame compatibility, track source dimensions, source-video dimensions/frame count, channel names, score length, and the video's SHA-256 against gripper provenance. Input file hashes are recorded in `occlusion/detection.json`.

## Exact current logic

### 1. Choose and maintain a reference silhouette

Let `A[t] = |O[t]|` and `direct_overlap[t] = |O[t] ∩ G[t]| / max(A[t], 1)`.

The initial reference `a` is the first nonempty object mask with direct overlap **≤ 5%**. Its tight crop becomes the silhouette template. Earlier frames are `insufficient_reference`; if no such frame exists, every frame has that status. Low overlap is a heuristic for a clean reference, not proof that the whole object is visible.

For each frame, translate the template near the current object-mask centroid. The search rectangle extends approximately 1.5 template widths/heights on each side of that centroid, clipped to the image. `cv2.matchTemplate(..., TM_CCORR)` chooses the placement maximizing the sum of pixel evidence under the template:

- `+1` inside the current object mask, including pixels also labeled link7;
- `0` inside link7 but outside the object;
- `−1` everywhere else.

The resulting full-image silhouette is `E[t]`. The template retains its exact shape and area: **no scaling, rotation, dilation, or filled bounding box**. An empty object mask or a search region too small to fit the template produces `insufficient_visible_object` and resets the episode.

After evaluating a frame, refresh the template and reference to that frame only when all three conditions hold:

```text
direct_overlap ≤ 5%
gripper-replaced fraction of E ≤ 5%
0.8 ≤ current object area / expected area ≤ 1.25
```

Refresh affects subsequent frames. It is not conditional on passing the track/support gates. These are low-contact tolerances, not strict gripper absence. Otherwise the existing reference remains frozen.

### 2. Measure missing pixels and gripper replacement

```text
L = E \ O                  expected pixels absent from the object mask
C = L ∩ G                  missing pixels occupied by link7
lost_fraction      = |L| / |E|
occluded_fraction  = |C| / |E|
explained_fraction = |C| / max(|L|, 1)
visible_support    = |E ∩ O| / |E|
```

Only **missing** expected pixels contribute to `C`. Direct `O ∩ G` overlap cannot start an episode. The denominator of `occluded_fraction` is the expected silhouette, not the shrunken current mask. `lost_fraction` is a spatial mismatch against an aligned reference; it is not simply the scalar decrease in mask area.

The algorithm does not explicitly measure gripper approach, a frame-to-frame increase in gripper coverage, or a rate of object-area decrease. Temporal information enters through the reference updates and episode state.

### 3. Measure affected tracks

At reference frame `a`, an eligible track must be visible and have its rounded coordinate inside `O[a]` but outside `G[a]`. This set stays fixed until the reference changes.

```text
track_fraction = eligible tracks whose current coordinates land in G[t]
                 / max(number of eligible tracks, 1)
```

Mask sampling uses `np.rint` coordinates; nonfinite or out-of-image coordinates count as misses. **Current visibility does not filter either numerator or denominator.** Currently invisible eligible tracks can still count as gripper hits. Conversely, loss of tracker visibility is not independently used as occlusion evidence. The detector does not extrapolate the reference track positions with the silhouette translation.

### 4. Start or continue an episode

A frame is assessable only if `visible_support ≥ 15%` and there are at least **five eligible reference tracks**. Otherwise its status is `insufficient_support` and it cannot be a candidate.

All onset gates are required:

```text
onset = assessable
        AND occluded_fraction ≥ 20%
        AND explained_fraction ≥ 60%
        AND track_fraction ≥ 20%
```

Continuation uses the previous frame's candidate state:

```text
continuation = previous_candidate
               AND assessable
               AND |E ∩ G| / |E| ≥ 20%
               AND track_fraction ≥ 20%

candidate = onset OR continuation
```

Continuation allows object-mask regrowth: `E ∩ G` includes direct object/gripper overlap, so missing-pixel evidence no longer has to pass. It still requires the same track threshold. A failed frame immediately resets the state; there is no grace period or gap bridging. The saved `continued` field is true only for continuation without onset.

Finally, contiguous candidate runs shorter than **two frames** are removed. `flagged` is the surviving result. This filtering happens after episode construction; `candidate` and `onset` can therefore be true when `flagged` is false. Unassessable frames are unflagged, which must not be interpreted as verified reliable frames.

These are the saved defaults for all three examples. The CLI exposes the three onset fraction thresholds; the other defaults live in `Config`.

## Why COSMOS3_0025 frame 162 is not caught

The screenshot is frame **162 at 6.75 s**. The reference is frame **154**, whose silhouette has **645 pixels**. Reconstructing the aligned mask and sampling the original track archive gives:

| Quantity | Exact count / fraction | Decision |
|---|---:|---|
| Current object area | 295 pixels | Descriptive |
| Expected silhouette still in object | 286 / 645 = **44.34%** | Passes 15% support |
| Missing expected pixels | 359 / 645 = **55.66%** | Descriptive |
| Missing pixels occupied by link7 | 344 / 645 = **53.33%** | Passes 20% replacement |
| Missing pixels explained by link7 | 344 / 359 = **95.82%** | Passes 60% explanation |
| Eligible reference tracks | **90** | Passes minimum of five |
| Eligible tracks currently inside link7 | 3 / 90 = **3.33%** | **Fails 20% track gate** |
| Total expected silhouette under link7 | 362 / 645 = **56.12%** | Passes continuation contact gate |

Thus `onset=false`, `candidate=false`, and `flagged=false`. Frame 162 is **not** an isolated candidate removed by the two-frame filter. It fails before that filter.

Continuation cannot rescue it: frame 161 was already not a candidate, and frame 162 independently fails the continuation track gate. Nearby decisions expose the same failure:

| Frame | Replaced silhouette | Tracks in link7 | Result |
|---|---:|---:|---|
| 158 | 22.95% | 21.11% | Onset, then removed as a one-frame run |
| 159 | 28.68% | 16.67% | Track gate fails; episode resets |
| 160 | 30.39% | 8.89% | Track gate fails |
| 161 | 44.03% | 8.89% | Track gate fails |
| **162** | **53.33%** | **3.33%** | **Track gate fails** |
| 163 | 57.21% | 2.22% | Track gate fails |
| 164 | 58.45% | 1.11% | Track gate fails |
| 165 | 66.51% | 22.22% | Onset, then removed as a one-frame run |
| 166 | 71.63% | 18.89% | Track gate fails |
| 167–174 | — | ≥ 20% each frame | Surviving flagged run |

The coral region in the replay already visualizes substantial inferred replacement. It does not itself determine the verdict. The white point cluster is also meaningful: **71 of the 90 eligible track coordinates land on the remaining current object mask at frame 162**, while only three land inside link7. Only 44 of the 90 are currently marked visible, but that visibility decline is not a detection gate. These counts establish where the tracker places its points; they do not establish whether those points still correspond to the correct physical surface locations. Drift or concentration on surviving visible texture is a plausible failure mechanism, not proven ground truth.

**The implementation equates “few predicted points inside link7” with insufficient tracking-risk evidence. That proxy can fail precisely when occlusion makes the predicted coordinates unreliable. Strong mask evidence is then vetoed by the tracking output whose reliability is in question.**

The displayed rigidity value is **0.2627041943 (26.27%)**, carried from frame 160 through frames 161–173. In [the scorer](../v1/rigidity.py), fewer than three currently visible selected pairs causes the previous score to be copied. Consequently, frame 162's plotted value is not a fresh measurement of its deformation. The detector appends `rigidity_carried` for reporting but never uses it to decide occlusion.

## What the other replays establish

| Case | Saved flagged intervals | Relevant behavior |
|---|---|---|
| COSMOS3_0025 | 114–115; 167–174 | Strong late mask evidence is rejected until the track gate also passes persistently. |
| COSMOS3_0056 | 111–151; 157–160; 162–170 | Frame 140 is caught through continuation despite only **12.23%** replacement; **97.70%** of eligible tracks are inside link7. This demonstrates the mask-regrowth exception. |
| COSMOS2.5_0065 | 21–92 | The link7 contour includes visible cup pixels. At onset frame 21, direct overlap covers **99.98%** of the object mask. Such contaminated segmentation makes the long interval uncertain as a physical-occlusion label. |

I inspected playback and overlays in the three local occlusion replays and reconstructed all **471 frames** from their original mask/track NPZs. Every computed detector field exactly matched the saved JSON, and all recorded input hashes matched. This verifies implementation/output consistency; it does not establish precision or recall against labeled physical occlusion.

## Implication for the next revision

Frame 162 exposes a mismatch between the intended policy and its implementation: accepting *small occlusion with largely unaffected tracks* does not imply rejecting *large, strongly gripper-explained loss whenever current track coordinates avoid the gripper*.

A focused next revision should test whether strong, persistent mask-replacement evidence can qualify without a mandatory current-coordinate track hit. Track overlap could corroborate borderline cases. Simply lowering the global track threshold would weaken protection everywhere, and the cup example shows why segmentation contamination still needs attention. This is a proposed direction, not a rule implemented by the current detector.

The detector itself leaves original scores unchanged. [occlusion_scores.py](occlusion_scores.py) separately averages original frame scores after frame 0, excluding only `flagged` frames. It retains unflagged carried and unassessable values, so frame 162 remains in the current filtered mean. This documentation review changes neither detection thresholds nor score outputs.
