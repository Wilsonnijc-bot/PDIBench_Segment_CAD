# Occlusion case review and a minimal proposed refinement

**Implementation status:** the user approved the final severity-dependent
proposal, and it is now implemented as `gripper-occlusion-v2`. The sections below
preserve the chronological analysis and counterfactual results; earlier rejected
proposals were not deployed. See [the current rule](OCCLUSION_DETECTION.md).

Reviewed 2026-09-30. This is an analysis and offline counterfactual, not an implemented detector change. See [the current algorithm](OCCLUSION_DETECTION.md). Frame numbers below are zero based; subtract one from the original rigidity replay's counter to compare. The user's LVP015 frames 32–44, if read from that original counter, correspond to 31–43; the diagnosis is unchanged.

**Current recommendation: retain the existing detector and add a severity-dependent onset route, described in the final section below.** The earlier mask-only and weighted proposals started LVP015 too early at frame 22. The user clarified that frame 29 is an approximate ideal onset, not an absolute hard boundary. Evaluate the visual transition rather than fitting an exact frame number.

## Evidence and scope

The saved detector covers all 42 non-`_0001` cases (4,634 frames), as well as the three excluded `_0001` cases. Thirteen of the 42 currently have catches. The counterfactual comparison below uses every saved row in those 42 cases.

For the seven newly requested cases, all 571 frame decisions were reconstructed from the original object-mask, link7-mask, and track NPZs. Every detector field matched the saved JSON exactly; all input hashes matched. COSMOS3_0025 and LVP0040 had already been similarly verified. I inspected decoded source-video crops with independently reconstructed mask overlays for the seven new cases, and inspected five additional cases affected by the proposal. This is targeted visual review, not exhaustive frame-by-frame ground-truth annotation.

Notation: `c` = replaced fraction of expected silhouette; `e` = fraction of missing pixels explained by link7; `p` = eligible reference tracks whose current coordinates land in link7. The current onset requires `c≥20% AND e≥60% AND p≥20%`, with sufficient visible support and reference tracks.

## Why the successful cases work

| Case | Current caught interval | Pattern |
|---|---|---|
| COSMOS2.5_0010 | 12–82 | At onset: `c=20.11%`, `e=98.61%`, `p=100%`. Track overlap stays 84.21–100% throughout the caught interval. The gripper traverses a substantial portion of the cup silhouette. |
| COSMOS2.5_0015 | 54–90 | At onset: `c=32.07%`, `e=97.7%` approximately, `p=26%`. During the interval, track overlap stays 20–52% and loss explanation stays above 90%. Earlier moderate contact usually remains below the replacement floor. |
| COSMOS3_0015 | 154–181 | At onset: `c=33.88%`, `e=90.49%`, `p=23.91%` approximately. Replacement and point overlap then grow together as the gripper moves across the cup. |

The common pattern is sustained agreement between mask replacement and track location. These examples support retaining the reference silhouette, spatial attribution to link7, severity threshold, and duration filtering.

Continuation is also doing useful work. COSMOS2.5_0010 has 48 onset-qualified frames and 23 continuation-only frames in its caught interval. Its explanation fraction falls below 60% after frame 59, yet contact and tracks support continuation. COSMOS3_0015 has four continuation-only frames at 178–181; COSMOS2.5_0015 retains frame 90 through continuation despite replacement falling to 17.4%. A replacement formula that forgets episode state would regress these cases.

Success does not mean every boundary is perfect: COSMOS2.5_0015 frame 53 and COSMOS3_0015 frame 153 already have substantial mask evidence but narrowly miss the track gate. Also, COSMOS2.5_0010 loses assessability at frames 83–84 when visible support falls below 15%, ending continuation. That is a separate limitation; this proposal does not change it.

## Why the insufficient cases fail

| Case / interval | Mask evidence | Track evidence | Actual blocker |
|---|---|---|---|
| COSMOS3_0025, frame 162 | `c=53.33%`, `e=95.82%` | `p=3.33%` | Mandatory track gate vetoes strong mask evidence. Nearby track dips repeatedly break episodes. |
| LVP0040, frames 33–41 | `c=20.01–27.79%`, `e=88.33–97.36%` | `p=12.05–19.28%` | Moderate but persistent replacement; track overlap stays just below 20%. |
| LVP015, frames 32–44 | `c=21.40–64.19%`, `e=89.27–95.15%` | `p=1.09–16.30%` | Every frame passes the mask and assessability gates; every frame fails the track gate. |

LVP015 uses reference frame 18 throughout this interval. At frame 32, replacement is 50.5% and track overlap is 4.35%; at frame 37 they are 63.7% and 11.96%; at frame 43 they are 52.4% and 3.26%. Even frame 44 still has 21.4% replacement, with 90.6% of loss explained by link7. None is a candidate, so the two-frame filter is not the cause.

The current gate measures predicted coordinates inside the gripper, not physical occlusion of the original tracked surface. Tracks can stay on surviving visible texture, drift, or become unreliable without landing in link7. Low overlap therefore cannot always veto strong mask evidence. These are plausible mechanisms; the saved coordinates do not establish which mechanism occurred for each point.

For LVP0040, do not force the entire requested interval into one label: frames 31–32 and 42 have replacement below 20%; frame 43 also fails loss attribution. Retaining those as unflagged is consistent with the existing minor-contact tolerance. LVP015 and COSMOS3_0025 supply much stronger replacement evidence than these LVP0040 boundary frames.

Lowering the mandatory point threshold from 20% to 10% is insufficient: a raw-input rerun catches only LVP015 frames 35–37, leaving most of the requested episode missed. It also cannot accept COSMOS3_0025 frame 162, whose point overlap is only 3.33%.

## Why the acceptable-contact baselines remain negative

| Baseline | Maximum replacement anywhere in clip | Peak frame | Current result |
|---|---:|---:|---|
| LVP0010 | 9.78% | 26 | No catches |
| LVP0065 | 2.39% | 41 | No catches |
| LVP0021 | 14.45% | 38 | No catches |

All three remain below 20% throughout. The important separator in these examples is replacement severity, not a high point-overlap threshold. Keep that severity floor as a hard requirement for any new onset route. High loss attribution alone must not trigger detection when the missing area is tiny.

These are negative controls supplied by the user, not proof that their masks are perfect. In particular, some LVP0010 frames show poor gripper-mask coverage. Preserving its current negative label verifies the requested behavior on current inputs, not robustness to every future segmentation change.

## Proposed narrow addition

Keep the existing final flags, including their existing onset, continuation, and two-frame filtering. Independently compute:

```text
strong_mask_candidate = current frame is assessable
                        AND replacement c ≥ 20%
                        AND loss explanation e ≥ 80%

strong_mask_flag = strong_mask_candidate belongs to a run of ≥3 consecutive frames

proposed_flag = existing_flag OR strong_mask_flag
```

This relaxes track dependence only when attribution is stronger (80%, versus the existing 60%) and persists longer (three frames, versus two). It does not add padding or bridge failed frames. Three-frame filtering is offline and retains the whole qualifying run, including its first two frames. At 16 and 24 fps, three frame periods are about 188 and 125 ms respectively; this is a frame-count heuristic, not a uniform duration guarantee.

Keep the two branches independent: new mask-only catches should not activate the old, more permissive continuation rule. That bounds the change and ensures all original catches survive. Report the added branch separately so reviewers can see why each frame was accepted.

### Counterfactual result on the requested cases

| Case | Existing | Existing + strong-mask branch |
|---|---|---|
| COSMOS2.5_0010 | 12–82 | 12–82 |
| COSMOS2.5_0015 | 54–90 | 53–90 |
| COSMOS3_0015 | 154–181 | 153–181 |
| COSMOS3_0025 | 114–115; 167–174 | 114–115; 158–175 |
| LVP0040 | None | 33–41 |
| LVP015 | None | 22–44 — rejected as too early; approximately 29 is preferred |
| LVP0010, LVP0065, LVP0021 | None | None |

User correction after this review: **LVP015 frame 22 is too early; approximately frame 29 is the ideal onset.** The user subsequently clarified that 29 is not an absolute hard gate. The proposed 22–44 interval starts too early, but this feedback does not provide exact per-frame ground-truth labels for 22–28. Its early start must not be presented as successful recovery, and the aggregate 95 additions are predictions, not 95 validated improvements. Do not hard-code a case-specific start frame into the detector.

### Weighting experiment

I also tested the same additive branch with the extra condition:

```text
0.85 × c + 0.15 × min(p, 0.20) ≥ 0.20
```

The hard `c≥20%`, `e≥80%`, assessability, and three-frame requirements remained. Capping `p` keeps very high point overlap from dominating the evidence. This is a transparent experimental score, not a calibrated probability.

It preserves the same successful and negative cases, but ends LVP0040 at frame 40 rather than 41, and LVP015 at 43 rather than 44. Across all 42 cases it adds 92 frames instead of 95, changing the same 11 cases. It still starts LVP015 at frame 22, so weighting does not resolve the user's corrected onset constraint. Neither version is ready for adoption.

Raising the new explanation threshold from 80% to 85% reduces additions to 86 frames in ten cases, but delays COSMOS3_0025's late catch to frame 161, missing its frame-160 score peak. Reducing persistence from three to two frames adds two more frames in LVP0056. These are sensitivity checks on the same examples, not independent validation or optimized thresholds.

## Changes elsewhere and remaining uncertainty

The proposed exception adds **95 / 4,634 frames (2.05%) across 11 / 42 cases**, removes none, and increases cases with catches from 13 to 17. Those modest frame counts do not guarantee a modest false-positive cost.

Outside the successful/missed examples requested here, it adds:

| Case | Added frames | Visual-review interpretation |
|---|---|---|
| COSMOS2.5_0044 | 83–92 | Similar finger contact persists after the original catch ends; track overlap oscillates around 20%. |
| COSMOS2.5_0046 | 83–92 | Similar contact persists while track overlap falls to 16–19%. |
| COSMOS2.5_0054 | 28–32; 37–41 | Gripper crosses the small yellow object; rotation and silhouette mismatch remain confounders. |
| COSMOS3_0056 | 161; 171 | Small extensions of the previously reviewed contact episode. |
| LVP0054 | 12–17 | Gripper body crosses the upper object silhouette despite zero current track overlap. |
| LVP0056 | 11; 19–26; 41–44 | Object appearance/shape and mask quality also change markedly. These additions need particular scrutiny to avoid excluding genuine deformation along with occlusion. |

The remaining five changed cases are COSMOS2.5_0015, COSMOS3_0015, COSMOS3_0025, LVP0015, and LVP0040 from the requested comparison.

The strongest supported conclusion is that **a mandatory track gate can miss substantial occlusion, but removing that gate based on persistent mask replacement alone can start too early**. The three negative controls support keeping the current 20% severity floor. They do not establish the precision of all newly added intervals. A refined proposal should start closer to LVP015's visually consequential episode around frame 29, while preserving the successful and acceptable-contact cases. LVP0056 also needs review before automatic score exclusion.

No detector implementation, detection JSON/CSV, or rigidity scores were changed. Additional replay HTML files only render existing saved detections. Carry-forward rigidity values and scoring exclusions remain separate issues; a carried score is not itself a new occlusion rule.

## Refined proposal: demand stronger evidence to start than to continue

This section supersedes the earlier mask-only recommendation. Frame 29 is the user's approximate ideal, not a hard per-case gate. The replacement estimates also have reference-shape and segmentation uncertainty; a one-frame difference should not be optimized as if the human label were exact.

Early LVP015 replacement is not merely an alignment artifact: at frame 25, the current mask has about 42.3% less total area than the reference and estimated gripper replacement is 41.1%. At frame 29 the corresponding values are 50.2% and 49.2%. Area shrinkage alone does not separate acceptable early contact from the desired onset. A small unconditional track weight does not solve that problem either.

The narrow refinement is to keep track corroboration for moderate replacement, but allow severe replacement to start an episode even when predicted point locations do not land in the gripper. Once evidence has established the episode, require continued substantial, well-explained replacement rather than requiring the onset evidence to recur in every frame.

Keep existing final flags unchanged. Independently evaluate a new branch with the same expected masks, reference updates, track membership, and assessability rules:

```text
base = assessable AND c ≥ 0.20 AND e ≥ 0.80

new_onset = base AND (p ≥ 0.10 OR c ≥ 0.50)
new_continue = previous_new_candidate AND base
new_candidate = new_onset OR new_continue

new_flag = new_candidate belongs to a run of ≥3 consecutive frames
final_proposed_flag = existing_flag OR new_flag
```

Here `c`, `e`, and `p` retain their definitions above. The new branch does not use direct overlap as replacement. It has no case names, rigidity-score thresholds, fixed frame numbers, gap bridging, or backfilling before onset. Three-frame filtering retains the candidate run from its onset; it does not require three consecutive fresh onset qualifications. A base failure resets this new branch immediately. Its state is separate from the original detector's more permissive continuation state.

Interpretation: moderate loss needs at least 10% track corroboration and stronger mask attribution than the original onset. Loss of at least half the expected silhouette can qualify on mask evidence alone. Continued loss of at least 20% remains flagged after a qualifying onset. These are provisional, interpretable thresholds tested on this review set, not independently calibrated optimal values.

### LVP015 onset and continuity

| Frame | Replacement `c` | Tracks `p` | New-branch interpretation |
|---|---:|---:|---|
| 22 | 27.9% | 0.0% | Too little point support for moderate replacement; no onset |
| 25 | 41.1% | 1.1% | Neither onset route qualifies |
| 28 | 49.2% | 5.4% | Neither onset route qualifies |
| 29 | 49.2% | 5.4% | Neither onset route qualifies |
| 30 | 54.3% | 6.5% | Severe replacement starts the new episode |
| 31 | 49.9% | 5.4% | Continued strong mask evidence retains the flag |
| 32–43 | 41.2–64.2% | 1.1–16.3% | Continued mask evidence retains the episode |
| 44 | 21.4% | 4.3% | Still above the continuation replacement floor |

The resulting interval is **30–44**, close to the user's approximate onset. Frame 30 is not a hard-coded boundary. Continuation avoids fragmenting the episode when replacement falls just below 50% or track overlap falls again.

### Full-set counterfactual

| Requested case | Current catches | Refined proposal |
|---|---|---|
| COSMOS2.5_0010 | 12–82 | 12–82 |
| COSMOS2.5_0015 | 54–90 | 53–90 |
| COSMOS3_0015 | 154–181 | 153–181 |
| COSMOS3_0025 | 114–115; 167–174 | 114–115; 158–175 |
| LVP0040 | None | 33–41 |
| LVP0015 | None | 30–44 |
| LVP0010, LVP0065, LVP0021 | None | None |

Across all 42 non-`_0001` cases, this adds **72 frames (1.55% of 4,634) in ten cases**, removes no existing catches, and changes the number of cases with catches from 13 to 16. The earlier unconstrained mask branch added 95 frames in 11 cases.

Other additions are COSMOS2.5_0044 83–92, COSMOS2.5_0046 83–92, COSMOS2.5_0054 28–32, COSMOS3_0056 161 and 171, and LVP0056 11 and 19–26. Compared with the previous proposal, the refined branch avoids all additions in LVP0054, the second COSMOS2.5_0054 interval, and the late LVP0056 interval. LVP0056 19–26 remains uncertain because apparent deformation and occlusion may coexist. Preserving original catches is guaranteed by the additive construction, not independent proof of correctness.

### Sensitivity and recommendation

Changing the severe-onset threshold from 50% to 45% moves LVP015 onset to frame 27; 55% moves it to 34. Both leave the proposed COSMOS3_0025 and LVP0040 intervals unchanged. Changing the reduced track threshold from 10% to 8% or 12% leaves all three missed-case intervals unchanged. Raising attribution from 80% to 85% delays COSMOS3_0025 to frame 162, losing its frame-160 peak. These are sensitivity checks on reviewed examples, not held-out validation.

I recommend this severity-dependent onset and mask-supported continuation as the next proposal to evaluate. It preserves point corroboration where mask loss is moderate, permits severe loss to overcome unreliable point coordinates, and avoids fitting an exact LVP015 frame number. No production detector or score outputs have been modified. Review the proposed transitions around LVP015 27–34 and the remaining LVP0056 additions before using this rule for automatic score exclusion.
