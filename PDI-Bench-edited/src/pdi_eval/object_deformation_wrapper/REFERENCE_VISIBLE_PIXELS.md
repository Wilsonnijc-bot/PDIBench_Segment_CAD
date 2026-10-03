# Frame-zero reference crops from visible object pixels

`reference_visible_pixels.py` is a separate offline consumer of the saved task-object SAM3 mask, link7 SAM3 mask, and `gripper-occlusion-v2` detection. It does not run a model or change those methods or any rigidity score.

For every frame `t`:

```text
available_pure_object[t] = task_object_mask[t] AND NOT link7_mask[t]
                           if the saved link7 mask passes the area guard;
                           empty otherwise
```

The detector's saved `reference_frame` and existing `align(crop(anchor_mask), object_mask[t], link7_mask[t])` reproduce its estimated full-object silhouette. The exporter checks that its area equals `detection.json`. It maps the available binary shape from that full silhouette's bounding box into frame 0's object bounding box using nearest-neighbor resizing, then intersects with frame 0's SAM3 object mask. This supplies an RGBA crop of the **actual frame 0 RGB pixels** under the mapped shape; no RGB pixels from the later frame are substituted.

The saved `masks.npz` contains `available_pure_object[T,H,W]`, `mapped_to_frame0[T,h0,w0]`, `mapping_valid[T]`, `frame0_object_mask[h0,w0]`, and `frame0_bbox_xyxy[4]`. Bounds are `[x0,y0,x1,y1]` with exclusive `x1,y1`. The per-frame `manifest.json` records area, estimate coverage, reference anchor, and mapping status. A mapped mask can be inspected even when `mapping_valid` is false, but it should not be treated as a usable reference crop.

Export all saved cases:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.reference_visible_pixels \
  --object-root results/object-deformation-selected45-20260929 \
  --output-root results/object-reference-crops-20261001
```

Render any zero-based frame from an existing export, without repeating alignment:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.reference_visible_pixels \
  --object-root results/object-deformation-selected45-20260929 \
  --output-root results/object-reference-crops-20261001 \
  --render-only --cases COSMOS3_0056 --frame 115
```

This writes `examples/frame_00115/current_available.png` from the selected frame and `examples/frame_00115/frame0_shape_crop.png` from frame 0. Both have transparent backgrounds. `examples/frame_00115/preview.png` shows them beside the complete `frame0_reference.png` crop. Failed-mask frames receive no pixel crops; their preview says unavailable. A failed or contaminated frame 0 receives no reference PNG. Rendering from saved masks checks that the occlusion audit has not changed since export; if it has, re-export the case first.

This is silhouette-coordinate registration using translation from the existing detector and a bounding-box scale into frame 0. It preserves the mapped binary shape, including holes, but does not establish exact material-point correspondence through rotation, deformation, or a bad SAM3 mask. If link7 fails its saved area guard, frame 0 has over 5% task-object/link7 overlap, or fewer than 80% of available pixels fall within the estimated full-object bounding box, `mapping_valid` is false. A frame with no visible object or no full-object estimate is also invalid. Failed-mask frames have an empty `available_pure_object` mask and `mask_valid=false`; other per-frame available masks remain saved. Re-exporting a case refreshes any example previews already present so stale pixel crops do not survive a new mask failure.

## Select ten frames per video by time interval

After exporting the masks, run:

```sh
PYTHONPATH=PDI-Bench-edited/src /opt/homebrew/bin/python3 -m pdi_eval.object_deformation_wrapper.frame_selection \
  --object-root results/object-deformation-selected45-20260929 \
  --crop-root results/object-reference-crops-20261001
```

Use `--cases CASE_NAME` to refresh a subset. The normal ten-slot rule is:

| Video interval | Ordinary selection quota |
|---|---:|
| 0–20% | 0 |
| 20–40% | 2 |
| 40–60% | 2 |
| 60–80% | 2 |
| 80–100% | 4 |

Intervals use frame positions `floor(5 * frame / total_frames)` with half-open boundaries. For a constant-rate source this divides its duration into five equal intervals. Integer frame boundaries round upward; no frame belongs to two intervals. A frame's percentage is `100 * frame / total_frames`.

Before reserving or ranking frames, version 5 calculates mean `available_area` in every interval, including all source frames and zeros. If the final interval's mean is strictly below **50% of both** the pooled first-80% mean and the immediately preceding 60–80% mean, activate a **per-frame gate**. Both baselines must be positive. The preceding-interval check distinguishes a new final collapse from already persistent low visibility.

When activated, reject only final frames with `available_area < 0.5 * first_80_percent_mean`. Frames at or above that threshold remain eligible if other gates pass, including substantial post-occlusion recoveries. **Keep quotas 0/2/2/2/4 and the original fallback:** missing final slots use unused eligible frames in 60–80% first, then 40–60%, then 20–40%. If no eligible late frames remain and enough earlier donors exist, actual counts become 0/2/2/6/0. An already empty final interval keeps its existing selections. `--last-interval-min-area-ratio` defaults to `0.5` and controls both warning and frame thresholds; `0` disables the gate. This is an area heuristic, not a semantic disappearance detector; it does not detect enlargement or guarantee that a retained mask belongs to the object.

Eligible frames have valid link7 masks, `mapping_status="mapped"`, and nonempty available and mapped masks. Within **each interval**, rank by descending `available_area`, counting actual current-frame mask pixels. Earlier frame numbers break ties. No ranking across the whole video or temporal-spacing constraint is used. Saved occlusion flags do not exclude otherwise usable available object pixels from area-priority slots.

First reserve the **immediate successor of every contiguous run of final saved `flagged` diagnoses**, if unflagged, `status="assessed"`, crop-eligible, and passing the activated final-frame area gate. This uses the existing detector's branch union unchanged. Required recovery frames consume their own interval's quota before area choices. A required recovery in the first 20% overrides its ordinary exclusion. If recovery reservations exceed interval quotas, all reservations keep priority; normal interval slots fill in chronological interval order up to ten total. More than ten mandatory recoveries raises an explicit conflict rather than dropping required frames.

If an interval has too few usable frames, fill missing slots from other allowed intervals, trying 80–100%, then 60–80%, 40–60%, and 20–40%. Each donor interval uses its own area ranking. This fallback was explicitly chosen to retain ten crops per usable video. Never fill from the first 20%. Optional `--strict-bins` leaves missing interval slots empty instead. If insufficient eligible frames remain, report a shortfall without duplicating or fabricating crops.

Low-area rejected frames cannot return through fallback. Retained substantial final frames are still selected normally. `--strict-bins` keeps the original quotas and leaves missing slots unfilled.

`selection.json` records method `time-quintiles-with-post-occlusion-v5`, per-interval boundaries, quotas, eligible area rankings, selections, selection reasons, fallback use, and every occlusion episode. `final_interval_policy` records all five means, both ratios, `collapse_detected`, `frame_area_threshold`, excluded indices, and `preserved_eligible_frames`. `suppressed` means the warning activated and no eligible final frames survived. Selected frames are saved in chronological order. `selected/frame_XXXXX/` contains `current_available.png`, `frame0_shape_crop.png`, `preview.png`, and the full decoded `original_frame.png`; manually requested `examples/` remain separate. `selection_index.json` summarizes processed cases and retains other cases during subset runs. `selection_gallery.html` displays the crop pairs and original-frame pictures with case search and filters for recovery frames, replacements, and exceptions. Click a picture to open it at full size. Open the gallery directly in a browser; it needs no server or network. Keep it beside the case directories so its relative image links work.

A failed or unassessable immediate successor does **not** prove occlusion disappeared. Report it as `successor_not_assessable`; an assessed successor with an invalid crop is `successor_crop_invalid`. Neither is silently replaced with a later recovery frame. A run reaching the final frame has no successor and is recorded separately. `status="incomplete"` means a crop-count shortfall or an unmet recovery requirement; interval fallbacks are explicit but do not make a completed ten-crop selection incomplete.

An immediate successor below the activated area threshold is recorded as `successor_below_final_interval_area_gate`, deliberately not selected, and not counted as an unmet recovery requirement. A substantial successor remains required; COSMOS3_0015 frame 182 is preserved. This does not change the saved occlusion diagnosis.

Re-exporting a case with an existing selection refreshes its automatic selection, selected crop directories, selection index, and gallery. Existing shape-alignment limitations still apply: area selection does not establish material-point correspondence or correct smaller segmentation errors.
