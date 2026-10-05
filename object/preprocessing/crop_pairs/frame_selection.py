"""Select time-quintile reference crops from saved masks and occlusion diagnoses.

Reserve immediate post-occlusion frames, then fill by visible pixel area within time quintiles.
No segmentation, occlusion detection, alignment, or rigidity scores are changed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

from object.preprocessing.occlusion.occlusion import atomic, sha256
from object.preprocessing.crop_pairs.reference_visible_pixels import render_saved_examples


def select_frames(manifest: dict, detection: dict, count: int = 10, *,
                  strict_bins: bool = False, last_interval_min_area_ratio: float = .5) -> dict:
    """Return a deterministic selection and explicit unsatisfied requirements.

    Only assessed, unflagged successors establish a diagnosed episode's end.
    Invalid immediate successors are reported, never moved to a later frame.
    Remaining slots rank area within quintiles; ten slots normally use [0, 2, 2, 2, 4].
    A collapsed final mean activates an area check per frame, preserving substantial recoveries.
    """
    if count not in (5, 10, 20):
        raise ValueError("Time-quintile selection supports 5, 10, or 20 slots")
    if not math.isfinite(last_interval_min_area_ratio) or not 0 <= last_interval_min_area_ratio <= 1:
        raise ValueError("Last-interval area ratio must be finite and between zero and one")
    rows, diagnoses = manifest["frames"], detection["frames"]
    if len(rows) != len(diagnoses) or len(rows) != manifest["frame_count"]:
        raise ValueError("Crop manifest and occlusion audit frame counts differ")
    if any(row["frame"] != t or diagnoses[t]["frame"] != t
           for t, row in enumerate(rows)):
        raise ValueError("Frame rows must be contiguous and zero based")
    eligible = {t for t, row in enumerate(rows)
                if row.get("mask_valid", True) and diagnoses[t].get("mask_valid", True)
                and not row.get("link7_overmask", False)
                and diagnoses[t].get("mask_overlap_fraction", row.get("link7_object_covered_fraction", 0)) <= .95
                and row["mapping_status"] == "mapped" and row["available_area"] > 0
                and row.get("mapped_reference_area", 0) > 0}
    length = len(rows)
    bucket = lambda t: min(4, 5 * t // length)
    scale = count // 5
    quotas = [0, scale, scale, scale, 2 * scale]
    default_quotas = list(quotas)
    interval_areas = [[row["available_area"] for t, row in enumerate(rows) if bucket(t) == b]
                      for b in range(5)]
    means = [sum(areas) / len(areas) if areas else 0.0 for areas in interval_areas]
    preceding_areas = [area for areas in interval_areas[:4] for area in areas]
    baseline = sum(preceding_areas) / len(preceding_areas) if preceding_areas else 0.0
    collapse_detected = bool(last_interval_min_area_ratio > 0 and interval_areas[4]
                            and baseline > 0 and means[3] > 0
                            and means[4] < last_interval_min_area_ratio * baseline
                            and means[4] < last_interval_min_area_ratio * means[3])
    frame_area_threshold = last_interval_min_area_ratio * baseline if collapse_detected else None
    excluded = {t for t, row in enumerate(rows)
                if collapse_detected and bucket(t) == 4 and row["available_area"] < frame_area_threshold}
    eligible.difference_update(excluded)
    preserved_final_frames = sorted(t for t in eligible if bucket(t) == 4)
    suppressed = collapse_detected and not preserved_final_frames
    # Keep the original quotas. Missing final slots use the existing
    # latest-interval fallback, while substantial late crops remain available.
    final_interval_policy = {
        "method": "mean-available-area-collapse-with-frame-retention",
        "min_area_ratio": last_interval_min_area_ratio,
        "interval_mean_available_area": means,
        "first_80_percent_mean_available_area": baseline,
        "final_to_first_80_ratio": means[4] / baseline if baseline > 0 else None,
        "final_to_preceding_interval_ratio": means[4] / means[3] if means[3] > 0 else None,
        "collapse_detected": collapse_detected,
        "frame_area_threshold": frame_area_threshold,
        "suppressed": suppressed,
        "excluded_frames": sorted(excluded),
        "preserved_eligible_frames": preserved_final_frames,
        "slot_policy": "unchanged quintile quotas; existing latest-interval fallback fills missing slots",
        "rule": "if final mean is below ratio times BOTH earlier means, exclude only final frames below ratio times first-80% mean; equality is retained",
    }
    ranked_bins = [sorted((t for t in eligible if bucket(t) == b),
                          key=lambda t: (-rows[t]["available_area"], t)) for b in range(5)]
    ranks = {t: rank for frames in ranked_bins for rank, t in enumerate(frames, 1)}
    required, episodes = [], []
    start = None
    for t, diagnosis in enumerate(diagnoses):
        if diagnosis["flagged"]:
            if start is None:
                start = t
            continue
        if start is None:
            continue
        event = {"start_frame": start, "last_occluded_frame": t - 1,
                 "immediate_successor_frame": t}
        if t in excluded:
            event["status"] = "successor_below_final_interval_area_gate"
        elif not diagnosis.get("mask_valid", True) or diagnosis["status"] != "assessed":
            event["status"] = "successor_not_assessable"
        elif t not in eligible:
            event["status"] = "successor_crop_invalid"
        else:
            event["status"] = "required"
            required.append(t)
        event["successor_detection_status"] = diagnosis["status"]
        event["successor_mapping_status"] = rows[t]["mapping_status"]
        episodes.append(event)
        start = None
    if start is not None:
        episodes.append({"start_frame": start, "last_occluded_frame": len(rows) - 1,
                         "immediate_successor_frame": None, "status": "occlusion_reaches_video_end"})
    if len(required) > count:
        raise ValueError(f"{len(required)} mandatory recovery frames exceed {count} slots; "
                         "cannot satisfy both rules")
    chosen = set(required)
    reasons = {t: "immediate_post_occlusion" for t in required}
    bins = []
    for b in range(5):
        in_bin = sum(bucket(t) == b for t in required)
        needed = max(0, quotas[b] - in_bin)
        added = []
        if b:
            for t in ranked_bins[b]:
                if len(added) >= needed or len(chosen) >= count:
                    break
                if t not in chosen:
                    chosen.add(t)
                    added.append(t)
                    reasons[t] = "largest_available_area_in_interval"
        bins.append({"interval": b + 1, "percent_range": [20 * b, 20 * (b + 1)],
                     "start_frame": (b * length + 4) // 5,
                     "end_frame_exclusive": ((b + 1) * length + 4) // 5,
                     "target_count": quotas[b], "eligible_count": len(ranked_bins[b]),
                     "mean_available_area": means[b], "suppressed": suppressed and b == 4,
                     "required_frames": [t for t in required if bucket(t) == b],
                     "area_ranked_frames": ranked_bins[b],
                     "unfilled_quota": max(0, needed - len(added))})
    # Empty intervals cannot yield valid crops. Optional fallback stays in the
    # allowed 20–100% range, trying the last interval first, then earlier ones.
    if not strict_bins:
        for b in (4, 3, 2, 1):
            for t in ranked_bins[b]:
                if len(chosen) >= count:
                    break
                if t not in chosen:
                    chosen.add(t)
                    reasons[t] = "empty_interval_fallback"
    for b, info in enumerate(bins):
        info["selected_count"] = sum(bucket(t) == b for t in chosen)
    selected = [{"frame": t, "available_area": rows[t]["available_area"],
                 "interval": bucket(t) + 1, "percent_range": [20 * bucket(t), 20 * (bucket(t) + 1)],
                 "video_position_percent": 100 * t / length,
                 "interval_area_rank": ranks[t], "occlusion_flagged": bool(diagnoses[t]["flagged"]),
                 "reason": reasons[t], "crop_directory": f"selected/frame_{t:05d}"}
                for t in sorted(chosen)]
    failures = [event for event in episodes if event["status"] in
                ("successor_not_assessable", "successor_crop_invalid")]
    return {"case": manifest["case"], "method": "time-quintiles-with-post-occlusion-v5",
            "target_count": count, "selected_count": len(selected),
            "eligible_count": sum(len(ranked_bins[b]) for b in range(1, 5)) + sum(bucket(t) == 0 for t in required), "shortfall": max(0, count - len(selected)),
            "status": "complete" if len(selected) == count and not failures else "incomplete",
            "ranking": "descending available_area within each time interval; earlier frame wins ties",
            "interval_quotas": quotas, "default_interval_quotas": default_quotas,
            "final_interval_policy": final_interval_policy,
            "intervals": bins, "strict_bins": strict_bins,
            "fallback_rule": "none" if strict_bins else "fill empty interval slots from latest available interval, excluding first 20%",
            "eligibility": "valid mask, object coverage by link7 <=95%, valid frame0 mapping, nonempty available and mapped masks, passes activated final-interval area gate",
            "overmask_excluded_frames": [t for t, row in enumerate(rows)
                if row.get("link7_overmask", False) or diagnoses[t].get("mask_overlap_fraction", row.get("link7_object_covered_fraction", 0)) > .95],
            "recovery_rule": "immediate unflagged, assessed, crop-valid successor of each flagged run passing activated final-interval area gate",
            "required_frames": required, "selected_frames": selected,
            "occlusion_episodes": episodes,
            "unsatisfied_recovery_count": len(failures)}


def select_case(case: Path, output: Path, count: int = 10, *, strict_bins: bool = False,
                last_interval_min_area_ratio: float = .5) -> dict:
    manifest_path = output / "manifest.json"
    detection_path = case / "occlusion/detection.json"
    manifest = json.loads(manifest_path.read_text())
    if sha256(detection_path) != manifest["occlusion_detection_sha256"]:
        raise ValueError("Occlusion audit changed; re-export this case before selecting frames")
    detection = json.loads(detection_path.read_text())
    if manifest["case"] != case.name or detection["case"] != case.name:
        raise ValueError("Crop manifest or occlusion audit belongs to another case")
    selection = select_frames(manifest, detection, count, strict_bins=strict_bins,
                              last_interval_min_area_ratio=last_interval_min_area_ratio)
    selection["crop_manifest_sha256"] = sha256(manifest_path)
    selection["occlusion_detection_sha256"] = manifest["occlusion_detection_sha256"]
    frames = [row["frame"] for row in selection["selected_frames"]]
    render_saved_examples(case, output, frames, subdirectory="selected")
    # Only this selector owns selected/. Keep manually requested examples intact.
    wanted = {f"frame_{t:05d}" for t in frames}
    for path in (output / "selected").glob("frame_*"):
        if path.is_dir() and path.name[6:].isdigit() and path.name not in wanted:
            shutil.rmtree(path)
    atomic(output / "selection.json", json.dumps(selection, indent=2) + "\n")
    summary = {key: selection[key] for key in
            ("case", "selected_count", "shortfall", "status", "required_frames",
             "selected_frames", "unsatisfied_recovery_count", "occlusion_episodes", "method",
             "strict_bins", "intervals", "interval_quotas", "default_interval_quotas",
             "final_interval_policy")}
    index_path = output.parent / "selection_index.json"
    previous = json.loads(index_path.read_text()) if index_path.is_file() else []
    merged = {row["case"]: row for row in previous}
    merged[case.name] = summary
    atomic(index_path, json.dumps([merged[name] for name in sorted(merged)], indent=2) + "\n")
    from object.replay.frame_selection_gallery import write_gallery
    write_gallery(output.parent)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--strict-bins", action="store_true",
                        help="Leave slots empty when an interval has too few usable crops")
    parser.add_argument("--last-interval-min-area-ratio", type=float, default=.5,
                        help="If final-20%% mean falls below this fraction of both earlier means, "
                             "exclude only final frames below this fraction of first-80%% mean; 0 disables")
    args = parser.parse_args()
    cases = args.cases or sorted(path.parent.name for path in args.crop_root.glob("*/manifest.json"))
    for name in cases:
        result = select_case(args.object_root / "cases" / name, args.crop_root / name, args.count,
                             strict_bins=args.strict_bins,
                             last_interval_min_area_ratio=args.last_interval_min_area_ratio)
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ("selected_frames", "occlusion_episodes", "intervals")}), flush=True)


if __name__ == "__main__":
    main()
