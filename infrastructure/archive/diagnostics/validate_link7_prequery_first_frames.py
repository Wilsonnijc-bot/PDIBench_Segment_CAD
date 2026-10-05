#!/usr/bin/env python3
"""Pack real Link 7 first frames, then review native V2 query sampling."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/validate_link7_prequery_first_frames.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def pack(cases: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for case in sorted(cases.iterdir()):
        if not case.is_dir():
            continue
        geometry_paths = list((case / "geometry-cache").glob("*.npz"))
        if len(geometry_paths) != 1:
            raise ValueError(f"expected one cached geometry archive: {case}")
        path = case / "path2_v2_tapip3d"
        with np.load(geometry_paths[0], allow_pickle=False) as data:
            pointmap = data["pointmaps"][0]
            pose = data["camera_poses"][0]
        with np.load(path / "segmentation.npz", allow_pickle=False) as data:
            index = data["object_names"].tolist().index("link7")
            mask = data["object_masks"][0, index]
        with np.load(path / "link7_initial_queries.npz", allow_pickle=False) as data:
            old_queries = data["queries"]
        capture = cv2.VideoCapture(str(path / "replay/interactive_exact-group/source.mp4"))
        ok, frame = capture.read()
        capture.release()
        if not ok or frame.shape[:2] != mask.shape:
            raise ValueError(f"source first frame and mask disagree: {case.name}")
        np.savez_compressed(output / f"{case.name}.npz", frame=frame, mask=mask,
                            pointmap=pointmap, pose=pose, old_queries=old_queries)
        print(f"packed {case.name}", flush=True)


def review(inputs: Path, output: Path) -> None:
    from robot.experiments.link7_depth_filter_v2.link7_depth_gate import link7_query_mask
    from infrastructure.shared.inference.tracking import TrackWrapper

    output.mkdir(parents=True, exist_ok=True)
    sampler = TrackWrapper.__new__(TrackWrapper)
    summary = {}
    for path in sorted(inputs.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            frame, mask = data["frame"], data["mask"]
            pointmap, pose = data["pointmap"], data["pose"]
            old_queries = data["old_queries"]
        source_h, source_w = mask.shape
        scale = min(1.0, 880 / max(source_h, source_w))
        tracker_h, tracker_w = int(source_h * scale), int(source_w * scale)
        image = cv2.resize(frame, (tracker_w, tracker_h))
        valid_tracker, stats = link7_query_mask(
            pointmap, pose, mask, (tracker_h, tracker_w),
        )
        mask_tracker = cv2.resize(mask.astype(np.uint8), (tracker_w, tracker_h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        new_queries = sampler._sample_region_queries(
            gray, valid_tracker, 100, "link7", snap_to_mask_pixels=True,
        )
        x = np.rint(new_queries[:, 1]).astype(int)
        y = np.rint(new_queries[:, 2]).astype(int)
        supported_queries = int(valid_tracker[y, x].sum())

        before = image.copy()
        after = image.copy()
        removed = mask_tracker & ~valid_tracker
        tint = after.copy()
        tint[removed] = (45, 45, 190)
        tint[valid_tracker] = (75, 150, 60)
        after = cv2.addWeighted(after, 0.68, tint, 0.32, 0)
        for _, u, v in old_queries:
            cv2.circle(before, (int(round(u * tracker_w / source_w)),
                                int(round(v * tracker_h / source_h))), 3,
                       (50, 80, 255), -1, cv2.LINE_AA)
        for _, u, v in new_queries:
            cv2.circle(after, (int(round(u)), int(round(v))), 3,
                       (30, 240, 255), -1, cv2.LINE_AA)
        cv2.putText(before, f"Old V2: {len(old_queries)} queries", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(after, f"Pre-query V2: {len(new_queries)} queries", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        preview = np.concatenate((before, after), axis=1)
        cv2.imwrite(str(output / f"{path.stem}.png"), preview)
        summary[path.stem] = {
            "masked_pointmap_pixels": stats["candidate_count"],
            "depth_supported_pointmap_pixels": stats["retained_count"],
            "eligible_tracker_pixels": stats["eligible_tracker_pixel_count"],
            "old_v2_query_count": len(old_queries),
            "new_v2_query_count": len(new_queries),
            "new_queries_on_supported_tracker_pixels": supported_queries,
            "preview": f"{path.stem}.png",
        }
        print(f"{path.stem}: {json.dumps(summary[path.stem])}", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pack", "review"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (pack if args.mode == "pack" else review)(args.input, args.output)


if __name__ == "__main__":
    main()
