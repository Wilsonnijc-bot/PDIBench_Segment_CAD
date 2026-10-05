"""Apply each frame's visible task-object shape to the frame-zero RGB object.

This is an offline consumer of saved SAM3 masks and the existing occlusion audit.
It does not change either segmentation, occlusion decisions, or rigidity scores.
Frame numbers are zero based.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from object.preprocessing.occlusion.occlusion import align, atomic, crop, sha256
from object.preprocessing.crop_pairs.paired_crops import METHOD as PAIR_METHOD, normalize_pair
from infrastructure.shared.contracts.mask_quality import MAX_LINK7_AREA_FRACTION, link7_area_validity


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the half-open (x0, y0, x1, y1) extent of a nonempty mask."""
    yy, xx = np.where(mask)
    if not len(xx):
        raise ValueError("Mask has no pixels")
    return int(xx.min()), int(yy.min()), int(xx.max() + 1), int(yy.max() + 1)


def available_pure_object(object_mask: np.ndarray, link7_mask: np.ndarray) -> np.ndarray:
    """The actual visible SAM3 object pixels, excluding every link7 mask pixel."""
    if object_mask.shape != link7_mask.shape:
        raise ValueError("Object and link7 masks must have the same shape")
    return np.asarray(object_mask, bool) & ~np.asarray(link7_mask, bool)


def map_available_to_frame0(
    available: np.ndarray, expected_full: np.ndarray, frame0_object: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Map the exact binary visible shape through full-object bounding-box coordinates.

    The existing occlusion detector supplies expected_full. Nearest-neighbor
    sampling keeps a hard binary boundary. The returned mask is in the cropped
    frame-zero object's pixel coordinates and is clipped to its SAM3 mask.
    """
    if available.shape != expected_full.shape or available.shape != frame0_object.shape:
        raise ValueError("All masks must use the same video pixel grid")
    x0, y0, x1, y1 = mask_bbox(expected_full)
    rx0, ry0, rx1, ry1 = mask_bbox(frame0_object)
    source = available[y0:y1, x0:x1].astype(np.uint8)
    mapped = cv2.resize(source, (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_NEAREST).astype(bool)
    reference = frame0_object[ry0:ry1, rx0:rx1].astype(bool)
    mapped &= reference
    metadata = {
        "expected_bbox_xyxy": [x0, y0, x1, y1],
        "frame0_bbox_xyxy": [rx0, ry0, rx1, ry1],
        "available_area": int(available.sum()),
        "available_inside_expected_area": int(source.sum()),
        "mapped_reference_area": int(mapped.sum()),
    }
    return mapped, metadata


def rgba_crop(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    rgb = frame[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1].astype(np.uint8) * 255
    if rgb.shape[:2] != alpha.shape:
        raise ValueError("RGB frame and mask dimensions differ")
    return np.dstack((rgb, alpha))


def read_frame(cap: cv2.VideoCapture, frame: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = cap.read()
    if not ok:
        raise ValueError(f"Cannot decode video frame {frame}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)):
        raise ValueError(f"Cannot write {path}")


def write_original_frame(path: Path, rgb: np.ndarray) -> None:
    """Save the complete decoded RGB frame without masks or annotations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"Cannot write {path}")


def write_preview(reference: Path, current: Path, mapped: Path, output: Path, frame: int) -> None:
    """Make one inspectable preview while keeping the source crops unannotated."""
    canvas = np.full((290, 780, 3), (29, 35, 37), dtype=np.uint8)
    for index, (path, label) in enumerate(((reference, "Frame 0 object"),
                                           (current, f"Frame {frame} available"),
                                           (mapped, "Frame 0 shape crop"))):
        x0 = index * 260
        cv2.putText(canvas, label, (x0 + 12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    .48, (238, 238, 238), 1, cv2.LINE_AA)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.is_file() else None
        if image is None:
            cv2.putText(canvas, "Unavailable", (x0 + 64, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        .52, (165, 170, 175), 1, cv2.LINE_AA)
            continue
        height, width = image.shape[:2]
        scale = min(220 / height, 236 / width)
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_NEAREST)
        alpha = image[:, :, 3:4].astype(np.float32) / 255
        background = np.full(image.shape[:2] + (3,), (68, 72, 74), dtype=np.uint8)
        rgb = (image[:, :, :3] * alpha + background * (1 - alpha)).astype(np.uint8)
        y0 = 45 + (220 - rgb.shape[0]) // 2
        x = x0 + (260 - rgb.shape[1]) // 2
        canvas[y0:y0 + rgb.shape[0], x:x + rgb.shape[1]] = rgb
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise ValueError(f"Cannot write {output}")


def load_case(case: Path) -> tuple[dict, np.ndarray, np.ndarray, Path]:
    detection = json.loads((case / "occlusion/detection.json").read_text())
    if detection["case"] != case.name or detection["method"] != "gripper-occlusion-v2":
        raise ValueError("Wrong occlusion audit for this case")
    paths = {}
    for name in ("object", "gripper"):
        item = detection["inputs"][name]
        path = Path(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise ValueError(f"Missing or changed {name} SAM3 input: {path}")
        paths[name] = path
    with np.load(paths["object"], allow_pickle=False) as z:
        if str(z["object_names"][0]) != "task_object":
            raise ValueError("Expected task_object in object channel 0")
        objects = z["object_masks"][:, 0].astype(bool)
    with np.load(paths["gripper"], allow_pickle=False) as z:
        names = z["object_names"].tolist()
        if names.count("link7") != 1:
            raise ValueError("Expected exactly one named link7 channel")
        grippers = z["object_masks"][:, names.index("link7")].astype(bool)
    if objects.shape != grippers.shape or len(objects) != detection["frame_count"]:
        raise ValueError("SAM3 masks and occlusion audit have different frame grids")
    video = case / "replay/source.mp4"
    if not video.is_file():
        raise ValueError(f"Missing source video: {video}")
    if sha256(video) != detection["source_video_sha256"]:
        raise ValueError("Source video differs from the occlusion audit input")
    cap = cv2.VideoCapture(str(video))
    video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                   int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    if video_shape != objects.shape:
        raise ValueError(f"Video and mask shapes differ: {video_shape}, {objects.shape}")
    return detection, objects, grippers, video


def render_saved_examples(case: Path, output: Path, frames: list[int], *,
                          subdirectory: str = "examples") -> None:
    """Render arbitrary frames from already exported masks without rerunning alignment."""
    if not frames:
        return
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["case"] != case.name:
        raise ValueError("Saved reference masks belong to another case")
    if sha256(case / "occlusion/detection.json") != manifest["occlusion_detection_sha256"]:
        raise ValueError("Occlusion audit changed; re-export this case before rendering frames")
    with np.load(output / "masks.npz", allow_pickle=False) as z:
        available = z["available_pure_object"]
        mapped = z["mapped_to_frame0"]
        valid = z["mapping_valid"]
        reference_bbox = tuple(int(x) for x in z["frame0_bbox_xyxy"])
    if any(t < 0 or t >= len(available) for t in frames):
        raise ValueError(f"Frame must be within 0..{len(available)-1}")
    cap = cv2.VideoCapture(str(case / "replay/source.mp4"))
    geometry_path = output / "pair_geometry.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.is_file() else {
        "method": PAIR_METHOD, "pairs": {}}
    try:
        frame0 = read_frame(cap, 0)
        for t in sorted(set(frames)):
            current = read_frame(cap, t)
            example = output / subdirectory / f"frame_{t:05d}"
            write_original_frame(example / "original_frame.png", current)
            current_path = example / "current_available.png"
            reference_path = example / "frame0_shape_crop.png"
            current_rgba = reference_rgba = None
            if available[t].any():
                current_bbox = mask_bbox(available[t])
                current_rgba = rgba_crop(current, available[t], current_bbox)
            else:
                current_path.unlink(missing_ok=True)
            if valid[t]:
                x0, y0, x1, y1 = reference_bbox
                reference_rgba = np.dstack((frame0[y0:y1, x0:x1], mapped[t].astype(np.uint8) * 255))
            else:
                reference_path.unlink(missing_ok=True)
            key = f"{subdirectory}/frame_{t:05d}"
            geometry["pairs"].pop(key, None)
            if (current_rgba is not None and reference_rgba is not None
                    and reference_rgba[:, :, 3].any()):
                current_rgba, reference_rgba, record = normalize_pair(current_rgba, reference_rgba)
                geometry["pairs"][key] = record
            if current_rgba is not None:
                write_png(current_path, current_rgba)
            if reference_rgba is not None:
                write_png(reference_path, reference_rgba)
            if key in geometry["pairs"]:
                record = geometry["pairs"][key]
                record["query_sha256"] = sha256(current_path)
                record["reference_sha256"] = sha256(reference_path)
            write_preview(output / "frame0_reference.png", current_path, reference_path,
                          example / "preview.png", t)
    finally:
        cap.release()
    atomic(geometry_path, json.dumps(geometry, indent=2) + "\n")


def export_case(case: Path, output: Path, example_frames: list[int]) -> dict:
    detection, objects, grippers, video = load_case(case)
    if not objects[0].any():
        raise ValueError("Frame 0 has no task-object mask")
    count, height, width = objects.shape
    if any(t < 0 or t >= count for t in example_frames):
        raise ValueError(f"Example frames must be within 0..{count-1}")
    overlap0 = float((objects[0] & grippers[0]).sum() / objects[0].sum())
    reference_quality = "clean_low_contact" if overlap0 <= .05 else "contaminated_by_link7_mask"
    reference_bbox = mask_bbox(objects[0])
    reference_mask = objects[0][reference_bbox[1]:reference_bbox[3],
                                reference_bbox[0]:reference_bbox[2]]
    available = available_pure_object(objects, grippers)
    object_covered_fraction = ((objects & grippers).sum(axis=(1, 2))
                               / np.maximum(1, objects.sum(axis=(1, 2))))
    mask_valid, _, _ = link7_area_validity(
        grippers, detection['config'].get('max_gripper_area_fraction', MAX_LINK7_AREA_FRACTION))
    mask_valid &= np.array([row.get('mask_valid', True) for row in detection['frames']], dtype=bool)
    available[~mask_valid] = False
    if not mask_valid[0]:
        reference_quality = 'failed_link7_mask_area'
    mapped = np.zeros((count, *reference_mask.shape), dtype=bool)
    valid = np.zeros(count, dtype=bool)
    rows = []
    for t, row in enumerate(detection["frames"]):
        info = {"frame": t, "available_area": int(available[t].sum()),
                "link7_object_covered_fraction": float(object_covered_fraction[t]),
                "link7_overmask": bool(object_covered_fraction[t] > .95),
                "mask_valid": bool(mask_valid[t]),
                "mapping_status": "unavailable", "occlusion_reference_frame": row["reference_frame"]}
        if not mask_valid[t]:
            info['mapping_status'] = 'failed_link7_mask_area'
            rows.append(info)
            continue
        anchor = row["reference_frame"]
        if row["expected_area"] is not None:
            if anchor is None or not 0 <= anchor < count:
                raise ValueError(f"Invalid occlusion reference frame at {t}")
            expected = align(crop(objects[anchor]), objects[t], grippers[t])
            if expected is None or int(expected.sum()) != row["expected_area"]:
                raise ValueError(f"Full-object silhouette disagrees with saved occlusion audit at {t}")
            mapped[t], geometry = map_available_to_frame0(available[t], expected, objects[0])
            coverage = geometry["available_inside_expected_area"] / max(1, geometry["available_area"])
            if reference_quality != "clean_low_contact":
                status = "unusable_frame0_reference"
            elif geometry["available_area"] == 0:
                status = "no_available_pixels"
            elif coverage < .80:
                status = "low_full_object_coverage"
            else:
                status = "mapped"
            info.update(geometry, available_inside_expected_fraction=float(coverage),
                        mapping_status=status)
            valid[t] = status == "mapped"
        rows.append(info)
    output.mkdir(parents=True, exist_ok=True)
    temp_masks = output / "masks.npz.tmp"
    with temp_masks.open("wb") as stream:
        np.savez_compressed(stream, available_pure_object=available,
                            mask_valid=mask_valid,
                            mapped_to_frame0=mapped, mapping_valid=valid,
                            frame0_object_mask=reference_mask,
                            frame0_bbox_xyxy=np.asarray(reference_bbox, dtype=np.int32))
    temp_masks.replace(output / "masks.npz")
    cap = cv2.VideoCapture(str(video))
    try:
        frame0 = read_frame(cap, 0)
        reference_path = output / "frame0_reference.png"
        if reference_quality == "clean_low_contact":
            write_png(reference_path, rgba_crop(frame0, objects[0], reference_bbox))
        else:
            reference_path.unlink(missing_ok=True)
    finally:
        cap.release()
    manifest = {
        "case": case.name, "frame_count": count, "image_hw": [height, width],
        "frame0_bbox_xyxy": list(reference_bbox),
        "frame0_object_link7_overlap_fraction": overlap0,
        "frame0_reference_quality": reference_quality,
        "mapped_frames": int(valid.sum()), "unmapped_frames": int((~valid).sum()),
        "failed_mask_frames": np.flatnonzero(~mask_valid).tolist(),
        "source_video": str(video.resolve()),
        "source_video_sha256": detection["source_video_sha256"],
        "occlusion_detection": str((case / "occlusion/detection.json").resolve()),
        "occlusion_detection_sha256": sha256(case / "occlusion/detection.json"),
        "object_segmentation_sha256": detection["inputs"]["object"]["sha256"],
        "link7_segmentation_sha256": detection["inputs"]["gripper"]["sha256"],
        "mapping": "expected-full-object bbox to frame0 SAM3-object bbox; nearest-neighbor binary resize; intersection with frame0 SAM3 mask",
        "mapping_valid_rule": "valid link7 mask in both frames, frame0 object/link7 overlap <= 5%, at least one available pixel, and at least 80% of available pixels inside the detector's expected-full-object bounding box",
        "available_definition": "task_object SAM3 mask AND NOT link7 SAM3 mask in valid-link7 frames; empty in failed-link7 frames",
        "frames": rows,
    }
    atomic(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    # Refresh previews from an earlier export too, so a newly failed mask cannot
    # leave an old current/reference crop on disk when --frame is omitted.
    existing = [int(path.name[6:]) for path in (output / "examples").glob("frame_*")
                if path.is_dir() and path.name[6:].isdigit()]
    render_saved_examples(case, output,
                          [t for t in set(example_frames) | set(existing) if 0 <= t < count])
    # Existing automatic selections must track the freshly exported manifest.
    if (output / "selection.json").is_file():
        from object.preprocessing.crop_pairs.frame_selection import select_case
        selection_config = json.loads((output / "selection.json").read_text())
        select_case(case, output, selection_config["target_count"],
                    strict_bins=selection_config.get("strict_bins", False))
    return {k: manifest[k] for k in ("case", "frame_count", "mapped_frames", "unmapped_frames", "frame0_reference_quality")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--frame", type=int, action="append", default=[],
                        help="Export current and frame-zero RGBA example crops for this frame; repeatable")
    parser.add_argument("--render-only", action="store_true",
                        help="Render --frame examples from masks.npz without recomputing all frames")
    args = parser.parse_args()
    cases = args.cases or sorted(p.name for p in (args.object_root / "cases").iterdir()
                                 if (p / "occlusion/detection.json").is_file())
    if args.render_only:
        if len(cases) != 1 or not args.frame:
            parser.error("--render-only requires exactly one --cases value and at least one --frame")
        render_saved_examples(args.object_root / "cases" / cases[0],
                              args.output_root / cases[0], args.frame)
        return
    summaries = []
    for name in cases:
        result = export_case(args.object_root / "cases" / name,
                             args.output_root / name, args.frame)
        summaries.append(result)
        print(json.dumps(result), flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic(args.output_root / "index.json", json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
