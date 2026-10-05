#!/usr/bin/env python3
"""Compare v2 occlusion-filtered task-object rigidity with v2 human labels."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATASETS = {"LVP": "LVP_ROBOWM", "Cosmos2.5": "COSMOS2.5", "Cosmos3 seed101": "COSMOS3"}
EXCLUDED_CASE = "COSMOS3_0001"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workbook_labels(path: Path) -> dict[str, dict]:
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//x:t", NS))
                      for item in root.findall("x:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    def cells(row):
        result = {}
        for cell in row.findall("x:c", NS):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            value = cell.find("x:v", NS)
            inline = cell.find("x:is", NS)
            text = (value.text if value is not None else
                    "".join(t.text or "" for t in inline.findall(".//x:t", NS))
                    if inline is not None else "")
            result[column] = shared[int(text)] if cell.attrib.get("t") == "s" and text else text
        return result

    rows = sheet.findall(".//x:sheetData/x:row", NS)
    headers = {name: column for column, name in cells(rows[0]).items()}
    required = {"Dataset", "Video ID", "Object deformation (0/1)2"}
    if not required <= headers.keys():
        raise ValueError(f"Missing workbook headers: {sorted(required - headers.keys())}")
    labels = {}
    for row in rows[1:]:
        values = cells(row)
        dataset = DATASETS[values[headers["Dataset"]]]
        number = re.search(r"(\d+)$", values[headers["Video ID"]])
        if not number:
            raise ValueError(f"Bad video ID at row {row.attrib['r']}")
        case = f"{dataset}_{int(number.group(1)):04d}"
        label = float(values[headers["Object deformation (0/1)2"]])
        if label not in (0.0, 0.5, 1.0) or case in labels:
            raise ValueError(f"Bad or duplicate label for {case}")
        labels[case] = {"label": label, "workbook_row": int(row.attrib["r"])}
    return labels


def auroc(rows: list[dict], score_key: str) -> float:
    positives = [r[score_key] for r in rows if r["human_label_v2"] == 1.0]
    negatives = [r[score_key] for r in rows if r["human_label_v2"] == 0.0]
    if not positives or not negatives:
        raise ValueError("AUROC needs both positive and negative labels")
    return sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))


def analyze(root: Path, workbook: Path) -> tuple[list[dict], dict]:
    labels = workbook_labels(workbook)
    cases = sorted(p for p in (root / "cases").iterdir() if p.is_dir())
    if len(cases) != 45 or set(p.name for p in cases) != set(labels):
        raise ValueError("The result cases and workbook labels are not the same 45 videos")
    detector_path = (_workspace_root() / 'object/preprocessing/occlusion/occlusion.py')
    detector_hash = sha256(detector_path)
    rows = []
    for case in cases:
        original_path = case / "score/rigidity.json"
        detection_path = case / "occlusion/detection.json"
        filtered_path = case / "score/rigidity_occlusion_filtered.json"
        original = json.loads(original_path.read_text())
        detection = json.loads(detection_path.read_text())
        filtered = json.loads(filtered_path.read_text())
        if detection["method"] != "gripper-occlusion-v2" or detection["detector_source_sha256"] != detector_hash:
            raise ValueError(f"Stale or wrong occlusion detector for {case.name}")
        if filtered["detection_method"] != "gripper-occlusion-v2":
            raise ValueError(f"Wrong filtered-score method for {case.name}")
        if filtered["inputs"] != {"naive_score_sha256": sha256(original_path),
                                  "detection_sha256": sha256(detection_path)}:
            raise ValueError(f"Stale filtered score for {case.name}")
        history = original["rigidity_history"]
        flags = detection["frames"]
        if len(history) != detection["frame_count"] or len(flags) != len(history):
            raise ValueError(f"Frame count mismatch for {case.name}")
        excluded = [i for i in range(1, len(history)) if flags[i]["flagged"] or not flags[i].get("mask_valid", True)]
        retained = [i for i in range(1, len(history)) if not flags[i]["flagged"] and flags[i].get("mask_valid", True)]
        expected = sum(history[i] for i in retained) / len(retained) if retained else None
        if (excluded != filtered["excluded_frames"] or retained != filtered["retained_frames"]
                or detection["flagged_frames"] != sum(bool(f["flagged"]) for f in flags)
                or not math.isclose(original["rigidity_score"], filtered["naive_rigidity_score"], rel_tol=1e-10)
                or (expected is None) != (filtered["filtered_rigidity_score"] is None)
                or (expected is not None and not math.isclose(expected, filtered["filtered_rigidity_score"], rel_tol=1e-10))):
            raise ValueError(f"Filtered score does not match frame flags for {case.name}")
        label = labels[case.name]
        reason = ("excluded_video" if case.name == EXCLUDED_CASE else
                  "ambiguous_human_label" if label["label"] == 0.5 else
                  "no_retained_frames" if expected is None else "included")
        rows.append({"case": case.name, "workbook_row": label["workbook_row"],
                     "human_label_v2": label["label"], "naive_rigidity_score": original["rigidity_score"],
                     "occlusion_filtered_rigidity_score": expected, "flagged_frames": detection["flagged_frames"],
                     "excluded_scored_frames": len(excluded), "retained_scored_frames": len(retained),
                     "total_frames": len(history), "auroc_status": reason})
    included = [r for r in rows if r["auroc_status"] == "included"]
    positives = sum(r["human_label_v2"] == 1.0 for r in included)
    negatives = sum(r["human_label_v2"] == 0.0 for r in included)
    summary = {"method": "gripper-occlusion-v2 with link7-frame-area-v1 and task-object-v1-occlusion-filtered-mean-v2",
               "human_label_column": "Object deformation (0/1)2", "positive_label": 1,
               "excluded_video": EXCLUDED_CASE, "ambiguous_label_policy": "Exclude 0.5 from binary AUROC",
               "score_direction": "Higher rigidity error predicts object deformation",
               "case_count": len(rows), "auroc_case_count": len(included),
               "positive_count": positives, "negative_count": negatives,
               "cases_with_flagged_frames": sum(r["flagged_frames"] > 0 for r in rows),
               "flagged_frames_all_cases": sum(r["flagged_frames"] for r in rows),
               "excluded_scored_frames_in_auroc_cases": sum(r["excluded_scored_frames"] for r in included),
               "naive_auroc": auroc(included, "naive_rigidity_score"),
               "occlusion_filtered_auroc": auroc(included, "occlusion_filtered_rigidity_score"),
               "workbook_sha256": sha256(workbook), "detector_source_sha256": detector_hash}
    summary["auroc_change"] = summary["occlusion_filtered_auroc"] - summary["naive_auroc"]
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/object-deformation-selected45-20260929"))
    parser.add_argument("--workbook", type=Path, default=Path("selected_45_matched_videos_styledv2.xlsx"))
    args = parser.parse_args()
    rows, summary = analyze(args.root, args.workbook)
    output = args.root / "occlusion"
    csv_path = output / "object_deformation_v2_auroc.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "object_deformation_v2_auroc.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
