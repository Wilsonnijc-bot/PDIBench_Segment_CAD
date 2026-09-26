"""Freeze the 45 matched workbook rows against the verified source corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

from .contracts import page_index, sha256_file, write_json


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATASETS = {"LVP": "LVP_ROBOWM", "Cosmos2.5": "COSMOS2.5",
            "Cosmos3 seed101": "COSMOS3"}


def workbook_rows(path: Path):
    with ZipFile(path) as archive:
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(".//x:sheetData/x:row", NS)
    headers = {}
    for cell in rows[0].findall("x:c", NS):
        column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
        value = cell.find("x:v", NS)
        inline = cell.find("x:is/x:t", NS)
        headers[column] = (value.text if value is not None else
                           inline.text if inline is not None else None)
    required = {"Source row", "Dataset", "Video ID", "Original Instruction / Task",
                "Robot deformation (0/1)", "Forearm deformation",
                "Gripper deformation", "Upper arm deformation"}
    if not required.issubset(headers.values()):
        raise ValueError(f"matched workbook is missing columns: {sorted(required - set(headers.values()))}")
    for row in rows[1:]:
        cells = {}
        for cell in row.findall("x:c", NS):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            value = cell.find("x:v", NS)
            inline = cell.find("x:is/x:t", NS)
            cells[headers[column]] = (value.text if value is not None else
                                      inline.text if inline is not None else None)
        yield int(row.attrib["r"]), cells


def build_selection(workbook: Path, source_manifest: Path, output_root: Path) -> dict:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("video_count") != 197:
        raise ValueError("expected the verified 197-video source manifest")
    sources = {}
    for item in manifest["videos"]:
        key = (item["dataset"], Path(item["relative_path"]).stem)
        if key in sources:
            raise ValueError(f"duplicate source video: {key}")
        sources[key] = item
    records = []
    for row, cells in workbook_rows(workbook):
        dataset = DATASETS[cells["Dataset"]]
        number = f"{int(re.search(r'[0-9]+$', cells['Video ID']).group()):04d}"
        source = sources.get((dataset, number))
        if source is None:
            raise ValueError(f"matched workbook video is absent from source manifest: {dataset}_{number}")
        def label(name: str):
            value = cells.get(name)
            return None if value in (None, "") else int(value)
        records.append({
            "dataset": dataset, "video_number": number, "video_id": cells["Video ID"],
            "raw_dataset": cells["Dataset"], "task": cells["Original Instruction / Task"],
            "workbook_row": row, "source_workbook_row": int(cells["Source row"]),
            "labels": {"robot_deformation": label("Robot deformation (0/1)"),
                       "upper_arm_deformation": label("Upper arm deformation"),
                       "forearm_deformation": label("Forearm deformation"),
                       "gripper_deformation": label("Gripper deformation"),
                       "deformation_degree": None},
            "relative_path": f"{number}.mp4",
            "staged_relative_path": f"{dataset}/{number}.mp4",
            "source_staged_relative_path": source["staged_relative_path"],
            "source_job_id": source.get("job_id"),
            "job_id": f"{dataset}_{number}-{source['sha256'][:12]}",
            "sha256": source["sha256"], "size_bytes": int(source["size_bytes"]),
            "replay_selected": True,
        })
    keys = [(item["dataset"], item["video_number"]) for item in records]
    counts = Counter(item["dataset"] for item in records)
    if len(records) != 45 or len(set(keys)) != 45 or set(counts.values()) != {15}:
        raise ValueError(f"expected 45 distinct videos, 15 per generator: {counts}")
    payload = {
        "schema_version": 1, "experiment": "selected-45-v1-persistent-mask",
        "workbook": str(workbook.resolve()), "workbook_sha256": sha256_file(workbook),
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_video_count": 197, "video_count": 45, "videos": records,
    }
    target = output_root / "selection.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != payload:
        raise FileExistsError(f"frozen selection differs: {target}")
    write_json(target, payload)
    page_index(output_root, payload)
    return payload
