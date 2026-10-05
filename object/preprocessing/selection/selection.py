"""Match the selected videos to their generation prompts without Excel dependencies."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATASETS = {"LVP": "LVP_ROBOWM", "Cosmos2.5": "COSMOS2.5", "Cosmos3 seed101": "COSMOS3"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def workbook_prompts(path: Path) -> dict[tuple[str, str], dict]:
    with ZipFile(path) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in item.findall(".//x:t", NS))
                      for item in strings.findall("x:si", NS)]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = sheet.findall(".//x:sheetData/x:row", NS)
    if not rows:
        raise ValueError("workbook has no rows")

    def values(row):
        cells = {}
        for cell in row.findall("x:c", NS):
            column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
            value = cell.find("x:v", NS)
            inline = cell.find("x:is", NS)
            text = (value.text if value is not None else
                    "".join(t.text or "" for t in inline.findall(".//x:t", NS))
                    if inline is not None else "")
            if cell.attrib.get("t") == "s" and text:
                text = shared[int(text)]
            cells[column] = text
        return cells

    headers = {column: name for column, name in values(rows[0]).items()}
    required = {"Dataset", "Video ID", "Generation Prompt"}
    if not required.issubset(headers.values()):
        raise ValueError(f"missing workbook columns: {sorted(required - set(headers.values()))}")
    result = {}
    for row in rows[1:]:
        cells = {headers[column]: value for column, value in values(row).items()
                 if column in headers}
        dataset = DATASETS.get(cells.get("Dataset"))
        video_id = cells.get("Video ID", "")
        match = re.search(r"(\d+)$", video_id)
        prompt = (cells.get("Generation Prompt") or "").strip()
        if dataset is None or match is None or not prompt:
            raise ValueError(f"invalid dataset, video ID, or prompt at workbook row {row.attrib['r']}")
        key = (dataset, f"{int(match.group(1)):04d}")
        if key in result:
            raise ValueError(f"duplicate workbook video: {key}")
        result[key] = {"generation_prompt": prompt, "workbook_row": int(row.attrib["r"]),
                       "workbook_video_id": video_id}
    return result


def build_manifest(workbook: Path, selection: Path) -> dict:
    prompt_rows = workbook_prompts(workbook)
    selected = json.loads(selection.read_text(encoding="utf-8"))
    videos = []
    for entry in selected["videos"]:
        key = (entry["dataset"], entry["video_number"])
        row = prompt_rows.pop(key, None)
        if row is None or row["workbook_row"] != entry["workbook_row"]:
            raise ValueError(f"selection/workbook mismatch for {key}")
        videos.append({**entry, **row, "case": f"{key[0]}_{key[1]}"})
    if prompt_rows or len(videos) != 45:
        raise ValueError(f"expected exactly 45 matched videos; extra workbook keys: {sorted(prompt_rows)}")
    return {"schema_version": 1, "workbook_sha256": sha256(workbook),
            "selection_sha256": sha256(selection), "video_count": 45, "videos": videos}
