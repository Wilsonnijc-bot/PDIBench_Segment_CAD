#!/usr/bin/env python3
"""Compare selected-45 V1 scores with the workbook's forearm/upper-arm labels."""

from __future__ import annotations

import json
import re
import statistics
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "results" / "selected-45-v1"
WORKBOOK = PROJECT / "selected_45_matched_videos_styled.xlsx"
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
DATASET_NAMES = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "Cosmos2.5", "COSMOS3": "Cosmos3 seed101"}
PARTS = (("Forearm", "AB", "AH", "link5"),
         ("Upper arm", "AD", "AJ", "link2"),
         ("Gripper", "AC", "AI", "link7"))
METRICS = (("Rigidity score (epsilon_rigidity)", "rigidity"),)
LABELS = (0.0, 0.5, 1.0)


def read_workbook_rows() -> dict[int, dict[str, str]]:
    with zipfile.ZipFile(WORKBOOK) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in item.findall(".//x:t", NS))
                   for item in shared_root.findall("x:si", NS)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    result = {}
    for row in sheet.findall(".//x:sheetData/x:row", NS):
        number = int(row.attrib["r"])
        if number == 1:
            continue
        cells = {}
        for cell in row.findall("x:c", NS):
            raw = cell.find("x:v", NS)
            if raw is None:
                continue
            value = raw.text or ""
            if cell.attrib.get("t") == "s":
                value = strings[int(value)]
            column = re.match(r"[A-Z]+", cell.attrib["r"])
            assert column is not None
            cells[column.group()] = value.strip()
        result[number] = cells
    return result


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def display_label(value: float | None) -> str:
    return "missing" if value is None else f"{value:g}"


def label_value(raw: str | None) -> float | None:
    try:
        value = float(raw) if raw is not None else None
    except ValueError:
        return None
    return value if value in LABELS else None


def values_for(records: list[dict], part: str, metric: str, label: float) -> list[float]:
    return [record["scores"][part][metric] for record in records
            if record["labels"][part] == label and part in record["scores"]]


def ordered_concordance(data: list[tuple[str, float, float]], *,
                        severe_vs_none: bool = False) -> tuple[float, int]:
    """Fraction of different-label pairs whose higher label has higher rigidity."""
    points = 0.0
    count = 0
    for i, (_, label, score) in enumerate(data):
        for _, other_label, other_score in data[i + 1:]:
            if label == other_label:
                continue
            if severe_vs_none and {label, other_label} != {0.0, 1.0}:
                continue
            direction = (label - other_label) * (score - other_score)
            points += 1.0 if direction > 0 else 0.5 if direction == 0 else 0.0
            count += 1
    return (points / count if count else float("nan"), count)


def main() -> None:
    workbook_rows = read_workbook_rows()
    selection = json.loads((ROOT / "selection.json").read_text())
    summary = json.loads((ROOT / "summary.json").read_text())
    if len(selection["videos"]) != 45 or len(workbook_rows) != 45:
        raise ValueError("Expected exactly 45 selected workbook rows")
    records = []
    for item in selection["videos"]:
        row_number = int(item["workbook_row"])
        row = workbook_rows[row_number]
        generator = item["dataset"]
        if row.get("C") != DATASET_NAMES[generator] or int(row["A"]) != int(item["video_number"]):
            raise ValueError(f"Selection/workbook identity mismatch at row {row_number}")
        case = f"{generator}_{int(item['video_number']):04d}"
        state = summary["cases"][case]["state"]
        scores = {}
        if state == "complete":
            metrics = json.loads((ROOT / "cases" / case / "v1" / "metrics.json").read_text())
            objects = metrics["modes"]["exact-group"]["objects"]
            for part, _, _, link in PARTS:
                obj = objects[link]
                if obj["status"] == "complete":
                    scores[part] = {"rigidity": float(obj["breakdown"]["epsilon_rigidity"])}
        flags = [part for part, _, flag_col, _ in PARTS if row.get(flag_col) == "3"]
        records.append({"case": case, "generator": generator, "row": row_number,
                        "labels": {part: label_value(row.get(col)) for part, col, _, _ in PARTS},
                        "flags": flags, "scores": scores, "state": state})

    lines = ["# Robot-part deformation labels versus V1 rigidity scores", "",
             "Workbook: [`selected_45_matched_videos_styled.xlsx`](../../selected_45_matched_videos_styled.xlsx), "
             "sheet `Selected 45`. Cases are joined using the workbook row stored in `selection.json`, "
             "then checked against generator and matched video number. Forearm uses column **AB** "
             "and **link5**; upper arm uses column **AD** and **link2**; gripper uses "
             "column **AC** and **link7**. The 0, 0.5, and 1 "
             "values are treated as the workbook's ordered deformation labels.", "",
             "The last three workbook columns (AH:AJ) are used **only** to flag rows containing `3`. "
             "Their other values do not enter any score calculation. A `3` marks a case for "
             "extra attention because it has relatively large deformation. The labels and "
             "the `3` flags remain separate; no label is overwritten.", "",
             "Scores come from `v1/metrics.json`, exact-group mode, field "
             "`breakdown.epsilon_rigidity`. This rigidity score measures the spread "
             "of tracked 3D point-pair distance ratios. Higher values indicate less "
             "consistent relative distances. Each table cell shows "
             "**median (number of scored videos)**. Missing labels and unscored links "
             "are excluded rather than set to zero.", ""]
    for part, column, _, link in PARTS:
        lines.extend([f"## {part}: {column} label versus {link} score", ""])
        for metric_title, metric in METRICS:
            lines.extend([f"### {metric_title}", "",
                          "| Label 0 | Label 0.5 | Label 1 |",
                          "| ---: | ---: | ---: |"])
            cells = []
            for label in LABELS:
                values = values_for(records, part, metric, label)
                cells.append(f"{fmt(statistics.median(values))} ({len(values)})" if values else "—")
            lines.append("| " + " | ".join(cells) + " |")
            lines.append("")
        missing = [record for record in records if record["labels"][part] is None]
        missing_text = ", ".join(f"{r['case']} (row {r['row']})" for r in missing) if missing else "none"
        lines.append(f"Workbook {part.lower()} labels absent or nonnumeric: {missing_text}.")
        lines.append("")

    lines.extend(["## Score association with deformation severity", "",
                  "The workbook defines 0 as almost no deformation, 0.5 as moderate, "
                  "and 1 as severe. **AUROC** compares labels 1 and 0 only: it is the "
                  "fraction of severe/none video pairs in which the severe video has "
                  "the higher rigidity score, counting ties as half. **Ordered-label "
                  "concordance** also includes moderate cases and compares every pair "
                  "with different labels. A value of 0.5 means no ordering.", "",
                  "| Mapping | Videos by label: 0 / 0.5 / 1 | AUROC (severe vs none) | Ordered concordance |",
                  "| --- | ---: | ---: | ---: |"])
    association = {}
    for part, link, cap in (("Upper arm", "link2", False),
                            ("Forearm", "link5", False),
                            ("Gripper", "link7", False),
                            ("Gripper", "link7", True)):
        data = [(record["generator"], record["labels"][part],
                 min(record["scores"][part]["rigidity"], 0.20) if cap
                 else record["scores"][part]["rigidity"])
                for record in records if record["labels"][part] is not None
                and part in record["scores"]]
        counts = [sum(label == value for _, label, _ in data) for value in LABELS]
        auc, auc_pairs = ordered_concordance(data, severe_vs_none=True)
        ordinal, ordinal_pairs = ordered_concordance(data)
        association[(part, cap)] = (auc, ordinal)
        name = f"{part} → {link}" + (" (cap 0.20)" if cap else "")
        lines.append(f"| {name} | {' / '.join(map(str, counts))} "
                     f"| {auc:.3f} ({auc_pairs} pairs) | {ordinal:.3f} ({ordinal_pairs} pairs) |")
    imputed_results = {}
    for assumed_score in (0.15, 0.20):
        data = [(record["generator"], record["labels"]["Gripper"],
                 min(record["scores"]["Gripper"]["rigidity"], 0.20)
                 if "Gripper" in record["scores"] else assumed_score)
                for record in records if record["labels"]["Gripper"] is not None]
        counts = [sum(label == value for _, label, _ in data) for value in LABELS]
        auc, auc_pairs = ordered_concordance(data, severe_vs_none=True)
        ordinal, ordinal_pairs = ordered_concordance(data)
        imputed_results[assumed_score] = (auc, ordinal, data)
        lines.append(f"| Gripper → link7 (cap 0.20; missing = {assumed_score:.2f}) "
                     f"| {' / '.join(map(str, counts))} | {auc:.3f} ({auc_pairs} pairs) "
                     f"| {ordinal:.3f} ({ordinal_pairs} pairs) |")
    lines.extend(["", "The calculation uses 42 videos for upper arm/link2 (two nonnumeric "
                  "labels and one failed V1 case) and 44 for forearm/link5 (one failed "
                  "case). Gripper/link7 has 37 scored videos; unscored link7 results "
                  "and the failed case are excluded. These values describe score "
                  "ordering in the selected set.", ""])

    gripper = [record for record in records if record["labels"]["Gripper"] is not None
               and "Gripper" in record["scores"]]
    missing_gripper = [record for record in records if "Gripper" not in record["scores"]]
    severe_missing = [record for record in missing_gripper
                      if record["labels"]["Gripper"] == 1.0]
    extreme_gripper = [record for record in records if "Gripper" in record["flags"]]
    extreme_scored = [record for record in extreme_gripper
                      if "Gripper" in record["scores"]]
    capped = [record for record in gripper
              if record["scores"]["Gripper"]["rigidity"] > 0.20]
    severe_raw = [record["scores"]["Gripper"]["rigidity"] for record in gripper
                  if record["labels"]["Gripper"] == 1.0]
    lines.extend(["## Link7 cap at 20%", "",
                  "The capped analysis uses `min(link7 rigidity, 0.20)`; it does not "
                  "change any stored score. The cap changes "
                  f"{len(capped)} of {len(gripper)} scored videos, all labeled severe. "
                  "Their original scores are "
                  + ", ".join(f"{record['case']} {record['scores']['Gripper']['rigidity']:.4f}"
                              for record in capped) + ".", "",
                  "Capping lowers the severe-label mean from "
                  f"{statistics.mean(severe_raw):.4f} to "
                  f"{statistics.mean(min(score, 0.20) for score in severe_raw):.4f}. "
                  "The severe-label median stays at 0.0723. AUROC and ordered-label "
                  "concordance stay the same because the capped scores remain above "
                  "the lower-label scores.", "",
                  f"**Coverage limit:** {len(severe_missing)} severe-label gripper videos "
                  f"lack a link7 score, including all {len(extreme_gripper)} cases marked "
                  f"`3` in the gripper column ({len(extreme_scored)} scored). "
                  "They do not enter either observed-only calculation; the cap cannot recover "
                  "their missing measurements.", "",
                  "## Hypothetical scores for unavailable link7 cases", "",
                  "As a sensitivity check, assign every unavailable link7 case a score "
                  "of 0.15 or 0.20. Apply the 0.20 cap to measured scores in both "
                  "scenarios. This includes all 45 videos and does not alter the "
                  "stored experiment outputs. The two assumed values give the same "
                  f"AUROC ({imputed_results[0.15][0]:.3f}) and ordered-label "
                  f"concordance ({imputed_results[0.15][1]:.3f}) because both "
                  "values exceed every measured label-0 and label-0.5 link7 score. "
                  "The higher assumption changes the score magnitude, though: "
                  "the severe-label mean is "
                  f"{statistics.mean(score for _, label, score in imputed_results[0.15][2] if label == 1):.4f} "
                  "at 0.15 versus "
                  f"{statistics.mean(score for _, label, score in imputed_results[0.20][2] if label == 1):.4f} "
                  "at 0.20. These are assumed scores, not measured rigidity.", ""])

    flagged = [record for record in records if record["flags"]]
    lines.extend(["## Cases marked `3` for extra attention", "",
                  "The `3` flags identify rows for closer review. Their ordinary labels "
                  "and corresponding link scores are shown below; a failed V1 case has no score.", "",
                  "| Video | Workbook row | `3` in | Forearm AB | Link5 rigidity | Upper arm AD | Link2 rigidity | Gripper AC | Link7 rigidity |",
                  "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for record in flagged:
        case = record["case"]
        case_link = (f"[{case}](cases/{case}/v1/metrics.json)" if record["scores"] else case)
        def score_text(part: str) -> str:
            score = record["scores"].get(part)
            return fmt(score["rigidity"]) if score else "—"
        cells = [case_link, str(record["row"]), ", ".join(record["flags"]),
                 display_label(record["labels"]["Forearm"]), score_text("Forearm"),
                 display_label(record["labels"]["Upper arm"]), score_text("Upper arm"),
                 display_label(record["labels"]["Gripper"]), score_text("Gripper")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "## Reading the comparison", "",
                  f"- {sum(r['state'] == 'complete' for r in records)} of 45 V1 cases are complete; "
                  "link2 and link5 are scored in every complete case. The failed case "
                  "`COSMOS2.5_0054` has workbook labels and `3` flags but no V1 score.",
                  "- Forearm/link5 rigidity does not separate the severity labels: "
                  f"AUROC is {association[('Forearm', False)][0]:.3f} and ordered-label "
                  f"concordance is {association[('Forearm', False)][1]:.3f}. Its medians are "
                  "0.0183, 0.0145, and 0.0152 for labels 0, 0.5, and 1.",
                  "- Upper-arm/link2 pooled rigidity medians rise from 0.0146 to 0.0152 "
                  "to 0.0331 for labels 0, 0.5, and 1, matching its higher AUROC.",
                  "- Gripper/link7 rigidity ranks severe cases above almost-no-deformation "
                  f"cases with AUROC {association[('Gripper', False)][0]:.3f}. "
                  "Capping values above 0.20 changes the mean but not the rank metrics.",
                  "- The `3`-flagged cases do not all have unusually high rigidity scores. "
                  "For example, `COSMOS2.5_0056` and `COSMOS2.5_0060` have link5 scores "
                  "0.0143 and 0.0120. "
                  "`COSMOS3_0054` has a `3` in "
                  "the gripper column only, and `COSMOS2.5_0054` has no V1 score.",
                  "- These are descriptive associations for 45 selected videos. They do not "
                  "establish whether the score detects the annotated deformation reliably.", ""])
    output = ROOT / "DEFORMATION_LABEL_ANALYSIS.md"
    output.write_text("\n".join(lines))
    print(output)


if __name__ == "__main__":
    main()
