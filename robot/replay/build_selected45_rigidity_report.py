#!/usr/bin/env python3
"""Summarize the frozen selected-45 V1 link scores from local metrics.json files."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import json
import statistics
from pathlib import Path

ROOT = (_workspace_root() / 'results/selected-45-v1')
LINKS = tuple(f"link{i}" for i in range(2, 8))
METRICS = (("rigidity", "epsilon_rigidity", "Rigidity score (epsilon_rigidity)"),)


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def mean_median(values: list[float]) -> str:
    return f"{fmt(statistics.mean(values))} / {fmt(statistics.median(values))}" if values else "—"


def main() -> None:
    selection = json.loads(((_workspace_root() / 'results/selected-45-v1/selection.json')).read_text())
    summary = json.loads(((_workspace_root() / 'results/selected-45-v1/summary.json')).read_text())
    rows = []
    for item in selection["videos"]:
        case = f"{item['dataset']}_{int(item['video_number']):04d}"
        state = summary["cases"][case]["state"]
        scores: dict[str, dict[str, float]] = {kind: {} for kind, _, _ in METRICS}
        if state == "complete":
            metrics = json.loads(((_workspace_root() / 'results/selected-45-v1/cases') / case / "v1" / "metrics.json").read_text())
            objects = metrics["modes"]["exact-group"]["objects"]
            for link in LINKS:
                obj = objects[link]
                if obj["status"] != "complete":
                    continue
                scores["rigidity"][link] = float(obj["breakdown"]["epsilon_rigidity"])
        rows.append({"case": case, "generator": item["dataset"], "state": state,
                     "scores": scores})

    lines = ["# Selected 45: V1 rigidity scores", "",
             "Source: each case's `v1/metrics.json`, `modes.exact-group.objects`. "
             "The row order follows `selection.json` (the selected workbook order). "
             "See [the deformation-label analysis](DEFORMATION_LABEL_ANALYSIS.md) for "
             "forearm/link5 and upper-arm/link2 comparisons with the workbook.", "",
             "**Score definition.** The rigidity score is each link's "
             "`breakdown.epsilon_rigidity`: the mean over frames of the robust spread "
             "of tracked 3D point-pair distance ratios relative to frame 0. "
             "Lower values mean the selected pair distances change more consistently. "
             "It can also respond to tracking, masking, or 3D reconstruction errors, "
             "and uniform scaling can leave this score low.", "",
             "**Totals.** Each video's total is the sum of its scored links; the pipeline does not "
             "emit a native whole-video score. `Per-link mean` divides that total by the number "
             "of scored links. `—` means the link was skipped or failed and is excluded, never zero. "
             "Totals for videos with fewer than six scored links are therefore not directly "
             "comparable; use the per-link mean for generator comparisons. Values here are "
             "computed from the four-decimal values stored in the source metrics.", ""]
    for kind, _, title in METRICS:
        lines.extend([f"## {title} by video", "",
                      "| Video | Generator | L2 | L3 | L4 | L5 | L6 | L7 | Scored | Total | Per-link mean |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in rows:
            scores = row["scores"][kind]
            values = list(scores.values())
            case_link = (f"[{row['case']}](cases/{row['case']}/v1/metrics.json)"
                         if values else row["case"])
            cells = [case_link, row["generator"]] + [fmt(scores.get(link)) for link in LINKS]
            cells += [str(len(values)), fmt(sum(values)) if values else "—",
                      fmt(statistics.mean(values)) if values else "—"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.extend(["", f"## {title}: generator mean / median", "",
                      "Each statistic below treats one completed video as one observation. "
                      "The score is excluded only when that video has no scored links.", "",
                      "| Generator | Completed / selected | Mean scored links | Total mean / median | Per-link mean / median |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for generator in ("LVP_ROBOWM", "COSMOS2.5", "COSMOS3"):
            group = [row for row in rows if row["generator"] == generator]
            good = [list(row["scores"][kind].values()) for row in group if row["scores"][kind]]
            totals = [sum(x) for x in good]
            normalized = [statistics.mean(x) for x in good]
            cells = [generator, f"{len(good)} / {len(group)}",
                     fmt(statistics.mean(map(len, good))) if good else "—",
                     mean_median(totals), mean_median(normalized)]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["## Link2, link5, and link7 medians by generator", "",
                  "Each cell is the median across videos with a completed score for that link; "
                  "`n` is the number of contributing videos. Unscored links are excluded.", ""])
    for kind, _, title in METRICS:
        lines.extend([f"### {title}", "",
                      "| Generator | Link2 median (n) | Link5 median (n) | Link7 median (n) |",
                      "| --- | ---: | ---: | ---: |"])
        for generator in ("LVP_ROBOWM", "COSMOS2.5", "COSMOS3"):
            group = [row for row in rows if row["generator"] == generator]
            cells = [generator]
            for link in ("link2", "link5", "link7"):
                values = [row["scores"][kind][link] for row in group
                          if link in row["scores"][kind]]
                cells.append(f"{fmt(statistics.median(values))} ({len(values)})" if values else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    failed = [row["case"] for row in rows if row["state"] != "complete"]
    lines.extend(["## Interpretation", "",
                  "- LVP_ROBOWM has the lowest normalized rigidity "
                  "mean/median (0.0264 / 0.0260), followed by COSMOS3 "
                  "(0.0407 / 0.0280) and COSMOS2.5 (0.0477 / 0.0412). "
                  "The COSMOS groups have larger mean-to-median gaps, indicating "
                  "some high-score videos pull up their means.",
                  "- Among link2, link5, and link7, LVP_ROBOWM has the lowest median "
                  "rigidity score on link2 and link7, while COSMOS3 is lowest "
                  "on link5. Link7 has only 11 scored videos in each COSMOS group, "
                  "compared with 15 in LVP_ROBOWM.", "",
                  "## Coverage and interpretation", "",
                  f"- Completed cases: {sum(row['state'] == 'complete' for row in rows)} / {len(rows)}. "
                  f"Failed: {', '.join(failed) if failed else 'none'}.",
                  "- A completed video may still have unscored links; the `Scored` column shows its denominator.",
                  "- Generator differences are descriptive for these selected videos; the three "
                  "generators contain matched video numbers, but no uncertainty interval or "
                  "significance test is asserted here.", ""])
    ((_workspace_root() / 'results/selected-45-v1/RIGIDITY_SCORES.md')).write_text("\n".join(lines))
    print((_workspace_root() / 'results/selected-45-v1/RIGIDITY_SCORES.md'))


if __name__ == "__main__":
    main()
