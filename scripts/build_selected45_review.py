#!/usr/bin/env python3
"""Build a truthful local review page for completed and failed selected-45 cases."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results/selected-45-v1"
VIDEO_FOLDERS = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "COSMOS2.5", "COSMOS3": "COSMOS3"}
MASK_CASE_PREFIXES = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "Cosmos25", "COSMOS3": "Cosmos3"}


def link(path: str, label: str) -> str:
    return f'<a href="{html.escape(path, quote=True)}">{html.escape(label)}</a>'


def failure_reason(case: Path, recorded: str) -> str:
    log = case / "end_to_end.log"
    if not log.is_file():
        return recorded
    messages = re.findall(r"^(?:RuntimeError|ValueError): (.+)$", log.read_text(errors="replace"), re.M)
    specifics = [message for message in messages if "command exited" not in message
                 and ("SAM3" in message or "VLM2" in message)]
    if specifics:
        return specifics[-1]
    return recorded


def score_cell(report: dict) -> str:
    if report.get("status") != "complete":
        return f'<span class="muted">{html.escape(report.get("status", "unavailable"))}</span>'
    grade = str(report.get("grade", ""))
    letter = grade.split(" ", 1)[0]
    return (f'<strong>{float(report["pdi_score"]):.4f}</strong> '
            f'<span class="grade" title="{html.escape(grade, quote=True)}">'
            f'{html.escape(letter)}</span>')


def cloud_panel(sample: str, case: Path, variant: str, reports: dict,
                label: str) -> str:
    interactive = case / variant / "replay/interactive_exact-group"
    if not (interactive / "index.html").is_file():
        raise ValueError(f"missing {variant} interactive point-cloud index: {sample}")
    options = []
    for name, report in reports.items():
        page = interactive / f"{name}_exact-group.html"
        pairs = page.with_name(f"{name}_exact-group_pairs.json")
        if report.get("status") == "complete":
            if not page.is_file() or not pairs.is_file():
                raise ValueError(f"missing {variant} point-cloud evidence for {sample} {name}")
            options.append((name, f"cases/{sample}/{variant}/replay/interactive_exact-group/"
                                  f"{name}_exact-group.html"))
    if not options:
        raise ValueError(f"{variant} has no interactive point cloud: {sample}")
    default = next((path for name, path in options if name == "link7"), options[0][1])
    frame_id = f"{variant}-cloud-{sample.replace('.', '-')}"
    selector = "".join(
        f'<option value="{html.escape(path, quote=True)}"'
        f'{" selected" if path == default else ""}>{html.escape(name)}</option>'
        for name, path in options
    )
    return (
        '<div class="cloud-panel">'
        f'<div class="cloud-head"><h5>{html.escape(label)} · interactive 3D point cloud</h5>'
        f'<label>Link <select class="cloud-link" data-frame="{html.escape(frame_id, quote=True)}">'
        f'{selector}</select></label></div>'
        f'<iframe id="{html.escape(frame_id, quote=True)}" class="cloud-frame" '
        f'title="{html.escape(sample + " " + label, quote=True)} interactive point cloud" '
        f'data-src="{html.escape(default, quote=True)}" loading="lazy"></iframe>'
        f'{link(f"cases/{sample}/{variant}/replay/interactive_exact-group/index.html", "Open all links in full page")}'
        '</div>'
    )


def base_comparison(sample: str, case: Path, refined_objects: dict) -> str:
    base = case / "base_v1"
    required = [base / "metrics.json", case / "v1/metrics.json"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        raise ValueError(f"incomplete local base-mask comparison: {sample}")
    base_metrics = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
    base_objects = base_metrics["modes"]["exact-group"]["objects"]
    if set(base_objects) != set(refined_objects):
        raise ValueError(f"base/refined link reports differ: {sample}")
    rows = []
    for name, old in base_objects.items():
        new = refined_objects[name]
        old_score = old.get("pdi_score") if old.get("status") == "complete" else None
        new_score = new.get("pdi_score") if new.get("status") == "complete" else None
        delta = (f'{float(new_score) - float(old_score):+.4f}'
                 if old_score is not None and new_score is not None else "—")
        rows.append(f'<tr><th scope="row">{html.escape(name)}</th><td>{score_cell(old)}</td>'
                    f'<td>{score_cell(new)}</td><td>{delta}</td></tr>')
    links = [link(f"cases/{sample}/base_v1/metrics.json", "Base-mask full metrics JSON"),
             link(f"cases/{sample}/v1/metrics.json", "Refined full metrics JSON")]
    return (
        '<div class="comparison-block"><h4>Full V1 case comparison</h4>'
        '<p class="notice">The base run uses this video’s original six-link SAM3 mask. '
        'The refined run replaces link7 only. Scores below are PDI values; '
        'the letter is the recorded grade.</p>'
        '<div class="table-scroll"><table><thead><tr><th>Link</th><th>Base PDI / grade</th>'
        f'<th>Refined PDI / grade</th><th>Refined − base</th></tr></thead><tbody>{"".join(rows)}'
        '</tbody></table></div>'
        f'<div class="links">{"".join(links)}</div>'
        '<div class="cloud-grid">'
        f'{cloud_panel(sample, case, "base_v1", base_objects, "Base mask")}'
        f'{cloud_panel(sample, case, "v1", refined_objects, "Refined mask")}'
        '</div></div>'
    )


def mask_comparison(entry: dict, sample: str, case: Path, refined_objects: dict,
                    has_base_comparison: bool) -> str:
    mask_case = f"{MASK_CASE_PREFIXES[entry['dataset']]}_{entry['video_number']}"
    relative = f"persistent_work/local/selected_45_v1/{mask_case}"
    naive = ROOT / relative / "naive" / mask_case / "naive_masking.mp4"
    refined = ROOT / relative / "sam" / mask_case / "masking.mp4"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (naive, refined)):
        raise ValueError(f"naive/refined mask replay is not local: {sample}")
    old_case = base_comparison(sample, case, refined_objects) if has_base_comparison else (
        '<p class="notice">Full base-mask V1 scoring was run for five frozen comparison cases; '
        'this case has the gripper-mask replays only.</p>'
    )
    badge = '<span class="badge">Full base-mask V1</span>' if has_base_comparison else ''
    return (
        f'<li class="mask-case"><details><summary><strong>{html.escape(sample)}</strong>'
        f'<span class="row">workbook row {entry["workbook_row"]}</span>{badge}</summary>'
        '<p class="notice">These first two videos compare the persistent-masking package’s '
        'naive frame-zero gripper trajectory with its validated refined gripper trajectory. '
        'The naive gripper video is distinct from the six-link base mask used for V1 scoring.</p>'
        '<div class="replay-grid">'
        f'<figure><figcaption>Naive gripper mask · full video</figcaption><video controls '
        f'preload="none" src="{html.escape(naive.relative_to(ROOT).as_posix(), quote=True)}">'
        '</video></figure>'
        f'<figure><figcaption>Validated refined gripper mask · full video</figcaption>'
        f'<video controls preload="none" src="{html.escape(refined.relative_to(ROOT).as_posix(), quote=True)}">'
        '</video></figure></div>'
        f'{old_case}</details></li>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "summary.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    selection = json.loads((ROOT / "selection.json").read_text())
    complete = []
    failed = []
    mask_reviews = []
    comparison_selection = json.loads(
        (ROOT / "base_comparison_selection.json").read_text(encoding="utf-8")
    )
    comparison_samples = {item["sample_id"] for item in comparison_selection["cases"]}
    reviewed_comparison_samples = set()
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        state = summary["cases"][sample]
        case = ROOT / "cases" / sample
        if state["state"] == "complete":
            interactive = case / "v1/replay/interactive_exact-group"
            if not (case / ".geometry_offloaded.json").is_file():
                raise ValueError(f"complete case is not locally verified: {sample}")
            metrics = json.loads((case / "v1/metrics.json").read_text(encoding="utf-8"))
            objects = metrics["modes"]["exact-group"]["objects"]
            scored = [name for name, report in objects.items()
                      if report.get("status") == "complete"]
            unscored = [name for name, report in objects.items()
                        if report.get("status") != "complete"]
            if not scored or not all((interactive / f"{name}_exact-group_pairs.json").is_file()
                                     and (interactive / f"{name}_exact-group.html").is_file()
                                     for name in scored):
                raise ValueError(f"missing interactive links or pair evidence: {sample}")
            links = [link(f"cases/{sample}/v1/replay/interactive_exact-group/index.html",
                          "Interactive index"),
                     link(f"cases/{sample}/v1/replay/combined_exact-group.mp4", "Combined MP4")]
            links += [link(f"cases/{sample}/v1/replay/interactive_exact-group/{name}_exact-group.html",
                           name) for name in scored]
            if unscored:
                links.append(f'<span>Unscored: {html.escape(", ".join(unscored))}</span>')
            complete.append(f'<li><strong>{html.escape(sample)}</strong> '
                            f'<span class="row">{entry["workbook_row"]}</span>'
                            f'<div class="links">{"".join(links)}</div></li>')
            if state.get("persistent_mask_status") == "completed_checks":
                mask_reviews.append(mask_comparison(
                    entry, sample, case, objects, sample in comparison_samples,
                ))
                if sample in comparison_samples:
                    reviewed_comparison_samples.add(sample)
        elif state["state"] == "failed":
            source = (ROOT / "review_sources" / VIDEO_FOLDERS[entry["dataset"]]
                      / f"{entry['video_number']}.mp4")
            if not source.is_file() or not (case / "status.json").is_file():
                raise ValueError(f"failed case diagnostics are not local: {sample}")
            reason = failure_reason(case, state.get("error", "No reason recorded"))
            images = []
            for filename, label in (("dinov2_boxes.jpg", "DINOv2 prompt boxes"),
                                    ("first_frame_mask.png", "First-frame SAM3 mask")):
                if (case / "base_generation" / filename).is_file():
                    images.append(f'<figure><img loading="lazy" src="cases/{html.escape(sample, quote=True)}/'
                                  f'base_generation/{filename}" alt="{html.escape(label, quote=True)}">'
                                  f'<figcaption>{html.escape(label)}</figcaption></figure>')
            diagnostics = [link(f"cases/{sample}/status.json", "Status JSON"),
                           link(f"cases/{sample}/end_to_end.log", "Case log")]
            prompt = case / "base_generation/sam3_prompt_diagnostics.json"
            if prompt.is_file():
                diagnostics.append(link(f"cases/{sample}/base_generation/sam3_prompt_diagnostics.json",
                                        "SAM3 prompt details"))
            failed.append(
                f'<li><details><summary><strong>{html.escape(sample)}</strong> '
                f'<span class="row">{entry["workbook_row"]}</span> '
                f'<span class="reason">{html.escape(reason)}</span></summary>'
                '<p class="notice">V1 scoring stopped before an interactive point-pair replay was produced. '
                'This is the original video and available mask diagnostics.</p>'
                f'<video controls preload="none" src="review_sources/{VIDEO_FOLDERS[entry["dataset"]]}/'
                f'{entry["video_number"]}.mp4"></video>'
                f'<div class="images">{"".join(images)}</div>'
                f'<div class="links">{"".join(diagnostics)}</div>'
                '</details></li>'
            )
    if reviewed_comparison_samples != comparison_samples:
        raise ValueError(
            f"missing frozen base-mask comparison replays: "
            f"{sorted(comparison_samples - reviewed_comparison_samples)}"
        )
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    exit_marker = ROOT / "run.exit"
    if exit_marker.is_file():
        batch_status = ("Batch finished successfully." if exit_marker.read_text().strip() == "0"
                        else "Batch finished with failed cases.")
    else:
        batch_status = "Batch is running."
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Selected 45 · V1 review</title>
	<style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f6f7f8;color:#17212b}}
main{{max-width:1160px;margin:auto;padding:32px 22px 80px}}h1{{margin:0 0 4px}}
.sub{{color:#566370;margin:0 0 24px}}h2{{margin:32px 0 10px}}
ul{{list-style:none;padding:0;margin:0;display:grid;gap:10px}}
li{{background:white;border:1px solid #dce2e8;border-radius:9px;padding:14px 16px}}
.row{{font-size:12px;color:#647380;margin-left:8px}}
.links{{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}}
a{{display:inline-block;color:#07579c;background:#eef5ff;border-radius:5px;padding:5px 9px;text-decoration:none}}
a:hover{{text-decoration:underline}}details summary{{cursor:pointer;list-style:auto}}
.reason{{color:#8b3b25;margin-left:8px}}.notice{{color:#606c78}}
video{{display:block;max-width:100%;width:560px;margin:12px 0;background:#111}}
.images{{display:flex;flex-wrap:wrap;gap:14px}}figure{{margin:0;max-width:480px}}
figure img{{display:block;max-width:100%;max-height:330px;border:1px solid #ddd}}
figcaption{{font-size:13px;color:#66717d;margin-top:4px}}
.mask-case details>summary{{padding:1px 0}}.mask-case details[open]>summary{{margin-bottom:14px}}
.badge{{display:inline-block;margin-left:10px;padding:2px 7px;border-radius:4px;
background:#e4f2ec;color:#246347;font-size:11px;font-weight:700;letter-spacing:.02em}}
.replay-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:14px 0}}
.replay-grid figure{{max-width:none;min-width:0}}.replay-grid figcaption{{font-weight:650;color:#344656}}
.replay-grid video{{width:100%;margin:7px 0 0;aspect-ratio:16/9;object-fit:contain}}
.comparison-block{{border-top:1px solid #dce2e8;margin-top:20px;padding-top:14px}}
h4{{font-size:18px;margin:0 0 5px}}h5{{font-size:15px;margin:0}}
.table-scroll{{overflow-x:auto;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:8px 10px;border-bottom:1px solid #e3e8ec;text-align:left;white-space:nowrap}}
thead th{{background:#f1f4f6;color:#425463;font-size:12px}}tbody th{{font-weight:650}}
.grade{{color:#567183;font-size:12px;margin-left:4px}}.muted{{color:#7a8790}}
.cloud-grid{{display:grid;grid-template-columns:minmax(0,1fr);gap:24px;margin-top:16px}}
.cloud-panel{{min-width:0}}.cloud-panel>a{{margin-top:8px}}
.cloud-head{{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:20px 0 8px}}
.cloud-head label{{font-size:13px;color:#425463}}select{{font:inherit;border:1px solid #bbc9d2;
border-radius:4px;background:white;color:#17212b;padding:5px 7px}}
.cloud-frame{{display:block;width:100%;height:620px;border:1px solid #dce2e8;border-radius:6px;
background:#f1f4f6}}.section-note{{max-width:900px;color:#566370;margin:0 0 13px}}
@media(max-width:760px){{.replay-grid{{grid-template-columns:1fr}}.cloud-frame{{height:520px}}
.cloud-head{{align-items:flex-start;flex-direction:column}}}}
</style><main><h1>Selected 45 · V1 review</h1>
<p class="sub">Snapshot {html.escape(now)} · {len(complete)} completed interactive replays · \
{len(failed)} failed case diagnostics. {batch_status}</p>
<h2>Mask and base-score comparisons</h2>
<p class="section-note">{len(mask_reviews)} validated refinements have side-by-side naive and refined gripper-mask videos.
The five marked cases also include full V1 scoring and 3D replay with the same-video base six-link mask.</p>
<ul>{''.join(mask_reviews)}</ul>
<h2>Completed cases</h2><ul>{''.join(complete)}</ul>
<h2>Failed cases</h2><ul>{''.join(failed)}</ul></main>
<script>
document.querySelectorAll('.mask-case details').forEach(details => {{
  details.addEventListener('toggle', () => {{
    if (!details.open) return;
    details.querySelectorAll('iframe[data-src]').forEach(frame => {{
      if (!frame.getAttribute('src')) frame.src = frame.dataset.src;
    }});
  }});
}});
document.querySelectorAll('.cloud-link').forEach(select => {{
  select.addEventListener('change', () => {{
    const frame = document.getElementById(select.dataset.frame);
    if (frame) frame.src = select.value;
  }});
}});
</script></html>"""
    destination = ROOT / "review.html"
    destination.write_text(page, encoding="utf-8")
    print(destination)
    print(f"complete={len(complete)} failed={len(failed)} mask_reviews={len(mask_reviews)}")


if __name__ == "__main__":
    main()
