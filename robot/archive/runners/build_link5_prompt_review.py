#!/usr/bin/env python3
"""Show the exact frame-0 Link 5 box and point preflight for two failed cases."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/build_link5_prompt_review.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import ast
import hashlib
import json
import shlex
import shutil
import struct
import subprocess
from pathlib import Path


ROOT = _SOURCE_PATH.parents[1] / "results/link7-point-filter-v2-20260926"
OUTPUT = ROOT / "link5-prompt-review"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
HOST = "root@region-9.autodl.pro"
REMOTE_PYTHON = "/root/autodl-tmp/pdi/env/qwen/bin/python"
CASES = ("LVP_ROBOWM_0040", "COSMOS3_0044")
RATIOS = ((.12, .55, 1), (.42, .55, 1), (.86, .35, 1),
          (.06, .28, 0), (.02, .57, 0))
EXTRACT = """import cv2,sys
from PIL import Image
cap=cv2.VideoCapture(sys.argv[1])
cap.set(cv2.CAP_PROP_POS_FRAMES,0)
ok,frame=cap.read()
cap.release()
if not ok: raise RuntimeError('Could not read source frame 0')
Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).save(sys.stdout.buffer,format='PNG')
"""


def frame_png(source: str) -> bytes:
    command = shlex.join((REMOTE_PYTHON, "-c", EXTRACT, source))
    return subprocess.check_output(("ssh", "-i", str(KEY), "-p", "26211", HOST, command))


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("source frame is not PNG")
    return struct.unpack(">II", data[16:24])


def make_case(sample: str, selection: dict[str, dict]) -> tuple[dict, str]:
    case = ROOT / "cases" / sample
    folder = OUTPUT / sample
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("dinov2_boxes.json", "sam3_prompt_diagnostics.json"):
        preserved = folder / name
        if not preserved.exists():
            shutil.copy2(case / "base_generation" / name, preserved)
    box_record = json.loads((folder / "dinov2_boxes.json").read_text())
    diagnostic = next(item for item in json.loads(
        (folder / "sam3_prompt_diagnostics.json").read_text())
        if item["target"] == "link5")
    box = next(item["box_xyxy"] for item in box_record["targets"] if item["name"] == "link5")
    if box != diagnostic["box_xyxy"] or box_record["frame_index"] != 0:
        raise ValueError(f"prompt box/frame mismatch: {sample}")
    recorded = ast.literal_eval(diagnostic["point_refinement_error"].split(": ", 1)[1])
    source = str((case / f"{sample}.mp4").readlink())
    if not source.startswith("/root/autodl-tmp/motionsmoothness-robot/videos/"):
        raise ValueError(f"unexpected source link: {sample}")
    png = frame_png(source)
    width, height = png_size(png)
    (folder / "frame_00000.png").write_bytes(png)
    x1, y1, x2, y2 = box
    points = []
    for number, (rx, ry, label) in enumerate(RATIOS):
        x = max(0, min(width - 1, round(x1 + rx * (x2 - x1))))
        y = max(0, min(height - 1, round(y1 + ry * (y2 - y1))))
        shown_xy, shown_label, inside = recorded[number]
        if [x, y] != shown_xy or label != shown_label:
            raise ValueError(f"computed point differs from recorded point: {sample} #{number+1}")
        points.append({"id": f"{'P' if label else 'N'}{number+1 if label else number-2}",
                       "x": x, "y": y, "label": label,
                       "inside_initial_mask": inside})
    record = {
        "sample_id": sample, "frame_index": 0, "image_size": [width, height],
        "source_video_sha256": selection[sample]["sha256"],
        "frame_png_sha256": hashlib.sha256(png).hexdigest(),
        "link5_box_xyxy": box, "points": points,
        "sam3_initial_candidate": diagnostic["candidates"][0],
        "point_prompt_submitted": False,
        "failure": diagnostic["point_refinement_error"],
        "source_diagnostics": [
            "dinov2_boxes.json",
            "sam3_prompt_diagnostics.json",
        ],
    }
    (folder / "provenance.json").write_text(json.dumps(record, indent=2) + "\n")
    zoom = [max(0, x1 - 65), max(0, y1 - 65),
            min(width, x2 + 65) - max(0, x1 - 65),
            min(height, y2 + 65) - max(0, y1 - 65)]
    rows = "".join(
        f"<tr><td><b class={'positive' if p['label'] else 'negative'}>{p['id']}</b></td>"
        f"<td>({p['x']}, {p['y']})</td><td>{'inside' if p['inside_initial_mask'] else '<strong>outside</strong>'}</td></tr>"
        for p in points)
    shapes = [f'<rect class="box" x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}"/>']
    for p in points:
        color = "#35e084" if p["label"] else "#ff6363"
        if not p["inside_initial_mask"]:
            shapes.append(f'<circle class="outside" cx="{p["x"]}" cy="{p["y"]}" r="16"/>')
        shapes.append(f'<circle cx="{p["x"]}" cy="{p["y"]}" r="9" fill="{color}" stroke="#09131a" stroke-width="3"/>')
        shapes.append(f'<text x="{p["x"]+13}" y="{p["y"]-11}">{p["id"]}</text>')
    card = f"""<section class="case" id="{sample}">
      <div class="case-head"><h2>{sample}</h2><span>frame 0 · prompt preflight failed</span></div>
      <div class="controls"><button onclick="setView('{sample}',true)">Zoom Link 5</button>
      <button onclick="setView('{sample}',false)">Full frame</button>
      <a href="{sample}/frame_00000.png">Raw source frame</a>
      <a href="{sample}/provenance.json">Point provenance</a></div>
      <svg id="view-{sample}" viewBox="{' '.join(map(str,zoom))}" data-zoom="{' '.join(map(str,zoom))}"
           data-full="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img"
           aria-label="Source frame with Link 5 box and five proposed prompt points">
        <image href="{sample}/frame_00000.png" x="0" y="0" width="{width}" height="{height}"/>
        {''.join(shapes)}
      </svg>
      <p class="legend"><span class="positive">● positive on Link 5</span>
      <span class="negative">● negative on wrist</span>
      <span class="outside-key">◌ outside initial SAM3 mask</span></p>
      <table><thead><tr><th>Point</th><th>Source pixel</th><th>Initial mask</th></tr></thead><tbody>{rows}</tbody></table>
      <p class="note">DINOv2 box: [{x1}, {y1}, {x2}, {y2}]. All five points were calculated,
      then the initial-mask check stopped the run before the five-point SAM3 prompt was submitted.</p>
    </section>"""
    return record, card


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selection = {f"{item['dataset']}_{item['video_number']}": item for item in
                 json.loads((ROOT / "selection.json").read_text())["videos"]}
    cards = [make_case(sample, selection)[1] for sample in CASES]
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Link 5 actual prompt box and points</title><style>
:root{font:15px/1.45 system-ui,sans-serif;color:#eaf3f0;background:#10191b}
body{margin:0 auto;padding:26px;max-width:1700px}h1{font-size:25px;margin:0 0 7px}h2{font-size:19px;margin:0}
p{margin:7px 0 18px;color:#b9cac5}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.case{background:#1c292b;border:1px solid #435655;padding:16px;min-width:0}.case-head{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.case-head span{font-size:12px;color:#efba77}.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:12px 0}
button{background:#294d48;border:1px solid #65a294;color:#fff;padding:6px 10px;cursor:pointer}a{color:#a6ddff}
svg{width:100%;height:auto;aspect-ratio:16/9;background:#070b0c;border:1px solid #405052}
.box{fill:none;stroke:#43caff;stroke-width:4}.outside{fill:none;stroke:#ffd65a;stroke-width:5;stroke-dasharray:5 4}
svg text{fill:#fff;font:bold 20px system-ui;paint-order:stroke;stroke:#10191b;stroke-width:5}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.positive{color:#35e084}.negative{color:#ff7676}.outside-key{color:#ffd65a}
table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:5px 8px;border-bottom:1px solid #3c4c4b}
.note{font-size:13px;margin-top:12px}.intro{max-width:980px}@media(max-width:1050px){.grid{grid-template-columns:1fr}}
</style></head><body><h1>Link 5 prompt box and points · actual frame 0</h1>
<p class="intro">These are the exact source frames and DINOv2 Link 5 boxes from the failed runs.
The points were computed from the recorded box-relative ratios. Yellow rings identify points outside
SAM3’s initial mask; the check stopped both cases before the five-point prompt was sent.</p>
<div class="grid">__CARDS__</div><script>
function setView(id,zoom){const svg=document.getElementById('view-'+id);svg.setAttribute('viewBox',zoom?svg.dataset.zoom:svg.dataset.full)}
</script></body></html>"""
    (OUTPUT / "index.html").write_text(page.replace("__CARDS__", "".join(cards)))
    print(OUTPUT / "index.html")


if __name__ == "__main__":
    main()
