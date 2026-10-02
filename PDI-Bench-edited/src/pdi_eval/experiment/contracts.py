"""Stable identities and artifact checks shared by the experiment interface."""

from __future__ import annotations

import hashlib
import html
import json
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATASET_FOLDERS = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "COSMOS2.5", "COSMOS3": "COSMOS3"}
CASE_PREFIXES = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "Cosmos25", "COSMOS3": "Cosmos3"}
MASK_VIDEO_FOLDERS = {
    "LVP_ROBOWM": "1irS6zoWSykw64DuaiwxyUVHdIffEo3Oa",
    "COSMOS2.5": "1rXdVXrwIjf9wMeCVxbmFWN7cZNcfdtb9",
    "COSMOS3": "15sRTmHwkBQO0FmDRnp0BePAArCad2_Gj",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checked_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def video_path(video_root: Path, entry: dict) -> Path:
    return video_root / DATASET_FOLDERS[entry["dataset"]] / f"{entry['video_number']}.mp4"


def case_name(entry: dict) -> str:
    return f"{CASE_PREFIXES[entry['dataset']]}_{entry['video_number']}"


def mask_is_valid(path: Path, expected_names: tuple[str, ...] | None = None) -> bool:
    if not path.is_file():
        return False
    import numpy as np
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(str(x) for x in archive["object_names"])
            masks = archive["object_masks"]
            ids = archive["object_ids"]
            expected = expected_names or tuple(f"link{i}" for i in range(2, 8))
            return (names == expected
                    and masks.ndim == 4 and masks.shape[1] == len(expected)
                    and ids.shape == (len(expected),))
    except (OSError, KeyError, ValueError):
        return False


def run_command(command: list[str], *, log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as output:
        output.write("\nCOMMAND " + shlex.join(command) + "\n")
        output.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
        output.write(f"EXIT_STATUS={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"command exited {result.returncode}; see {log}")


def read_selection(output_root: Path) -> dict:
    path = output_root / "selection.json"
    checked_file(path, "frozen selection")
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("video_count") != len(selection.get("videos", [])):
        raise ValueError("selection video count disagrees with its records")
    ids = [f"{item['dataset']}_{item['video_number']}" for item in selection["videos"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selection must contain distinct videos")
    return selection


def status_summary(output_root: Path, selection: dict) -> dict:
    cases = {}
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        path = output_root / "cases" / sample / "status.json"
        cases[sample] = json.loads(path.read_text()) if path.is_file() else {"state": "pending"}
    counts = {state: sum(item.get("state") == state for item in cases.values())
              for state in ("pending", "running", "complete", "failed")}
    return {"video_count": len(cases), "counts": counts, "cases": cases}


def page_index(output_root: Path, selection: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        case = output_root / "cases" / sample
        path = case / "status.json"
        state = json.loads(path.read_text()).get("state", "unknown") if path.is_file() else "pending"
        links = []
        for label, relative in (
            ("V1 replay", "v1/replay/combined_exact-group.mp4"),
            ("V1 interactive", "v1/replay/interactive_exact-group/index.html"),
            ("V1 metrics", "v1/metrics.json"),
            ("Base-mask V1 replay", "base_v1/replay/combined_exact-group.mp4"),
            ("Base-mask V1 interactive", "base_v1/replay/interactive_exact-group/index.html"),
            ("Base-mask V1 metrics", "base_v1/metrics.json"),
        ):
            if (case / relative).is_file():
                link = f"cases/{sample}/{relative}"
                links.append(f'<a href="{html.escape(link)}">{label}</a>')
        rows.append(f"<tr><td>{html.escape(sample)}</td><td>{entry['workbook_row']}</td>"
                    f"<td>{html.escape(state)}</td><td>{' · '.join(links)}</td></tr>")
    page = ("<!doctype html><meta charset='utf-8'><title>PDI V1 replays</title>"
            "<style>body{font:15px system-ui;margin:40px;max-width:1100px}table{border-collapse:collapse;width:100%}"
            "td,th{padding:10px;border-bottom:1px solid #bbb;text-align:left}a{margin-right:10px}</style>"
            "<h1>PDI V1 replay experiment</h1><p>Persistent mask and shared geometry; "
            "exact-group CoTracker.</p><table><tr><th>Case</th><th>Workbook row</th>"
            "<th>Status</th><th>Artifacts</th></tr>" + "".join(rows) + "</table>")
    (output_root / "index.html").write_text(page, encoding="utf-8")
    write_json(output_root / "summary.json", status_summary(output_root, selection))
