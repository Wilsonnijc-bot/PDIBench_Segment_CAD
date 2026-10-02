"""Refresh Link 7 comparison replay UI without rerunning inference or scoring."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "results/link7-point-filter-v2-20260926/cases"
TEMPLATE = (ROOT / "PDI-Bench-edited/src/pdi_eval/utils/rigidity_replay.html")


def main() -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    for comparison_path in sorted(CASES.glob("*/comparison.json")):
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        for path_name, directory in (("path1", "path1_v1_cotracker"),
                                     ("path2", "path2_v2_tapip3d")):
            page = comparison_path.parent / directory / "replay/interactive_exact-group/link7_exact-group.html"
            if not page.exists():
                continue
            old = page.read_text(encoding="utf-8")
            marker = "const data="
            data = json.loads(old.split(marker, 1)[1].split(";\n", 1)[0])
            policy = comparison["paths"][path_name]
            data["version"] = policy["filter"].upper()
            tracker = "TAPIP3D" if policy["tracker"] == "tapip3d" else "CoTracker3"
            data["tracker_label"] = tracker
            mask = policy["mask"]
            data["mask_label"] = ("persistent refined mask" if mask == "persistent_refined"
                                  else "base mask fallback (persistent refinement failed)"
                                  if mask.startswith("base_fallback") else "base mask")
            payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
            page.write_text(template.replace("CoTracker", tracker).replace("__REPLAY_DATA__", payload),
                            encoding="utf-8")
            print(page)


if __name__ == "__main__":
    main()
