"""Refresh Link 7 comparison replay UI without rerunning inference or scoring."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/refresh_link7_replay_ui.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import json
from pathlib import Path


ROOT = _SOURCE_PATH.parents[1]
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
