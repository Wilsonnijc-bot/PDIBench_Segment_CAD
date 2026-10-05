"""Freeze the selected-45 inputs and historical, non-target link references."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/prepare_link5_link7_four_way.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import json
import shutil
from pathlib import Path

ROOT = _SOURCE_PATH.parents[1]
SOURCE = ROOT / "results/selected-45-v1"
TARGET = ROOT / "results/link5-link7-four-way-selected45-20260927"


def sha(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    selection = json.loads((SOURCE / "selection.json").read_text())
    if selection["video_count"] != 45:
        raise ValueError("source selection is not the frozen 45")
    TARGET.mkdir(parents=True, exist_ok=True)
    (TARGET / "input").mkdir(exist_ok=True)
    for relative in ("selection.json", "input/selected_45_matched_videos_styled.xlsx"):
        source = SOURCE / relative
        target = TARGET / relative
        if target.is_file() and sha(target) != sha(source):
            raise ValueError(f"frozen input differs: {target}")
        shutil.copy2(source, target)
    references = {"schema_version": 1, "source_run": "selected-45-v1",
                  "cases": {}}
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        folder = SOURCE / "cases" / sample / "v1"
        metrics_path = folder / "metrics.json"
        manifest_path = folder / "manifest.json"
        if not metrics_path.is_file() or not manifest_path.is_file():
            references["cases"][sample] = {"status": "unavailable"}
            continue
        metrics = json.loads(metrics_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if manifest["input"]["sha256"] != entry["sha256"]:
            raise ValueError(f"historical video identity mismatch: {sample}")
        objects = metrics["modes"]["exact-group"]["objects"]
        references["cases"][sample] = {
            "status": "historical_reference",
            "metrics_sha256": sha(metrics_path),
            "manifest_sha256": sha(manifest_path),
            "segmentation_sha256": manifest["segmentation"]["sha256"],
            "links": {name: {"status": objects[name]["status"],
                              "epsilon_rigidity": objects[name].get("breakdown", {}).get("epsilon_rigidity")}
                      for name in ("link2", "link3", "link4", "link6")},
        }
    target = TARGET / "historical_references.json"
    payload = json.dumps(references, indent=2, sort_keys=True) + "\n"
    if target.is_file() and target.read_text() != payload:
        raise ValueError("historical references already differ from the frozen source")
    target.write_text(payload)
    print(f"prepared {selection['video_count']} cases at {TARGET}")


if __name__ == "__main__":
    main()
