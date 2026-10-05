#!/usr/bin/env python3
"""Score the five frozen selected-45 base masks outside the general runner."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/scripts/run_selected45_base_comparisons.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import argparse
import json
import os
import sys
from pathlib import Path

BENCHMARK = _SOURCE_PATH.parents[1]
sys.path.insert(0, str(BENCHMARK / "src"))

from pdi_eval.experiment.contracts import (  # noqa: E402
    checked_file, mask_is_valid, page_index, read_selection, sha256_file,
    video_path, write_json,
)
from pdi_eval.experiment.runner import (  # noqa: E402
    LINK_NAMES, command, ensure_video_alias, score_argv, v1_outputs_complete,
)
from pdi_eval.experiment.spec import load_spec  # noqa: E402


def frozen_cases(output_root: Path, selection: dict) -> list[tuple[str, dict]]:
    record = json.loads(
        (output_root / "base_comparison_selection.json").read_text(encoding="utf-8")
    )
    chosen = record.get("cases")
    if (selection.get("video_count") != 45 or record.get("schema_version") != 1
            or record.get("max_cases") != 5 or not isinstance(chosen, list)
            or len(chosen) != 5):
        raise ValueError("expected the five frozen selected-45 comparison cases")
    by_sample = {f"{entry['dataset']}_{entry['video_number']}": entry
                 for entry in selection["videos"]}
    samples = [item.get("sample_id") for item in chosen]
    if (len(set(samples)) != 5
            or samples != [name for name in by_sample if name in samples]):
        raise ValueError("comparison cases must be distinct and in workbook order")
    cases = []
    for item, sample in zip(chosen, samples):
        entry = by_sample.get(sample)
        if (entry is None or item.get("source_sha256") != entry["sha256"]
                or item.get("workbook_row") != entry["workbook_row"]):
            raise ValueError(f"comparison differs from frozen selection: {sample}")
        cases.append((sample, entry))
    return cases


def compare_case(spec, sample: str, entry: dict) -> None:
    case = spec.output_root / "cases" / sample
    status = json.loads((case / "status.json").read_text(encoding="utf-8"))
    if (status.get("state") != "complete"
            or status.get("source_sha256") != entry["sha256"]
            or status.get("persistent_mask_status") != "completed_checks"):
        raise ValueError(f"{sample} requires a completed V1 run with validated refinement")

    source = video_path(spec.video_root, entry)
    checked_file(source, "selected video")
    if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
        raise ValueError(f"source differs from frozen selection: {sample}")
    alias = case / f"{sample.replace('.', '_')}.mp4"
    ensure_video_alias(alias, source, entry["sha256"])

    base = case / "base_segmentation.npz"
    refined = case / "refined_segmentation.npz"
    if not mask_is_valid(base) or not mask_is_valid(refined):
        raise ValueError(f"base or refined segmentation is invalid: {sample}")
    provenance = json.loads(
        (case / "base_segmentation_source.json").read_text(encoding="utf-8")
    )
    base_hash = sha256_file(base)
    refined_hash = sha256_file(refined)
    if (provenance.get("source_video_sha256") != entry["sha256"]
            or provenance.get("base_segmentation_sha256") != base_hash
            or provenance.get("method") != "DINOv2-guided SAM3 full-video six-link segmentation"
            or status.get("scoring_segmentation_sha256") != refined_hash
            or not v1_outputs_complete(
                case / "v1", video_sha256=entry["sha256"],
                segmentation_sha256=refined_hash,
            )):
        raise ValueError(f"refined V1 or mask provenance does not match: {sample}")

    destination = case / "base_v1"
    if not v1_outputs_complete(
        destination, video_sha256=entry["sha256"],
        segmentation_sha256=base_hash,
    ):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(BENCHMARK / "src")
        command(
            score_argv(spec, alias, base, destination, case),
            case / "base_comparison.log", environment, spec.gpu_lock,
        )
        if not v1_outputs_complete(
            destination, video_sha256=entry["sha256"],
            segmentation_sha256=base_hash,
        ):
            raise RuntimeError(f"base-mask V1 comparison is incomplete: {sample}")

    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    reports = metrics["modes"]["exact-group"]["objects"]
    write_json(destination / "comparison.json", {
        "sample_id": sample,
        "source_sha256": entry["sha256"],
        "base_segmentation_sha256": base_hash,
        "refined_segmentation_sha256": refined_hash,
        "scored_links": [name for name in LINK_NAMES
                         if reports[name]["status"] == "complete"],
        "unscored_links": {
            name: reports[name].get("error_type", reports[name]["status"])
            for name in LINK_NAMES if reports[name]["status"] != "complete"
        },
    })
    print(f"{sample}: base comparison complete", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Selected-45 output root containing the frozen selection")
    args = parser.parse_args()
    os.environ["PDI_RUN_ROOT"] = str(args.run_root.absolute())
    spec = load_spec(args.spec)
    if spec.output_root != args.run_root.absolute():
        raise ValueError("spec output root differs from --run-root")
    selection = read_selection(spec.output_root)
    cases = frozen_cases(spec.output_root, selection)
    for sample, entry in cases:
        compare_case(spec, sample, entry)
    page_index(spec.output_root, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
