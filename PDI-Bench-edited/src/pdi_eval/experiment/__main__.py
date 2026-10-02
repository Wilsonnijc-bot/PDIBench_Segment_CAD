"""Public command line interface for frozen PDI experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .contracts import (ROOT, checked_file, mask_is_valid, page_index,
                        read_selection, sha256_file, status_summary, video_path, write_json)
from .spec import load_spec


def select(args: argparse.Namespace) -> int:
    from .manifest import build_manifest

    full = build_manifest(args.workbook, args.source_manifest, replay_all=True)
    if full["source"]["video_count"] != 197:
        raise ValueError("expected the 197-video source manifest")
    requested = json.loads(args.cases.read_text(encoding="utf-8"))
    ids = [item["sample_id"] for item in requested]
    if len(ids) != 10 or len(set(ids)) != 10:
        raise ValueError("selection must contain ten distinct sample IDs")
    lookup = {f"{item['dataset']}_{item['video_number']}": item for item in full["videos"]}
    missing = set(ids) - lookup.keys()
    if missing:
        raise ValueError(f"selected videos are missing from workbook/source intersection: {sorted(missing)}")
    records = [{**lookup[item["sample_id"]], "selection_category": item.get("category", ""),
                "selection_reason": item.get("reason", "")} for item in requested]
    payload = {"schema_version": 1, "experiment": "v1-persistent-mask-replay-10",
               "workbook": str(args.workbook.resolve()), "workbook_sha256": sha256_file(args.workbook),
               "source_manifest": str(args.source_manifest.resolve()),
               "source_manifest_sha256": sha256_file(args.source_manifest),
               "source_video_count": 197, "video_count": 10, "videos": records}
    target = args.output_root / "selection.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != payload:
        raise FileExistsError(f"frozen selection already exists with different contents: {target}")
    write_json(target, payload)
    page_index(args.output_root, payload)
    print(f"Selected workbook rows: {[item['workbook_row'] for item in records]}")
    return 0


def check(spec_path: Path, *, assets: bool = True) -> dict:
    spec = load_spec(spec_path)
    selection = read_selection(spec.output_root)
    if spec.workflow in {"selected45_link5_link7_four_way", "selected45_link5_only"} and selection["video_count"] != 45:
        raise ValueError("Selected-45 profile requires the frozen 45 videos")
    if selection["video_count"] not in (10, 45):
        raise ValueError("this experiment profile requires ten or 45 selected videos")
    if spec.selection != spec.output_root / "selection.json":
        raise ValueError("spec selection must be output_root/selection.json")
    checked_file(spec.workbook, "human-label workbook")
    if sha256_file(spec.workbook) != selection["workbook_sha256"]:
        raise ValueError("workbook differs from frozen selection")
    for entry in selection["videos"]:
        source = video_path(spec.video_root, entry)
        checked_file(source, "selected source video")
        if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"selected source video differs from manifest: {source}")
        case = spec.output_root / "cases" / f"{entry['dataset']}_{entry['video_number']}"
        base = case / "base_segmentation.npz"
        provenance = case / "base_segmentation_source.json"
        if base.exists() or provenance.exists() or not spec.generate_base_masks:
            expected_names = ("link5",) if spec.workflow == "selected45_link5_only" else None
            if not mask_is_valid(base, expected_names):
                raise ValueError(f"validated base mask is missing: {base}")
            checked_file(provenance, "base mask provenance")
            record = json.loads(provenance.read_text(encoding="utf-8"))
            if (record.get("source_video_sha256") != entry["sha256"]
                    or record.get("base_segmentation_sha256") != sha256_file(base)):
                raise ValueError(f"base mask provenance mismatch: {case}")
            expected_method = ("DINOv2-guided SAM3 full-video single-link segmentation"
                               if expected_names else "DINOv2-guided SAM3 full-video six-link segmentation")
            if spec.generate_base_masks and record.get("method") != expected_method:
                raise ValueError(f"base mask was not generated in this run: {case}")
    if assets:
        for path, label in ((spec.sam_python, "SAM Python"),
                            (spec.pdi_python, "PDI Python"),
                            (spec.tracker_checkpoint, "CoTracker checkpoint"),
                            (ROOT / "assets/replay/plotly.min.js", "offline replay library"),
                            (ROOT / "third_party/mega_sam/checkpoints/megasam_final.pth", "MegaSAM checkpoint")):
            checked_file(path, label)
        if spec.workflow != "selected45_link5_only":
            checked_file(spec.qwen_python, "Qwen Python")
            if not spec.qwen_model.is_dir():
                raise FileNotFoundError(f"Qwen model is missing: {spec.qwen_model}")
        if spec.generate_base_masks:
            for path, label in ((spec.segmentation_python, "segmentation Python"),
                                (spec.sam3_checkpoint, "SAM3 checkpoint"),
                                (spec.sam3_bpe, "SAM3 BPE"),
                                (spec.dinov2_model / "model.safetensors", "DINOv2 weights")):
                checked_file(path, label)
            if not spec.reference_dir.is_dir():
                raise FileNotFoundError(f"reference directory is missing: {spec.reference_dir}")
            runtime = os.environ.copy()
            runtime["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}"
            probes = [
                (spec.segmentation_python,
                 "import sys, torch, sam3, transformers; "
                 "from pathlib import Path; "
                 "from pdi_eval.perception.dinov2_reference_boxes import Dinov2DenseEncoder; "
                 "Dinov2DenseEncoder(Path(sys.argv[1])); "
                 "assert torch.ones(2, device='cuda').sum().item() == 2", 
                 [str(spec.dinov2_model)], "DINOv2/SAM3"),
                (spec.pdi_python,
                 "import torch, pdi_eval.experiment.score_v1; "
                 "assert torch.ones(2, device='cuda').sum().item() == 2",
                 [], "V1 scorer"),
            ]
            if spec.workflow != "selected45_link5_only":
                probes.append((spec.qwen_python,
                    "import sys, torch, transformers; from transformers import AutoConfig; "
                    "AutoConfig.from_pretrained(sys.argv[1], local_files_only=True); "
                    "import persistent_masking.pipeline; "
                    "assert torch.ones(2, device='cuda').sum().item() == 2",
                    [str(spec.qwen_model)], "persistent masking/Qwen"))
            for python, script, arguments, label in probes:
                result = subprocess.run([str(python), "-c", script, *arguments],
                                        cwd=ROOT, env=runtime, capture_output=True,
                                        text=True, timeout=120, check=False)
                if result.returncode:
                    raise RuntimeError(f"{label} runtime preflight failed: {result.stderr[-2000:]}")
        if spec.workflow == "selected45_link5_link7_four_way":
            for path, label in ((spec.tapip3d_python, "TAPIP3D Python"),
                                (spec.tapip3d_checkpoint, "TAPIP3D checkpoint"),
                                (spec.tapip3d_repository / "utils/inference_utils.py", "TAPIP3D source"),
                                (spec.output_root / "historical_references.json", "historical link references")):
                checked_file(path, label)
    return {"cases": len(selection["videos"]), "pipeline": "v1",
            "concurrency": spec.concurrency, "output_root": str(spec.output_root)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("select", help="freeze ten workbook/source matches")
    for name in ("workbook", "source-manifest", "cases", "output-root"):
        selection.add_argument("--" + name, type=Path, required=True)
    matched = sub.add_parser("select-matched", help="freeze the 45 matched workbook rows")
    for name in ("workbook", "source-manifest", "output-root"):
        matched.add_argument("--" + name, type=Path, required=True)
    for name, help_text in (("check", "verify selected inputs and model assets"),
                            ("run", "run the frozen experiment"),
                            ("resume", "resume incomplete cases"),
                            ("status", "inspect case states without starting models"),
                            ("export", "refresh the replay index and summary"),
                            ("replay-pairs", "refresh V1 selected-pair evidence and HTML")):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--spec", type=Path, required=True)
        if name in {"run", "resume"}:
            command.add_argument("--sample", action="append", default=[],
                                 help="four-way profile: run only this frozen sample ID")
    sub.add_parser("score", help="score one video with V1; use --help for options")
    for name in ("video-manifest", "batch-v1", "export-v1-batch", "merge-mask",
                 "migrate-checkpoint"):
        sub.add_parser(name, help=f"{name} operation; use --help for options")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough = {"video-manifest": "video_manifest", "batch-v1": "batch_v1",
                   "export-v1-batch": "export_v1_batch", "merge-mask": "mask_merge",
                   "migrate-checkpoint": "migration"}
    if argv and argv[0] in passthrough:
        from importlib import import_module
        return import_module(f".{passthrough[argv[0]]}", __package__).main(argv[1:])
    if argv and argv[0] == "score":
        if any(arg == "--version" or arg.startswith("--version=") for arg in argv[1:]):
            raise SystemExit("score no longer accepts --version; V1 is the only pipeline")
        from .score_v1 import main as score
        return score(argv[1:])
    args = build_parser().parse_args(argv)
    if args.command == "select":
        return select(args)
    if args.command == "select-matched":
        from .matched_selection import build_selection
        payload = build_selection(args.workbook, args.source_manifest, args.output_root)
        print(f"Selected {payload['video_count']} matched videos")
        return 0
    spec = load_spec(args.spec)
    if getattr(args, "sample", []) and spec.workflow not in {"selected45_link5_link7_four_way", "selected45_link5_only"}:
        raise ValueError("--sample is supported only for selected-45 profiles")
    if args.command == "check":
        print(json.dumps(check(args.spec), indent=2))
        return 0
    selection = read_selection(spec.output_root)
    if args.command in ("status", "export"):
        if args.command == "export":
            if spec.workflow == "selected45_link5_link7_four_way":
                from .link5_link7_four_way import export_index
            elif spec.workflow == "selected45_link5_only":
                from .link5_only import export_index
                export_index(spec.output_root, selection)
            else:
                page_index(spec.output_root, selection)
        print(json.dumps(status_summary(spec.output_root, selection), indent=2))
        return 0
    if args.command == "replay-pairs":
        if spec.workflow == "selected45_link5_link7_four_way":
            raise ValueError("four-way pair replays are exported by each native scorer path")
        from .pair_export import refresh_pair_replays
        print(json.dumps(refresh_pair_replays(spec), indent=2))
        return 0
    check(args.spec)
    if args.command == "run" and any(
        (spec.output_root / "cases" / f"{entry['dataset']}_{entry['video_number']}" / "status.json").exists()
        for entry in selection["videos"]
        if not args.sample or f"{entry['dataset']}_{entry['video_number']}" in args.sample
    ):
        raise FileExistsError("case status exists; use resume")
    if spec.workflow == "selected45_link5_link7_four_way":
        from .link5_link7_four_way import run_experiment
    elif spec.workflow == "selected45_link5_only":
        from .link5_only import run_experiment
    else:
        from .runner import run_experiment
    return run_experiment(spec, samples=args.sample) if spec.workflow in {"selected45_link5_link7_four_way", "selected45_link5_only"} else run_experiment(spec)


if __name__ == "__main__":
    raise SystemExit(main())
