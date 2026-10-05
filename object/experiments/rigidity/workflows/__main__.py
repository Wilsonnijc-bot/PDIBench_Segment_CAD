"""Prepare, run, and inspect the selected-45 task-object rigidity batch."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

from object.preprocessing.selection.selection import build_manifest, sha256, write_json

BENCHMARK = (_workspace_root())
DATASET_FOLDERS = {"LVP_ROBOWM": "LVP", "COSMOS2.5": "COSMOS2.5", "COSMOS3": "COSMOS3"}


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def prepare(args) -> None:
    manifest = build_manifest(args.workbook, args.selection)
    target = args.output_root / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError(f"prepared manifest already differs: {target}")
    write_json(target, manifest)
    print(json.dumps({"status": "prepared", "videos": 45, "manifest": str(target)}))


def _invoke(command: list[str], log: Path, *, environment: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        result = subprocess.run(command, cwd=BENCHMARK, env=environment,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
        stream.write(f"EXIT_STATUS={result.returncode}\n")
    return result.returncode


def _summary(root: Path, manifest: dict) -> dict:
    lock_path = root / "summary.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _write_summary(root, manifest)


def _write_summary(root: Path, manifest: dict) -> dict:
    rows = []
    for entry in manifest["videos"]:
        path = root / "cases" / entry["case"] / "status.json"
        state = json.loads(path.read_text()) if path.is_file() else {"status": "pending"}
        rows.append({"case": entry["case"], "workbook_row": entry["workbook_row"],
                     "status": state.get("status", "pending"),
                     "target_object": state.get("target_object", ""),
                     "rigidity_score": state.get("rigidity_score", ""),
                     "reason": state.get("reason", "")})
    with (root / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"video_count": len(rows), "complete": sum(r["status"] == "complete" for r in rows),
               "unscorable": sum(r["status"] == "unscorable" for r in rows), "cases": rows}
    write_json(root / "summary.json", summary)
    return summary


def _replay_ready(folder: Path, scored: dict | None = None) -> bool:
    replay = folder / "replay"
    required = (replay / "index.html", replay / "task_object_exact-group.html",
                replay / "task_object_exact-group_pairs.json", replay / "source.mp4",
                replay / "plotly.min.js", replay / "replay.json")
    if not all(path.is_file() and path.stat().st_size for path in required):
        return False
    try:
        record = json.loads((replay / "replay.json").read_text())
        return (record.get("status") == "complete" and
                (scored is None or record.get("rigidity_score") == scored.get("rigidity_score")))
    except (OSError, ValueError):
        return False


def run(args) -> int:
    root = args.output_root
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("video_count") != 45:
        raise ValueError("expected the prepared 45-video manifest")
    if args.case:
        known = {entry["case"] for entry in manifest["videos"]}
        unknown = set(args.case) - known
        if unknown:
            raise ValueError(f"unknown case(s): {sorted(unknown)}")
    for asset in (args.sam_python, args.sam3_checkpoint, args.sam3_bpe,
                  args.tracker_checkpoint):
        if not asset.exists():
            raise FileNotFoundError(asset)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str((_workspace_root())) + os.pathsep + str(BENCHMARK)
    for entry in manifest["videos"]:
        case = entry["case"]
        if args.case and case not in args.case:
            continue
        folder = root / "cases" / case
        folder.mkdir(parents=True, exist_ok=True)
        status_path = folder / "status.json"
        prior = json.loads(status_path.read_text()) if status_path.is_file() else {}
        if (prior.get("status") == "unscorable" or
                prior.get("status") == "complete" and _replay_ready(folder)):
            print(json.dumps({"case": case, "status": "already_" + prior["status"]}), flush=True)
            continue
        source = args.video_root / DATASET_FOLDERS[entry["dataset"]] / f"{entry['video_number']}.mp4"
        state = {"case": case, "status": "running", "workbook_row": entry["workbook_row"],
                 "workbook_sha256": manifest["workbook_sha256"],
                 "generation_prompt": entry["generation_prompt"],
                 "source_video": str(source), "expected_video_sha256": entry["sha256"]}
        write_json(status_path, state)
        try:
            if not source.is_file() or source.stat().st_size != entry["size_bytes"] or sha256(source) != entry["sha256"]:
                raise ValueError("source video differs from frozen selection")
            alias = folder / f"{case.replace('.', '_')}.mp4"
            if not alias.exists():
                alias.symlink_to(source.resolve())
            if sha256(alias) != entry["sha256"]:
                raise ValueError("case video alias differs from frozen selection")
            prompt_file = folder / "generation_prompt.txt"
            prompt_file.write_text(entry["generation_prompt"] + "\n", encoding="utf-8")
            masking = folder / "masking"
            grounding_file = masking / "grounding.json"
            segmentation = masking / "segmentation.npz"
            grounding = json.loads(grounding_file.read_text()) if grounding_file.is_file() else {}
            if (grounding.get("status") != "complete" or
                    not segmentation.is_file() or
                    sha256(segmentation) != grounding.get("segmentation_sha256") or
                    grounding.get("video_sha256") != entry["sha256"]):
                with gpu_lock(args.gpu_lock):
                    code = _invoke([
                        str(args.sam_python), "-m", "object.preprocessing.segmentation.segment",
                        "--video", str(alias), "--prompt-file", str(prompt_file),
                        "--output", str(masking), "--sam3-checkpoint", str(args.sam3_checkpoint),
                        "--sam3-bpe", str(args.sam3_bpe),
                    ], folder / "segment.log", environment=environment)
                grounding = json.loads(grounding_file.read_text()) if grounding_file.is_file() else {}
                if code or grounding.get("status") != "complete":
                    raise ValueError("grounding or SAM3 failed; see masking/grounding.json and segment.log")
            state["target_object"] = grounding["target_object"]
            state["segmentation_sha256"] = sha256(segmentation)
            write_json(status_path, state)
            score_dir = folder / "score"
            score_path = score_dir / "rigidity.json"
            scored = json.loads(score_path.read_text()) if score_path.is_file() else {}
            if (scored.get("status") != "complete" or
                    scored.get("input", {}).get("video_sha256") != entry["sha256"] or
                    scored.get("input", {}).get("segmentation_sha256") != state["segmentation_sha256"]):
                with gpu_lock(args.gpu_lock):
                    code = _invoke([
                        str(args.pdi_python), "-m", "object.experiments.rigidity.scoring.score",
                        "--video", str(alias), "--segmentation", str(segmentation),
                        "--output", str(score_dir), "--geometry-cache", str(folder / "geometry-cache"),
                        "--tracker-checkpoint", str(args.tracker_checkpoint),
                    ], folder / "score.log", environment=environment)
                if code:
                    raise ValueError("rigidity scoring failed; see score.log")
                scored = json.loads(score_path.read_text())
            if not _replay_ready(folder, scored):
                code = _invoke([
                    str(args.pdi_python), "-m", "object.experiments.rigidity.replay.replay",
                    "--score", str(score_path),
                    "--tracks", str(score_dir / "cotracker_exact-group.npz"),
                    "--segmentation", str(segmentation),
                    "--pointmaps", str(scored["geometry"]["cache_path"]),
                    "--video", str(alias), "--output", str(folder / "replay"),
                    "--target-name", grounding["target_object"],
                ], folder / "replay.log", environment=environment)
                if code or not _replay_ready(folder, scored):
                    raise ValueError("interactive rigidity replay failed; see replay.log")
            state.update(status="complete", rigidity_score=scored["rigidity_score"],
                         rigidity_strategy=scored["rigidity_strategy"],
                         interactive_replay="replay/index.html")
        except Exception as exc:
            state.update(status="unscorable", reason=str(exc), error_type=type(exc).__name__)
        finally:
            write_json(status_path, state)
            summary = _summary(root, manifest)
            print(json.dumps({"case": case, "status": state["status"],
                              "complete": summary["complete"],
                              "unscorable": summary["unscorable"]}), flush=True)
    summary = _summary(root, manifest)
    if args.case:
        states = {row["case"]: row["status"] for row in summary["cases"]}
        return 0 if all(states[name] in {"complete", "unscorable"} for name in args.case) else 1
    return 0 if summary["complete"] + summary["unscorable"] == 45 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepared = sub.add_parser("prepare")
    prepared.add_argument("--workbook", type=Path, required=True)
    prepared.add_argument("--selection", type=Path, required=True)
    prepared.add_argument("--output-root", type=Path, required=True)
    running = sub.add_parser("run")
    running.add_argument("--output-root", type=Path, required=True)
    running.add_argument("--video-root", type=Path, required=True)
    running.add_argument("--sam-python", type=Path, required=True)
    running.add_argument("--pdi-python", type=Path, default=Path(sys.executable))
    running.add_argument("--sam3-checkpoint", type=Path, required=True)
    running.add_argument("--sam3-bpe", type=Path, required=True)
    running.add_argument("--tracker-checkpoint", type=Path, required=True)
    running.add_argument("--gpu-lock", type=Path, required=True)
    running.add_argument("--case", action="append")
    checking = sub.add_parser("status")
    checking.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return 0
    if args.command == "run":
        return run(args)
    manifest = json.loads((args.output_root / "manifest.json").read_text())
    print(json.dumps(_summary(args.output_root, manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
