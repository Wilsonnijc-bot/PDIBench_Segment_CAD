#!/usr/bin/env python3
"""Run a resumable two-worker SAM3 and exact-group batch entirely on the GPU host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from export_batch_metrics_csv import LINK_NAMES, export_batch_csv


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_segmentation(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            names = tuple(str(value) for value in archive["object_names"].tolist())
            return names == LINK_NAMES and archive["object_masks"].shape[1] == len(LINK_NAMES)
    except (OSError, KeyError, ValueError):
        return False


def valid_metrics(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        metrics = read_json(path)
        names = tuple(metrics["modes"]["exact-group"]["objects"])
        return names == LINK_NAMES
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


class BatchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.batch_root = args.batch_root.resolve()
        self.code_root = args.code_root.resolve()
        self.gpu_root = args.gpu_root.resolve()
        self.manifest_path = args.manifest.resolve()
        self.manifest = read_json(self.manifest_path)
        self.csv_path = self.batch_root / "metrics.csv"
        self.csv_lock = threading.Lock()
        self.print_lock = threading.Lock()
        self.pdi_python = self.gpu_root / "env/pdi-bench/bin/python"
        self.sam3_python = self.gpu_root / "env/sam3/bin/python"
        self.reference_dir = self.batch_root / "references"

    def log(self, message: str) -> None:
        with self.print_lock:
            print(f"[{now()}] {message}", flush=True)

    def export_csv(self) -> None:
        with self.csv_lock:
            export_batch_csv(self.manifest_path, self.batch_root, self.csv_path)

    def command(
        self, command: list[str], log_path: Path, environment: dict[str, str]
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"\n[{now()}] $ {' '.join(command)}\n")
            stream.flush()
            result = subprocess.run(
                command,
                cwd=self.code_root,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed with status {result.returncode}; see {log_path}"
            )

    def clean_transient_geometry(self, job_id: str, job_root: Path) -> None:
        mega_root = self.code_root / "third_party/mega_sam"
        paths = [
            job_root / "geometry-cache",
            mega_root / "work_space" / job_id,
            mega_root / "outputs" / job_id,
            mega_root / "outputs_cvd" / job_id,
        ]
        for path in paths:
            if path.exists():
                shutil.rmtree(path)

    def run_job(self, entry: dict[str, Any]) -> str:
        job_id = entry["job_id"]
        job_root = self.batch_root / "jobs" / job_id
        status_path = job_root / "status.json"
        metrics_path = job_root / "output/metrics.json"
        if valid_metrics(metrics_path):
            self.log(f"skip complete {job_id}")
            return "complete"

        staged_video = self.batch_root / "videos" / entry["staged_relative_path"]
        status = {
            **entry,
            "state": "running",
            "started_at": now(),
            "worker_thread": threading.current_thread().name,
        }
        write_json(status_path, status)
        self.export_csv()
        self.log(f"start {job_id}")
        try:
            if not staged_video.is_file():
                raise FileNotFoundError(f"staged video is missing: {staged_video}")
            if staged_video.stat().st_size != entry["size_bytes"]:
                raise RuntimeError(f"staged size mismatch: {staged_video}")
            if sha256_file(staged_video) != entry["sha256"]:
                raise RuntimeError(f"staged SHA-256 mismatch: {staged_video}")

            input_dir = job_root / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            video_alias = input_dir / f"{job_id}.mp4"
            if video_alias.is_symlink() or video_alias.exists():
                video_alias.unlink()
            video_alias.symlink_to(staged_video)
            intermediate = job_root / "intermediate"
            segmentation = intermediate / "segmentation.npz"

            common_env = os.environ.copy()
            common_env["PYTHONPATH"] = str(self.code_root / "src")
            sam3_env = common_env.copy()
            sam3_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            if not valid_segmentation(segmentation):
                self.command(
                    [
                        str(self.sam3_python),
                        "-m",
                        "pdi_eval.perception.sam3_dinov2_segment",
                        "--input",
                        str(video_alias),
                        "--reference-dir",
                        str(self.reference_dir),
                        "--output-npz",
                        str(segmentation),
                        "--dinov2-model",
                        str(self.gpu_root / "models/dinov2/base-f9e44c814b77203eaa57a6bdbbd535f21ede1415"),
                        "--sam3-checkpoint",
                        str(self.gpu_root / "models/sam3/sam3.pt"),
                        "--sam3-bpe",
                        str(self.gpu_root / "models/sam3/bpe_simple_vocab_16e6.txt.gz"),
                        "--text-prompt",
                        "visual",
                        "--link-text-prompt",
                        "link4=entire white oval on top of the black circle",
                        "--link-text-prompt",
                        "link5=entire white elongated robot arm link surrounding and to the right of the black inset",
                        "--link-text-prompt",
                        "link7=entire white quadrangular robot gripper",
                        "--require-franka-links",
                        "--reference-spatial-priors",
                        "--padding-fraction",
                        "0.10",
                        "--minimum-tracked-fraction",
                        "0.0",
                    ],
                    job_root / "sam3.log",
                    sam3_env,
                )
            if not valid_segmentation(segmentation):
                raise RuntimeError("SAM3 output does not contain exactly links 2-7")

            output_dir = job_root / "output"
            self.command(
                [
                    str(self.pdi_python),
                    str(self.code_root / "evaluation/run_multi_object.py"),
                    "--config",
                    str(self.code_root / "configs/default.yaml"),
                    "--input",
                    str(video_alias),
                    "--segmentation-npz",
                    str(segmentation),
                    "--output-dir",
                    str(output_dir),
                    "--geometry-cache-dir",
                    str(job_root / "geometry-cache"),
                    "--tracker-checkpoint",
                    str(self.gpu_root / "models/tracker/scaled_offline.pth"),
                    "--tracking-mode",
                    "exact-group",
                    "--disable-replay",
                ],
                job_root / "pdi.log",
                common_env,
            )
            if not valid_metrics(metrics_path):
                raise RuntimeError("PDI output does not contain exact-group metrics for links 2-7")
            status.update(state="complete", completed_at=now(), error="")
            write_json(status_path, status)
            self.log(f"complete {job_id}")
            return "complete"
        except Exception as exc:  # Continue the corpus after isolated video failures.
            status.update(
                state="failed",
                completed_at=now(),
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            write_json(status_path, status)
            self.log(f"failed {job_id}: {exc}")
            return "failed"
        finally:
            self.clean_transient_geometry(job_id, job_root)
            self.export_csv()

    def run(self) -> int:
        self.batch_root.mkdir(parents=True, exist_ok=True)
        self.export_csv()
        entries = self.manifest["videos"]
        self.log(f"batch start videos={len(entries)} workers={self.args.workers}")
        counts = {"complete": 0, "failed": 0}
        with ThreadPoolExecutor(max_workers=self.args.workers, thread_name_prefix="gpu") as pool:
            futures = {pool.submit(self.run_job, entry): entry for entry in entries}
            for future in as_completed(futures):
                outcome = future.result()
                counts[outcome] += 1
                self.log(
                    f"progress complete={counts['complete']} failed={counts['failed']} "
                    f"remaining={len(entries) - sum(counts.values())}"
                )
        self.export_csv()
        summary = {
            "status": "complete",
            "completed_at": now(),
            "videos": len(entries),
            **counts,
            "csv": str(self.csv_path),
        }
        write_json(self.batch_root / "summary.json", summary)
        self.log(json.dumps(summary, sort_keys=True))
        return 0 if counts["failed"] == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-root", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 2:
        parser.error("workers must be 1 or 2 on the configured 40 GB GPU")
    return BatchRunner(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
