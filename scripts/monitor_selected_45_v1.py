#!/usr/bin/env python3
"""Copy completed selected-45 cases and offload verified remote geometry caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import time
from pathlib import Path
from zipfile import ZipFile


PROJECT = Path(__file__).resolve().parents[1]
LOCAL = PROJECT / "results/selected-45-v1"
REMOTE = Path("/root/autodl-tmp/pdi/experiments/v1-v2-persistent-10-20260923/selected-45-v1")
IDENTITY = PROJECT / ".tmp/autodl_pdi_ed25519"
HOST = "root@region-9.autodl.pro"
PORT = "26211"
SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2")
SSH = ["ssh", "-i", str(IDENTITY), *SSH_OPTIONS, "-p", PORT, HOST]
RSYNC_SSH = (f"ssh -i {shlex.quote(str(IDENTITY))} "
             "-o BatchMode=yes -o ConnectTimeout=10 "
             f"-o ServerAliveInterval=15 -o ServerAliveCountMax=2 -p {PORT}")
HASH_SCRIPT = (
    "import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
    "files=(f for f in p.rglob('*') if f.is_file() and not f.is_symlink()); "
    "print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() "
    "for f in files},sort_keys=True))"
)
PRUNE_SCRIPT = (
    "import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); "
    "s=json.loads((p/'status.json').read_text()); "
    "assert s['state']=='complete'; "
    "cache=p/'geometry-cache'; "
    "[(f.unlink()) for f in cache.glob('*.npz') if f.is_file()]"
)


def remote(*arguments: str) -> str:
    result = subprocess.run([*SSH, shlex.join(arguments)], capture_output=True,
                            text=True, check=True)
    return result.stdout


def copy(source: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--partial", "--timeout=300", "-e", RSYNC_SSH,
                    f"{HOST}:{source}", str(destination)], check=True)


def local_hashes(case: Path) -> dict[str, str]:
    return {str(path.relative_to(case)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in case.rglob("*") if path.is_file() and not path.is_symlink()}


def verify_case(case: Path, expected_source_hash: str) -> None:
    status = json.loads((case / "status.json").read_text(encoding="utf-8"))
    manifest = json.loads((case / "v1/manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((case / "v1/metrics.json").read_text(encoding="utf-8"))
    timing = json.loads((case / "v1/timing.json").read_text(encoding="utf-8"))
    objects = metrics.get("modes", {}).get("exact-group", {}).get("objects", {})
    expected_links = {f"link{i}" for i in range(2, 8)}
    if set(objects) != expected_links:
        raise ValueError(f"missing V1 link reports: {case}")
    scored = {name for name, report in objects.items()
              if report.get("status") == "complete"}
    if not scored or set(status.get("scored_links", sorted(scored))) != scored:
        raise ValueError(f"scored-link status differs from V1 metrics: {case}")
    required = ("base_segmentation.npz", "base_segmentation_source.json",
                "v1/cotracker_exact-group.npz", "v1/replay/combined_exact-group.mp4",
                "v1/replay/interactive_exact-group/index.html",
                "v1/replay/interactive_exact-group/source.mp4",
                *(f"v1/replay/interactive_exact-group/{name}_exact-group_pairs.json"
                  for name in scored))
    if not all((case / name).is_file() and (case / name).stat().st_size > 0
               for name in required):
        raise ValueError(f"incomplete local artifacts: {case}")
    if (status.get("state") != "complete" or status.get("source_sha256") != expected_source_hash
            or manifest.get("status") != "complete"
            or manifest.get("tracking_modes") != ["exact-group"]
            or manifest.get("input", {}).get("sha256") != expected_source_hash
            or timing.get("status") != "complete"):
        raise ValueError(f"inconsistent local V1 result: {case}")


def sync_base_comparison(sample: str, case: Path, expected_source_hash: str) -> bool:
    marker = case / ".base_comparison_synced.json"
    if marker.is_file():
        return False
    remote_case = REMOTE / "cases" / sample
    local_base = case / "base_v1"
    copy(str(remote_case / "base_v1") + "/", local_base)
    copy(str(remote_case / "status.json"), case)
    status = json.loads((case / "status.json").read_text(encoding="utf-8"))
    comparison_path = local_base / "comparison.json"
    comparison = (json.loads(comparison_path.read_text(encoding="utf-8"))
                  if comparison_path.is_file() else status.get("base_comparison", {}))
    if (status.get("state") != "complete"
            or (not comparison_path.is_file() and comparison.get("state") != "complete")
            or (comparison_path.is_file()
                and (comparison.get("sample_id") != sample
                     or comparison.get("source_sha256") != expected_source_hash))):
        raise ValueError(f"base comparison status is incomplete: {sample}")
    base = case / "base_segmentation.npz"
    if hashlib.sha256(base.read_bytes()).hexdigest() != comparison["base_segmentation_sha256"]:
        raise ValueError(f"base comparison mask hash differs: {sample}")
    manifest = json.loads((local_base / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((local_base / "metrics.json").read_text(encoding="utf-8"))
    objects = metrics["modes"]["exact-group"]["objects"]
    scored = {name for name, report in objects.items() if report.get("status") == "complete"}
    if (manifest.get("status") != "complete"
            or manifest.get("input", {}).get("sha256") != expected_source_hash
            or manifest.get("segmentation", {}).get("sha256") != comparison["base_segmentation_sha256"]
            or set(objects) != {f"link{i}" for i in range(2, 8)}
            or not scored or set(comparison.get("scored_links", [])) != scored):
        raise ValueError(f"base comparison metrics differ: {sample}")
    required = ("cotracker_exact-group.npz", "replay/combined_exact-group.mp4",
                "replay/interactive_exact-group/index.html",
                *(f"replay/interactive_exact-group/{name}_exact-group_pairs.json"
                  for name in scored))
    if not all((local_base / name).is_file() and (local_base / name).stat().st_size > 0
               for name in required):
        raise ValueError(f"base comparison replay or points are missing: {sample}")
    observed = json.loads(remote("python3", "-c", HASH_SCRIPT, str(remote_case / "base_v1")))
    if observed != local_hashes(local_base):
        raise ValueError(f"remote/local base comparison hashes differ: {sample}")
    remote("python3", "-c", PRUNE_SCRIPT, str(remote_case))
    marker.write_text(json.dumps({"case": sample, "verified_local": True,
                                  "file_count": len(observed)}, indent=2) + "\n",
                      encoding="utf-8")
    print(f"verified base comparison {sample}", flush=True)
    return True


def one_pass() -> tuple[int, bool]:
    LOCAL.mkdir(parents=True, exist_ok=True)
    summary_text = remote("cat", str(REMOTE / "summary.json"))
    summary = json.loads(summary_text)
    (LOCAL / "summary.json").write_text(summary_text, encoding="utf-8")
    copy(str(REMOTE / "index.html"), LOCAL)
    selection = json.loads((LOCAL / "selection.json").read_text(encoding="utf-8"))
    source_hashes = {f"{item['dataset']}_{item['video_number']}": item["sha256"]
                     for item in selection["videos"]}
    comparison_samples = set(json.loads(remote(
        "python3", "-c",
        "import json,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
        "print(json.dumps([p.parent.parent.name for p in root.glob('*/base_v1/comparison.json')]))",
        str(REMOTE / "cases"),
    )))
    offloaded = 0
    for sample, state in summary["cases"].items():
        if state.get("state") != "complete":
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sample):
            raise ValueError(f"invalid case name: {sample}")
        case = LOCAL / "cases" / sample
        marker = case / ".geometry_offloaded.json"
        if marker.exists():
            if (state.get("base_comparison", {}).get("state") == "complete"
                    or sample in comparison_samples):
                sync_base_comparison(sample, case, source_hashes[sample])
            continue
        print(f"copying {sample}", flush=True)
        copy(str(REMOTE / "cases" / sample) + "/", case)
        verify_case(case, source_hashes[sample])
        remote_case = REMOTE / "cases" / sample
        observed = json.loads(remote("python3", "-c", HASH_SCRIPT, str(remote_case)))
        if observed != local_hashes(case):
            raise ValueError(f"remote/local file hashes differ: {sample}")
        cache_files = [name for name in observed if name.startswith("geometry-cache/")
                       and name.endswith(".npz")]
        if len(cache_files) != 1:
            raise ValueError(f"completed case must have one geometry archive: {sample}")
        with ZipFile(case / cache_files[0]) as archive:
            fields = {name.removesuffix(".npy") for name in archive.namelist()}
        if not {"pointmaps", "camera_poses", "focal_length"}.issubset(fields):
            raise ValueError(f"incomplete MegaSAM geometry archive: {sample}")
        remote("python3", "-c", PRUNE_SCRIPT, str(remote_case))
        marker.write_text(json.dumps({"case": sample, "cache_files": cache_files,
                                      "verified_local": True}, indent=2) + "\n",
                          encoding="utf-8")
        if (state.get("base_comparison", {}).get("state") == "complete"
                or sample in comparison_samples):
            sync_base_comparison(sample, case, source_hashes[sample])
        offloaded += 1
        print(f"verified and offloaded {sample}", flush=True)
    exit_code = remote("python3", "-c",
                       "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
                       "print(p.read_text().strip() if p.exists() else 'RUNNING')",
                       str(REMOTE / "run.exit")).strip()
    if exit_code != "RUNNING":
        copy(str(REMOTE / "run.log"), LOCAL)
        copy(str(REMOTE / "run.exit"), LOCAL)
    print(f"cases: {summary['counts']}; offloaded this pass: {offloaded}; runner: {exit_code}",
          flush=True)
    return offloaded, exit_code != "RUNNING"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            _, finished = one_pass()
            if finished or not args.loop:
                return
        except Exception as error:
            print(f"monitor error: {error}", flush=True)
            if not args.loop:
                raise
        time.sleep(45)


if __name__ == "__main__":
    main()
