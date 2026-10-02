#!/usr/bin/env python3
"""Copy completed Link 7 comparison cases and offload verified geometry."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "results/link7-point-filter-v2-20260926"
REMOTE = "/root/autodl-tmp/pdi/experiments/link7-point-filter-v2-20260926"
HOST = "root@region-9.autodl.pro"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
SSH = ["ssh", "-i", str(KEY), "-p", "26211", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3", HOST]
RSYNC_SSH = (f"ssh -i {shlex.quote(str(KEY))} -p 26211 -o BatchMode=yes "
             "-o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3")


def remote(*args: str) -> str:
    result = subprocess.run([*SSH, shlex.join(args)], capture_output=True,
                            text=True, check=True)
    return result.stdout.strip()


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_and_verify(sample: str) -> Path:
    source = f"{HOST}:{REMOTE}/cases/{sample}/"
    target = LOCAL / "cases" / sample
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".geometry_offloaded.json"
    base = ["rsync", "-a", "--checksum", "--partial", "--timeout=300",
            "--bwlimit=8000",
            "-e", RSYNC_SSH]
    exclude = ["--exclude=geometry-cache/"] if marker.is_file() else []
    subprocess.run([*base, *exclude, source, str(target) + "/"], check=True)
    verification = subprocess.run([*base, *exclude, "--dry-run", "--itemize-changes",
                                   source, str(target) + "/"],
                                  capture_output=True, text=True, check=True)
    for line in verification.stdout.splitlines():
        code, path = line.split(maxsplit=1)
        if code.startswith(">f") and (target / path).is_file():
            observed = remote("sha256sum", f"{REMOTE}/cases/{sample}/{path}").split()[0]
            if observed == checksum(target / path):
                continue  # macOS and Linux rsync can disagree on this checksum.
        raise ValueError(f"remote/local files differ for {sample}: {line}")
    return target


def sync_case(sample: str) -> None:
    target = copy_and_verify(sample)
    marker = target / ".geometry_offloaded.json"
    status = json.loads((target / "status.json").read_text())
    if status["state"] != "complete" or not (target / "comparison.json").is_file():
        raise ValueError(f"case is not complete: {sample}")
    for path in ("path1_v1_cotracker/manifest.json",
                 "path2_v2_tapip3d/manifest.json", "comparison.json"):
        if not (target / path).is_file():
            raise ValueError(f"missing {path}: {sample}")
    caches = list((target / "geometry-cache").glob("*.npz"))
    if len(caches) != 1:
        raise ValueError(f"expected one geometry cache: {sample}")
    if not marker.is_file():
        observed = remote("sha256sum", f"{REMOTE}/cases/{sample}/geometry-cache/{caches[0].name}").split()[0]
        if observed != checksum(caches[0]):
            raise ValueError(f"geometry cache checksum differs: {sample}")
        remote("rm", "--", f"{REMOTE}/cases/{sample}/geometry-cache/{caches[0].name}")
        marker.write_text(json.dumps({
            "case": sample, "verified_local_sha256": observed,
        }, indent=2) + "\n")
    (target / ".case_synced.json").write_text(json.dumps({"case": sample,
        "all_remote_result_files_verified": True}, indent=2) + "\n")
    print(f"verified and offloaded {sample}", flush=True)


def sync_failed_case(sample: str) -> None:
    target = copy_and_verify(sample)
    status = json.loads((target / "status.json").read_text())
    if status["state"] != "failed":
        raise ValueError(f"case is not failed: {sample}")
    (target / ".case_synced.json").write_text(json.dumps({"case": sample,
        "state": "failed", "all_remote_result_files_verified": True}, indent=2) + "\n")
    print(f"verified failed case {sample}", flush=True)


def preoffload_staged_cache(sample: str) -> bool:
    """Free GPU disk using the already local, identical selected-45 cache."""
    original = list((ROOT / "results/selected-45-v1/cases" / sample
                     / "geometry-cache").glob("*.npz"))
    if len(original) != 1:
        return False
    target = LOCAL / "cases" / sample
    marker = target / ".geometry_offloaded.json"
    if marker.is_file():
        return True
    remote_path = f"{REMOTE}/cases/{sample}/geometry-cache/{original[0].name}"
    observed = remote("sha256sum", remote_path).split()[0]
    if observed != checksum(original[0]):
        raise ValueError(f"staged geometry cache differs: {sample}")
    local_cache = target / "geometry-cache" / original[0].name
    local_cache.parent.mkdir(parents=True, exist_ok=True)
    if not local_cache.is_file() or checksum(local_cache) != observed:
        shutil.copy2(original[0], local_cache)
    if checksum(local_cache) != observed:
        raise ValueError(f"local geometry cache copy differs: {sample}")
    remote("rm", "--", remote_path)
    marker.write_text(json.dumps({"case": sample,
        "verified_local_sha256": observed, "source": "selected-45-v1"}, indent=2) + "\n")
    print(f"pre-offloaded verified cache {sample}", flush=True)
    return True


def one_pass() -> bool:
    selection = json.loads((LOCAL / "selection.json").read_text())
    states = json.loads(remote("python3", "-c",
        "import json,pathlib,sys; r=pathlib.Path(sys.argv[1]); "
        "print(json.dumps({p.parent.name:json.loads(p.read_text()).get('state','pending') "
        "for p in r.glob('*/status.json')}))",
        f"{REMOTE}/cases"))
    completed = []
    failed = []
    for entry in selection["videos"]:
        sample = f"{entry['dataset']}_{entry['video_number']}"
        marker = LOCAL / "cases" / sample / ".case_synced.json"
        if marker.is_file():
            continue
        if states.get(sample) == "complete":
            completed.append(sample)
        elif states.get(sample) == "failed":
            failed.append(sample)
    for sample in completed:
        preoffload_staged_cache(sample)
    for sample in completed:
        sync_case(sample)
    for sample in failed:
        sync_failed_case(sample)
    exit_code = remote("python3", "-c",
                       "from pathlib import Path; import sys; p=Path(sys.argv[1]); print(p.read_text().strip() if p.is_file() else 'RUNNING')",
                       f"{REMOTE}/run.exit")
    print(f"runner={exit_code}", flush=True)
    return exit_code != "RUNNING"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            finished = one_pass()
        except (OSError, ValueError, subprocess.CalledProcessError) as error:
            if not args.loop:
                raise
            print(f"sync connection interrupted: {error}", flush=True)
            finished = False
        if finished or not args.loop:
            break
        time.sleep(45)
