#!/usr/bin/env python3
"""Resume matched Link 7 tracker runs and exact V1 pair replays for selected 45.

This local driver stages each saved MegaSAM cache on the GPU host, runs the
unchanged V1 scorer once per tracker, exports the original interactive pair
replay through a wrapper, and copies the artifacts back. It never regenerates
geometry or edits the original selected-45 result folders locally.
"""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.pdibench.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import argparse
import hashlib
import os
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = (_workspace_root())
SELECTED = (_workspace_root() / 'results/selected-45-v1')
OUTPUT = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison')
HOST = "root@region-9.autodl.pro"
PORT = "26211"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
REMOTE_CODE = os.environ.get("PDI_REMOTE_CODE_ROOT", "/root/autodl-tmp/pdi/code/workspace")
PDI_PYTHON = "/root/autodl-tmp/pdi/env/pdi-bench/bin/python"
TAPIP_PYTHON = "/root/autodl-tmp/pdi/env/tapip3d/bin/python"
TAPIP_REPO = "/root/autodl-tmp/pdi/code/TAPIP3D"
TAPIP_CHECKPOINT = "/root/autodl-tmp/pdi/models/tapip3d/tapip3d_final.pth"
TRACKER_CHECKPOINT = "/root/autodl-tmp/pdi/models/tracker/scaled_offline.pth"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ssh(command: list[str], log) -> str:
    remote = " ".join(shlex.quote(str(arg)) for arg in command)
    result = subprocess.run(
        ["ssh", "-i", str(KEY), "-p", PORT, HOST, remote],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log.write(f"$ {remote}\n{result.stdout}\n")
    log.flush()
    if result.returncode:
        raise RuntimeError(f"remote command failed ({result.returncode}): {remote}\n{result.stdout[-2000:]}")
    return result.stdout.strip()


def _remote_exists(path: str, log) -> bool:
    result = subprocess.run(
        ["ssh", "-i", str(KEY), "-p", PORT, HOST,
         "test -f " + shlex.quote(path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(f"remote existence check failed: {path}: {result.stdout}")
    log.write(f"remote exists={result.returncode == 0}: {path}\n")
    return result.returncode == 0


def _rsync(source: str, target: str, log, *, exclude: str | None = None) -> None:
    command = ["rsync", "-a", "--checksum"]
    if exclude:
        command += ["--exclude", exclude]
    command += ["-e", f"ssh -i {KEY} -p {PORT}", source, target]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    log.write(f"$ {' '.join(command)}\n{result.stdout}\n")
    log.flush()
    if result.returncode:
        raise RuntimeError(f"rsync failed ({result.returncode}): {result.stdout[-2000:]}")


def _run_case(case: Path, log) -> dict:
    sample = case.name
    metrics = json.loads((case / "v1/metrics.json").read_text(encoding="utf-8"))
    remote_video = metrics["video"]
    remote_cache = metrics["geometry"]["cache_path"]
    remote_v1 = str(Path(remote_cache).parent.parent / "v1")
    local_cache = case / "geometry-cache" / Path(remote_cache).name
    if not local_cache.is_file():
        raise FileNotFoundError(f"saved MegaSAM cache is missing: {local_cache}")
    cached_before = _remote_exists(remote_cache, log)
    if not cached_before:
        _rsync(str(local_cache), HOST + ":" + remote_cache, log)
    remote_hash = _ssh(["sha256sum", remote_cache], log).split()[0]
    if remote_hash != _sha256(local_cache):
        raise ValueError(f"staged MegaSAM cache hash differs: {sample}")
    common = [PDI_PYTHON, "-m", "robot.workflows", "score",
              "--input", remote_video,
              "--segmentation-npz", f"{remote_v1}/segmentation.npz",
              "--geometry-cache-dir", str(Path(remote_cache).parent),
              "--tracker-checkpoint", TRACKER_CHECKPOINT,
              "--tracking-mode", "exact-group", "--disable-replay",
              "--config", f"{remote_v1}/run_config.yaml"]
    for backend, folder, archive in (
        ("cotracker3", "link7_cotracker3", "cotracker_exact-group.npz"),
        ("tapip3d", "link7_tapip3d", "tapip3d_link7_exact-group.npz"),
    ):
        remote_folder = f"{remote_v1}/{folder}"
        if not (_remote_exists(f"{remote_folder}/manifest.json", log)
                and _remote_exists(f"{remote_folder}/{archive}", log)
                and _remote_exists(f"{remote_folder}/link7_initial_queries.npz", log)):
            command = ["env", "PYTHONPATH=.", *common, "--link7-tracker", backend,
                       "--output-dir", remote_folder]
            if backend == "tapip3d":
                command += ["--tapip3d-python", TAPIP_PYTHON,
                            "--tapip3d-repository", TAPIP_REPO,
                            "--tapip3d-checkpoint", TAPIP_CHECKPOINT]
            _ssh(["bash", "-lc", "cd " + shlex.quote(REMOTE_CODE) + " && " +
                  " ".join(shlex.quote(arg) for arg in command)], log)
        remote_metrics = json.loads(_ssh(
            [PDI_PYTHON, "-c",
             "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps({'cache_hit':d['geometry']['cache_hit'],'link7':d['modes']['exact-group']['objects']['link7']['status']}))",
             f"{remote_folder}/metrics.json"], log))
        if not remote_metrics["cache_hit"]:
            raise ValueError(f"{backend} did not reuse the saved MegaSAM cache: {sample}")
    replay_folder = f"{remote_v1}/link7_pair_replay"
    if not _remote_exists(f"{replay_folder}/manifest.json", log):
        command = [PDI_PYTHON, "-m", "infrastructure.shared.replay.link7_pair_replay",
                   "--video", remote_video,
                   "--segmentation", f"{remote_v1}/segmentation.npz",
                   "--pointmaps", remote_cache,
                   "--cotracker-dir", f"{remote_v1}/link7_cotracker3",
                   "--tapip3d-dir", f"{remote_v1}/link7_tapip3d",
                   "--output-dir", replay_folder]
        _ssh(["bash", "-lc", "cd " + shlex.quote(REMOTE_CODE) +
              " && PYTHONPATH=. " + " ".join(shlex.quote(arg) for arg in command)], log)
    local = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/cases') / sample
    for remote_name, local_name, exclude in (
        ("link7_cotracker3", "cotracker3", None),
        ("link7_tapip3d", "tapip3d", "link7_tapip3d_input.npz"),
        ("link7_pair_replay", "replay", None),
    ):
        (local / local_name).mkdir(parents=True, exist_ok=True)
        _rsync(f"{HOST}:{remote_v1}/{remote_name}/", str(local / local_name) + "/",
               log, exclude=exclude)
    replay = json.loads((local / "replay/manifest.json").read_text(encoding="utf-8"))
    if not replay["identical_initial_queries"] or replay["selected_point_count"] < 2:
        raise ValueError(f"invalid local pair replay manifest: {sample}")
    if not cached_before:
        _ssh(["rm", "--", remote_cache], log)
    return {"state": "complete", "selected_points": replay["selected_point_count"],
            "methods": replay["methods"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", help="Run a named case; repeat as needed")
    args = parser.parse_args()
    if not KEY.is_file():
        parser.error(f"SSH key is missing: {KEY}")
    cases = sorted(((_workspace_root() / 'results/selected-45-v1/cases')).iterdir())
    selected = set(args.case or [])
    cases = [case for case in cases if case.is_dir() and
             (not selected or case.name in selected) and
             (case / "status.json").is_file() and
             json.loads((case / "status.json").read_text())["state"] == "complete"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary_path = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/summary.json')
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {"cases": {}}
    for case in cases:
        manifest = (_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/cases') / case.name / "replay/manifest.json"
        if manifest.is_file() and summary["cases"].get(case.name, {}).get("state") == "complete":
            continue
        ((_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/logs')).mkdir(exist_ok=True)
        with ((_workspace_root() / 'results/selected-45-v1/link7-tracker-comparison/logs') / f"{case.name}.log").open("a", encoding="utf-8") as log:
            log.write(f"\nStarted {datetime.now(timezone.utc).isoformat()}\n")
            try:
                result = _run_case(case, log)
            except Exception as exc:
                result = {"state": "failed", "error": str(exc)}
                log.write(f"ERROR: {exc}\n")
        summary["cases"][case.name] = result
        summary["updated_at"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{case.name}: {result['state']}", flush=True)
    return 1 if any(value["state"] == "failed" for value in summary["cases"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
