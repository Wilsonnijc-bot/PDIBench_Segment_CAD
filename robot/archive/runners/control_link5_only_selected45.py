#!/usr/bin/env python3
"""Stage verified geometry, collect each case, and offload the remote cache."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/control_link5_only_selected45.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import time
from pathlib import Path

ROOT = _SOURCE_PATH.parents[1]
OLD = ROOT / "results/link5-link7-four-way-selected45-20260927"
LOCAL = ROOT / "results/link5-only-selected45-updated-mask-20260929"
REMOTE = "/root/autodl-tmp/pdi/experiments/link5-only-selected45-updated-mask-20260929"
HOST = "root@region-9.autodl.pro"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
SSH = ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=20", "-p", "26211", HOST]
RSYNC_SSH = f"ssh -i {shlex.quote(str(KEY))} -o BatchMode=yes -o ConnectTimeout=15 -p 26211"


def remote(*args: str) -> str:
    for attempt in range(3):
        result = subprocess.run([*SSH, shlex.join(args)], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode != 255:
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"remote command failed ({result.returncode}): {result.stderr.strip()}")


def rsync(source: str, target: str, *args: str) -> str:
    command = ["rsync", "-a", "--partial", "--timeout=300", "-e",
               RSYNC_SSH, *args, source, target]
    for attempt in range(3):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout
        if result.returncode not in {12, 30, 255}:
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"rsync failed ({result.returncode}): {result.stderr.strip()}")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_cache(sample: str) -> Path:
    files = list((OLD / "cases" / sample / "geometry-cache").glob("*.npz"))
    if len(files) != 1:
        raise ValueError(f"expected exactly one historical geometry cache: {sample}")
    return files[0]


def update_ledger(summary: dict, exit_code: int) -> None:
    ledger = ROOT / "experiment_GPU_record.md"
    source = ledger.read_text()
    heading = "## Selected-45 updated Link 5 masking and V1 rigidity — 2026-09-29"
    start = source.index(heading)
    next_heading = source.find("\n## ", start + len(heading))
    end = len(source) if next_heading < 0 else next_heading
    section = source[start:end]
    complete = summary["counts"]["complete"]
    failed = summary["counts"]["failed"]
    scores = [item.get("epsilon_rigidity") for item in summary["cases"].values()
              if isinstance(item.get("epsilon_rigidity"), (int, float))]
    median = f"; median epsilon_rigidity {statistics.median(scores):.6f}" if scores else ""
    failed_ids = [name for name, item in summary["cases"].items()
                  if item["state"] == "failed"]
    result = (f"{complete}/45 cases complete, {failed} failed; remote exit {exit_code}"
              f"{median}. All terminal cases were retrieved with SHA-256 verification."
              + (f" Failed cases: {', '.join(failed_ids)}." if failed_ids else ""))
    section = re.sub(r"(?m)^- \*\*Status:\*\*.*$",
                     f"- **Status:** {'completed' if failed == 0 and exit_code == 0 else 'failed'}.",
                     section, count=1)
    section = re.sub(r"(?m)^- \*\*Results:\*\*.*$",
                     f"- **Results:** {result}", section, count=1)
    section = re.sub(r"(?m)^- \*\*Artifacts:\*\*.*$",
                     "- **Artifacts:** Local `results/link5-only-selected45-updated-mask-20260929/index.html`, "
                     "`summary.json`, `run_status.json`, `run.log`, and per-case masks, scores, "
                     "VLM guard records, and interactive replays; remote run root recorded above.",
                     section, count=1)
    temporary = ledger.with_suffix(".md.tmp")
    temporary.write_text(source[:start] + section + source[end:])
    temporary.replace(ledger)


def stage(sample: str) -> None:
    source = source_cache(sample)
    folder = f"{REMOTE}/cases/{sample}/geometry-cache"
    remote("mkdir", "-p", folder)
    rsync(str(source), f"{HOST}:{folder}/")
    if remote("sha256sum", f"{folder}/{source.name}").split()[0] != sha(source):
        raise ValueError(f"staged geometry cache differs for {sample}")
    remote("touch", f"{folder}/ready")
    print(f"STAGED {sample}", flush=True)


def collect(sample: str, state: dict) -> None:
    local_case = LOCAL / "cases" / sample
    local_case.mkdir(parents=True, exist_ok=True)
    source = f"{HOST}:{REMOTE}/cases/{sample}/"
    options = ("--exclude=/geometry-cache/", f"--exclude=/{sample.replace('.', '_')}.mp4")
    rsync(source, str(local_case) + "/", *options)
    hash_script = ("import hashlib,json,sys; from pathlib import Path; "
                   "root=Path(sys.argv[1]); "
                   "print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() "
                   "for p in root.rglob('*') if p.is_file() and not p.is_symlink() "
                   "and 'geometry-cache' not in p.relative_to(root).parts}))")
    remote_hashes = json.loads(remote("python3", "-c", hash_script,
                                    f"{REMOTE}/cases/{sample}"))
    local_hashes = {str(path.relative_to(local_case)): sha(path)
                    for path in local_case.rglob("*")
                    if path.is_file() and not path.is_symlink()
                    and "geometry-cache" not in path.relative_to(local_case).parts}
    if local_hashes != remote_hashes:
        differences = sorted(set(local_hashes) ^ set(remote_hashes))
        differences += [name for name in local_hashes.keys() & remote_hashes.keys()
                        if local_hashes[name] != remote_hashes[name]]
        raise ValueError(f"case transfer failed SHA-256 verification: {sample}: {differences[:5]}")
    if json.loads((local_case / "status.json").read_text())["state"] != state["state"]:
        raise ValueError(f"case state changed during transfer: {sample}")
    if state["state"] == "complete":
        output = local_case / "v1_cotracker3"
        manifest = json.loads((output / "manifest.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        if (manifest["status"] != "complete" or manifest["score_links"] != ["link5"]
                or metrics["modes"]["exact-group"]["objects"]["link5"]["status"]
                    != state["link5_status"]):
            raise ValueError(f"scorer manifest disagrees with status: {sample}")
        if not (output / "replay/interactive_exact-group/link5_exact-group.html").is_file():
            raise ValueError(f"interactive replay missing: {sample}")
        if sha(local_case / "base_segmentation.npz") != state["mask_sha256"]:
            raise ValueError(f"retrieved mask differs from scored mask: {sample}")
    cache_dir = f"{REMOTE}/cases/{sample}/geometry-cache"
    remote_files = remote("sh", "-c", f"test -d {shlex.quote(cache_dir)} && "
                          f"find {shlex.quote(cache_dir)} -maxdepth 1 -type f -name '*.npz' || true")
    for filename in remote_files.splitlines():
        path = Path(filename)
        local_file = local_case / "geometry-cache" / path.name
        local_file.parent.mkdir(exist_ok=True)
        original = source_cache(sample)
        digest = remote("sha256sum", filename).split()[0]
        if path.name == original.name and digest == sha(original):
            if not local_file.exists():
                os.link(original, local_file)
        else:
            rsync(f"{HOST}:{filename}", str(local_file))
        if sha(local_file) != digest:
            raise ValueError(f"geometry cache transfer failed: {sample}")
    # The Mac copy is verified, so keep only the small remote status and log.
    remote("rm", "-rf", cache_dir, f"{REMOTE}/cases/{sample}/base_generation",
           f"{REMOTE}/cases/{sample}/v1_cotracker3")
    remote("rm", "-f", f"{REMOTE}/cases/{sample}/base_segmentation.npz",
           f"{REMOTE}/cases/{sample}/{sample.replace('.', '_')}.mp4")
    (local_case / "collected.ok").write_text(state["state"] + "\n")
    print(f"COLLECTED {sample} {state['state']}", flush=True)


def main() -> None:
    selection = json.loads((LOCAL / "selection.json").read_text())
    samples = [f"{entry['dataset']}_{entry['video_number']}" for entry in selection["videos"]]
    collected = set()
    staged = set()
    for sample in samples:
        local_case = LOCAL / "cases" / sample
        local_status = local_case / "status.json"
        if local_status.is_file() and (local_case / "collected.ok").is_file():
            state = json.loads(local_status.read_text())
            if state.get("state") in {"complete", "failed"}:
                collected.add(sample)
    while len(collected) < len(samples):
        status_script = ("import json,sys; from pathlib import Path; "
                         "root=Path(sys.argv[1]); "
                         "print(json.dumps({p.parent.name:json.loads(p.read_text()) "
                         "for p in root.glob('cases/*/status.json')}))")
        remote_statuses = json.loads(remote("python3", "-c", status_script, REMOTE))
        for sample in samples:
            if sample in collected:
                continue
            state = remote_statuses.get(sample, {})
            if state.get("state") in {"complete", "failed"}:
                collect(sample, state)
                collected.add(sample)
                staged.discard(sample)
        for sample in samples:
            if len(staged) >= 2:
                break
            if sample not in staged and sample not in collected:
                stage(sample)
                staged.add(sample)
        print(f"PROGRESS collected={len(collected)}/{len(samples)} staged={sorted(staged)}",
              flush=True)
        if len(collected) < len(samples):
            time.sleep(20)
    import sys
    sys.path.insert(0, str(ROOT / "PDI-Bench-edited/src"))
    from pdi_eval.experiment.link5_only import export_index
    summary = export_index(LOCAL, selection)
    exit_path = f"{REMOTE}/run.exit"
    while True:
        raw = remote("sh", "-c", f"test -s {shlex.quote(exit_path)} && cat {shlex.quote(exit_path)} || true")
        if raw:
            break
        time.sleep(10)
    exit_code = int(raw)
    rsync(f"{HOST}:{REMOTE}/run.log", str(LOCAL / "run.log"))
    completed = summary["counts"]["complete"]
    failed = summary["counts"]["failed"]
    record = {"video_count": len(samples), "complete": completed, "failed": failed,
              "remote_exit_status": exit_code,
              "verified": completed + failed == len(samples) and (exit_code == 0) == (failed == 0)}
    (LOCAL / "run_status.json").write_text(json.dumps(record, indent=2) + "\n")
    if not record["verified"]:
        raise RuntimeError(f"run completion is inconsistent: {record}")
    update_ledger(summary, exit_code)
    print("ALL_CASES_COLLECTED", flush=True)


if __name__ == "__main__":
    main()
