"""Stage verified caches, run bounded GPU tmux batches, and offload results."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/run_link5_link7_four_way_45.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import hashlib
import fcntl
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = _SOURCE_PATH.parents[1]
LOCAL = ROOT / "results/link5-link7-four-way-selected45-20260927"
REMOTE = "/root/autodl-tmp/pdi/experiments/link5-link7-four-way-selected45-20260927"
CODE = "/root/autodl-tmp/pdi/code/PDI-Bench-edited"
HOST = "root@region-9.autodl.pro"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
SSH = ["ssh", "-i", str(KEY), "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=20",
       "-o", "ServerAliveCountMax=3", "-p", "26211", HOST]
RSYNC_SSH = (f"ssh -i {KEY} -o BatchMode=yes -o ConnectTimeout=15 "
             "-o ServerAliveInterval=20 -o ServerAliveCountMax=3 -p 26211")
sys.path.insert(0, str(ROOT / "PDI-Bench-edited/src"))
from robot.workflows.score_v1 import _source_fingerprint

SCORER_FINGERPRINT = _source_fingerprint(ROOT / "PDI-Bench-edited")
COMPATIBLE_SCORER_FINGERPRINTS = {
    "465e424e2b4de2e124a104f19db86c917a8a339034f3b03bd6f3ef12cda0e686",
    "81a6319132638cf34f2b674f58cdf4d7bceb1b88a48fc1582dab24ff1fb62b03",
    "24aa66a575ad1312def5b08ce7bfd52f4891534d7e9cb08c653a17ed1d4b6e1a",
}
PRIORITY = (
    "COSMOS2.5_0010", "COSMOS3_0010", "COSMOS2.5_0015",
    "LVP_ROBOWM_0021", "COSMOS3_0030",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote(*arguments: str) -> str:
    for attempt in range(3):
        result = subprocess.run([*SSH, shlex.join(arguments)], text=True,
                                capture_output=True, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        if result.returncode != 255 or attempt == 2:
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr)
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable SSH retry state")


def rsync(source: str, target: str, *, exclude: str | None = None) -> None:
    command = ["rsync", "-a", "--partial", "--timeout=300", "-e", RSYNC_SSH]
    if exclude:
        command += ["--exclude", exclude]
    subprocess.run([*command, source, target], check=True)


def cache_for(sample: str, video_hash: str) -> Path:
    candidates = [
        (ROOT / "results/selected-45-v1/cases" / sample,
         "v1/manifest.json"),
        (ROOT / "results/link7-point-filter-v2-20260926/cases" / sample,
         "path2_v2_tapip3d/manifest.json"),
    ]
    for case, manifest_name in candidates:
        manifest_path = case / manifest_name
        caches = list((case / "geometry-cache").glob("*.npz"))
        if not manifest_path.is_file() or len(caches) != 1:
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest["input"]["sha256"] != video_hash:
            raise ValueError(f"cache video identity differs: {sample}")
        if Path(manifest["shared_geometry_cache"]["cache_path"]).name != caches[0].name:
            raise ValueError(f"cache key differs from source manifest: {sample}")
        return caches[0]
    raise FileNotFoundError(f"no validated local geometry cache for {sample}")


def stage_cache(sample: str, source: Path) -> None:
    folder = f"{REMOTE}/cases/{sample}/geometry-cache"
    remote("mkdir", "-p", folder)
    destination = f"{folder}/{source.name}"
    existing = remote("bash", "-lc", f"if test -s {shlex.quote(destination)}; then sha256sum {shlex.quote(destination)} | cut -d' ' -f1; fi")
    source_hash = sha(source)
    if existing != source_hash:
        rsync(str(source), f"{HOST}:{destination}")
    if remote("sha256sum", destination).split()[0] != source_hash:
        raise ValueError(f"remote cache hash mismatch: {sample}")


def remote_status(sample: str) -> str:
    case = f"{REMOTE}/cases/{sample}"
    script = """import json,sys
from pathlib import Path
from robot.experiments.link7_tracker_filter_comparison.link5_link7_four_way import PATHS, _complete, COMPATIBLE_SCORER_FINGERPRINTS
case=Path(sys.argv[1]); expected_fingerprint=sys.argv[2]
status_path=case/'status.json'
if not status_path.is_file(): print('pending'); sys.exit()
status=json.loads(status_path.read_text())
if status.get('state')!='complete' or len(status.get('paths',{}))!=4:
    print('pending'); sys.exit()
source=status.get('source_sha256'); mask=status.get('selected_mask_sha256')
low=status.get('tracked_fraction',{}).get('link7',0)<0.80
for name,filter_name,tracker in PATHS:
    path=status['paths'].get(name,{})
    if (path.get('filter')!=filter_name or path.get('tracker')!=tracker
            or path.get('selected_mask_sha256')!=mask):
        print('pending'); sys.exit()
    if low and path.get('link7',{}).get('error_type')!='insufficient_sam3_coverage':
        print('pending'); sys.exit()
    if low and name!='v1_cotracker3': continue
    links=('link5',) if low else (('link5','link7') if name=='v1_cotracker3' else ('link7',))
    if not _complete(case/name,source,mask,filter_name,tracker,links):
        print('pending'); sys.exit()
    manifest=json.loads((case/name/'manifest.json').read_text())
    if manifest['benchmark_source_sha256'] not in (COMPATIBLE_SCORER_FINGERPRINTS | {expected_fingerprint}):
        print('pending'); sys.exit()
print('complete')
"""
    return remote("env", f"PYTHONPATH={CODE}/src:{CODE}",
                  "/root/autodl-tmp/pdi/env/pdi-bench/bin/python", "-c", script,
                  case, SCORER_FINGERPRINT)


def run_batch(index: int, samples: list[str]) -> None:
    if all(remote_status(sample) == "complete" for sample in samples):
        return
    attempt = 1
    while True:
        batch_id = f"b{index:02d}-a{attempt}"
        marker = f"{REMOTE}/batches/{batch_id}.exit"
        value = remote("bash", "-lc", f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi")
        if not value:
            break
        attempt += 1
    pending = [sample for sample in samples if remote_status(sample) != "complete"]
    if not pending:
        return
    session = f"pdi-l57-{batch_id}"
    if not remote("bash", "-lc", f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo running || true"):
        command = ["bash", f"{CODE}/scripts/run_link5_link7_batch.sh", batch_id, *pending]
        remote("tmux", "new-session", "-d", "-s", session, shlex.join(command))
    print(f"launched {batch_id}: {', '.join(pending)}", flush=True)
    while True:
        value = remote("bash", "-lc", f"if test -f {shlex.quote(marker)}; then cat {shlex.quote(marker)}; fi")
        session = f"pdi-l57-{batch_id}"
        if value:
            if value != "0":
                raise RuntimeError(f"GPU batch {batch_id} exited {value}; inspect {REMOTE}/batches/{batch_id}.log")
            if any(remote_status(sample) != "complete" for sample in samples):
                raise RuntimeError(f"GPU batch {batch_id} exited 0 without complete cases")
            return
        print(f"waiting for {batch_id}", flush=True)
        time.sleep(45)


def sync_case(sample: str, cache: Path) -> None:
    destination = LOCAL / "cases" / sample
    destination.mkdir(parents=True, exist_ok=True)
    rsync(f"{HOST}:{REMOTE}/cases/{sample}/", str(destination) + "/",
          exclude="geometry-cache/")
    script = (
        "import hashlib,json,os,pathlib,sys; root=pathlib.Path(sys.argv[1]); "
        "files={}; "
        "[(files.setdefault(str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest())) "
        "for p in root.rglob('*') if p.is_file() and not p.is_symlink() "
        "and 'geometry-cache' not in p.relative_to(root).parts]; "
        "print(json.dumps(files,sort_keys=True))"
    )
    manifest = json.loads(remote("python3", "-c", script, f"{REMOTE}/cases/{sample}"))
    for relative, digest in manifest.items():
        local = destination / relative
        if not local.is_file() or sha(local) != digest:
            raise ValueError(f"result transfer hash mismatch: {sample}/{relative}")
    status = json.loads((destination / "status.json").read_text())
    if status["state"] != "complete" or len(status.get("paths", {})) != 4:
        raise ValueError(f"incomplete local case: {sample}")
    fingerprints = {
        json.loads((destination / name / "manifest.json").read_text())[
            "benchmark_source_sha256"]
        for name in status["paths"]
        if (destination / name / "manifest.json").is_file()
    }
    if len(fingerprints) != 1:
        raise ValueError(f"scorer fingerprints disagree or are missing: {sample}")
    scored_fingerprint = fingerprints.pop()
    staged = f"{REMOTE}/cases/{sample}/geometry-cache/{cache.name}"
    if remote("sha256sum", staged).split()[0] != sha(cache):
        raise ValueError(f"staged cache changed during scoring: {sample}")
    local_cache = destination / "geometry-cache" / cache.name
    local_cache.parent.mkdir(exist_ok=True)
    if not local_cache.exists():
        try:
            os.link(cache, local_cache)
        except OSError:
            import shutil
            shutil.copy2(cache, local_cache)
    if sha(local_cache) != sha(cache):
        raise ValueError(f"local cache reference differs: {sample}")
    remote("rm", "--", staged)
    (destination / ".case_synced.json").write_text(json.dumps({
        "sample_id": sample, "remote_regular_files_verified": len(manifest),
        "geometry_sha256": sha(cache), "geometry_source": str(cache),
        "scorer_fingerprint": scored_fingerprint,
    }, indent=2) + "\n")
    print(f"verified and offloaded {sample}", flush=True)


def main() -> None:
    stop_after_batch = int(os.environ.get("PDI_STOP_AFTER_BATCH", "999"))
    selection = json.loads((LOCAL / "selection.json").read_text())
    if selection["video_count"] != 45:
        raise ValueError("expected the frozen 45-video selection")
    remote_fingerprint = remote(
        "env", f"PYTHONPATH={CODE}/src:{CODE}",
        "/root/autodl-tmp/pdi/env/pdi-bench/bin/python", "-c",
        "from pathlib import Path; from robot.workflows.score_v1 import _source_fingerprint; "
        "print(_source_fingerprint(Path('/root/autodl-tmp/pdi/code/PDI-Bench-edited')))",
    )
    if remote_fingerprint != SCORER_FINGERPRINT:
        raise ValueError("local and remote scorer fingerprints differ; deploy code before running")
    entries = selection["videos"]
    lookup = {f"{entry['dataset']}_{entry['video_number']}": entry
              for entry in entries}
    if len(lookup) != 45 or any(sample not in lookup for sample in PRIORITY):
        raise ValueError("priority cases differ from the frozen selection")
    first = entries[:3]
    first_ids = {f"{entry['dataset']}_{entry['video_number']}" for entry in first}
    entries = (first + [lookup[sample] for sample in PRIORITY if sample not in first_ids]
               + [entry for entry in entries[3:]
                  if f"{entry['dataset']}_{entry['video_number']}" not in PRIORITY])
    # The first eleven cases used the original batch IDs. LVP 0015 completed
    # remotely in b11, while LVP 0010 is deferred to the second four-case batch.
    batches = ([(index, [entry]) for index, entry in enumerate(entries[:7])]
               + [(index, entries[index:index + 2]) for index in (7, 9)]
               + [(12, [entries[12]])]
               + [(13, entries[13:17])]
               + [(index, ([entries[11]] if index == 17 else [])
                   + entries[index:index + (3 if index == 17 else 4)])
                  for index in [17, *range(20, len(entries), 4)]])
    for index, batch in batches:
        pairs = [(f"{entry['dataset']}_{entry['video_number']}",
                  cache_for(f"{entry['dataset']}_{entry['video_number']}", entry["sha256"]))
                 for entry in batch]
        pending = [(sample, cache) for sample, cache in pairs
                   if remote_status(sample) != "complete"]
        if pending:
            free_kib = int(remote("bash", "-lc", "df -Pk /root/autodl-tmp | tail -1 | awk '{print $4}'"))
            cache_kib = sum((cache.stat().st_size + 1023) // 1024
                            for _, cache in pending)
            if free_kib - cache_kib < 2 * 1024 * 1024:
                raise RuntimeError(
                    f"GPU data volume lacks 2 GiB work reserve for batch {index}: "
                    f"free={free_kib} KiB, caches={cache_kib} KiB")
        for sample, cache in pairs:
            if remote_status(sample) != "complete":
                stage_cache(sample, cache)
        run_batch(index, [sample for sample, _ in pairs])
        for sample, cache in pairs:
            marker = LOCAL / "cases" / sample / ".case_synced.json"
            synced = (json.loads(marker.read_text()) if marker.is_file() else {})
            if synced.get("scorer_fingerprint") not in (COMPATIBLE_SCORER_FINGERPRINTS | {SCORER_FINGERPRINT}):
                sync_case(sample, cache)
        free_kib = int(remote("bash", "-lc", "df -Pk /root/autodl-tmp | tail -1 | awk '{print $4}'"))
        if free_kib < 1024 * 1024:
            raise RuntimeError("GPU data volume below 1 GiB free after offload")
        if index >= stop_after_batch:
            print(f"STOPPED AFTER BATCH {index}: results locally verified", flush=True)
            return
    print("COMPLETE 45/45 locally verified", flush=True)


if __name__ == "__main__":
    LOCAL.mkdir(parents=True, exist_ok=True)
    with (LOCAL / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        main()
