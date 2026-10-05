"""Stage the remaining caches for an autonomous remote run, then collect results."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'scripts/run_link5_link7_remaining_45.py'
        import sys as _layout_sys
        _layout_sys.path[:0] = [str(_layout_root / "infrastructure/compat" / "scripts"), str(_layout_root / "infrastructure/compat"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited/src"), str(_layout_root / "infrastructure/compat" / "PDI-Bench-edited")]
        break


import fcntl
import json
import subprocess
import sys
import time

import run_link5_link7_four_way_45 as base

SESSION = "pdi-l57-remaining-v1"
CONTROL = f"{base.REMOTE}/control"
BATCHES = f"{base.REMOTE}/batches"
MANIFEST = f"{CONTROL}/remaining-v1.groups"
OVERALL_EXIT = f"{BATCHES}/remaining-v1.exit"


def ordered_entries() -> list[dict]:
    selection = json.loads((base.LOCAL / "selection.json").read_text())
    if selection["video_count"] != 45:
        raise ValueError("expected the frozen 45-video selection")
    entries = selection["videos"]
    lookup = {f"{entry['dataset']}_{entry['video_number']}": entry
              for entry in entries}
    first = entries[:3]
    first_ids = {f"{entry['dataset']}_{entry['video_number']}" for entry in first}
    return (first + [lookup[sample] for sample in base.PRIORITY if sample not in first_ids]
            + [entry for entry in entries[3:]
               if f"{entry['dataset']}_{entry['video_number']}" not in base.PRIORITY])


def groups() -> list[tuple[str, list[tuple[str, object]]]]:
    entries = ordered_entries()
    result = []
    for index in range(20, len(entries), 4):
        pairs = []
        for entry in entries[index:index + 4]:
            sample = f"{entry['dataset']}_{entry['video_number']}"
            pairs.append((sample, base.cache_for(sample, entry["sha256"])))
        result.append((f"b{index:02d}", pairs))
    if len(result) != 7 or sum(len(pairs) for _, pairs in result) != 25:
        raise ValueError("remaining groups differ from frozen selection")
    return result


def stage_and_launch() -> None:
    planned = groups()
    entries = ordered_entries()
    first_twenty = [f"{entry['dataset']}_{entry['video_number']}" for entry in entries[:20]]
    if any(not (base.LOCAL / "cases" / sample / ".case_synced.json").is_file()
           for sample in first_twenty):
        raise RuntimeError("first 20 videos are not locally verified; remaining run will not start")
    fingerprint = base.remote(
        "env", f"PYTHONPATH={base.CODE}/src:{base.CODE}",
        "/root/autodl-tmp/pdi/env/pdi-bench/bin/python", "-c",
        "from pathlib import Path; from robot.workflows.score_v1 import _source_fingerprint; "
        f"print(_source_fingerprint(Path('{base.CODE}')))",
    )
    if fingerprint != base.SCORER_FINGERPRINT:
        raise ValueError("local and remote scorer fingerprints differ")
    base.remote("mkdir", "-p", CONTROL, BATCHES)
    local_manifest = base.LOCAL / "control/remaining-v1.groups"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text("".join(
        batch_id + " " + " ".join(sample for sample, _ in pairs) + "\n"
        for batch_id, pairs in planned))
    base.rsync(str(local_manifest), f"{base.HOST}:{MANIFEST}")
    overall = base.remote("bash", "-lc", f"if test -f {OVERALL_EXIT}; then cat {OVERALL_EXIT}; fi")
    if overall and overall != "0":
        raise RuntimeError(f"remote run exited {overall}; inspect {BATCHES}/remaining-v1.log")
    if not overall and not base.remote("bash", "-lc", f"tmux has-session -t {SESSION} 2>/dev/null && echo running || true"):
        command = (f"bash {base.CODE}/scripts/run_link5_link7_remaining.sh {MANIFEST} "
                   f"> {BATCHES}/remaining-v1.log 2>&1")
        base.remote("tmux", "new-session", "-d", "-s", SESSION, command)
    print(f"remote tmux {SESSION} active", flush=True)
    for batch_id, pairs in planned:
        exit_value = base.remote(
            "bash", "-lc", f"if test -f {BATCHES}/{batch_id}.exit; then cat {BATCHES}/{batch_id}.exit; fi")
        if exit_value:
            if exit_value != "0":
                raise RuntimeError(f"{batch_id} exited {exit_value}; no retry will be attempted")
            print(f"already completed {batch_id}", flush=True)
            continue
        free_kib = int(base.remote(
            "bash", "-lc", "df -Pk /root/autodl-tmp | tail -1 | awk '{print $4}'"))
        cache_kib = sum((cache.stat().st_size + 1023) // 1024 for _, cache in pairs)
        if free_kib - cache_kib < 3 * 1024 * 1024:
            raise RuntimeError(f"GPU data volume lacks 3 GiB reserve for {batch_id}")
        for sample, cache in pairs:
            base.stage_cache(sample, cache)
            print(f"staged {batch_id} {sample}", flush=True)
        base.remote("touch", f"{CONTROL}/{batch_id}.ready")
        print(f"ready {batch_id}", flush=True)
    print("All remaining groups staged; remote loop runs independently", flush=True)


def collect() -> None:
    value = base.remote("bash", "-lc", f"if test -f {OVERALL_EXIT}; then cat {OVERALL_EXIT}; fi")
    if value != "0":
        raise RuntimeError(f"remote loop has not completed successfully: {value or 'running'}")
    for batch_id, pairs in groups():
        batch_value = base.remote("bash", "-lc", f"cat {BATCHES}/{batch_id}.exit")
        if batch_value != "0":
            raise RuntimeError(f"{batch_id} exited {batch_value}")
        for sample, cache in pairs:
            marker = base.LOCAL / "cases" / sample / ".case_synced.json"
            synced = json.loads(marker.read_text()) if marker.is_file() else {}
            if synced.get("scorer_fingerprint") in (
                    base.COMPATIBLE_SCORER_FINGERPRINTS | {base.SCORER_FINGERPRINT}):
                continue
            if base.remote_status(sample) != "complete":
                raise RuntimeError(f"remote case incomplete: {sample}")
            base.sync_case(sample, cache)
        print(f"collected and verified {batch_id}", flush=True)
    print("COMPLETE 45/45 locally verified", flush=True)


def collect_completed() -> None:
    for batch_id, pairs in groups():
        batch_value = base.remote(
            "bash", "-lc", f"if test -f {BATCHES}/{batch_id}.exit; then cat {BATCHES}/{batch_id}.exit; fi")
        if not batch_value:
            continue
        for sample, cache in pairs:
            marker = base.LOCAL / "cases" / sample / ".case_synced.json"
            synced = json.loads(marker.read_text()) if marker.is_file() else {}
            if synced.get("scorer_fingerprint") in (
                    base.COMPATIBLE_SCORER_FINGERPRINTS | {base.SCORER_FINGERPRINT}):
                continue
            if base.remote_status(sample) != "complete":
                print(f"incomplete {sample}; preserving its remote artifacts", flush=True)
                continue
            base.sync_case(sample, cache)
        print(f"fetched completed cases from {batch_id} (exit {batch_value})", flush=True)


def collect_when_done() -> None:
    while True:
        try:
            value = base.remote(
                "bash", "-lc", f"if test -f {OVERALL_EXIT}; then cat {OVERALL_EXIT}; fi")
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255:
                raise
            print("SSH unavailable while waiting for remote completion; retrying", flush=True)
            time.sleep(60)
            continue
        if value:
            if value != "0":
                raise RuntimeError(f"remote loop exited {value}; results need inspection")
            break
        time.sleep(60)
    while True:
        try:
            collect()
            return
        except subprocess.CalledProcessError as exc:
            print(f"result transfer interrupted ({exc.returncode}); retrying", flush=True)
            time.sleep(60)


if __name__ == "__main__":
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] not in
                             {"--collect", "--collect-completed", "--collect-when-done"}):
        raise SystemExit("usage: run_link5_link7_remaining_45.py [--collect|--collect-completed|--collect-when-done]")
    base.LOCAL.mkdir(parents=True, exist_ok=True)
    with (base.LOCAL / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if len(sys.argv) == 2 and sys.argv[1] == "--collect-when-done":
            collect_when_done()
        elif len(sys.argv) == 2 and sys.argv[1] == "--collect-completed":
            collect_completed()
        elif len(sys.argv) == 2:
            collect()
        else:
            stage_and_launch()
