"""Fetch the final selected-45 case after its GPU batch finishes."""

from __future__ import annotations

import fcntl
import json
import subprocess
import time

import run_link5_link7_four_way_45 as base


SAMPLE = "LVP_ROBOWM_0060"
EXIT = f"{base.REMOTE}/batches/b45.exit"


def main() -> None:
    while True:
        try:
            value = base.remote(
                "bash", "-lc", f"if test -f {EXIT}; then cat {EXIT}; fi")
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255:
                raise
            print("SSH unavailable; checking b45 again", flush=True)
            time.sleep(60)
            continue
        if value:
            if value != "0":
                raise RuntimeError(f"b45 exited {value}; preserve failed-case evidence")
            break
        time.sleep(60)

    with (base.LOCAL / "orchestrator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selection = json.loads((base.LOCAL / "selection.json").read_text())
        entry = next(item for item in selection["videos"]
                     if f"{item['dataset']}_{item['video_number']}" == SAMPLE)
        cache = base.cache_for(SAMPLE, entry["sha256"])
        while True:
            try:
                if base.remote_status(SAMPLE) != "complete":
                    raise RuntimeError(f"b45 exited 0 without complete {SAMPLE}")
                base.sync_case(SAMPLE, cache)
                break
            except subprocess.CalledProcessError as exc:
                print(f"transfer interrupted ({exc.returncode}); retrying", flush=True)
                time.sleep(60)
    count = len(list((base.LOCAL / "cases").glob("*/.case_synced.json")))
    print(f"b45 fetched and hash verified; locally verified {count}/45", flush=True)


if __name__ == "__main__":
    main()
