"""Release the final Link 5 retry after b40 and b44 finish successfully."""

from __future__ import annotations

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import run_link5_link7_four_way_45 as base


CONTROL = f"{base.REMOTE}/control"
BATCHES = f"{base.REMOTE}/batches"
SOURCES = (
    "src/pdi_eval/perception/link5_point_guard.py",
    "src/pdi_eval/experiment/link5_link7_four_way.py",
)


def marker(name: str) -> str:
    return base.remote(
        "bash", "-lc",
        f"if test -f {BATCHES}/{name}.exit; then cat {BATCHES}/{name}.exit; fi",
    )


def main() -> None:
    while True:
        try:
            b40, b44 = marker("b40"), marker("b44")
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255:
                raise
            print("SSH unavailable; waiting to inspect b40/b44", flush=True)
            time.sleep(60)
            continue
        if (b40 and b40 != "0") or (b44 and b44 != "0"):
            raise RuntimeError(f"prior group failed: b40={b40!r}, b44={b44!r}")
        if b40 == b44 == "0":
            break
        time.sleep(60)

    if marker("b45"):
        raise RuntimeError("b45 already has an exit marker")
    if base.remote("bash", "-lc", f"test -f {CONTROL}/b45.ready && echo yes || true"):
        raise RuntimeError("b45 is already released")

    for relative in SOURCES:
        source = Path(__file__).resolve().parents[1] / "PDI-Bench-edited" / relative
        destination = f"{base.CODE}/{relative}"
        base.rsync(str(source), f"{base.HOST}:{destination}")
        expected = hashlib.sha256(source.read_bytes()).hexdigest()
        actual = base.remote("sha256sum", destination).split()[0]
        if actual != expected:
            raise RuntimeError(f"deployed source hash mismatch: {relative}")
        print(f"deployed {relative} {actual}", flush=True)

    result = base.remote(
        "env", f"PYTHONPATH={base.CODE}/src:{base.CODE}",
        "/root/autodl-tmp/pdi/env/sam3/bin/python", "-m", "unittest",
        "discover", "-s", f"{base.CODE}/tests", "-p", "test_link5_point_guard.py",
    )
    print(result or "remote guard tests passed", flush=True)
    base.remote("touch", f"{CONTROL}/b45.ready")
    print("released b45 as the final group", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"B45_RELEASE_FAILED: {exc}", file=sys.stderr, flush=True)
        raise
