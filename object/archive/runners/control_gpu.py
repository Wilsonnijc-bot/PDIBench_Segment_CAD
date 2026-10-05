"""Stage one verified geometry cache at a time, run GPU cases, and retrieve results."""

from __future__ import annotations

# Source ownership changed; retain historical resource/provenance paths.
from pathlib import Path as _LayoutPath
_SOURCE_PATH = _LayoutPath(__file__).absolute()
for _layout_root in _SOURCE_PATH.resolve().parents:
    if (_layout_root / "documentation/architecture/layout.json").is_file():
        _SOURCE_PATH = _layout_root / 'infrastructure/compat' / 'PDI-Bench-edited/src/pdi_eval/object_deformation_wrapper/control_gpu.py'
        break


import argparse
import hashlib
import html
import json
import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = _SOURCE_PATH.parents[4]
LOCAL = ROOT / "results/object-deformation-selected45-20260929"
REMOTE = "/root/autodl-tmp/pdi/experiments/object-deformation-selected45-20260929"
GPU_SCRIPT = "/root/autodl-tmp/pdi/code/PDI-Bench-edited/src/pdi_eval/object_deformation_wrapper/run_gpu.sh"
HOST = "root@region-9.autodl.pro"
KEY = Path.home() / ".ssh/pdi_tapip3d_ed25519"
SSH = ["ssh", "-i", str(KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
       "-o", "ServerAliveInterval=20", "-p", "26211", HOST]
RSYNC_SSH = (f"ssh -i {shlex.quote(str(KEY))} -o BatchMode=yes "
             "-o ConnectTimeout=15 -o ServerAliveInterval=20 -p 26211")


def remote(*arguments: str) -> str:
    last = None
    for attempt in range(3):
        result = subprocess.run([*SSH, shlex.join(arguments)], text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout.strip()
        last = result
        if result.returncode != 255:
            break
        time.sleep(2 ** attempt)
    raise RuntimeError(f"remote command failed ({last.returncode}): {last.stderr.strip()}")


def rsync(source: str, target: str, *options: str) -> str:
    for attempt in range(3):
        result = subprocess.run(["rsync", "-a", "--partial", "--timeout=300",
                                 "-e", RSYNC_SSH, *options, source, target],
                                text=True, capture_output=True)
        if result.returncode == 0:
            return result.stdout
        if result.returncode not in (12, 255) or attempt == 2:
            raise RuntimeError(f"rsync failed ({result.returncode}): {result.stderr.strip()}")
        time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_copy(case: str, local: Path) -> None:
    source = f"{REMOTE}/cases/{case}"
    script = (
        "import hashlib,json,pathlib,sys; "
        "root=pathlib.Path(sys.argv[1]); "
        "print(json.dumps({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() "
        "for p in root.rglob('*') if p.is_file() and not p.is_symlink() "
        "and 'geometry-cache' not in p.relative_to(root).parts},sort_keys=True))"
    )
    expected = json.loads(remote("python3", "-c", script, source))
    if not expected:
        raise ValueError(f"remote case {case} has no regular result files")
    for relative, digest in expected.items():
        path = local / relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"retrieved case {case} differs at {relative}")


def cache_source(case: str) -> Path:
    for root in (ROOT / "results/selected-45-v1/cases" / case,
                 ROOT / "results/link5-link7-four-way-selected45-20260927/cases" / case):
        files = list((root / "geometry-cache").glob("*.npz"))
        if len(files) == 1:
            return files[0]
    raise FileNotFoundError(f"no local verified geometry cache for {case}")


def status(case: str) -> dict:
    path = f"{REMOTE}/cases/{case}/status.json"
    raw = remote("sh", "-c", f"test -f {shlex.quote(path)} && cat {shlex.quote(path)} || true")
    return json.loads(raw) if raw else {}


def stage_cache(case: str, source: Path) -> None:
    destination = f"{REMOTE}/cases/{case}/geometry-cache"
    remote("mkdir", "-p", destination)
    rsync(str(source), f"{HOST}:{destination}/")
    remote_hash = remote("sha256sum", f"{destination}/{source.name}").split()[0]
    if remote_hash != sha256(source):
        raise ValueError(f"staged geometry cache hash differs for {case}")


def run_case(case: str, index: int) -> None:
    exit_file = f"{REMOTE}/cases/{case}/control.exit"
    log = f"{REMOTE}/cases/{case}/control.log"
    session = f"pdi-object-{index:02d}"
    replay = f"{REMOTE}/cases/{case}/replay/index.html"
    if remote("sh", "-c", f"test -f {shlex.quote(exit_file)} && echo done || true") == "done":
        if remote("sh", "-c", f"test -s {shlex.quote(replay)} && echo ready || true") != "ready":
            remote("rm", "-f", exit_file)
    if remote("sh", "-c", f"test -f {shlex.quote(exit_file)} && echo done || true") != "done":
        command = (f"bash {shlex.quote(GPU_SCRIPT)} {shlex.quote(case)} > {shlex.quote(log)} 2>&1; "
                   f"echo $? > {shlex.quote(exit_file)}")
        # A resumed controller leaves a live case session alone.
        active = remote("sh", "-c", f"tmux has-session -t {shlex.quote(session)} 2>/dev/null && echo active || true")
        if active != "active":
            remote("tmux", "new-session", "-d", "-s", session, command)
    while True:
        result = remote("sh", "-c", f"test -f {shlex.quote(exit_file)} && cat {shlex.quote(exit_file)} || true")
        if result:
            if result != "0":
                raise RuntimeError(f"GPU case {case} exited {result}; see {log}")
            return
        time.sleep(20)


def collect(case: str, source_cache: Path) -> None:
    local = LOCAL / "cases" / case
    local.mkdir(parents=True, exist_ok=True)
    excluded = ("--exclude=/geometry-cache/", f"--exclude=/{case.replace('.', '_')}.mp4")
    source = f"{HOST}:{REMOTE}/cases/{case}/"
    rsync(source, str(local) + "/", *excluded)
    verify_copy(case, local)
    remote_cache = remote("sh", "-c", f"find {shlex.quote(REMOTE + '/cases/' + case + '/geometry-cache')} -maxdepth 1 -type f -name '*.npz' -print")
    caches = [Path(line) for line in remote_cache.splitlines() if line]
    score = local / "score/rigidity.json"
    state = json.loads((local / "status.json").read_text())
    if state["status"] == "complete":
        result = json.loads(score.read_text())
        selected = Path(result["geometry"]["cache_path"])
        if selected not in caches:
            raise ValueError(f"scored geometry cache missing for {case}")
    else:
        selected = next((item for item in caches if item.name == source_cache.name), None)
        if selected is None:
            raise ValueError(f"staged geometry cache missing for {case}")
    remote_hash = remote("sha256sum", str(selected)).split()[0]
    local_cache = local / "geometry-cache" / selected.name
    local_cache.parent.mkdir(exist_ok=True)
    if sha256(source_cache) == remote_hash:
        if not local_cache.exists():
            try:
                os.link(source_cache, local_cache)
            except OSError:
                import shutil
                shutil.copy2(source_cache, local_cache)
    else:
        rsync(f"{HOST}:{selected}", str(local_cache))
    if sha256(local_cache) != remote_hash:
        raise ValueError(f"geometry cache retrieval failed for {case}")
    if state["status"] == "complete":
        result = json.loads(score.read_text())
        if result["input"]["video_sha256"] != state["expected_video_sha256"]:
            raise ValueError("retrieved score video identity mismatch")
        if result["input"]["segmentation_sha256"] != sha256(local / "masking/segmentation.npz"):
            raise ValueError("retrieved score segmentation identity mismatch")
        replay = local / "replay/replay.json"
        if not replay.is_file() or not (local / "replay/index.html").is_file():
            raise ValueError("interactive replay is missing from completed case")
        replay_record = json.loads(replay.read_text())
        if (replay_record["score_sha256"] != sha256(score)
                or replay_record["segmentation_sha256"] != sha256(local / "masking/segmentation.npz")
                or replay_record["track_sha256"] != sha256(local / "score/cotracker_exact-group.npz")
                or replay_record["geometry_sha256"] != remote_hash):
            raise ValueError("interactive replay provenance differs from scored evidence")
    remote("rm", "-rf", f"{REMOTE}/cases/{case}/geometry-cache")
    rsync(f"{HOST}:{REMOTE}/summary.json", str(LOCAL / "summary.json"))
    rsync(f"{HOST}:{REMOTE}/scores.csv", str(LOCAL / "scores.csv"))


def process_case(index: int, entry: dict) -> dict:
    case = entry["case"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", case):
        raise ValueError(f"invalid case identifier: {case}")
    local_case = LOCAL / "cases" / case
    local_status = local_case / "status.json"
    if local_status.is_file():
        previous = json.loads(local_status.read_text())
        if (previous.get("status") in {"complete", "unscorable"}
                and previous.get("expected_video_sha256") == entry["sha256"]
                and len(list((local_case / "geometry-cache").glob("*.npz"))) == 1
                and (previous["status"] == "unscorable" or
                     (local_case / "score/rigidity.json").is_file()
                     and (local_case / "replay/index.html").is_file())):
            return {"case": case, "status": "already_collected"}
    source_cache = cache_source(case)
    state = status(case)
    replay_missing = state.get("status") == "complete" and remote(
        "sh", "-c", f"test -s {shlex.quote(REMOTE + '/cases/' + case + '/replay/index.html')} && echo ready || true") != "ready"
    if state.get("status") not in {"complete", "unscorable"} or replay_missing:
        stage_cache(case, source_cache)
        run_case(case, index)
    collect(case, source_cache)
    state = json.loads((LOCAL / "cases" / case / "status.json").read_text())
    return {"case": case, "status": state["status"],
            "rigidity_score": state.get("rigidity_score")}


def refresh_index() -> dict:
    rsync(f"{HOST}:{REMOTE}/summary.json", str(LOCAL / "summary.json"))
    rsync(f"{HOST}:{REMOTE}/scores.csv", str(LOCAL / "scores.csv"))
    summary = json.loads((LOCAL / "summary.json").read_text())
    rows = []
    for entry in summary["cases"]:
        case = entry["case"]
        replay = LOCAL / "cases" / case / "replay/index.html"
        label = html.escape(case)
        if replay.is_file():
            label = f'<a href="cases/{html.escape(case, quote=True)}/replay/index.html">{label}</a>'
        score = entry.get("rigidity_score", "")
        shown_score = f"{float(score):.6f}" if score != "" and score is not None else "—"
        rows.append("<tr><td>" + label + "</td><td>" +
                    html.escape(str(entry.get("status", "pending"))) + "</td><td>" +
                    html.escape(str(entry.get("target_object", ""))) + "</td><td>" +
                    shown_score + "</td></tr>")
    document = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
                "<title>Task-object rigidity</title><style>body{font:16px system-ui;"
                "max-width:1000px;margin:3rem auto;padding:0 1rem;color:#17212b}"
                "table{width:100%;border-collapse:collapse}td,th{padding:.65rem;"
                "border-bottom:1px solid #d8dfe6;text-align:left}a{color:#07559a}"
                "</style><h1>Task-object rigidity</h1>"
                f"<p>{summary['complete']} complete · {summary['unscorable']} unscorable"
                f" · {summary['video_count']} selected videos</p>"
                "<table><thead><tr><th>Video / replay</th><th>Status</th>"
                "<th>Manipulated object</th><th>Rigidity score</th></tr></thead><tbody>" +
                "".join(rows) + "</tbody></table></html>")
    (LOCAL / "index.html").write_text(document, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        parser.error("workers must be 1 or 2")
    manifest = json.loads((LOCAL / "manifest.json").read_text())
    selected = [(index, entry) for index, entry in enumerate(manifest["videos"])
                if not args.case or entry["case"] in args.case]
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_case, index, entry): entry["case"]
                   for index, entry in selected}
        for future in as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failures.append((futures[future], str(exc)))
                print(json.dumps({"case": futures[future], "controller_error": str(exc)}), flush=True)
    summary = refresh_index()
    print(json.dumps({"complete": summary["complete"],
                      "unscorable": summary["unscorable"],
                      "video_count": summary["video_count"]}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} controller cases failed: {failures[:3]}")


if __name__ == "__main__":
    main()
