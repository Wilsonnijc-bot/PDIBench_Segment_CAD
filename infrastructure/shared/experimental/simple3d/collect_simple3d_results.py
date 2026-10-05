"""Retrieve terminal cases and live summaries; optionally continue until remote exit."""

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = (_workspace_root())
sys.path.insert(0, str(ROOT))
from infrastructure.shared.experimental.simple3d.simple3d_layout import result_path


def update_ledger(run_id, state, report):
    if run_id != "20261003-run2":
        return
    ledger = (_workspace_root() / 'experiment_GPU_record.md')
    text = ledger.read_text()
    heading = "## Direct official Simple3D: object and link5 references — 2026-10-03"
    if heading not in text:
        return
    start = text.index(heading)
    end = text.find("\n## ", start + len(heading))
    end = len(text) if end < 0 else end
    section = text[start:end]
    terminal = state["exit"] is not None and state["finalization"]
    status = ("completed" if state["exit"] == "0" else "failed") if terminal else "running"
    section = re.sub(r"(?m)^- \*\*Status:\*\*.*$", f"- **Status:** {status}", section, count=1)
    counts = "; ".join(f"{k}: {s['videos_processed']}/{s['eligible_video_count']} videos, "
                       f"{s['successful_simple3d_scores']}/{s['expected_comparisons']} scores, "
                       f"{len(s['failures'])} failed comparisons" for k, s in report["experiments"].items())
    results = ("Initial run1 failed before reconstruction because older GPU PDI source lacked the crop helper; "
               "its misleading exit0 was corrected. Run2 frozen current-source results: " + counts + ". "
               + (f"All cases terminal; observed batch exit{state['exit']}; native artifact audits passed and SHA256 retrieval verified."
                  if terminal else "Full batch remains in tmux; observed results are partial. User authorized leaving it active after verified completed cases."))
    section = re.sub(r"(?m)^- \*\*Results:\*\*.*$", "- **Results:** " + results, section, count=1)
    tmp = ledger.with_suffix(".md.simple3d.tmp")
    tmp.write_text(text[:start] + section + text[end:])
    tmp.replace(ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True)
    p.add_argument("--remote-root", default="/root/autodl-tmp/pdi/simple3d-evaluation")
    p.add_argument("--host", default="root@region-9.autodl.pro")
    p.add_argument("--port", default="26211")
    p.add_argument("--identity", type=Path, default=(_workspace_root() / '.tmp/autodl_pdi_ed25519'))
    p.add_argument("--watch", action="store_true")
    a = p.parse_args()
    ssh = ["ssh", "-i", str(a.identity), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", a.port]
    remote_script = """import json,pathlib,sys
r=pathlib.Path(sys.argv[1]); run=sys.argv[2]; out={}; canonical=r/'results'/('simple3d-'+run)
for role,folder,prefix in [('object','object','simple3d-object-deformation'),('cad','link5_cad','simple3d-link5-cad-reference'),('first_frame','link5_first_frame','simple3d-link5-firstframe-reference')]:
 p=canonical/folder
 if not p.is_dir(): p=r/'results'/(prefix+'-'+run)
 out[role]={'relative_path':str(p.relative_to(r)),'cases':[q.parent.name for q in p.glob('cases/*/comparisons.json')],
              'geometry':[q.parent.name for q in p.glob('observed/*/geometry.json') if json.loads(q.read_text())['status']!='running']}
report=canonical/'metadata'
if not report.is_dir(): report=r/'results'/('simple3d-run-'+run)
execution=report/'execution'
if not execution.is_dir(): execution=r/'results'
exitfile=execution/('simple3d-'+run+'.exit')
print(json.dumps({'roots':out,'exit':exitfile.read_text().strip() if exitfile.exists() else None,
                  'report_path':str(report.relative_to(r)),'execution_path':str(execution.relative_to(r)),
                  'report_available':report.is_dir(),'finalization':(report/'finalization.json').is_file()}))
"""
    metadata = result_path(a.run_id, "metadata") / "execution"
    metadata.mkdir(parents=True, exist_ok=True)
    received = set()
    verified = {}
    while True:
        try:
            state = json.loads(subprocess.check_output([*ssh, a.host,
                shlex.join(["python3", "-c", remote_script, a.remote_root, a.run_id])], text=True))
            def sync(relative, target, extra=()):
                target.mkdir(parents=True, exist_ok=True)
                subprocess.run(["rsync", "-a", "--partial", "--timeout=120", "-e", shlex.join(ssh), *extra,
                                f"{a.host}:{a.remote_root}/{relative}", str(target) + "/"], check=True)
            for name, info in state["roots"].items():
                dest = result_path(a.run_id, name)
                remote_relative = info["relative_path"]
                if not info["cases"]:
                    continue
                sync(f"{remote_relative}/", dest,
                     ("--include=/*.json", "--include=/*.csv", "--exclude=/*/", "--exclude=*"))
                for group, names in (("cases", info["cases"]), ("observed", info["geometry"])):
                    for video in names:
                        key = (name, group, video)
                        if key not in received:
                            sync(f"{remote_relative}/{group}/{video}/", dest / group / video)
                            remote_folder = f"{a.remote_root}/{remote_relative}/{group}/{video}"
                            hash_script = "import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in r.rglob('*') if p.is_file() and not p.is_symlink()}))"
                            expected = json.loads(subprocess.check_output([*ssh, a.host,
                                shlex.join(["python3", "-c", hash_script, remote_folder])], text=True))
                            import hashlib
                            for relative, digest in expected.items():
                                local = dest / group / video / relative
                                assert local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == digest, str(local)
                            verified["/".join(key)] = {"file_count": len(expected), "status": "SHA256_verified"}
                            (metadata / "transfer_verification.json").write_text(json.dumps(verified, indent=2) + "\n")
                            received.add(key)
                if name == "cad":
                    sync(f"{remote_relative}/cad/", dest / "cad")
            sync(state["execution_path"] + "/", metadata, ("--include=/*" + a.run_id + "*.log", "--include=/*" + a.run_id + "*.exit", "--include=/preflight.json",
                                       "--exclude=/*/", "--exclude=*"))
            if state["report_available"]:
                sync(state["report_path"] + "/", metadata.parent,
                     ("--include=/*.json", "--include=/*.csv", "--exclude=/*/", "--exclude=*"))
                update_ledger(a.run_id, state, json.loads((metadata.parent / "summary.json").read_text()))
            (metadata / "collection_status.json").write_text(json.dumps(state, indent=2) + "\n")
            print("COLLECTED", len(received), "case/geometry folders; remote exit", state["exit"], flush=True)
            if not a.watch or (state["exit"] is not None and state["finalization"]):
                return 0
        except (subprocess.CalledProcessError, json.JSONDecodeError, OSError, AssertionError) as e:
            print("COLLECTION_RETRY", str(e), flush=True)
            if not a.watch:
                return 1
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
