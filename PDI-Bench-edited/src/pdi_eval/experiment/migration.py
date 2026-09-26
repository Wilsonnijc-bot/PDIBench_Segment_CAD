#!/usr/bin/env python3
"""Split one interrupted ten-case masking checkpoint into ten resumable case records."""

import argparse
import copy
import json
from pathlib import Path

from .contracts import case_name, sha256_file, write_json
from .runner import mask_work


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--global-work", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_root.resolve()
    global_work = args.global_work.resolve()
    selection = json.loads((output / "selection.json").read_text())
    source = global_work / "provenance.json"
    record = json.loads(source.read_text())
    wanted = [case_name(entry) for entry in selection["videos"]]
    if record["cases"] != wanted or len(wanted) != 10:
        raise ValueError("checkpoint cases differ from the frozen ten-case selection")
    for case in wanted:
        result = record["results"][case]
        if result["status"] not in {"vlm1_pending", "vlm2_pending", "no_confirmed_deformation"}:
            raise ValueError(f"cannot split unexpected {case} status {result['status']}")
        work = mask_work(output, case)
        target = work / "provenance.json"
        if target.exists():
            raise FileExistsError(target)
        cloned = copy.deepcopy(record)
        cloned["level"] = "local"
        cloned["status"] = "running"
        cloned["cases"] = [case]
        cloned["results"] = {case: copy.deepcopy(result)}
        cloned["migration"] = {"source": str(source), "source_sha256": sha256_file(source),
                                "reason": "resume one video end to end with two case workers"}
        write_json(target, cloned)
        artifact = global_work / "artifacts" / case / "vlm1_initial_frame.png"
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        destination = work / "artifacts" / case
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "vlm1_initial_frame.png").write_bytes(artifact.read_bytes())
        print(case, result["status"], len(result.get("diagnoses", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
