#!/usr/bin/env python3
"""Build a deterministic, dataset-qualified manifest for remote video staging."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def job_slug(dataset: str, relative_path: str, sha256: str) -> str:
    stem = str(Path(relative_path).with_suffix(""))
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{dataset}_{stem}").strip("_.-")
    return f"{slug}-{sha256[:12]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for specification in args.dataset:
        if "=" not in specification:
            parser.error(f"invalid dataset specification: {specification!r}")
        dataset, root_value = specification.split("=", 1)
        root = Path(root_value).resolve()
        if not root.is_dir():
            parser.error(f"dataset directory is missing: {root}")
        for video in sorted(root.rglob("*.mp4")):
            relative_path = video.relative_to(root).as_posix()
            digest = sha256_file(video)
            records.append(
                {
                    "dataset": dataset,
                    "relative_path": relative_path,
                    "staged_relative_path": f"{dataset}/{relative_path}",
                    "job_id": job_slug(dataset, relative_path, digest),
                    "sha256": digest,
                    "size_bytes": video.stat().st_size,
                }
            )
    job_ids = [record["job_id"] for record in records]
    if len(job_ids) != len(set(job_ids)):
        raise RuntimeError("dataset-qualified job IDs are not unique")
    payload = {
        "schema_version": 1,
        "video_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "videos": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"videos": len(records), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
