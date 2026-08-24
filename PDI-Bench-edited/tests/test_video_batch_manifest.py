from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "evaluation" / "build_video_batch_manifest.py"


def run_builder(output: Path, *datasets: tuple[str, Path]) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(SCRIPT)]
    for name, root in datasets:
        command.extend(["--dataset", f"{name}={root}"])
    command.extend(["--replays-per-dataset", "2", "--output", str(output)])
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_repeated_dataset_roots_deduplicate_identical_relative_paths(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "0000.mp4").write_bytes(b"zero")
    (first / "0002.mp4").write_bytes(b"two")
    (second / "0000.mp4").write_bytes(b"zero")
    (second / "0001.mp4").write_bytes(b"one")
    output = tmp_path / "manifest.json"

    result = run_builder(output, ("COSMOS3", first), ("COSMOS3", second))

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    assert manifest["video_count"] == 3
    assert manifest["replay_count"] == 2
    assert [item["relative_path"] for item in manifest["videos"]] == [
        "0000.mp4",
        "0001.mp4",
        "0002.mp4",
    ]
    assert [item["replay_selected"] for item in manifest["videos"]] == [
        True,
        True,
        False,
    ]


def test_repeated_dataset_roots_reject_conflicting_duplicates(tmp_path: Path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "0000.mp4").write_bytes(b"first")
    (second / "0000.mp4").write_bytes(b"second")

    result = run_builder(
        tmp_path / "manifest.json", ("COSMOS3", first), ("COSMOS3", second)
    )

    assert result.returncode != 0
    assert "conflicting duplicate video for COSMOS3/0000.mp4" in result.stderr
