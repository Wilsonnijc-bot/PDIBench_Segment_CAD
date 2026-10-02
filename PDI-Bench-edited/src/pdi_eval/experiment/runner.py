"""Run frozen PDI cases end to end with bounded workers and GPU access.

Each worker completes one video (persistent mask, V1, replays) before
taking another. Only model calls hold the GPU lock; VLM2 API and validation
can overlap the other worker's GPU work. Existing stage outputs are resumed.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contracts import (
    ROOT, MASK_VIDEO_FOLDERS, case_name, checked_file, mask_is_valid,
    page_index, read_selection, run_command, sha256_file, video_path, write_json,
)
from .spec import ExperimentSpec

LINK_NAMES = tuple(f"link{i}" for i in range(2, 8))
MINIMUM_TRACKED_FRACTION = 0.80
SKIPPABLE_LOW_COVERAGE_LINKS = frozenset({"link3", "link4"})
COMPATIBLE_LINK5_GUARD_SHA256 = frozenset({
    "10ffb235e15bf4242ace338500f4dcf219e1be64f664470559e6307f5b879927",
})
COMPATIBLE_SEGMENTATION_SHA256 = frozenset({
    "fbc92f008b520c4e34154307af43fa46c61d140e24cd6b40aa59dc5b1fcb2310",
    "ee7c0222a59d8641eb1d0061d711286bbed047e7e1ae8670154267a20b1a548f",
})


def mask_coverage(path: Path) -> dict[str, float]:
    """Measure each link on the saved standard SAM3 pass."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        names = tuple(str(name) for name in archive["object_names"].tolist())
        masks = np.asarray(archive["object_masks"], dtype=bool)
    if (not names or any(name not in LINK_NAMES for name in names)
            or len(set(names)) != len(names) or masks.ndim != 4
            or masks.shape[1] != len(names)):
        raise ValueError(f"invalid named-link segmentation: {path}")
    tracked = np.any(masks, axis=(2, 3))
    return {name: float(np.mean(tracked[:, index]))
            for index, name in enumerate(names)}


def validate_mask_coverage(
    path: Path, *, defer_low_link7: bool = False,
) -> tuple[dict[str, float], tuple[str, ...]]:
    coverage = mask_coverage(path)
    low = tuple(name for name, fraction in coverage.items()
                if fraction < MINIMUM_TRACKED_FRACTION)
    allowed = SKIPPABLE_LOW_COVERAGE_LINKS | ({"link7"} if defer_low_link7 else set())
    disallowed = tuple(name for name in low if name not in allowed)
    if disallowed:
        raise ValueError(
            f"SAM3 coverage below {MINIMUM_TRACKED_FRACTION:.0%} for required "
            f"links {disallowed}: {coverage}"
        )
    return coverage, low


@contextmanager
def gpu_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def command(argv: list[str], log: Path, environment: dict[str, str], lock: Path | None = None):
    if lock is None:
        run_command(argv, log=log, environment=environment)
    else:
        with gpu_lock(lock):
            run_command(argv, log=log, environment=environment)


def v1_outputs_complete(
    destination: Path, *, video_sha256: str, segmentation_sha256: str,
) -> bool:
    required = (
        destination / "metrics.json",
        destination / "manifest.json",
        destination / "timing.json",
        destination / "segmentation.npz",
        destination / "cotracker_exact-group.npz",
        destination / "replay/combined_exact-group.mp4",
        destination / "replay/combined_exact-group_first_frame.png",
        destination / "replay/combined_exact-group.json",
        destination / "replay/interactive_exact-group/index.html",
        destination / "replay/interactive_exact-group/source.mp4",
        destination / "replay/interactive_exact-group/plotly.min.js",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        timing = json.loads((destination / "timing.json").read_text(encoding="utf-8"))
        objects = metrics.get("modes", {}).get("exact-group", {}).get("objects", {})
        if set(objects) != set(LINK_NAMES):
            return False
        complete = [name for name, report in objects.items()
                    if report.get("status") == "complete"]
        if not complete or any(report.get("status") not in {"complete", "failed", "skipped"}
                               for report in objects.values()):
            return False
        if not all((pair := destination / "replay/interactive_exact-group" /
                    f"{name}_exact-group_pairs.json").is_file()
                   and pair.stat().st_size > 0 for name in complete):
            return False
        return (
            manifest.get("status") == "complete"
            and manifest.get("tracking_modes") == ["exact-group"]
            and manifest.get("input", {}).get("sha256") == video_sha256
            and manifest.get("segmentation", {}).get("sha256") == segmentation_sha256
            and timing.get("status") == "complete"
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def score_argv(args: ExperimentSpec, video: Path, segmentation: Path,
               destination: Path, case: Path) -> list[str]:
    return [str(args.pdi_python), "-m", "pdi_eval.experiment", "score",
            "--config", str(ROOT / "configs/default.yaml"),
            "--input", str(video), "--segmentation-npz", str(segmentation),
            "--output-dir", str(destination),
            "--geometry-cache-dir", str(case / "geometry-cache"),
            "--tracker-checkpoint", str(args.tracker_checkpoint),
            "--tracking-mode", "exact-group",
            "--skip-low-coverage-link", "link3",
            "--skip-low-coverage-link", "link4",
            "--minimum-tracked-fraction", str(MINIMUM_TRACKED_FRACTION)]


def mask_work(output: Path, case: str) -> Path:
    return output / "persistent_work" / "local" / output.name.replace(".", "_").replace("-", "_") / case


def ensure_video_alias(alias: Path, source: Path, expected_hash: str) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    if not alias.exists() and not alias.is_symlink():
        alias.symlink_to(source)
    checked_file(alias, "selected video alias")
    if sha256_file(alias) != expected_hash:
        raise ValueError(f"video alias differs from the frozen selection: {alias}")


def clean_megasam_intermediates(sample: str) -> None:
    """Drop per-video MegaSAM scratch after scoring has finished.

    The replay, CoTracker tracks, and point pairs live in the case's V1 output.
    Geometry cache stays in the case until the offload monitor verifies it.
    """
    video_id = sample.replace("COSMOS2.5", "COSMOS2_5")
    if not re.fullmatch(r"[A-Za-z0-9_]+", video_id):
        raise ValueError(f"invalid MegaSAM video id: {video_id}")
    root = ROOT / "third_party/mega_sam"
    for directory in ("work_space", "cache_flow", "reconstructions"):
        path = root / directory / video_id
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
    for directory, suffix in (("outputs", "_droid.npz"),
                              ("outputs_cvd", "_sgd_cvd_hr.npz")):
        path = root / directory / f"{video_id}{suffix}"
        if path.is_file() and not path.is_symlink():
            path.unlink()


def ensure_base_segmentation(args: ExperimentSpec, entry: dict, case: Path,
                             source: Path, log: Path,
                             required_links: tuple[str, ...] | None = None,
                             selected_target: str | None = None) -> Path:
    base = case / "base_segmentation.npz"
    provenance = case / "base_segmentation_source.json"
    expected_names = (selected_target,) if selected_target else None
    method = ("DINOv2-guided SAM3 full-video single-link segmentation"
              if selected_target else "DINOv2-guided SAM3 full-video six-link segmentation")
    if base.exists() or provenance.exists():
        if not mask_is_valid(base, expected_names) or not provenance.is_file():
            raise ValueError(f"incomplete or invalid base segmentation: {case}")
        record = json.loads(provenance.read_text(encoding="utf-8"))
        if (record.get("source_video_sha256") != entry["sha256"]
                or record.get("base_segmentation_sha256") != sha256_file(base)):
            raise ValueError(f"base segmentation provenance mismatch: {case}")
        if args.generate_base_masks and record.get("method") != method:
            raise ValueError(f"base mask was not generated in this run: {case}")
        if args.workflow in {"selected45_link5_link7_four_way", "selected45_link5_only"}:
            segmentation_hash = sha256_file(ROOT / "src/pdi_eval/perception/sam3_dinov2_segment.py")
            guard_hash = sha256_file(ROOT / "src/pdi_eval/perception/link5_point_guard.py")
            if (record.get("segmentation_source_sha256") not in
                    (COMPATIBLE_SEGMENTATION_SHA256 | {segmentation_hash})
                    or record.get("link5_guard_source_sha256") not in
                    (COMPATIBLE_LINK5_GUARD_SHA256 | {guard_hash})):
                raise ValueError(f"base mask implementation differs on resume: {case}")
        if required_links is None:
            validate_mask_coverage(base, defer_low_link7=True)
        elif any(mask_coverage(base)[name] < MINIMUM_TRACKED_FRACTION for name in required_links):
            raise ValueError(f"base segmentation lacks required link coverage: {required_links}")
        return base
    if not args.generate_base_masks:
        raise ValueError(f"validated base mask must be staged for this case: {case}")
    assert all((args.segmentation_python, args.reference_dir, args.dinov2_model,
                args.sam3_checkpoint, args.sam3_bpe))
    generated = case / "base_generation" / "segmentation.npz"
    generated.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1")
    argv = [str(args.segmentation_python), "-u", "-m",
            "pdi_eval.perception.sam3_dinov2_segment", "--input", str(source),
            "--reference-dir", str(args.reference_dir), "--output-npz", str(generated),
            "--dinov2-model", str(args.dinov2_model),
            "--sam3-checkpoint", str(args.sam3_checkpoint),
            "--sam3-bpe", str(args.sam3_bpe), "--text-prompt", "visual",
            "--require-franka-links", "--link5-vlm-guard",
            "--reference-spatial-priors",
            "--padding-fraction", "0.10", "--minimum-tracked-fraction", "0",
            "--link-text-prompt", "link4=entire white oval on top of the black circle",
            "--link-text-prompt", "link7=entire white quadrangular robot gripper"]
    if selected_target:
        argv += ["--selected-target", selected_target]
    if not mask_is_valid(generated, expected_names):
        command(argv, log, environment, args.gpu_lock)
    if not mask_is_valid(generated, expected_names):
        raise ValueError(f"generated segmentation is invalid: {generated}")
    if required_links is None:
        coverage, low = validate_mask_coverage(generated, defer_low_link7=True)
    else:
        coverage = mask_coverage(generated)
        low = tuple(name for name, fraction in coverage.items()
                    if fraction < MINIMUM_TRACKED_FRACTION)
        if any(name in low for name in required_links):
            raise ValueError(f"generated mask lacks required link coverage: {required_links}")
    shutil.copy2(generated, base)
    write_json(provenance, {
        "source_video_sha256": entry["sha256"],
        "base_segmentation_sha256": sha256_file(base),
        "method": method,
        "selected_target": selected_target,
        "reference_dir": str(args.reference_dir),
        "sam3_prompt_profile": "descriptive link4/link5/link7; link5 three-positive three-negative points reviewed in two VLM calls",
        "minimum_tracked_fraction_for_scoring": MINIMUM_TRACKED_FRACTION,
        "tracked_fraction": coverage,
        "low_coverage_links_skipped_in_scoring": [
            name for name in low if name in SKIPPABLE_LOW_COVERAGE_LINKS
        ],
        "low_coverage_link7_deferred_to_persistent_masking": "link7" in low,
        "temporal_disambiguation_retry": False,
        "segmentation_source_sha256": sha256_file(ROOT / "src/pdi_eval/perception/sam3_dinov2_segment.py"),
        "link5_guard_source_sha256": sha256_file(ROOT / "src/pdi_eval/perception/link5_point_guard.py"),
    })
    return base


def mask_environment(args: ExperimentSpec, case: str) -> dict[str, str]:
    from persistent_masking.interface.secrets import load_env_file

    load_env_file(ROOT / ".env.vlm")
    result = os.environ.copy()
    result.update(
        PYTHONPATH=str(ROOT),
        PERSISTENT_MASKING_VIDEO_ROOT=str(args.output_root / "source_video_links"),
        PDI_PMASK_RUN_NAME=args.output_root.name.replace(".", "_").replace("-", "_"),
        PDI_PMASK_LEVEL="local", PDI_PMASK_CASES=case,
        PDI_PMASK_VIDEO_ROOT=str(args.output_root / "source_video_links"),
        PDI_PMASK_WORK_ROOT=str(args.output_root / "persistent_work"),
        PDI_PMASK_REVIEW_ROOT=str(args.output_root / "mask_review"),
        PDI_PMASK_INCLUDE_NAIVE_REPLAY="0",
        PDI_PMASK_QWEN_MODEL=str(args.qwen_model),
        PDI_PMASK_QWEN_PYTHON=str(args.qwen_python),
    )
    return result


def persistent_mask(args: ExperimentSpec, entry: dict, log: Path) -> tuple[Path, dict]:
    case = case_name(entry)
    work = mask_work(args.output_root, case)
    environment = mask_environment(args, case)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required for persistent masking")
    common = ["--work", str(work), "--cases", case, "--level", "local",
              "--ffmpeg", ffmpeg, "--examples", *(
                  str(ROOT / f"persistent_masking/vlm_interface/images/reference_{i}.png")
                  for i in range(1, 4))]
    provenance = work / "provenance.json"
    # The masking pipeline persists its terminal outcome. Keep that outcome on
    # resume so a malformed VLM response cannot make the whole video retry the
    # same failed refinement indefinitely. Its current-run base mask remains
    # available for V1 scoring below.
    if not provenance.is_file():
        naive = work / "naive" / case / "provenance.json"
        if not naive.is_file():
            command([str(args.sam_python), "-u", "-m", "persistent_masking.naive_sam3",
                     "--work", str(work / "naive"), "--cases", case, "--ffmpeg", ffmpeg],
                    log, environment, args.gpu_lock)
        command([str(args.sam_python), "-u", "-m", "persistent_masking.pipeline",
                 "prepare", *common], log, environment)
    for stage, lock in (("select_frames", args.gpu_lock), ("prompt_sam", None),
                        ("segment", args.gpu_lock), ("validate", None)):
        python = args.qwen_python if stage == "select_frames" else args.sam_python
        command([str(python), "-u", "-m", "persistent_masking.pipeline",
                 stage, *common], log, environment, lock)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    result = record["results"][case]
    if result.get("source_sha256") != entry["sha256"]:
        raise ValueError(f"persistent mask source differs from frozen selection: {case}")
    status = result["status"]
    if status == "completed_checks":
        source = work / "sam" / case / "seed_masks.npz"
        checked_file(source, "persistent full-video mask")
        if result.get("masks_sha256") != sha256_file(source):
            raise ValueError(f"persistent mask hash mismatch: {case}")
    elif status not in {"no_confirmed_deformation", "reseed_frame_unavailable",
                        "failed_vlm2", "failed_sam", "failed_mask_validation",
                        "failed_validation", "failed_prepare"}:
        raise RuntimeError(f"persistent mask did not validate: {case}: {status}")
    review = args.output_root / "mask_review" / "local" / work.parent.name / case
    if not review.exists():
        command([str(args.sam_python), "-m", "persistent_masking.export_selected_replay",
                 "--work", str(work), "--destination", str(review)], log, environment)
    return work, result


def run_case(args: ExperimentSpec, entry: dict) -> bool:
    from .mask_merge import build_refined_segmentation

    sample = f"{entry['dataset']}_{entry['video_number']}"
    case = args.output_root / "cases" / sample
    case.mkdir(parents=True, exist_ok=True)
    log = case / "end_to_end.log"
    status_path = case / "status.json"
    source = video_path(args.video_root, entry)
    refined = case / "refined_segmentation.npz"
    alias = case / f"{sample.replace('.', '_')}.mp4"
    status = {"sample_id": sample, "workbook_row": entry["workbook_row"],
              "source_sha256": entry["sha256"], "state": "running",
              "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(status_path, status)
    failure_traceback = None
    try:
        checked_file(source, "source video")
        if source.stat().st_size != entry["size_bytes"] or sha256_file(source) != entry["sha256"]:
            raise ValueError("source video differs from frozen 197-video selection")
        base = ensure_base_segmentation(args, entry, case, source, log)
        ensure_video_alias(alias, source, entry["sha256"])
        source_link = args.output_root / "source_video_links" / MASK_VIDEO_FOLDERS[entry["dataset"]] / f"{entry['video_number']}.mp4"
        ensure_video_alias(source_link, source, entry["sha256"])
        work, mask_record = persistent_mask(args, entry, log)
        status["persistent_mask_status"] = mask_record["status"]
        if mask_record["status"] == "completed_checks":
            if not mask_is_valid(refined):
                build_refined_segmentation(video=alias, base_segmentation=base,
                    persistent_work=work, case=case_name(entry), output_npz=refined)
            if not mask_is_valid(refined):
                raise ValueError("refined six-link segmentation is invalid")
            selected_mask = refined
            status["mask_policy"] = "validated persistent link7 replacement"
        else:
            selected_mask = base
            status["mask_policy"] = (
                f"base mask; persistent mask status {mask_record['status']}"
            )
        coverage, low = validate_mask_coverage(selected_mask)
        status["sam3_tracked_fraction"] = coverage
        status["low_coverage_links"] = list(low)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        destination = case / "v1"
        selected_mask_sha256 = sha256_file(selected_mask)
        if not v1_outputs_complete(
            destination, video_sha256=entry["sha256"],
            segmentation_sha256=selected_mask_sha256,
        ):
            command(score_argv(args, alias, selected_mask, destination, case),
                    log, environment, args.gpu_lock)
            if not v1_outputs_complete(
                destination, video_sha256=entry["sha256"],
                segmentation_sha256=selected_mask_sha256,
            ):
                raise RuntimeError("V1 scorer exited without complete matching metrics and replays")
        metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
        object_reports = metrics["modes"]["exact-group"]["objects"]
        status["scored_links"] = [name for name in LINK_NAMES
                                  if object_reports[name]["status"] == "complete"]
        status["unscored_links"] = {
            name: object_reports[name].get("error_type", object_reports[name]["status"])
            for name in LINK_NAMES if object_reports[name]["status"] != "complete"
        }
        status.update(state="complete", completed_at=datetime.now(timezone.utc).isoformat(),
                      scoring_segmentation_sha256=selected_mask_sha256)
    except Exception as error:
        status.update(state="failed", error=str(error),
                      completed_at=datetime.now(timezone.utc).isoformat())
        failure_traceback = traceback.format_exc()
    try:
        clean_megasam_intermediates(sample)
    except Exception:
        with log.open("a", encoding="utf-8") as output:
            output.write("MegaSAM scratch cleanup failed:\n" + traceback.format_exc())
    if failure_traceback:
        with log.open("a", encoding="utf-8") as output:
            output.write(failure_traceback)
    write_json(status_path, status)
    print(f"{sample}: {status['state']}", flush=True)
    return status["state"] == "complete"


def run_experiment(args: ExperimentSpec) -> int:
    selection = read_selection(args.output_root)
    if args.selection != args.output_root / "selection.json":
        raise ValueError("selection must be the output root's selection.json")
    if sha256_file(args.workbook) != selection["workbook_sha256"]:
        raise ValueError("workbook differs from frozen selection")
    for path, label in ((args.sam_python, "SAM/Qwen Python"),
                        (args.qwen_python, "Qwen Python"),
                        (args.pdi_python, "PDI Python"),
                        (args.tracker_checkpoint, "CoTracker checkpoint")):
        checked_file(path, label)
    if not args.qwen_model.is_dir():
        raise FileNotFoundError(args.qwen_model)
    if not (ROOT / "assets/replay/plotly.min.js").is_file():
        raise FileNotFoundError("offline Plotly replay asset")
    checked_file(ROOT / "third_party/mega_sam/checkpoints/megasam_final.pth",
                 "MegaSAM checkpoint")
    with ThreadPoolExecutor(max_workers=args.concurrency) as workers:
        futures = {workers.submit(run_case, args, entry): entry
                   for entry in selection["videos"]}
        success = True
        for future in as_completed(futures):
            success = future.result() and success
            page_index(args.output_root, selection)
    return 0 if success else 1
