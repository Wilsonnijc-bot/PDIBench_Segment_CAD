"""Simple check/run/status/export commands for the persistent masking pipeline."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from persistent_masking.interface.config import PROJECT_ROOT, role_config, run_config
from persistent_masking.interface.secrets import load_env_file


DATASET_FOLDERS = {
    "LVP": "1irS6zoWSykw64DuaiwxyUVHdIffEo3Oa",
    "Cosmos25": "1rXdVXrwIjf9wMeCVxbmFWN7cZNcfdtb9",
    "Cosmos3": "15sRTmHwkBQO0FmDRnp0BePAArCad2_Gj",
}


def locations(config):
    work = config["work_root"] / config["level"] / config["name"]
    if config["level"] == "local":
        work /= config["cases"][0]
    review = config["review_root"] / config["level"] / config["name"]
    return work, review


def video_path(config, case):
    dataset, number = case.split("_")
    return config["video_root"] / DATASET_FOLDERS[dataset] / f"{number}.mp4"


def ffmpeg_path(config):
    selected = config["ffmpeg_binary"] or os.environ.get("FFMPEG_BINARY")
    if selected:
        return Path(selected).expanduser().resolve()
    found = shutil.which("ffmpeg")
    if found:
        return Path(found).resolve()
    try:
        import imageio_ffmpeg
        return Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    except (ImportError, RuntimeError):
        return None


def preflight(config, *, require_new):
    """Check requirements without calling paid APIs or starting model inference."""
    errors = []
    work, review = locations(config)
    if require_new:
        if work.exists():
            errors.append(f"Work directory already exists: {work}; choose a new RUN_NAME")
        if review.exists():
            errors.append(f"Review directory already exists: {review}; choose a new RUN_NAME")
    sam_python = PROJECT_ROOT / "env-sam/bin/python"
    if not sam_python.is_file():
        errors.append(f"Missing SAM environment: {sam_python}; see setup/README.md")
    for role in ("vlm1", "vlm2"):
        settings = role_config(role)
        if settings["backend"] == "local_gpu":
            if not Path(settings["model"]).exists():
                errors.append(f"{role.upper()} model is missing: {settings['model']}")
            if not Path(settings["python"]).is_file():
                errors.append(f"{role.upper()} Python environment is missing: {settings['python']}")
        elif not os.environ.get(settings["api_key_env"]):
            errors.append(f"{role.upper()} needs {settings['api_key_env']} in .env.vlm or the environment")
    for path in (PROJECT_ROOT / "models/sam3/sam3.pt",
                 PROJECT_ROOT / "models/dinov2"):
        if not path.exists():
            errors.append(f"Required model/checkpoint is missing: {path}")
    references = PROJECT_ROOT / "results_v1/references/by_link/palm"
    if not any(references.glob("*.png")):
        errors.append(f"Missing naive-SAM reference images: {references}")
    for number in range(1, 4):
        path = PROJECT_ROOT / f"persistent_masking/vlm_interface/images/reference_{number}.png"
        if not path.is_file():
            errors.append(f"Missing VLM2 reference image: {path}")
    for case in config["cases"]:
        path = video_path(config, case)
        if not path.is_file() or not path.stat().st_size:
            errors.append(f"Missing or empty source video for {case}: {path}")
    ffmpeg = ffmpeg_path(config)
    if ffmpeg is None or not ffmpeg.is_file() or not os.access(ffmpeg, os.X_OK):
        errors.append("FFmpeg not found; set FFMPEG_BINARY in interface/config.py")
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA GPU is unavailable; run on the configured GPU host")
        else:
            assert (torch.ones(1, device="cuda") + 1).item() == 2
    except (ImportError, AssertionError, RuntimeError) as error:
        errors.append(f"CUDA check failed: {type(error).__name__}")
    return errors, ffmpeg


def show(config, errors, ffmpeg):
    work, review = locations(config)
    print(f"Run: {config['name']} ({config['level']})")
    print(f"Cases: {', '.join(config['cases'])}")
    for role, job in (("vlm1", "frame selection"), ("vlm2", "SAM point prompting")):
        settings = role_config(role)
        print(f"{role.upper()} ({job}): {settings['model']} [{settings['backend']}]")
    print(f"Work: {work}")
    print(f"Review: {review}")
    print(f"Naive replay in review: {'yes' if config['include_naive_replay'] else 'no'}")
    if ffmpeg:
        print(f"FFmpeg: {ffmpeg}")
    if errors:
        print("Not ready:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Ready. No model or API call was made.")
    return 0


def run_stage(name, command, *, env, log):
    print(f"Running {name}", flush=True)
    log.write("COMMAND " + shlex.join(command) + "\n")
    log.flush()
    with subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    log.write(f"STAGE_EXIT_STATUS={code}\n")
    log.flush()
    return code


def export_review(config, work, review):
    command = [sys.executable, "-m", "persistent_masking.export_selected_replay",
               "--work", str(work), "--destination", str(review)]
    if config["include_naive_replay"]:
        command.append("--include-naive")
    return subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=False).returncode


def run_foreground(config, ffmpeg):
    work, review = locations(config)
    work.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["PERSISTENT_MASKING_VIDEO_ROOT"] = str(config["video_root"])
    env["FFMPEG_BINARY"] = str(ffmpeg)
    examples = [str(PROJECT_ROOT / f"persistent_masking/vlm_interface/images/reference_{n}.png")
                for n in range(1, 4)]
    cases = config["cases"]
    naive = [sys.executable, "-u", "-m", "persistent_masking.naive_sam3",
             "--work", str(work / "naive"), "--cases", *cases, "--ffmpeg", str(ffmpeg)]
    pipeline = [sys.executable, "-u", "-m", "persistent_masking.pipeline", "continue",
                "--level", config["level"], "--work", str(work), "--cases", *cases,
                "--ffmpeg", str(ffmpeg), "--examples", *examples]
    result = 1
    with (work / "run.log").open("w") as log:
        try:
            if run_stage("naive SAM3", naive, env=env, log=log) or run_stage("refined pipeline", pipeline, env=env, log=log):
                print(f"Run failed. See {work / 'run.log'}")
                return 1
            record = json.loads((work / "provenance.json").read_text())
            statuses = {case: record["results"][case]["status"] for case in cases}
            if export_review(config, work, review):
                print(f"Review export failed; run data remains at {work}")
                return 1
            print(f"Review ready: {review}")
            for case, status in statuses.items():
                print(f"  {case}: {status}")
            result = 0 if all(status == "completed_checks" for status in statuses.values()) else 1
            return result
        finally:
            log.write(f"EXIT_STATUS={result}\n")
            log.flush()


def run_detached(config):
    tmux = shutil.which("tmux")
    if not tmux:
        raise ValueError("tmux is unavailable; use `run` without --detach")
    session = f"pm_{config['level']}_{config['name']}"
    if subprocess.run([tmux, "has-session", "-t", session],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise ValueError(f"tmux session already exists: {session}")
    inner = f"cd {shlex.quote(str(PROJECT_ROOT))} && exec {shlex.join([sys.executable, '-m', 'persistent_masking.interface', 'run'])}"
    subprocess.run([tmux, "new-session", "-d", "-s", session, inner], check=True)
    print(f"Started tmux session: {session}")
    print(f"Check it with: ./persistent_masking/pmask status")


def status(config):
    work, review = locations(config)
    print(f"Run: {config['name']} ({config['level']})")
    print(f"Work: {work}")
    print(f"Review: {review if review.exists() else 'not exported yet'}")
    record_path = work / "provenance.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        print(f"Pipeline status: {record['status']}")
        for case in record["cases"]:
            print(f"  {case}: {record['results'].get(case, {}).get('status', 'pending')}")
    elif work.exists():
        print("Pipeline status: naive SAM3 is running or preparing")
    else:
        print("Pipeline status: not started")
    log = work / "run.log"
    if log.is_file():
        markers = [line for line in log.read_text(errors="replace").splitlines()
                   if line.startswith("EXIT_STATUS=")]
        if markers:
            print(markers[-1])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Configure, run, and review persistent masking")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Inspect the run and requirements; make no model/API calls")
    run_parser = commands.add_parser("run", help="Run configured cases and export review files")
    run_parser.add_argument("--detach", action="store_true", help="Keep running in a tmux session")
    commands.add_parser("status", help="Show progress and per-case results")
    commands.add_parser("export", help="Export an existing run into a new review directory")
    args = parser.parse_args(argv)
    try:
        config = run_config()
        load_env_file(PROJECT_ROOT / ".env.vlm")
        if args.command == "status":
            status(config)
            return 0
        if args.command == "export":
            work, review = locations(config)
            if not (work / "provenance.json").is_file():
                raise ValueError(f"Run provenance is missing: {work / 'provenance.json'}")
            if review.exists():
                raise ValueError(f"Review already exists: {review}; exports never overwrite it")
            return export_review(config, work, review)
        errors, ffmpeg = preflight(config, require_new=True)
        if args.command == "check":
            return show(config, errors, ffmpeg)
        if errors:
            raise ValueError("Run preflight failed:\n- " + "\n- ".join(errors))
        if args.detach:
            run_detached(config)
            return 0
        return run_foreground(config, ffmpeg)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
