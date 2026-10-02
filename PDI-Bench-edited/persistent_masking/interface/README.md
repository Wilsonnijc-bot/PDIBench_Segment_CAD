# Persistent masking: configure and run

This is the everyday interface for the naive SAM3 path, VLM-guided repair,
validation, and a small review folder. You do not need to call the individual
pipeline scripts.

## 1. Edit one configuration file

Open [`config.py`](config.py) and set:

| Setting | Meaning |
| --- | --- |
| `RUN_NAME` | A new, unique name for this attempt. Existing work and review folders are never overwritten. |
| `LEVEL` | `local` for one diagnostic case, or `global` for a frozen set of cases, including regression cases. |
| `CASES` | Video IDs such as `Cosmos3_0048`, `Cosmos25_0003`, or `LVP_0049`. |
| `VIDEO_ROOT` | Directory containing the three source-video dataset folders. |
| `WORK_ROOT` | Intermediate masks, logs, and full provenance. |
| `REVIEW_ROOT` | Compact images and replays to inspect. |
| `INCLUDE_NAIVE_REPLAY` | Include the existing frame-zero SAM3 replay beside the refined replay. |
| `VLM1` | First-deformation-frame selection: choose `local_gpu` or `cloud_api`, then model and backend settings. |
| `VLM2` | SAM3 point prompting: choose `local_gpu` or `cloud_api`, then model and backend settings. |
| `VLM2_MALFORMED_FALLBACK` | One GPT-6 Luna request with high reasoning if a VLM2 point response is malformed. Uses the same 302.ai URL and VLM2 API key. |

The defaults select local Qwen3.5-9B for VLM1 and Gemini 3.8 Flash through
302.AI for VLM2. Each VLM can be configured independently. A valid
`positive_points: null` still advances to the next confirmed deformation
frame. A malformed VLM2 response triggers one Luna request for that point
prompt; both responses are retained in provenance. The result must pass the
same point validation before SAM3 runs.

For a cloud role, put the corresponding key in the ignored project-root
`.env.vlm` file, using the variable named by that role's `api_key_env` setting.
For example, with the default VLM2 setting:

```text
VLM2_API_KEY=your-key-here
```

Use plain `KEY=value` lines (quotes and `export` are optional). Do not put
secrets in `config.py` or commit `.env.vlm`.

## 2. Check, run, inspect

Run these commands from the project root on the configured GPU host after
following [`setup/`](../../setup/):

```bash
./persistent_masking/pmask check
./persistent_masking/pmask run --detach
./persistent_masking/pmask status
```

`check` verifies paths, models, API-key presence, FFmpeg, and CUDA without
running inference or making an API call. Drop `--detach` to run in the
foreground. Detached mode uses tmux; the work folder's `run.log` remains the
durable log even after the tmux session exits. A cloud run may incur API costs.

If the pipeline completed but review export did not, keep the same config and
run `./persistent_masking/pmask export`. Export requires a new review directory
and never overwrites an existing one. For another experiment, choose a new
`RUN_NAME` before `run`.

## Where the files go

For `LEVEL = "local"`, intermediate work goes to
`WORK_ROOT/local/RUN_NAME/CASE/`. For `LEVEL = "global"`, it goes to
`WORK_ROOT/global/RUN_NAME/`, with all configured cases together. Work contains
the actual naive masks, pipeline artifacts, `provenance.json`, and `run.log`.

The compact review is at `REVIEW_ROOT/LEVEL/RUN_NAME/`, with one folder per
case. A completed case contains:

```text
CASE/
  qwen_input_1.png      # vlm1_input_1.png if VLM1 is not Qwen
  qwen_input_2.png
  qwen_input_3.png
  first_deformed_frame.png
  points.png
  masking.mp4
  naive_masking.mp4       # when INCLUDE_NAIVE_REPLAY is True
```

The review root also has a short `README.md` and
`provenance/{run,manifest}.json`. A failed or skipped case is reported there
without fabricated images or a refined replay. These are experiment results,
not promoted canonical results; review a complete global run before promotion.

For developer understanding of the exact VLM2 system prompt and annotated
images, see [`vlm_interface/`](../vlm_interface/README.md). It remains a
human-readable reference, not a second place to configure models.
