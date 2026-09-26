# VLM1–VLM2–SAM3 pipeline

See [the complete specification](PIPELINE_SPEC.md). The runner is
`persistent_masking.pipeline`; point placement uses
`persistent_masking.vlm2_sam_prompting` and the backend-neutral
`persistent_masking.vlm_client`.

VLM2 receives exactly four images: original video frame `earliest_deformed + 1`,
then three annotated examples. VLM1 alone receives the nondeformed reference.
Positive output order is dark1, dark2, white; negative output order is forearm,
wrist. SAM3 reseeds on the identical frame.

Configure both roles only in [`vlm_interface/config.py`](vlm_interface/config.py). Set either role
to `local_gpu`, or use `cloud_api` with `chat_completions` or `responses`. The
latest exact VLM2 request is written to `vlm2_interface/README.md`, with the
target and reference PNGs beside it, in both its case artifact directory and
the run root.
