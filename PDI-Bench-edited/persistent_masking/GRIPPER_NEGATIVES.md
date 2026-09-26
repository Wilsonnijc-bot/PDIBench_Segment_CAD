# Gripper negative-point replay test

The accepted visual target is `qwen_results/Cosmos3_0048/masking.mp4`: follow the gripper through deformation, excluding the long forearm and upper white wrist housing. The three existing Qwen positive coordinates are reused unchanged. There is no junction-negative mode.

`gripper_negatives.py` reads the retained Qwen provenance directly. `ground` requests arm and wrist boxes and points; `segment` submits those points to SAM3 and propagates in both directions over the entire source video; `replay` writes source RGB beside the cyan mask at source FPS. Geometry checks and prompt membership are diagnostics, not segmentation accuracy. The replay remains the result to judge.

On the GPU host, from the project root:

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 env-qwen/bin/python -m persistent_masking.gripper_negatives ground --work negative_work
OMP_NUM_THREADS=2 env-sam/bin/python -m persistent_masking.gripper_negatives segment --work negative_work
FFMPEG_BINARY=/path/to/ffmpeg env-sam/bin/python -m persistent_masking.gripper_negatives replay --work negative_work
```

`--cases` selects cases; `--overwrite` explicitly repeats an existing stage. `--targets wrist --full-context` supports the contextual wrist retry used for Cosmos3_0048. `--mode positive` and `--mode arm` are optional internal controls; the delivered test uses `both`. Four existing cases without valid three-positive correction seeds are skipped explicitly.

Use `python -m persistent_masking.export_gripper_negatives --work <downloaded-work>` to publish the current tests. Only inputs, combined point previews, and `masking.mp4` enter the viewing folders. Compact coordinates, prompts, raw answers and validation enter `qwen_results/provenance/arm_wrist.json`. Internal masks and diagnostics stay outside that tree. The exporter publishes actual tests, including inaccurate placements and mask failures; it does not claim automatic acceptance.
