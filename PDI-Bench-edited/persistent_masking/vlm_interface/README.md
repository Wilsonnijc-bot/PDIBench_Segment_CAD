# VLM configuration, prompts, and reference images

## Model configuration

Edit only [the user-facing configuration](../interface/config.py) to change either
VLM role between a local GPU model and a cloud API. This folder is retained as
the developer-readable record of the prompts and reference images sent to VLM2;
its `config.py` is a compatibility import for existing code.

The current selection is:

- **VLM1 — deformation-frame selection:** local GPU, `Qwen3.5-9B`.
- **VLM2 — SAM point prompting:** 302.AI China endpoint, Gemini 3.8 Flash (`gemini-3.8-flash`).

API secrets are not stored in the configuration file; the configured environment-variable
name resolves through the ignored project-root `.env.vlm` file.

## VLM2 prompt interface

This is the small human-readable interface for VLM2, the SAM point-prompting
model. The three images below are the default annotated references sent as
Images 2–4. Image 1 is the run-specific reseeding frame and is written into the
run's own `vlm2_interface/images/target_frame.png` when VLM2 is called.

## System prompt

```text
You are a SAM3 prompting expert. Your objective is to help SAM3
mask the entire gripper correctly, starting from Image 1, the source-video frame
on which SAM3 will be seeded. Images 2,
3, and 4 are annotated point-placement references. Inspect the images at original
detail, follow the requested anatomical roles exactly, and return only the JSON
schema requested by the user task. Coordinates are integer x,y values normalized
to 0 through 1000 relative to Image 1. Never infer points on invisible material.
```

## Positive-point task

```text
Place three positive SAM points. Return exactly THREE points in this order:
(1) FIRST DARK POINT, strictly inside lower dark/black gripper material below the
white palm and shiny collar, toward the hanging fingers; choose a broad interior
and avoid rim, forearm trim, collar, wrist, white palm, background, thin outlines,
and held objects. (2) SECOND DARK POINT, strictly inside a different, clearly
spatially separated lower dark/black material region, with the same exclusions.
(3) WHITE PALM POINT, at the center of the hanging WHITE gripper block below the
shiny collar, away from collar, dark rim, fingers, wrist, held object, background,
and boundaries. Return only JSON
{"positive_points":[[dark1_x,dark1_y],[dark2_x,dark2_y],[white_x,white_y]]}.
If any required location is not visible, return {"positive_points":null}.
```

## Negative-point task

```text
Place two exclusion points for SAM. Return exactly TWO points in this order:
(1) ARM NEGATIVE, well inside the long forearm link with black rim and white inset,
between elbow and terminal wrist housing, in its elbow-side half; exclude outlines,
joints, gripper, held object, and background. (2) WRIST NEGATIVE, well inside the
white upright end-cap at the gripper-side end of the black-rimmed forearm, on its
upper white surface near the dark top cap; exclude outlines, joints, gripper, held
object, and background. Return only JSON
{"negative_points":[[arm_x,arm_y],[wrist_x,wrist_y]]}. If either location is
not identifiable, return {"negative_points":null}.
```

## Reference images sent to VLM2

### Image 2 — annotated reference 1

![Annotated reference 1](images/reference_1.png)

### Image 3 — annotated reference 2

![Annotated reference 2](images/reference_2.png)

### Image 4 — annotated reference 3

![Annotated reference 3](images/reference_3.png)
