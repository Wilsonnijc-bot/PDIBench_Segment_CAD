"""Prompts for the two role-based VLM stages."""
from generation.link_crop_wrapper.run_segment_vlm import LOOSE_V2_SYSTEM_PROMPT
from persistent_masking.palm_recovery import PALM_FOCUS, PALM_RESPONSE, PALM_VISIBILITY


VLM1_SYSTEM_PROMPT = LOOSE_V2_SYSTEM_PROMPT + "\n\n" + PALM_VISIBILITY + "\n\n" + PALM_FOCUS
VLM1_RESPONSE_REQUEST = PALM_RESPONSE

VLM2_SYSTEM_PROMPT = """You are a SAM3 prompting expert. Your objective is to help SAM3
mask the entire gripper correctly, starting from Image 1, the source-video frame
on which SAM3 will be seeded. Images 2,
3, and 4 are annotated point-placement references. Inspect the images at original
detail, follow the requested anatomical roles exactly, and return only the JSON
schema requested by the user task. Coordinates are integer x,y values normalized
to 0 through 1000 relative to Image 1. Never infer points on invisible material."""

VLM2_POSITIVE_PROMPT = """Place three positive SAM points. Return exactly THREE points in this order:
(1) FIRST DARK POINT, strictly inside lower dark/black gripper material below the
white palm and shiny collar, toward the hanging fingers; choose a broad interior
and avoid rim, forearm trim, collar, wrist, white palm, background, thin outlines,
and held objects. (2) SECOND DARK POINT, strictly inside a different, clearly
spatially separated lower dark/black material region, with the same exclusions.
(3) WHITE PALM POINT, at the center of the hanging WHITE gripper block below the
shiny collar, away from collar, dark rim, fingers, wrist, held object, background,
and boundaries. Return only JSON
{\"positive_points\":[[dark1_x,dark1_y],[dark2_x,dark2_y],[white_x,white_y]]}.
If any required location is not visible, return {\"positive_points\":null}."""

VLM2_NEGATIVE_PROMPT = """Place two exclusion points for SAM. Return exactly TWO points in this order:
(1) ARM NEGATIVE, well inside the long forearm link with black rim and white inset,
between elbow and terminal wrist housing, in its elbow-side half; exclude outlines,
joints, gripper, held object, and background. (2) WRIST NEGATIVE, well inside the
white upright end-cap at the gripper-side end of the black-rimmed forearm, on its
upper white surface near the dark top cap; exclude outlines, joints, gripper, held
object, and background. Return only JSON
{\"negative_points\":[[arm_x,arm_y],[wrist_x,wrist_y]]}. If either location is
not identifiable, return {\"negative_points\":null}."""
