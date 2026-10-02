"""Independent VLM reviews of Link 5 wrist negatives and forearm positives."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_FRAME = PROJECT_ROOT / "assets/link5_guard/lvp040_reference_frame.png"
REFERENCE_BOX = (452, 0, 818, 132)
REFERENCE_POINTS = [(510, 67), (570, 67), (770, 40),
                    (495, 27), (478, 92), (807, 82)]
GUARD_MODEL = "gemini-3.8-flash"
REVIEW_SPECS = {
    "negative": {
        "indices": (3, 4),
        "field": "negative_points_xy",
        "system_prompt": (
            "You review SAM3 point prompts for the robot's Link 5 forearm. "
            "Image 1 is a full-frame annotated example of ideal N1 and N2 "
            "placement; Image 2 is the current full frame with the proposed Link 5 "
            "box and N1 and N2. Both images show only these two red point markers. "
            "Reason carefully from the images, but return only the requested "
            "answer. Judge only the red negative points N1 and N2."
        ),
    },
    "positive": {
        "indices": (1, 2),
        "field": "positive_points_xy",
        "system_prompt": (
            "You review SAM3 point prompts for the robot's Link 5 forearm. "
            "Image 1 is a full-frame annotated example of ideal P2 and P3 "
            "placement; Image 2 is the current full frame with the proposed Link 5 "
            "box and P2 and P3. Both images show only these two green point markers. "
            "Reason carefully from the images, but return only the requested "
            "answer. Judge only the green positive points P2 and P3."
        ),
    },
}


def _annotate(image, box, points, indices):
    from PIL import ImageDraw, ImageFont

    image = image.convert("RGB").resize((image.width * 2, image.height * 2))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    green = (157, 255, 55)
    red = (255, 76, 54)
    draw.rectangle(tuple(int(v * 2) for v in box), outline=green, width=4)
    for index in indices:
        x, y = (int(v * 2) for v in points[index])
        color = green if index < 3 else red
        radius = 7
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     fill=color, outline=(15, 26, 24), width=2)
        label = f"P{index + 1}" if index < 3 else f"N{index - 2}"
        draw.text((x + radius + 3, y - radius - 3), label, fill=(0, 0, 0),
                  stroke_width=3, stroke_fill=(255, 255, 255), font=font)
    return image


def _parse_decision(answer: str, task: str, width: int, height: int,
                    original: np.ndarray) -> tuple[str, np.ndarray | None]:
    """Validate response shape and frame safety only; no anatomy or material veto."""
    answer = answer.strip()
    if answer.startswith("```") and answer.endswith("```"):
        lines = answer.splitlines()
        if len(lines) >= 3 and lines[0].lower() in {"```", "```json"}:
            answer = "\n".join(lines[1:-1]).strip()
    if answer == "PASS":
        return "PASS", None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError("guard answer must be PASS or a REJECT JSON object") from exc
    field = REVIEW_SPECS[task]["field"]
    if not isinstance(data, dict) or data.get("decision") != "REJECT":
        raise ValueError("guard REJECT answer has the wrong decision")
    if set(data) != {"decision", field}:
        raise ValueError("guard REJECT answer has unexpected fields")
    raw = data[field]
    if (not isinstance(raw, list) or len(raw) != 2
            or any(not isinstance(point, list) or len(point) != 2
                   or any(type(value) is not int for value in point)
                   for point in raw)):
        raise ValueError("guard REJECT answer needs exactly two integer [x,y] points")
    replacement = np.asarray(raw, dtype=np.int32)
    if (np.any(replacement < 0) or np.any(replacement[:, 0] >= width)
            or np.any(replacement[:, 1] >= height)):
        raise ValueError("guard coordinates are outside the full frame")
    if np.linalg.norm(replacement[0] - replacement[1]) < 8:
        raise ValueError("guard points are too close together")
    fixed_indices = (0, 1, 2, 5) if task == "negative" else (0, 3, 4, 5)
    if np.any(np.all(replacement[:, None] == original[list(fixed_indices)][None], axis=2)):
        raise ValueError("guard point coincides with a fixed opposite-label point")
    return "REJECT", replacement


def _review(task, images, prompt, width, height, original):
    from persistent_masking.interface.config import role_config
    from persistent_masking.vlm_client import VLMClient

    guard_config = role_config("vlm2")
    guard_config["model"] = GUARD_MODEL
    client = VLMClient(f"link5_{task}_guard", guard_config)
    try:
        record = client.ask(
            images, prompt, system_prompt=REVIEW_SPECS[task]["system_prompt"]
        )
    except Exception as exc:
        reason = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        client_records = getattr(client, "records", [])
        record = {
            "role": f"link5_{task}_guard", "requested_model": client.model,
            "response_error": reason,
            "response_diagnostic": (client_records[-1].get("response_diagnostic")
                                    if client_records else None),
        }
        return "DEFAULT_UNREVIEWED", None, record, reason
    try:
        decision, replacement = _parse_decision(
            record["answer"], task, width, height, original
        )
    except ValueError as exc:
        record["parse_error"] = str(exc)
        return "DEFAULT_UNREVIEWED", None, record, str(exc)
    return decision, replacement, record, None


def review_link5_points(image, box_xyxy, points, output_dir: Path):
    """Run one two-image VLM call for N1/N2 and one for P2/P3."""
    from PIL import Image

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from persistent_masking.interface.secrets import load_env_file

    load_env_file(PROJECT_ROOT / ".env.vlm")
    output_dir.mkdir(parents=True, exist_ok=True)
    original = np.asarray(points, dtype=np.int32)
    if original.shape != (6, 2):
        raise ValueError("Link 5 guard expects three positives and three negatives")
    width, height = image.size
    reference_frame = Image.open(REFERENCE_FRAME)
    selected = original.copy()
    reviews = {}
    attempts = []
    for task, spec in REVIEW_SPECS.items():
        indices = spec["indices"]
        reference = _annotate(reference_frame, REFERENCE_BOX, REFERENCE_POINTS, indices)
        current = _annotate(image, box_xyxy, original, indices)
        reference_name = f"link5_guard_{task}_reference.png"
        current_name = f"link5_guard_{task}_current.png"
        reference.save(output_dir / reference_name)
        current.save(output_dir / current_name)
        if task == "negative":
            prompt = (
                "Ideal Link 5 prompt: two red negative points lie INSIDE the "
                "adjacent left wrist housing, just outside the forearm's black "
                "edge. The wrist is the pale end-cap attached to the forearm, "
                "not the background, metal collar, or gripper.\n"
                "Check ONLY N1 and N2 in Image 2 against Image 1: (1) are the "
                "centers of both points outside the forearm itself, including "
                "its black edge; (2) are the centers of both points on visible "
                "white wrist material rather than background? A point near the "
                "wrist silhouette still PASSES if its center is on wrist material. "
                "Do not reject merely because a point could be placed more "
                "centrally. If both checks pass for both points, reply with "
                "exactly PASS. Otherwise reply with exactly "
                '{"decision":"REJECT","negative_points_xy":[[n1x,n1y],[n2x,n2y]]}, '
                "placing BOTH replacement points well inside the wrist. On "
                "REJECT, choose two clearly interior wrist points, not tiny edge "
                "nudges. Place N1 in the broad upper white wrist cap and N2 in "
                "the left white side housing, both LEFT of the black forearm "
                "rim; leave clear wrist material between each point and every "
                "outer silhouette or black forearm edge. "
                f"Current N1={original[3].tolist()}, N2={original[4].tolist()}. "
            )
        else:
            prompt = (
                "Ideal Link 5 prompt: two green positive points lie INSIDE the "
                "white forearm link. P2 lies on the pale left inset enclosed by "
                "the black border; P3 lies on the right-hand white forearm "
                "panel, to the right of the black inset and left of the "
                "circular joint.\n"
                "Check ONLY P2 and P3 in Image 2 against Image 1: (1) is P2's "
                "center on the visible pale inset rather than its black rim, "
                "wrist, or background; (2) is P3's center on the visible "
                "right-hand white panel rather than the black inset, dark "
                "joint rim, pale circular joint disk, or background? A point "
                "near a boundary still PASSES if its center is on the intended "
                "white link surface. Do not reject merely because a point "
                "could be placed more centrally. If both checks pass for both "
                "points, reply with exactly PASS. Otherwise reply with exactly "
                '{"decision":"REJECT","positive_points_xy":[[p2x,p2y],[p3x,p3y]]}. '
                "Return both coordinates. Keep a valid point unchanged and "
                "move any invalid point clearly inside its intended white "
                "region, not a tiny edge nudge. "
                f"Current P2={original[1].tolist()}, P3={original[2].tolist()}. "
            )
        prompt += (
            f"Coordinates are integer pixels of the FULL Image 2 frame before "
            f"display enlargement, width={width}, height={height}; top-left is "
            "(0,0). If the anatomy is unclear, do not invent points; reply with "
            "exactly REJECT."
        )
        decision, replacement, record, reason = _review(
            task, [reference, current], prompt, width, height, original
        )
        if replacement is not None:
            selected[list(indices)] = replacement
        attempts.append(record)
        reviews[task] = {
            "decision": decision,
            "original_points_xy": original[list(indices)].tolist(),
            "selected_points_xy": selected[list(indices)].tolist(),
            "fallback_reason": reason,
            "reference_image": reference_name,
            "current_image": current_name,
        }
    _annotate(image, box_xyxy, selected, range(6)).save(
        output_dir / "link5_guard_selected.png"
    )
    decisions = [review["decision"] for review in reviews.values()]
    decision = ("DEFAULT_UNREVIEWED" if "DEFAULT_UNREVIEWED" in decisions
                else "REJECT" if "REJECT" in decisions else "PASS")
    result = {
        "decision": decision,
        "original_points_xy": original.tolist(),
        "selected_points_xy": selected.tolist(),
        "fallback_to_default": "DEFAULT_UNREVIEWED" in decisions,
        "reviews": reviews,
        "attempts": attempts,
        "selected_image": "link5_guard_selected.png",
    }
    (output_dir / "link5_guard.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return selected, result
