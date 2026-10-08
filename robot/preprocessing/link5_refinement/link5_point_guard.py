"""Independent VLM reviews of Link 5 wrist negatives and forearm positives."""

from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()


import json
import hashlib
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = (_workspace_root())
REFERENCE_FRAME = (_workspace_root() / 'documentation/data/references/link5_guard/lvp040_reference_frame.png')
POSITIVE_REFERENCE_FRAME = (_workspace_root() / 'documentation/data/references/link5_guard/positive_three_points_reference.png')
REFERENCE_BOX = (452, 0, 818, 132)
REFERENCE_POINTS = [(510, 67), (570, 67), (770, 40),
                    (495, 27), (478, 92), (807, 82)]
GUARD_MODEL = "gemini-3.8-flash"
GUARD_REASONING = {"gemini-3.8-flash": "high", "gpt-6-luna": "xhigh"}
NEGATIVE_BOX_MARGIN_PIXELS = 5
NEGATIVE_MAX_ATTEMPTS = 3


class NegativeGuardExhausted(ValueError):
    """The negative guard consumed its call budget; do not restart the stage."""

    failure = {'category': 'link5_negative_guard_exhausted', 'retryable': False}


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
        "indices": (0, 1, 2),
        "field": "positive_points_xy",
        "system_prompt": (
            "You review SAM3 point prompts for the robot's Link 5 forearm. "
            "Image 1 is an annotated reference example of ideal P1, P2, and P3 "
            "placement; Image 2 is the current full frame with the proposed Link 5 "
            "box and proposed P2 and P3 only. P1 has no default placement. "
            "Reason carefully from the images, but return only the requested "
            "answer. Place P1 and review P2 and P3, returning all three green "
            "positive points explicitly."
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
    if answer == "PASS" and task == "positive":
        raise ValueError("positive guard must explicitly place P1, P2, and P3; PASS has no P1 coordinate")
    if answer == "PASS":
        return "PASS", None
    try:
        data = json.loads(answer)
    except json.JSONDecodeError as exc:
        raise ValueError("guard answer must be placement/review JSON, or PASS for negatives") from exc
    field = REVIEW_SPECS[task]["field"]
    expected_decision = "PLACE" if task == "positive" else "REJECT"
    if not isinstance(data, dict) or data.get("decision") != expected_decision:
        raise ValueError(f"guard answer must have decision {expected_decision}")
    if set(data) != {"decision", field}:
        raise ValueError("guard REJECT answer has unexpected fields")
    raw = data[field]
    count = len(REVIEW_SPECS[task]["indices"])
    if (not isinstance(raw, list) or len(raw) != count
            or any(not isinstance(point, list) or len(point) != 2
                   or any(type(value) is not int for value in point)
                   for point in raw)):
        raise ValueError(f"guard answer needs exactly {count} integer [x,y] points")
    replacement = np.asarray(raw, dtype=np.int32)
    if (np.any(replacement < 0) or np.any(replacement[:, 0] >= width)
            or np.any(replacement[:, 1] >= height)):
        raise ValueError("guard coordinates are outside the full frame")
    distances = np.linalg.norm(replacement[:, None] - replacement[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 8:
        raise ValueError("guard points are too close together")
    fixed_indices = [i for i in range(len(original)) if i not in REVIEW_SPECS[task]["indices"]]
    if np.any(np.all(replacement[:, None] == original[list(fixed_indices)][None], axis=2)):
        raise ValueError("guard point coincides with a fixed opposite-label point")
    return expected_decision, replacement


def _negative_box_check(points, box_xyxy, margin_pixels):
    """Require N1 inside/near the green box; report N2 without vetoing it."""
    x1, y1, x2, y2 = box_xyxy
    points = np.asarray(points)
    inside = ((points[:, 0] >= x1 - margin_pixels)
              & (points[:, 0] <= x2 + margin_pixels)
              & (points[:, 1] >= y1 - margin_pixels)
              & (points[:, 1] <= y2 + margin_pixels))
    return {
        "passed": bool(inside[0]),
        "required_points": ["N1"],
        "n2_inside_box": bool(inside[1]),
        "box_xyxy": list(box_xyxy),
        "margin_pixels": margin_pixels,
        "candidate_points_xy": points.tolist(),
        "outside_points": [f"N{i + 1}" for i, valid in enumerate(inside) if not valid],
    }


def guard_config(config=None, *, configured_model=True):
    """Link5-only reasoning policy; leave other links' VLM defaults unchanged."""
    from robot.preprocessing.link7_persistent.interface.config import role_config
    settings = dict(config) if config is not None else role_config("vlm2")
    if not configured_model: settings["model"] = GUARD_MODEL
    if settings["backend"] == "cloud_api" and settings["model"] in GUARD_REASONING:
        settings["reasoning_effort"] = GUARD_REASONING[settings["model"]]
        # Reasoning and answer share the provider's output budget. 4096 tokens
        # previously caused missing-text completions on harder reviews.
        token_field = "max_completion_tokens" if "luna" in settings["model"] else "max_tokens"
        settings[token_field] = max(65536, int(settings.get(token_field, 0)))
        settings.pop("max_tokens" if token_field == "max_completion_tokens" else "max_completion_tokens", None)
        minimum_timeout = 1200 if "luna" in settings["model"] else 600
        settings["timeout_seconds"] = max(minimum_timeout, int(settings.get("timeout_seconds", 0)))
        if "luna" in settings["model"]: settings["temperature"] = None
    return settings


def _review(task, images, prompt, width, height, original, *, configured_model=False, config=None):
    from robot.preprocessing.link7_persistent.vlm_client import VLMClient

    settings = guard_config(config, configured_model=configured_model)
    client = VLMClient(f"link5_{task}_guard", settings)
    receipt_settings = {
        "reasoning_effort": settings.get("reasoning_effort"),
        "api_style": settings.get("api_style"),
        "api_base": settings.get("api_base"),
        "output_token_limit": settings.get("max_completion_tokens", settings.get("max_tokens")),
        "timeout_seconds": settings.get("timeout_seconds", 240),
    }
    started = time.monotonic()
    try:
        record = client.ask(
            images, prompt, system_prompt=REVIEW_SPECS[task]["system_prompt"]
        )
    except Exception as exc:
        from infrastructure.shared.contracts.vlm_failure import failure_record
        reason = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else type(exc).__name__
        client_records = getattr(client, "records", [])
        record = {
            **(client_records[-1] if client_records else {}),
            **receipt_settings,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "role": f"link5_{task}_guard", "requested_model": client.model,
            "response_error": reason,
            "failure": failure_record(exc),
            "response_diagnostic": (client_records[-1].get("response_diagnostic")
                                    if client_records else None),
        }
        return "DEFAULT_UNREVIEWED", None, record, reason
    record.update(receipt_settings, elapsed_seconds=round(time.monotonic() - started, 3))
    try:
        decision, replacement = _parse_decision(
            record["answer"], task, width, height, original
        )
    except ValueError as exc:
        record["parse_error"] = str(exc)
        return "DEFAULT_UNREVIEWED", None, record, str(exc)
    return decision, replacement, record, None


def review_link5_points(image, box_xyxy, points, output_dir: Path, *,
                        reference_frame=None, positive_reference_frame=None,
                        required=False, vlm_config=None,
                        negative_box_margin_pixels=NEGATIVE_BOX_MARGIN_PIXELS,
                        negative_max_attempts=NEGATIVE_MAX_ATTEMPTS,
                        negative_only=False):
    """Review N1/N2 and explicitly place P1/P2/P3; P1 has no default fallback."""
    from PIL import Image

    if type(negative_box_margin_pixels) is not int or negative_box_margin_pixels < 0:
        raise ValueError('negative box margin must be a nonnegative integer')
    if type(negative_max_attempts) is not int or negative_max_attempts < 1:
        raise ValueError('negative max attempts must be a positive integer')
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from robot.preprocessing.link7_persistent.interface.secrets import load_env_file

    load_env_file((_workspace_root() / '.env.vlm'))
    output_dir.mkdir(parents=True, exist_ok=True)
    original = np.asarray(points, dtype=np.int32).copy()
    if original.shape not in ((5, 2), (6, 2)):
        raise ValueError("Link 5 guard expects three positives and two or three negatives")
    if negative_only:
        if np.any(original[:3] < 0):
            raise ValueError('Negative-only review requires frozen positive coordinates')
    else:
        original[0] = (-1, -1)  # Discard the legacy box-relative P1.
    width, height = image.size
    positive_path = Path(positive_reference_frame or POSITIVE_REFERENCE_FRAME)
    reference_frame = Image.open(reference_frame or REFERENCE_FRAME)
    selected = original.copy()
    reviews = {}
    attempts = []
    for task, spec in REVIEW_SPECS.items():
        if negative_only and task != 'negative':
            continue
        indices = spec["indices"]
        reference = (Image.open(positive_path).convert('RGB')
                     if task == 'positive' else
                     _annotate(reference_frame, REFERENCE_BOX, REFERENCE_POINTS, indices))
        current = _annotate(image, box_xyxy, original, (1, 2) if task == 'positive' else indices)
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
                "Ideal Link 5 prompt: three green positive points lie INSIDE the "
                "white forearm link. P1 lies just inside the LEFT interior edge "
                "of the rounded/circular pale left lobe enclosed by the black "
                "border, with clear pale material between its center and the "
                "black rim. This is the rounded LEFT end of the inset, not the "
                "adjacent wrist housing, metal collar, or right circular joint. "
                "P2 lies farther right on the pale inset enclosed by the black "
                "border; P3 lies on the right-hand white forearm panel, to the "
                "right of the black inset and left of the circular joint.\n"
                "Use Image 1 to place P1 in Image 2. P1 has NO proposed or default "
                "coordinate; determine it from the visible robot anatomy. "
                "Then check P2 and P3 in Image 2 against Image 1: (1) is P2's "
                "center on the visible pale inset rather than its black rim, "
                "wrist, or background; (2) is P3's center on the visible "
                "right-hand white panel rather than the black inset, dark "
                "joint rim, pale circular joint disk, or background? Keep a "
                "valid P2 or P3 unchanged; move any invalid point clearly inside "
                "its intended white region, not a tiny edge nudge. "
                "Always reply with exactly "
                '{"decision":"PLACE","positive_points_xy":[[p1x,p1y],[p2x,p2y],[p3x,p3y]]}. '
                "Return all three coordinates, even when P2 and P3 are valid. "
                "Do not reply PASS. "
                f"Current P2={original[1].tolist()}, P3={original[2].tolist()}. "
            )
        prompt += (
            f"Coordinates are integer pixels of the FULL Image 2 frame before "
            f"display enlargement, width={width}, height={height}; top-left is "
            "(0,0). If the anatomy is unclear, do not invent points; reply with "
            "exactly REJECT."
        )
        review_args = {'configured_model': True} if required else {}
        if vlm_config is not None: review_args.update(config=vlm_config, configured_model=True)
        limit = negative_max_attempts if task == 'negative' else 1
        for attempt_index in range(limit):
            # Retry the exact same images and prompts, without applying failed points.
            decision, replacement, record, reason = _review(
                task, [reference, current], prompt, width, height,
                selected if task == 'positive' else original, **review_args
            )
            record['attempt_number'] = attempt_index + 1
            if task == 'negative' and decision == 'DEFAULT_UNREVIEWED' and record.get('parse_error'):
                record['accepted'] = False
                record['retry_reason'] = 'parse_failure'
                attempts.append(record)
                if attempt_index + 1 < limit:
                    continue
                decision, replacement = 'FAILED_PARSE_RETRIES', None
                reason = f'Negative response parsing failed after {limit} attempts'
                break
            if task == 'negative' and decision != 'DEFAULT_UNREVIEWED':
                candidate = original[list(indices)] if replacement is None else replacement
                check = _negative_box_check(candidate, box_xyxy, negative_box_margin_pixels)
                record['negative_box_check'] = check
                if not check['passed']:
                    record['accepted'] = False
                    record['retry_reason'] = 'n1_outside_box'
                    attempts.append(record)
                    if attempt_index + 1 < limit:
                        continue
                    decision, replacement = 'FAILED_BOX_CHECK', None
                    reason = f'N1 outside green box after {limit} attempts'
                    break
                record['accepted'] = True
            attempts.append(record)
            break
        if replacement is not None:
            selected[list(indices)] = replacement
        reviews[task] = {
            "decision": decision,
            "original_points_xy": original[list(indices)].tolist(),
            "selected_points_xy": selected[list(indices)].tolist(),
            "fallback_reason": reason,
            "reference_image": reference_name,
            "current_image": current_name,
            "attempt_count": attempt_index + 1,
        }
        if decision in ('FAILED_BOX_CHECK', 'FAILED_PARSE_RETRIES') or (required and decision == "DEFAULT_UNREVIEWED"): break
    placed_indices = [i for i, xy in enumerate(selected) if np.all(xy >= 0)]
    _annotate(image, box_xyxy, selected, placed_indices).save(
        output_dir / "link5_guard_selected.png"
    )
    decisions = [review["decision"] for review in reviews.values()]
    decision = ('FAILED_PARSE_RETRIES' if 'FAILED_PARSE_RETRIES' in decisions
                else 'FAILED_BOX_CHECK' if 'FAILED_BOX_CHECK' in decisions
                else "DEFAULT_UNREVIEWED" if "DEFAULT_UNREVIEWED" in decisions
                else "REJECT" if any(d in ('REJECT', 'PLACE') for d in decisions) else "PASS")
    result = {
        "decision": decision,
        "original_points_xy": original.tolist(),
        "selected_points_xy": selected.tolist(),
        "fallback_to_default": "DEFAULT_UNREVIEWED" in decisions,
        "reviews": reviews,
        "attempts": attempts,
        "selected_image": "link5_guard_selected.png",
        "positive_placement_policy": "frozen_cached_positives" if negative_only else "vlm_places_all_three_no_default_p1",
        "positive_guard_called": not negative_only,
        "p1_default_used": False,
        "negative_box_policy": {
            "margin_pixels": negative_box_margin_pixels,
            "max_attempts": negative_max_attempts,
            "retry_inputs": "identical_images_and_prompts",
            "required_points": ["N1"],
            "retry_parse_failures": True,
        },
        "positive_reference": str(positive_path),
        "positive_reference_sha256": hashlib.sha256(positive_path.read_bytes()).hexdigest() if positive_path.is_file() else None,
    }
    (output_dir / "link5_guard.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if decision == 'FAILED_BOX_CHECK':
        raise NegativeGuardExhausted('Link5 negative guard exhausted green-box retries; see link5_guard.json')
    if decision == 'FAILED_PARSE_RETRIES':
        raise NegativeGuardExhausted('Link5 negative guard exhausted parsing retries; see link5_guard.json')
    if (required and result['fallback_to_default']) or np.any(selected[0] < 0):
        from infrastructure.shared.contracts.vlm_failure import raise_timeout
        for attempt in attempts: raise_timeout(attempt)
        raise ValueError('Link5 VLM guard returned no usable point review; see link5_guard.json')
    return selected, result
