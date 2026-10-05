"""One Luna request, with at most one Gemini fallback, for task-object grounding."""

from __future__ import annotations

import base64
import io
import json
import math
import os
import urllib.request
from dataclasses import dataclass

API_BASE = "https://api.302ai.cn/v1/chat/completions"
SYSTEM_PROMPT = """You locate the physical, non-robot object that the robot directly manipulates in a generated robot video. Use the generation prompt to determine the intended task object and the first video frame to locate it. Examples: for putting a cup on a plate choose the cup; for pushing a cube choose the cube; for closing a drawer choose the drawer; for pressing a button choose the button. Exclude the robot, gripper, background, and passive receptacles. If several objects are named, choose the one directly moved, pressed, or opened by the robot. Return exactly one tight positive bounding box around the visible target and one positive point well inside its visible material, away from boundaries. Coordinates are normalized 0..1000 relative to the full first frame. Return only JSON: {"object_name":"short name","box_xyxy":[x1,y1,x2,y2],"point_xy":[x,y]}."""


@dataclass(frozen=True)
class Grounding:
    object_name: str
    box_xyxy: tuple[int, int, int, int]
    point_xy: tuple[int, int]


def parse_grounding(answer: str, width: int, height: int) -> Grounding:
    start = answer.find("{")
    if start < 0:
        raise ValueError("VLM response has no JSON object")
    value, _ = json.JSONDecoder().raw_decode(answer[start:])
    if not isinstance(value, dict):
        raise ValueError("VLM response is not a JSON object")
    name = value.get("object_name")
    if not isinstance(name, str) or not name.strip() or len(name) > 100:
        raise ValueError("invalid object_name")
    box, point = value.get("box_xyxy"), value.get("point_xy")
    if not isinstance(box, list) or len(box) != 4 or not isinstance(point, list) or len(point) != 2:
        raise ValueError("expected one box and one point")
    coords = box + point
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or not 0 <= v <= 1000 for v in coords):
        raise ValueError("coordinates must be finite values in [0,1000]")
    x1, y1, x2, y2 = box
    px, py = point
    if not (x1 < px < x2 and y1 < py < y2 and x2 - x1 >= 10 and y2 - y1 >= 10):
        raise ValueError("positive point must be strictly inside a nondegenerate box")
    result_box = (max(0, int(x1 * width / 1000)), max(0, int(y1 * height / 1000)),
                  min(width, math.ceil(x2 * width / 1000)), min(height, math.ceil(y2 * height / 1000)))
    result_point = (min(width - 1, int(px * width / 1000)),
                    min(height - 1, int(py * height / 1000)))
    if result_box[2] - result_box[0] < 3 or result_box[3] - result_box[1] < 3:
        raise ValueError("box is too small in source pixels")
    return Grounding(name.strip(), result_box, result_point)


def call_302(image, generation_prompt: str, model: str) -> dict:
    key = os.environ.get("VLM2_API_KEY")
    if not key:
        raise RuntimeError("VLM2_API_KEY is unavailable")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
                {"type": "text", "text": "Generation prompt:\n" + generation_prompt +
                 "\nIdentify the directly manipulated object in this first frame."},
            ]},
        ],
    }
    if model == "gpt-6-luna":
        payload["reasoning_effort"] = "high"
        payload["max_completion_tokens"] = 4096
    else:
        payload["max_tokens"] = 4096
        payload["temperature"] = 0
    request = urllib.request.Request(API_BASE, data=json.dumps(payload).encode(),
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            result = json.load(response)
    except Exception as exc:
        raise RuntimeError(f"302.AI {model} request failed: {type(exc).__name__}") from None
    choice = result.get("choices", [{}])[0]
    answer = choice.get("message", {}).get("content")
    if isinstance(answer, list):
        answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"{model} returned no text (finish_reason={choice.get('finish_reason')})")
    return {"model_requested": model, "model_returned": result.get("model"),
            "response_id": result.get("id"), "usage": result.get("usage"),
            "system_prompt": SYSTEM_PROMPT, "generation_prompt": generation_prompt,
            "answer": answer}
