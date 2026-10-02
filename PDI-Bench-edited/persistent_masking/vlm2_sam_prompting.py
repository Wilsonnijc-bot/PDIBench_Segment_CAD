"""Validated VLM1 frame selection -> VLM2 SAM point prompting."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from persistent_masking.frame_lineage import resolve_frames, validate_lineage
from persistent_masking.vlm2_interface import write_interface
from persistent_masking.vlm_client import VLMClient
from persistent_masking.vlm_interface.config import role_config
from persistent_masking.vlm_prompts import (
    VLM2_NEGATIVE_PROMPT,
    VLM2_POSITIVE_PROMPT,
    VLM2_SYSTEM_PROMPT,
)


def answer(raw, key):
    text = raw.get('answer', '') if isinstance(raw, dict) else str(raw)
    start = text.find('{')
    if start < 0:
        raise ValueError('VLM2 returned no JSON')
    data, end = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(data, dict) or set(data) != {key}:
        raise ValueError('Unexpected point response keys')
    count = {'positive_points': 3, 'negative_points': 2}[key]
    value = data[key]
    if not isinstance(value, list) or len(value) != count:
        raise ValueError(f'VLM2 must return exactly {count} {key}')
    for point in value:
        if (not isinstance(point, list) or len(point) != 2 or
                any(type(x) is not int or not math.isfinite(x) or not 0 <= x <= 1000 for x in point)):
            raise ValueError('Expected integer normalized coordinates in [0,1000]')
    return value


def _explicit_null_positive(raw):
    text = raw.get('answer', '') if isinstance(raw, dict) else str(raw)
    start = text.find('{')
    if start < 0:
        return False
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return False
    return isinstance(data, dict) and data == {'positive_points': None}


def ask_point_response(route, images, prompt, key, role, calls, *, persist=None,
                       lineage=None, fallback_budget=None, fallback_on_null=False):
    """Use the alternate VLM2 route within a shared per-case retry budget."""
    try:
        raw = route.ask(images, prompt, system_prompt=VLM2_SYSTEM_PROMPT)
    except ValueError as error:
        malformed = str(error)
        calls.append({'role': role, 'requested_model': getattr(route, 'model', 'primary_vlm2'),
                      'error': malformed})
    else:
        calls.append({**raw, 'role': role})
        if key == 'positive_points' and _explicit_null_positive(raw) and not fallback_on_null:
            if persist:
                persist(dict(lineage=lineage, calls=calls))
            return raw  # A valid null asks the existing flow to try another frame.
        try:
            answer(raw, key)
        except ValueError as error:
            malformed = str(error)
        else:
            if persist:
                persist(dict(lineage=lineage, calls=calls))
            return raw
    if persist:
        persist(dict(lineage=lineage, calls=calls))
    if fallback_budget is not None:
        if fallback_budget['remaining'] < 1:
            raise ValueError(f'VLM2 {key} failed after the one alternate attempt: {malformed}')
        fallback_budget['remaining'] -= 1
    fallback = VLMClient('vlm2_sam_prompting_malformed_fallback',
                         role_config('vlm2_malformed_fallback'))
    repaired = fallback.ask(images, prompt, system_prompt=VLM2_SYSTEM_PROMPT)
    calls.append({**repaired, 'role': role + '_luna_fallback',
                  'fallback_for': 'malformed_primary_response',
                  'primary_error': malformed})
    if persist:
        persist(dict(lineage=lineage, calls=calls))
    answer(repaired, key)
    return repaired


def to_pixels(positive, negative, width, height):
    points = np.asarray(positive + negative, dtype=float) / 1000 * [width, height]
    if (points < 0).any() or (points >= [width, height]).any():
        raise ValueError('Points outside source pixels; boundary coordinate 1000 is not interior')
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() < 5:
        raise ValueError('Points duplicated or conflicting (<5 source pixels apart)')
    return points.tolist()


def run_case(video, diagnoses, examples, out, route=None, persist=None):
    out = Path(out)
    if len(examples) != 3:
        raise ValueError('Exactly three annotated examples required')
    lineage = resolve_frames(video, diagnoses, out)
    clean = Image.open(out/'vlm2_reseed_frame.png').convert('RGB')
    images = [clean] + [Image.open(p).convert('RGB') for p in examples]
    config = role_config('vlm2')
    route = route or VLMClient('vlm2_sam_prompting', config)
    calls = []
    fallback_budget = {'remaining': 1}
    write_interface(out/'vlm2_interface', system_prompt=VLM2_SYSTEM_PROMPT,
                    user_prompt=VLM2_POSITIVE_PROMPT, images=images,
                    role='positive_points', model=str(config['model']), backend=config['backend'])
    raw = ask_point_response(route, images, VLM2_POSITIVE_PROMPT, 'positive_points',
                             'positive_points', calls, persist=persist, lineage=lineage,
                             fallback_budget=fallback_budget, fallback_on_null=True)
    positive_response = raw
    if persist:
        persist(dict(lineage=lineage, calls=calls))
    write_interface(out/'vlm2_interface', system_prompt=VLM2_SYSTEM_PROMPT,
                    user_prompt=VLM2_NEGATIVE_PROMPT, images=images,
                    role='negative_points', model=str(config['model']), backend=config['backend'])
    raw = ask_point_response(route, images, VLM2_NEGATIVE_PROMPT, 'negative_points',
                             'negative_points', calls, persist=persist, lineage=lineage,
                             fallback_budget=fallback_budget)
    if persist:
        persist(dict(lineage=lineage, calls=calls))
    # Validate the final positive response together with the negative response.
    positive = answer(positive_response, 'positive_points')
    negative = answer(raw, 'negative_points')
    points = to_pixels(positive, negative, *clean.size)
    frame = lineage['vlm2_reseed_frame']
    validate_lineage(lineage, sam_frame=frame, vlm2_frame=frame)
    preview = clean.copy()
    draw = ImageDraw.Draw(preview)
    for i, (x, y) in enumerate(points):
        color = 'lime' if i < 3 else 'red'
        draw.ellipse((x-6,y-6,x+6,y+6), fill=color, outline='white', width=1)
        draw.text((x+8,y+2),str(i+1),fill=color)
    preview.save(out/'points.png')
    return dict(**lineage, points=points, labels=[1,1,1,0,0], frame=frame,
                video=str(video), source_hw=[clean.height,clean.width], positive_count=3,
                status='ready', positive_points=positive, negative_points=negative, calls=calls)
