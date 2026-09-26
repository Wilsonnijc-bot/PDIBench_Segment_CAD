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
    if set(data) != {key}:
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
    write_interface(out/'vlm2_interface', system_prompt=VLM2_SYSTEM_PROMPT,
                    user_prompt=VLM2_POSITIVE_PROMPT, images=images,
                    role='positive_points', model=str(config['model']), backend=config['backend'])
    raw = route.ask(images, VLM2_POSITIVE_PROMPT, system_prompt=VLM2_SYSTEM_PROMPT)
    positive_response = raw
    calls.append({**raw, 'role': 'positive_points'})
    text = raw.get('answer', '') if isinstance(raw, dict) else str(raw)
    # Walk the VLM1-confirmed deformation frames in order until VLM2 returns
    # three positives. This keeps each retry on a clean source frame.
    deformed_count = sum(d.get('state') == 'deformed' and not d.get('parse_error') for d in diagnoses)
    rank = 1
    while '"positive_points":null' in text.replace(' ', '') and rank < deformed_count:
        lineage = resolve_frames(video, diagnoses, out, deformed_rank=rank)
        clean = Image.open(out/'vlm2_reseed_frame.png').convert('RGB')
        images = [clean] + [Image.open(p).convert('RGB') for p in examples]
        write_interface(out/'vlm2_interface', system_prompt=VLM2_SYSTEM_PROMPT,
                        user_prompt=VLM2_POSITIVE_PROMPT, images=images,
                        role=f'positive_points_fallback_{rank+1}',
                        model=str(config['model']), backend=config['backend'])
        retry = route.ask(images, VLM2_POSITIVE_PROMPT, system_prompt=VLM2_SYSTEM_PROMPT)
        positive_response = retry
        calls.append({**retry, 'role': f'positive_points_fallback_deformation_{rank+1}'})
        text = retry.get('answer', '') if isinstance(retry, dict) else str(retry)
        rank += 1
    if persist:
        persist(dict(lineage=lineage, calls=calls))
    write_interface(out/'vlm2_interface', system_prompt=VLM2_SYSTEM_PROMPT,
                    user_prompt=VLM2_NEGATIVE_PROMPT, images=images,
                    role='negative_points', model=str(config['model']), backend=config['backend'])
    raw = route.ask(images, VLM2_NEGATIVE_PROMPT, system_prompt=VLM2_SYSTEM_PROMPT)
    calls.append({**raw, 'role': 'negative_points'})
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
