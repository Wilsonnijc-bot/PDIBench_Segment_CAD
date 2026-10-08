"""Local Responses endpoint probe using exact saved Link5 guard inputs.

This does not change production routes, invoke SAM, or alter Link7 calls.
Credentials are read from the process environment and never saved in receipts.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .common import read, sha, write
from robot.preprocessing.link5_refinement.link5_point_guard import _annotate, _parse_decision


def export_previews(source, destination, result):
    """Draw actual parsed output coordinates on the clean saved frame0."""
    from .refine_guard_local import successful_masking
    root = next(p for p in source.parents if (p / 'training/train20_manifest.json').is_file())
    video = source.parents[1].name
    masking = successful_masking(root, video)
    diagnostic = next(d for d in read(masking / 'sam3_prompt_diagnostics.json') if d['target'] == 'link5')
    image = Image.open(source.parent / 'inputs/frame0.png').convert('RGB')
    points = np.asarray(result['selected_points_xy'], np.int32)
    box = diagnostic['sam3_prompt_box_xyxy']
    result['selected_image'] = 'points.png'
    result['output_images'] = {}
    for view, indices, filename in (
        ('selected', range(len(points)), 'points.png'),
        ('positive_output', range(3), 'positive_points.png'),
        ('negative_output', range(3, len(points)), 'negative_points.png'),
    ):
        _annotate(image, box, points, indices).save(destination / filename)
        result['output_images'][view] = filename


def response_text(response):
    if isinstance(response.get('output_text'), str):
        return response['output_text']
    return ''.join(part.get('text', '') for item in response.get('output', [])
                   if item.get('type') == 'message' for part in item.get('content', [])
                   if part.get('type') == 'output_text')


def ask(base, key, payload, timeout):
    """One request; accept JSON or streamed Responses completion without retries."""
    started = time.monotonic()
    timings = {}
    request = urllib.request.Request(base.rstrip('/') + '/responses',
        data=json.dumps(payload).encode(), headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
        })
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:
            timings.update(http_status=reply.status,
                headers_seconds=round(time.monotonic() - started, 3),
                content_type=reply.headers.get('Content-Type'))
            if 'text/event-stream' not in (timings['content_type'] or ''):
                result = json.load(reply)
            else:
                result = None
                for line in reply:
                    if time.monotonic() - started > timeout:
                        raise TimeoutError('Responses probe exceeded its total time allowance')
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if not data or data == b'[DONE]':
                        continue
                    event = json.loads(data)
                    timings.setdefault('first_event_seconds', round(time.monotonic() - started, 3))
                    if event.get('type') == 'response.output_text.delta':
                        timings.setdefault('first_text_seconds', round(time.monotonic() - started, 3))
                    if event.get('type') in ('response.completed', 'response.failed', 'response.incomplete'):
                        result = event.get('response', {})
                        break
                    if event.get('type') == 'error':
                        raise ValueError(json.dumps(event).replace(key, '[REDACTED]'))
                if result is None:
                    raise ValueError('Responses stream ended without a terminal response')
    except urllib.error.HTTPError as error:
        timings.update(http_status=error.code,
            error=error.read(4000).decode(errors='replace').replace(key, '[REDACTED]'))
        result = None
    except Exception as error:
        timings.update(error_type=type(error).__name__, error=str(error).replace(key, '[REDACTED]'))
        result = None
    timings['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result, timings


def run(source, destination, base, model, key_env, timeout):
    key = os.environ[key_env]
    prior = read(source / 'link5_guard.json')
    frame_path = source.parent / 'inputs/frame0.png'
    image = Image.open(frame_path).convert('RGB')
    original = np.asarray(prior['original_points_xy'], np.int32)
    selected = original.copy()
    attempts, reviews = [], {}
    destination.mkdir(parents=True, exist_ok=True)
    for task in ('negative', 'positive'):
        old = next(a for a in prior['attempts'] if a['role'] == f'link5_{task}_guard')
        spec = prior['reviews'][task]
        inputs = [source / spec[name] for name in ('reference_image', 'current_image')]
        hashes = [sha(p) for p in inputs]
        if hashes != old['image_sha256']:
            raise ValueError('Saved guard inputs do not match the original API call')
        content = [{'type': 'input_text', 'text': old['user_prompt']}]
        content += [{'type': 'input_image', 'detail': 'high',
            'image_url': 'data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode()}
            for p in inputs]
        payload = dict(model=model, instructions=old['system_prompt'],
            input=[dict(role='user', content=content)], reasoning=dict(effort='medium'),
            max_output_tokens=65536, store=False, stream=True)
        response, timings = ask(base, key, payload, timeout)
        record = dict(role=f'link5_{task}_guard', requested_model=model,
            api_base=base, api_style='responses', reasoning_effort='medium',
            output_token_limit=65536, timeout_seconds=timeout, store=False, stream=True,
            system_prompt=old['system_prompt'], user_prompt=old['user_prompt'],
            image_sha256=hashes, input_paths=[str(p.resolve()) for p in inputs], **timings)
        decision, replacement = 'DEFAULT_UNREVIEWED', None
        if response is not None:
            # Exclude any echoed request/images from the saved response metadata.
            record.update({name: response.get(name) for name in
                ('id', 'model', 'status', 'usage', 'error', 'incomplete_details', 'reasoning')})
            record['answer'] = response_text(response)
            if response.get('status') == 'completed' and record['answer']:
                try:
                    decision, replacement = _parse_decision(record['answer'], task, *image.size, original)
                except ValueError as error:
                    record['parse_error'] = str(error)
            else:
                record['response_error'] = 'No complete usable response'
        if replacement is not None:
            indices = (3, 4) if task == 'negative' else (1, 2)
            selected[list(indices)] = replacement
        attempts.append(record)
        reviews[task] = dict(decision=decision,
            selected_points_xy=selected[[3, 4] if task == 'negative' else [1, 2]].tolist())
        result = dict(original_points_xy=original.tolist(), selected_points_xy=selected.tolist(),
            reviews=reviews, attempts=attempts, fallback_to_default=any(
                r['decision'] == 'DEFAULT_UNREVIEWED' for r in reviews.values()),
            source_guard_sha256=sha(source / 'link5_guard.json'),
            frame0_sha256=sha(frame_path), sam_rerun=False, production_config_changed=False,
            provider_cost_usd=None, cost_note='Token usage recorded; this endpoint billing rate is unknown.')
        write(destination / 'link5_guard.json', result)
        print(json.dumps(dict(task=task, decision=decision, **timings,
            answer=record.get('answer'), model=record.get('model'), usage=record.get('usage'))), flush=True)
        if decision == 'DEFAULT_UNREVIEWED':
            break
    complete = len(reviews) == 2 and not result['fallback_to_default']
    result.update(status='complete' if complete else 'failed',
        decision='PASS' if complete and all(r['decision'] == 'PASS' for r in reviews.values())
                 else 'REJECT' if complete else 'DEFAULT_UNREVIEWED')
    if complete:
        export_previews(source, destination, result)
    write(destination / 'link5_guard.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--base', default='https://coding.wangchaoming.com/v1')
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--key-env', default='LINK5_PROBE_API_KEY')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    run(args.source, args.destination, args.base, args.model, args.key_env, args.timeout)
