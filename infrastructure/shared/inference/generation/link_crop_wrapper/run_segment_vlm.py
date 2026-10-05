"""Run segment-specific pairwise VLM diagnoses from a wrapper call manifest."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import time
from pathlib import Path


STRICT_SYSTEM_PROMPT = """You are an absolutely strict, exceptionally meticulous visual inspector for structural deformation in generated robot videos. You receive exactly two crops of the same robot segment and must examine every recoverable geometric detail with great care.

Image 1 is the segment in the initial frame. Image 2 is the same segment in one later generated-video frame. Decide whether Image 2 preserves the same physically legitimate rigid structure.

Absolutely strictly inspect silhouette continuity, local thickness, taper, length-to-width ratio, internal black/white panel proportions, joint attachment, part count, and the physical continuity of every visible boundary. Treat even a localized structural inconsistency as deformation when it is visibly supported.

At the same time, reason correctly about transformation dynamics. A rigid segment may translate, rotate in three dimensions, articulate at its joints, move toward or away from the camera, become foreshortened, or be partly self-occluded. Perspective can alter apparent length and scale, and motion blur can soften or directionally trail an edge. Accept these effects only when they provide a specific, coherent physical explanation for all visible changes while the implied underlying rigid shape remains recoverable.

Do not use motion, perspective, blur, lighting, or occlusion as a generic excuse for a silhouette that bends, bulges, pinches, melts, stretches, shrinks, splits, duplicates, disconnects, changes thickness implausibly, or loses structure. Be conservative about explanations and meticulous about detection. Nearby background and held objects are not robot geometry. Return unclear only when the crop truly prevents a reliable judgment."""

LOOSE_SYSTEM_PROMPT = """You are a tolerant visual inspector for major structural deformation in generated robot videos. You receive exactly two crops of the same robot segment.

Image 1 is the segment in the initial frame. Image 2 is the same segment in one later generated-video frame. Decide whether the segment in Image 2 is still recognizably and functionally the same robot part.

Allow normal translation, three-dimensional rotation, articulation, perspective, foreshortening, scale change with depth, self-occlusion, lighting variation, and motion blur. Also allow small pixel-level defects, minor boundary wobble, slight local smearing, modest thickness or proportion variation, small panel-color inconsistencies, partial overlap, and other limited generation artifacts within a reasonable range. If the segment remains identifiable as the same part, preserves its basic connectivity and role, and has no obvious major structural failure, classify it as normal.

Flag deformation only for clear, substantial failures such as a segment becoming unrecognizable, severe bending or melting, major stretching or shrinking, obvious splitting or duplication, loss of an essential component, grossly implausible thickness change, or a clear mechanical disconnection. Ambiguous or subtle differences should normally be accepted. Nearby background and held objects are not robot geometry. Return unclear only when visibility is too poor to tell whether the segment remains identifiable."""

LOOSE_V2_SYSTEM_PROMPT = """You are a tolerant visual inspector for major structural deformation in generated robot videos. You receive exactly two crops of the same robot segment.

Image 1 is the segment in the initial frame. Image 2 is the same segment in one later generated-video frame. Decide whether the segment in Image 2 is still recognizably and functionally the same robot part.

Allow normal translation, three-dimensional rotation, articulation, perspective, foreshortening, scale change with depth, self-occlusion, lighting variation, and motion blur. Also allow small pixel-level defects, minor boundary wobble, slight local smearing, modest thickness or proportion variation, small panel-color inconsistencies, partial overlap, and other limited generation artifacts within a reasonable range. If the segment remains identifiable as the same part, preserves its basic connectivity and role, and has no obvious major structural failure, classify it as normal.

Flag deformation only for clear, substantial failures such as a segment becoming unrecognizable, severe bending or melting, major stretching or shrinking, obvious splitting or duplication, loss of an essential component, grossly implausible thickness change, or a clear mechanical disconnection. Do not report disconnection merely because a connection is occluded or outside the crop; a physical break or abnormal gap must be visible. Classify the segment as normal when its visible differences are limited to the allowed variations above. Nearby background is not robot geometry. Return unclear only when the requested segment's main structure is not visible enough to evaluate reliably."""

MODEST_SYSTEM_PROMPT = """You are a careful visual inspector for structural deformation in generated robot videos. Use moderate sensitivity: ignore small pixel-level artifacts, but detect noticeable abnormal shape changes even when the part remains recognizable. You receive exactly two crops of the same robot segment.

Image 1 is the segment in the initial frame. Image 2 is the same segment in one later generated-video frame. Compare the underlying mechanical structure, expected subparts, connectivity, proportions, silhouette, and stable material-color regions.

Allow translation, three-dimensional rotation, articulation at real joints, perspective, foreshortening, depth-related scale change, partial occlusion, lighting variation, and motion blur when they plausibly explain the difference. Ignore isolated pixel noise, tiny boundary wobble, compression artifacts, minor color speckles, and slightly soft edges that do not noticeably change shape. Do not use viewing or rendering effects to excuse a clear abnormal shape change.

Classify the segment as deformed when Image 2 shows a noticeable, visibly supported structural change after accounting for legitimate viewing and motion effects. Examples include misshaped, missing, duplicated, or fused components; broken attachment; abnormal rigid-link bending; implausible proportions; silhouette distortion; bulging, pinching, stretching, shrinking, melting, or warping; and material-color regions mixing because the part changed shape. Use mild for a small but clearly noticeable deformation, moderate for an obvious deformation that preserves the overall part, and severe for major structural failure. Ambiguous evidence alone is not deformation. Return unclear only when visibility prevents a reliable judgment."""

PROMPT_PROFILES = {
    "strict-v1": STRICT_SYSTEM_PROMPT,
    "loose-v1": LOOSE_SYSTEM_PROMPT,
    "loose-v2": LOOSE_V2_SYSTEM_PROMPT,
    "modest-v1": MODEST_SYSTEM_PROMPT,
}

BLACK_PADDING_GUIDANCE = """A crop may contain a solid-black region with a sharp straight boundary. This explicitly marks image area outside the source frame. Treat black padding as unavailable visual evidence, not as robot geometry, disappearance, occlusion, deformation, or background. The exact requested and fallback-resolved frames from the baseline evaluation have been preserved; use the reported black-padding fraction when judging how much visual evidence remains."""

SEGMENT_FOCUS = {
    "upperarm": "Inspect upper-arm link rigidity, shoulder and elbow connections, taper, panel proportions, and length-to-width consistency.",
    "forearm": "Inspect forearm rigidity, wrist and elbow connections, parallel edges, joint housing, thickness, and scale-depth consistency.",
    "gripper": "Inspect finger count and shape, visible symmetry, finger-to-body proportions, rigid attachment, and separation from any held object.",
}

LOOSE_V2_SEGMENT_FOCUS = {
    "upperarm": """Inspect the upper arm at the level of overall structure and connectivity, not small pixel, panel-edge, surface-texture, highlight, or mask-boundary differences. Allow a small amount of apparent bending around the shoulder as part of its built-in shape and articulation. Flag bending only when it is clearly noticeable beyond a plausible joint range or visibly warps the rigid upper-arm body.""",
    "forearm": """Inspect overall forearm rigidity and its wrist and elbow connections. Allow a small amount of apparent bending around the dark-and-white link intersection as part of its built-in geometry and articulation. Flag bending only when it is clearly noticeable beyond a plausible range or visibly warps the rigid forearm body. Do not use apparent length or thickness change alone when orientation, depth, or crop scale differs substantially.""",
    "gripper": """Inspect finger count and shape, visible symmetry, finger-to-body proportions, rigid attachment, and separation from held objects. Ignore objects between the jaws unless they are visibly continuous with robot material. Normal jaw opening, closing, overlap, or partial finger occlusion must not be treated as fused, missing, or deformed fingers.""",
}

MODEST_SEGMENT_FOCUS = {
    "upperarm": """Target anatomy: the upper arm is the elongated, mostly white, slightly tapered rigid housing between the large dark-ringed joint above and the smaller joint below. Inspect only this white housing; the joints are boundary landmarks, not part of its geometry.
Check each point explicitly:
1. Length-to-width ratio: decide whether its proportions remain mechanically legitimate after accounting for perspective, foreshortening, and depth.
2. Bending: distinguish normal rotation at the boundary joints from noticeable bending, curving, bowing, or kinking within the rigid white housing.
3. Silhouette: flag a clearly visible abnormal outline change, including bulging, pinching, melting, stretching, shrinking, splitting, flattening, or an abrupt thickness change, even when the upper arm remains recognizable.
4. Connections: verify continuity up to both joint boundaries when visible.
Allow lighting and shadow to alter its mostly white appearance; color variation alone is not deformation.""",
    "forearm": """Expected structure: the forearm is a long, rigid, two-tone segment with approximately half white and half dark-colored structure and distinct white/dark regions between the elbow and wrist.
Check each point explicitly:
1. Length-to-width ratio: decide whether the long forearm retains a mechanically plausible proportion after accounting for perspective, foreshortening, and depth.
2. Bending: distinguish normal elbow or wrist articulation from abnormal bending, bowing, or kinking within the rigid forearm itself.
3. Silhouette consistency: compare the overall outline and long edges. Flag a noticeable bulge, pinch, melt, stretch, shrink, split, waviness, or abrupt thickness change even when the forearm remains recognizable.
4. Two-tone structure: verify that the white and dark-colored regions remain present, distinct, and coherently attached rather than disappearing, bleeding together, or being structurally rearranged.
5. Connections: verify continuity with the elbow and wrist joint housings when those connections are visible.
Do not mistake shading, reflection, occlusion, or a held/background object for mixing of the forearm's own colors.""",
    "gripper": """Expected structure: the gripper has a white, rectangular palm with two dark-colored fingers.
Check each point explicitly:
1. Finger presence and count: determine whether both dark-colored fingers are still present; flag a finger that disappears, duplicates, fuses with the other finger, or becomes noticeably misshaped when visibility is sufficient.
2. Finger shape: check whether each finger retains a plausible rigid shape, attachment, relative size, and separation while allowing normal opening, closing, rotation, perspective, and partial occlusion. Flag a clearly warped, bent, swollen, shortened, elongated, or otherwise abnormal finger even when it remains recognizable.
3. White palm: verify that the white rectangular palm remains present and does not melt, warp, split, collapse, stretch, or shrink into an implausible shape.
4. Color and part boundaries: check whether the white palm and dark fingers remain distinct. Flag color mixing only when it reflects visible warping, fusion, or structural corruption of the gripper itself.
5. Connectivity: verify that both fingers remain attached to the palm and that the palm remains attached to the wrist when visible.
Do not count a grasped object, object color, shadow, reflection, or occlusion as an extra, missing, or mixed gripper component.""",
}


def segment_focus(prompt_profile: str, segment: str) -> str:
    focus_map = segment_focus_map(prompt_profile)
    try:
        return focus_map[segment]
    except KeyError as exc:
        raise ValueError(f"Unsupported segment: {segment}") from exc


def segment_focus_map(prompt_profile: str) -> dict[str, str]:
    if prompt_profile == "modest-v1":
        return MODEST_SEGMENT_FOCUS
    if prompt_profile == "loose-v2":
        return LOOSE_V2_SEGMENT_FOCUS
    return SEGMENT_FOCUS


def uses_black_padding_guidance(prompt_profile: str) -> bool:
    return prompt_profile == "modest-v1"


RESPONSE_REQUEST = """Compare Image 2 with Image 1 for the requested segment. Return exactly one JSON object with only these fields:
- segment: the requested segment name
- state: one of normal, deformed, unclear
- issue: a concise problem name, or none
- evidence: one concise sentence grounded in visible geometry
- severity: one of none, mild, moderate, severe, unclear
- probability: probability of structural deformation from 0.0 to 1.0

Use state deformed exactly when probability is at least 0.5. Use state normal only when probability is below 0.5. Use state unclear when visibility prevents a reliable binary judgment. Do not include markdown or text outside the JSON object."""

RESPONSE_KEYS = ("segment", "state", "issue", "evidence", "severity", "probability")
STATES = {"normal", "deformed", "unclear"}
SEVERITIES = {"none", "mild", "moderate", "severe", "unclear"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--prompt-profile", choices=tuple(PROMPT_PROFILES), required=True
    )
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-videos", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_response(raw: str, expected_segment: str) -> dict:
    match = re.search(r"\{[^{}]+\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError("Response did not contain a JSON object")
    payload = match.group(0)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        try:
            data = json.loads(payload, strict=False)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(payload)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(
                    "Response object is not valid JSON or a Python dictionary literal"
                ) from exc
    if not isinstance(data, dict):
        raise ValueError("Response object is not a dictionary")
    if set(data) != set(RESPONSE_KEYS):
        raise ValueError(f"Unexpected response keys: {sorted(data)}")
    segment = str(data["segment"]).strip().lower()
    state = str(data["state"]).strip().lower()
    severity = str(data["severity"]).strip().lower()
    probability = float(data["probability"])
    issue = str(data["issue"]).strip()
    evidence = str(data["evidence"]).strip()
    if segment != expected_segment:
        raise ValueError(f"Expected segment {expected_segment}, received {segment}")
    if state not in STATES:
        raise ValueError(f"Invalid state: {state}")
    if severity not in SEVERITIES:
        raise ValueError(f"Invalid severity: {severity}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Probability is outside [0, 1]")
    if state == "deformed" and probability < 0.5:
        raise ValueError("Deformed state is inconsistent with probability")
    if state == "normal" and probability >= 0.5:
        if issue.lower() == "none" and severity == "none":
            # StableI2I sometimes emits confidence in its chosen normal state
            # despite the request for deformation probability.
            probability = 1.0 - probability
        else:
            raise ValueError("Normal state is inconsistent with probability")
    if not issue or not evidence:
        raise ValueError("Issue and evidence must not be empty")
    return {
        "segment": segment,
        "state": state,
        "issue": issue,
        "evidence": evidence,
        "severity": severity,
        "probability": probability,
    }


def build_messages(
    call_root: Path, row: dict[str, str], prompt_profile: str = "strict-v1"
):
    from PIL import Image

    segment = row["segment"]
    reference_image = Image.open(call_root / row["reference_crop"]).convert("RGB")
    candidate_image = Image.open(call_root / row["candidate_crop"]).convert("RGB")
    try:
        system_prompt = PROMPT_PROFILES[prompt_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt profile: {prompt_profile}") from exc
    system_prompt += "\n\n" + segment_focus(prompt_profile, segment)
    if uses_black_padding_guidance(prompt_profile):
        system_prompt += "\n\n" + BLACK_PADDING_GUIDANCE
    actual_frame = int(row["candidate_frame"])

    def padding_description(prefix: str) -> str:
        black_value = row.get(f"{prefix}_black_pixel_fraction", "")
        if black_value == "":
            visible_value = row.get(f"{prefix}_visible_pixel_fraction", "")
            if visible_value == "":
                return "Black-padding metadata is unavailable."
            black_fraction = 1.0 - float(visible_value)
        else:
            black_fraction = float(black_value)
        pads = ", ".join(
            f"{side}={int(row[f'{prefix}_pad_{side}'])}px"
            for side in ("left", "top", "right", "bottom")
            if row.get(f"{prefix}_pad_{side}", "") != ""
            and int(row[f"{prefix}_pad_{side}"]) > 0
        )
        if pads:
            return f"Solid-black out-of-frame area: {black_fraction:.1%}; {pads}."
        return "Solid-black out-of-frame area: 0.0%."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Requested segment: {segment}. Image 1 is its initial "
                        f"reference crop. {padding_description('reference')}"
                    ),
                },
                {"type": "image", "image": reference_image},
                {
                    "type": "text",
                    "text": (
                        "Image 2 is the candidate crop at "
                        f"frame {actual_frame}, "
                        f"t={float(row['timestamp_seconds']):.3f}s. "
                        f"{padding_description('candidate')}"
                    ),
                },
                {"type": "image", "image": candidate_image},
                {"type": "text", "text": RESPONSE_REQUEST},
            ],
        },
    ]
    return messages, [reference_image, candidate_image]


def model_class_name(architectures: list[str] | None) -> str:
    supported = {
        "Qwen2_5_VLForConditionalGeneration",
        "Qwen3VLForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
    }
    matches = supported.intersection(architectures or [])
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one supported Qwen VL architecture, found {architectures or []}"
        )
    return matches.pop()


def checkpoint_key_mapping(class_name: str) -> dict[str, str] | None:
    if class_name == "Qwen3_5ForConditionalGeneration":
        # Vision-OPD was saved through a training wrapper. Transformers 5.5 can
        # stream-remap those text weights without copying the 18.8 GB checkpoint.
        return {r"language_model.model": "language_model"}
    return None


def load_model(model_path: Path):
    import torch
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_path)
    class_name = model_class_name(getattr(config, "architectures", None))
    model_class = getattr(transformers, class_name)
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    load_options = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "attn_implementation": "sdpa",
    }
    if mapping := checkpoint_key_mapping(class_name):
        load_options["key_mapping"] = mapping
    model = model_class.from_pretrained(model_path, **load_options).eval()
    return processor, model, class_name


def generate(model, processor, messages: list[dict], max_new_tokens: int) -> str:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
    trimmed = [
        output[len(source) :] for source, output in zip(inputs.input_ids, generated)
    ]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def call_key(row: dict) -> tuple[str, str, int, str]:
    return (
        str(row["dataset"]),
        str(row["video_id"]),
        int(row.get("requested_candidate_frame") or row["candidate_frame"]),
        str(row.get("requested_segment") or row["segment"]),
    )


def validate_call_manifest(rows: list[dict], expected_videos: int | None = None) -> None:
    if not rows:
        raise ValueError("Call manifest is empty")
    by_video: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        by_video.setdefault((str(row["dataset"]), str(row["video_id"])), []).append(row)
    if expected_videos is not None and len(by_video) != expected_videos:
        raise ValueError(f"Expected {expected_videos} videos, found {len(by_video)}")
    for video, video_rows in by_video.items():
        if len(video_rows) != 15:
            raise ValueError(f"Expected 15 calls for {video}, found {len(video_rows)}")
        counts = {
            segment: sum(row["segment"] == segment for row in video_rows)
            for segment in SEGMENT_FOCUS
        }
        if counts != {segment: 5 for segment in SEGMENT_FOCUS}:
            raise ValueError(f"Invalid segment call counts for {video}: {counts}")
    keys = [call_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Call manifest contains duplicate comparison keys")


def shard_call_rows(
    rows: list[dict], shard_count: int, shard_index: int
) -> list[dict]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    video_keys = sorted({(row["dataset"], row["video_id"]) for row in rows})
    selected = set(video_keys[shard_index::shard_count])
    return [row for row in rows if (row["dataset"], row["video_id"]) in selected]


def main() -> None:
    args = build_parser().parse_args()
    calls_path = args.calls.expanduser().resolve()
    call_root = calls_path.parent
    rows = read_csv(calls_path)
    validate_call_manifest(rows, args.expected_videos)
    rows = shard_call_rows(rows, args.shard_count, args.shard_index)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_root / "segment_diagnoses.jsonl"
    csv_path = output_root / "segment_diagnoses.csv"
    prompt_text = [
        f"[PROMPT PROFILE: {args.prompt_profile}]",
        PROMPT_PROFILES[args.prompt_profile],
    ]
    if uses_black_padding_guidance(args.prompt_profile):
        prompt_text.extend(
            ["\n[CROP PADDING GUIDANCE]", BLACK_PADDING_GUIDANCE]
        )
    focus_map = segment_focus_map(args.prompt_profile)
    for segment, focus in focus_map.items():
        prompt_text.extend([f"\n[{segment.upper()} FOCUS]", focus])
    prompt_text.extend(["\n[RESPONSE REQUEST]", RESPONSE_REQUEST])
    (output_root / "prompt.txt").write_text(
        "\n".join(prompt_text) + "\n", encoding="utf-8"
    )

    existing: dict[tuple[str, str, int, str], dict] = {}
    if args.resume:
        existing = {call_key(record): record for record in read_jsonl(jsonl_path)}
    processor, model, class_name = load_model(args.model.expanduser().resolve())
    print(
        f"Loaded {class_name}; processing {len(rows)} segment calls with "
        f"{args.prompt_profile} on shard {args.shard_index}/{args.shard_count}",
        flush=True,
    )

    for index, row in enumerate(rows, start=1):
        key = call_key(row)
        prior = existing.get(key)
        if prior is not None and not prior.get("parse_error"):
            print(
                f"[{index:02d}/{len(rows)}] resume {key[0]}/{key[1]} "
                f"frame={key[2]} segment={key[3]}",
                flush=True,
            )
            continue
        started = time.monotonic()
        messages, images = build_messages(call_root, row, args.prompt_profile)
        try:
            raw_response = generate(model, processor, messages, args.max_new_tokens)
            try:
                parsed = parse_response(raw_response, row["segment"])
                parse_error = ""
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                parsed = {field: "" for field in RESPONSE_KEYS}
                parse_error = str(error)
        finally:
            for image in images:
                image.close()
        record = {
            "dataset": row["dataset"],
            "video_id": row["video_id"],
            "requested_candidate_frame": int(
                row.get("requested_candidate_frame") or row["candidate_frame"]
            ),
            "candidate_frame": int(row["candidate_frame"]),
            "timestamp_seconds": float(row["timestamp_seconds"]),
            "score_rank": int(row["score_rank"]),
            "s2_score": float(row["s2_score"]),
            "requested_segment": row["segment"],
            "source_links": row["source_links"],
            "reference_mask_status": row["reference_mask_status"],
            "candidate_mask_status": row["candidate_mask_status"],
            "prompt_profile": args.prompt_profile,
            **parsed,
            "parse_error": parse_error,
            "raw_response": raw_response,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        existing[key] = record
        ordered = sorted(existing.values(), key=call_key)
        write_jsonl(jsonl_path, ordered)
        write_csv(csv_path, ordered)
        print(
            f"[{index:02d}/{len(rows)}] {key[0]}/{key[1]} frame={key[2]} "
            f"segment={key[3]} state={record['state'] or 'parse-error'}",
            flush=True,
        )
    print(f"Wrote {len(existing)} segment diagnoses to {output_root}")


if __name__ == "__main__":
    main()
