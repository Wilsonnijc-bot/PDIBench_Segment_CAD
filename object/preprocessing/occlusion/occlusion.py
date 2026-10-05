"""Conservative, offline gripper-occlusion audit; never changes rigidity scores.

Run with --object-root and --gripper-root. Frame numbers are zero based.
Only numpy and OpenCV are required; no model inference is performed.
"""
from __future__ import annotations

from pathlib import Path as _LayoutPath
from infrastructure.deformation_detect.layout import root as _workspace_root
_SOURCE_PATH = _LayoutPath(__file__).resolve()

import argparse
import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2
import numpy as np
from infrastructure.shared.contracts.mask_quality import MASK_QUALITY_METHOD, MAX_LINK7_AREA_FRACTION, link7_area_validity


@dataclass(frozen=True)
class Config:
    min_occluded_fraction: float = .20
    min_explained_fraction: float = .60
    min_track_fraction: float = .20
    min_run: int = 2
    min_visible_support: float = .15
    clean_overlap: float = .05
    severity_min_explained_fraction: float = .80
    severity_min_track_fraction: float = .10
    severity_min_occluded_fraction: float = .50
    severity_min_run: int = 3
    max_gripper_area_fraction: float = MAX_LINK7_AREA_FRACTION


def sample(mask, points):
    finite = np.isfinite(points).all(axis=1)
    xy = np.rint(np.nan_to_num(points)).astype(np.int64)
    h, w = mask.shape
    valid = finite & (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    out = np.zeros(len(points), dtype=bool)
    out[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return out


def runs(flags):
    edges = np.diff(np.r_[False, flags, False].astype(int))
    return [[int(a), int(b-1)] for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0])]


def align(template, obj, grip):
    """Translation-only silhouette match near current visible object.

    Reward support in object pixels, penalize missing pixels outside gripper.
    Gripper pixels are neutral, so moving the template into gripper alone cannot
    beat matching actual visible object. No dilation or inflated bounding box.
    """
    yy, xx = np.where(obj)
    if not len(xx):
        return None
    th, tw = template.shape
    h, w = obj.shape
    cx, cy = float(np.mean(xx)), float(np.mean(yy))
    x0, y0 = max(0, int(cx - 1.5*tw)), max(0, int(cy - 1.5*th))
    x1, y1 = min(w, int(cx + 1.5*tw)+1), min(h, int(cy + 1.5*th)+1)
    if y1-y0 < th or x1-x0 < tw:
        return None
    evidence = np.where(obj[y0:y1, x0:x1], 1.,
                        np.where(grip[y0:y1, x0:x1], 0., -1.)).astype(np.float32)
    scores = cv2.matchTemplate(evidence, template.astype(np.float32), cv2.TM_CCORR)
    _, _, _, (dx, dy) = cv2.minMaxLoc(scores)
    expected = np.zeros_like(obj)
    expected[y0+dy:y0+dy+th, x0+dx:x0+dx+tw] = template
    return expected


def crop(mask):
    yy, xx = np.where(mask)
    return mask[yy.min():yy.max()+1, xx.min():xx.max()+1].copy()


def detect(objects, grippers, tracks, visibility, config=Config()):
    if objects.shape != grippers.shape or objects.ndim != 3:
        raise ValueError('Object and link7 masks must have identical [T,H,W] shapes')
    if tracks.shape[:2] != visibility.shape or tracks.shape[0] != len(objects) or tracks.shape[2] != 2:
        raise ValueError('Tracks/visibility must match segmentation frames')
    areas = objects.sum(axis=(1, 2))
    mask_valid, gripper_areas, gripper_fractions = link7_area_validity(
        grippers, config.max_gripper_area_fraction)
    overlap = (objects & grippers).sum(axis=(1, 2)) / np.maximum(areas, 1)
    # Start from the first genuinely low-contact observation, not a maximum-area
    # frame chosen using rigidity scores. Earlier frames remain unassessable.
    clean = np.flatnonzero(mask_valid & (areas > 0) & (overlap <= config.clean_overlap))
    anchor = int(clean[0]) if len(clean) else None
    template = crop(objects[anchor]) if anchor is not None else None
    rows = []
    active = False
    severity_active = False
    for t, (obj, grip) in enumerate(zip(objects, grippers)):
        row = dict(frame=t, object_area=int(areas[t]), reference_frame=anchor,
                   gripper_area=int(gripper_areas[t]),
                   gripper_area_fraction=float(gripper_fractions[t]),
                   mask_valid=bool(mask_valid[t]),
                   mask_failure_reason=None if mask_valid[t] else 'link7_area_too_large',
                   expected_area=None, lost_fraction=None, occluded_fraction=None,
                   explained_fraction=None, track_fraction=None, visible_support=None,
                   mask_overlap_fraction=float(overlap[t]), onset=False, continued=False,
                   legacy_candidate=False, legacy_flagged=False,
                   severity_onset=False, severity_continued=False,
                   severity_candidate=False, severity_flagged=False,
                   candidate=False, flagged=False, status='insufficient_reference')
        if not mask_valid[t]:
            row['status'] = 'failed_link7_mask_area'
            active = severity_active = False; rows.append(row); continue
        if template is None or t < anchor:
            active = severity_active = False; rows.append(row); continue
        expected = align(template, obj, grip)
        if expected is None:
            row['status'] = 'insufficient_visible_object'; active = severity_active = False; rows.append(row); continue
        total = int(expected.sum())
        # Only missing object pixels can count as replaced. Direct overlap is
        # ambiguous (link7 can leak onto the object), never occlusion evidence.
        missing = expected & ~obj
        covered = missing & grip
        support = (expected & obj).sum() / max(1, total)
        occluded = covered.sum() / max(1, total)
        explained = covered.sum() / max(1, missing.sum())
        # Fixed anchor membership avoids shrinking the denominator to surviving
        # visible tracks. Include currently invisible tracks in gripper hits.
        eligible = sample(objects[anchor] & ~grippers[anchor], tracks[anchor]) & visibility[anchor]
        track_fraction = float((sample(grip, tracks[t]) & eligible).sum() / max(1, eligible.sum()))
        assessable = support >= config.min_visible_support and eligible.sum() >= 5
        onset = assessable and occluded >= config.min_occluded_fraction and explained >= config.min_explained_fraction and track_fraction >= config.min_track_fraction
        # Once replacement has established an episode, retain its risk flag while
        # gripper contact and affected tracks persist. This catches later frames
        # whose segmentation regrows over the gripper without starting a new
        # episode from overlap alone.
        continued = bool(active and assessable and (expected & grip).sum()/max(1,total) >= config.min_occluded_fraction and track_fraction >= config.min_track_fraction)
        candidate = bool(onset or continued)
        active = candidate
        # Independent additive branch: moderate replacement needs some track
        # corroboration; severe replacement can establish risk without it.
        # Continuation always requires actual missing-pixel replacement, unlike
        # the legacy branch. Never seed legacy continuation from this state.
        severity_base = (assessable and occluded >= config.min_occluded_fraction
                         and explained >= config.severity_min_explained_fraction)
        severity_onset = bool(severity_base and (
            track_fraction >= config.severity_min_track_fraction
            or occluded >= config.severity_min_occluded_fraction))
        severity_continued = bool(severity_active and severity_base)
        severity_active = bool(severity_onset or severity_continued)
        row.update(expected_area=total, lost_fraction=float(missing.sum()/max(1,total)),
                   occluded_fraction=float(occluded), explained_fraction=float(explained),
                   track_fraction=track_fraction, visible_support=float(support),
                   mask_overlap_fraction=float(overlap[t]), onset=bool(onset), continued=bool(continued and not onset),
                   legacy_candidate=candidate, severity_onset=severity_onset,
                   severity_continued=bool(severity_continued and not severity_onset),
                   severity_candidate=severity_active, candidate=bool(candidate or severity_active),
                   status='assessed' if assessable else 'insufficient_support')
        rows.append(row)
        # Refresh only when gripper is absent from BOTH observed and expected
        # silhouette, and mask area remains reasonably stable. Freeze at contact.
        if overlap[t] <= config.clean_overlap and occluded <= config.clean_overlap and .8 <= areas[t]/max(1,total) <= 1.25:
            template, anchor = crop(obj), t
    # Filter each branch before taking the union: a short candidate from one
    # branch must not make a short candidate from the other branch survive.
    for branch, min_run in [('legacy', config.min_run), ('severity', config.severity_min_run)]:
        flags = np.array([r[branch+'_candidate'] for r in rows])
        for start, end in runs(flags):
            if end-start+1 < min_run:
                flags[start:end+1] = False
        for row, flag in zip(rows, flags):
            row[branch+'_flagged'] = bool(flag)
    for row in rows:
        row['flagged'] = row['legacy_flagged'] or row['severity_flagged']
    return rows


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(content, encoding='utf-8'); temp.replace(path)


def process(case, gripper_root, config):
    sources = {'object': case/'masking/segmentation.npz',
               'gripper': gripper_root/'cases'/case.name/'v1_cotracker3/segmentation.npz',
               'tracks': case/'score/cotracker_exact-group.npz'}
    provenance = gripper_root/'cases'/case.name/'base_segmentation_source.json'
    video_sha = sha256(case/'replay/source.mp4')
    if json.loads(provenance.read_text())['source_video_sha256'] != video_sha:
        raise ValueError('Object replay and link segmentation use different source videos')
    with np.load(sources['object'], allow_pickle=False) as z:
        if str(z['object_names'][0]) != 'task_object':
            raise ValueError('Expected task_object at object_masks[:,0]')
        objects = z['object_masks'][:, 0]
    with np.load(sources['gripper'], allow_pickle=False) as z:
        if str(z['object_names'][5]) != 'link7':
            raise ValueError('Expected link7 at object_masks[:,5]')
        grippers = z['object_masks'][:, 5]
    with np.load(sources['tracks'], allow_pickle=False) as z:
        names = z['object_names'].tolist(); idx = names.index('task_object')
        lo, hi = z['object_offsets'][idx:idx+2]
        tracks, visibility = z['tracks'][:,lo:hi], z['visibility'][:,lo:hi]
        metadata = json.loads(str(z['metadata_json']))
        if tuple(metadata['source_hw']) != objects.shape[1:]:
            raise ValueError('Track coordinate dimensions differ from segmentation')
    video = cv2.VideoCapture(str(case/'replay/source.mp4'))
    dims = (int(video.get(cv2.CAP_PROP_FRAME_COUNT)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))
    video.release()
    if dims != objects.shape:
        raise ValueError(f'Video/mask alignment mismatch: {dims}, {objects.shape}')
    rows = detect(objects, grippers, tracks, visibility, config)
    score = json.loads((case/'score/rigidity.json').read_text())
    history = score['rigidity_history']
    if len(history) != len(rows):
        raise ValueError('Score/mask frame count mismatch')
    carried = set(score.get('evidence', {}).get('carried_frames', []))
    for row, rigidity in zip(rows, history):
        row.update(rigidity_score=rigidity, rigidity_carried=row['frame'] in carried)
    result = dict(method='gripper-occlusion-v2', case=case.name, frame_index_base=0,
                  mask_quality_method=MASK_QUALITY_METHOD,
                  failed_mask_frames=np.flatnonzero([not r['mask_valid'] for r in rows]).tolist(),
                  failed_mask_intervals=runs([not r['mask_valid'] for r in rows]),
                  detector_source_sha256=sha256(_SOURCE_PATH),
                  source_video_sha256=video_sha, config=asdict(config), inputs={k:{'path':str(v.resolve()),'sha256':sha256(v)} for k,v in sources.items()},
                  flagged_intervals=runs([r['flagged'] for r in rows]),
                  flagged_frames=sum(r['flagged'] for r in rows), frame_count=len(rows), frames=rows)
    out = case/'occlusion'
    atomic(out/'detection.json', json.dumps(result, indent=2, allow_nan=False)+'\n')
    buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    atomic(out/'frames.csv',buf.getvalue())
    return {k:result[k] for k in ('case','flagged_intervals','flagged_frames','frame_count',
                                 'failed_mask_frames','failed_mask_intervals')}


def write_report(root, results):
    lines = ["# Gripper occlusion audit", "",
        "Method: **gripper-occlusion-v2**. Conservative risk flags, not proof of physical occlusion. Original per-frame rigidity values remain unchanged; the separate filtered mean excludes flagged and failed-mask frames. Frame indices below are **zero based** (the original replay counter is frame + 1).", "",
        "## Rule", "",
        "First fail any frame whose link7 mask occupies >=25% of the image (the configurable --max-gripper-area-fraction default). Count true mask pixels, not the bounding box. Failed rows have mask_valid=false and status=failed_link7_mask_area; they supply neither a reference nor occlusion evidence and reset both episode states. Filtered scoring excludes them independently of occlusion flags. This size heuristic does not identify the table or catch small segmentation errors.", "",
        "Use a recent low-contact object silhouette as the expected mask E, translated to match the current object mask O. Link7 is G. No mask dilation, learned classifier, or rigidity-score threshold is used.", "",
        "- Missing object pixels: L = E minus O.",
        "- Gripper-replaced pixels: C = L intersect G.",
        "- Start a flag when |C|/|E| >= 20%, |C|/|L| >= 60%, and >= 20% of anchor object tracks fall inside link7. All three conditions are required.",
        "- Continue an established episode while >= 20% of E is under link7 and >= 20% of anchor tracks remain under link7. This permits object-mask regrowth during ongoing contact. `onset` and `continued` distinguish the evidence in every row.",
        "- Remove runs shorter than two frames; do not bridge gaps or pad intervals.",
        "- Independently start a severity-supported episode when replacement >= 20%, loss explanation >= 80%, and either anchor-track overlap >= 10% or replacement >= 50%.",
        "- Continue the severity-supported episode while replacement >= 20% and loss explanation >= 80%, without requiring track overlap. All frames must remain assessable; any failure resets that branch.",
        "- Keep severity-supported runs of at least three frames, including the first two. Filter the original and severity branches independently, then take their union. New-branch episodes never activate original-branch continuation. No backfilling before onset.",
        "- `onset` / `continued` and `legacy_candidate` / `legacy_flagged` describe the original branch. `severity_onset`, `severity_continued`, `severity_candidate`, and `severity_flagged` describe the added branch. `candidate` and `flagged` are the respective unions; `flagged` drives occlusion-based score exclusion; failed masks are also excluded.",
        "- Direct object/link7 overlap cannot start an episode. Missing masks, fewer than five anchor tracks, or less than 15% silhouette support are marked unassessable, not safe.", "",
        "The percentages above are defaults; each detection.json records the actual configuration. Anchor tracks are selected using anchor visibility and object membership; currently invisible tracks stay in the denominator. Current tracker visibility is not itself an occlusion signal.", "",
        "## Expected-mask estimate", "",
        "Start at the first nonempty object mask with <= 5% direct link7 overlap. Translate its cropped silhouette within a local search around the current object centroid. Match visible object pixels and penalize unexplained missing pixels; gripper-only pixels are neutral. Refresh the template only at low contact and when area is within 80–125% of its previous size. Freeze the template during contact. This is a translation-only approximation; large rotations, scaling, and segmentation errors can defeat it.", "",
        "## Example checks and limitations", "",
        "- COSMOS3_0056: the established episode includes frames 137–140, including the maximum score at frame 140. These are continuation flags: the object mask regrows during contact, so fresh missing-pixel evidence alone would miss them.",
        "- COSMOS3_0025: the new branch retains the late replacement episode despite low track overlap, including frame 162. Frames 161–173 are carried rigidity values, not fresh per-frame measurements.",
        "- LVP_ROBOWM_0010 and LVP_ROBOWM_0015: the saved persistent link7 masks select the tabletop. All frames fail the area guard; neither case has an occlusion verdict or a retained filtered score.",
        "- LVP_ROBOWM_0040: moderate replacement has sufficient evidence under the reduced track threshold. LVP_ROBOWM_0065 and LVP_ROBOWM_0021 remain negative controls for acceptable contact.",
        "- COSMOS2.5_0065: link7 visibly includes the cup itself from early frames. Overlap alone would flag nearly the whole clip; the missing-pixel requirement delays onset to frame 21. The remaining long interval is still uncertain because the gripper segmentation is contaminated. Repair that mask before using these flags as automatic exclusions.",
        "- Examples were reviewed using replay playback and decoded video frames with mask contours. LVP_ROBOWM_0056 remains uncertain where occlusion and changing object shape coexist. No exhaustive hand-labeled validation set exists, so precision/recall and causal attribution are not established.", "",
        "## Cases", "",
        "| Case | Flagged / frames | Zero-based intervals | Failed mask frames | Evidence |", "|---|---:|---|---:|---|"]
    for r in results:
        name=r['case']
        if 'error' in r:
            lines.append(f"| {name} | error | {r['error']} | — | — |")
            continue
        spans=', '.join(f"{a}–{b}" for a,b in r['flagged_intervals']) or 'none'
        lines.append(f"| [{name}](../cases/{name}/replay/task_object_exact-group.html) | {r['flagged_frames']} / {r['frame_count']} | {spans} | {len(r.get('failed_mask_frames', []))} | [CSV](../cases/{name}/occlusion/frames.csv) · [JSON](../cases/{name}/occlusion/detection.json) |")
    lines += ["", "## Reproduce", "", "```sh", "PYTHONPATH=. python3 -m object.preprocessing.occlusion.occlusion --object-root results/object-deformation-selected45-20260929 --gripper-root results/link5-link7-four-way-selected45-20260927", "```", "", "Optional: `--cases`, `--min-occluded-fraction` (shared replacement floor), `--min-explained-fraction`, `--min-track-fraction` (original branch), and `--severity-min-explained-fraction`, `--severity-min-track-fraction`, `--severity-min-occluded-fraction` (new branch). Requires NumPy and OpenCV. Source-video hashes, frame counts, mask dimensions, link identity, track coordinates, and score length are checked. Inputs and detector source are hashed in each detection.json.", "", "Use `flagged` as an occlusion-risk annotation. Do not replace flagged scores with zero or silently average them away. `rigidity_carried` records a separate existing scorer limitation.", ""]
    atomic(root/'occlusion'/'README.md', '\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-root', type=Path, required=True)
    parser.add_argument('--gripper-root', type=Path, required=True)
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--min-occluded-fraction', type=float, default=.20)
    parser.add_argument('--min-explained-fraction', type=float, default=.60)
    parser.add_argument('--min-track-fraction', type=float, default=.20)
    parser.add_argument('--severity-min-explained-fraction', type=float, default=.80)
    parser.add_argument('--severity-min-track-fraction', type=float, default=.10)
    parser.add_argument('--severity-min-occluded-fraction', type=float, default=.50)
    parser.add_argument('--max-gripper-area-fraction', type=float, default=MAX_LINK7_AREA_FRACTION,
                        help='Fail link7 masks occupying this fraction of the image or more (default: 0.25)')
    args = parser.parse_args(); results=[]
    config = Config(min_occluded_fraction=args.min_occluded_fraction, min_explained_fraction=args.min_explained_fraction, min_track_fraction=args.min_track_fraction,
                    severity_min_explained_fraction=args.severity_min_explained_fraction,
                    severity_min_track_fraction=args.severity_min_track_fraction,
                    severity_min_occluded_fraction=args.severity_min_occluded_fraction,
                    max_gripper_area_fraction=args.max_gripper_area_fraction)
    if any(not 0 < x <= 1 for x in (config.min_occluded_fraction,config.min_explained_fraction,config.min_track_fraction,
                                  config.severity_min_explained_fraction,config.severity_min_track_fraction,config.severity_min_occluded_fraction,
                                  config.max_gripper_area_fraction)):
        parser.error('Fractions must lie in (0, 1]')
    for case in sorted((args.object_root/'cases').iterdir()):
        if not case.is_dir() or (args.cases and case.name not in args.cases): continue
        try:
            result=process(case,args.gripper_root,config)
        except (ValueError,KeyError,OSError) as exc:
            result={'case':case.name,'error':str(exc)}
        results.append(result);print(json.dumps(result),flush=True)
    # A subset rerun must not erase the other cases from the index.
    if args.cases and (args.object_root/'occlusion'/'summary.json').exists():
        previous = json.loads((args.object_root/'occlusion'/'summary.json').read_text())
        merged = {r['case']:r for r in previous}
        merged.update({r['case']:r for r in results})
        results = [merged[k] for k in sorted(merged)]
    atomic(args.object_root/'occlusion'/'summary.json', json.dumps(results,indent=2)+'\n')
    write_report(args.object_root, results)


if __name__ == '__main__':
    main()
