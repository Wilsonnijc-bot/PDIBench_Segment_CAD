"""Frame-lineage guard for VLM1 -> VLM2 -> SAM3.

VLM1 diagnosis records are the sole authority for the SAM reseeding frame.
"""
import json
from pathlib import Path
import cv2
from PIL import Image


def earliest_deformed(diagnoses):
    frames=[int(x['frame']) for x in diagnoses if x.get('state')=='deformed' and not x.get('parse_error')]
    if not frames: raise ValueError('No VLM1-confirmed deformed frame')
    return min(frames)


def read_original_frame(video, frame):
    cap=cv2.VideoCapture(str(video));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));
    if frame<0 or frame>=count: cap.release(); raise ValueError(f'Reseed frame {frame} outside source video ({count} frames)')
    cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release()
    if not ok: raise ValueError(f'Cannot read original source frame {frame}')
    return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)


def resolve_frames(video, diagnoses, out, deformed_rank=0):
    """Persist the first VLM1 frame and exact original-video VLM2 frame."""
    if not diagnoses: raise ValueError('VLM1 diagnosis list is empty')
    vlm1_initial=min(int(d['frame']) for d in diagnoses)
    deformed=sorted(int(d['frame']) for d in diagnoses if d.get('state')=='deformed' and not d.get('parse_error'))
    if len(deformed) <= deformed_rank: raise ValueError('Not enough VLM1-confirmed deformed frames for fallback')
    first=deformed[deformed_rank]; reseed=first+1
    vlm1=read_original_frame(video,vlm1_initial); vlm2=read_original_frame(video,reseed)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    Image.fromarray(vlm1).save(out/'vlm1_initial_frame.png')
    Image.fromarray(vlm2).save(out/'vlm2_reseed_frame.png')
    manifest={'vlm1_initial_frame':vlm1_initial,'earliest_deformed_frame':first,
              'vlm2_reseed_frame':reseed,'source_video':str(video),
              'invariant':'vlm2_reseed_frame == selected_deformed_frame + 1; both VLM2 and SAM3 use this frame',
              'fallback_rank':deformed_rank}
    return manifest


def validate_lineage(manifest, sam_frame, vlm2_frame):
    expected=int(manifest['vlm2_reseed_frame'])
    if 'earliest_deformed_frame' in manifest and expected != int(manifest['earliest_deformed_frame'])+1:
        raise AssertionError('Reseeding frame must be earliest_deformed + 1')
    if int(sam_frame)!=expected or int(vlm2_frame)!=expected:
        raise AssertionError(f'Frame mismatch: expected {expected}, VLM2={vlm2_frame}, SAM3={sam_frame}')
    return True
