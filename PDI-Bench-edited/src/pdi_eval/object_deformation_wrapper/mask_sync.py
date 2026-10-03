"""Refresh occlusion, its viewer, and crop selection from accepted VLM3 masks."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from pdi_eval.experiment.mask_merge import build_repaired_segmentation
from .occlusion import Config, atomic, process, sha256
from .occlusion_replay import export as export_replay
from .reference_visible_pixels import export_case
from .frame_selection import select_case


def sync_case(*, case: Path, repair_record: Path, base_segmentation: Path,
              gripper_root: Path, crop_root: Path) -> dict:
    """Default accepted-mask handoff; replays and crops use the same archive.

    Output roots must be dedicated to this repair. Original object masks, tracks,
    videos and scores are reused; only derived occlusion and crop data change.
    """
    case, repair_record, base_segmentation, gripper_root, crop_root = (
        Path(p).resolve() for p in (case, repair_record, base_segmentation, gripper_root, crop_root))
    repair = json.loads(repair_record.read_text())
    if repair['case'] != case.name:
        raise ValueError('VLM3 repair and object case differ')
    if repair.get('accepted') and sha256(Path(repair['output_masks'])) != repair['output_masks_sha256']:
        raise ValueError('Accepted VLM3 mask archive changed before synchronization')
    object_path = case/'masking/segmentation.npz'
    if sha256(object_path) != repair['object_input']['sha256']:
        raise ValueError('VLM3 task-object input changed before synchronization')
    detection_path = case/'occlusion/detection.json'
    config = Config(**json.loads(detection_path.read_text())['config']) if detection_path.exists() else Config()
    gripper_case = gripper_root/'cases'/case.name
    selected = gripper_case/'v1_cotracker3/segmentation.npz'
    if selected.exists():
        # Resume only this exact accepted repair, never silently reuse stale masks.
        manifest = json.loads(selected.with_suffix('.json').read_text())
        if (manifest['vlm3_record_sha256'] != sha256(repair_record)
                or manifest['output_segmentation_sha256'] != sha256(selected)):
            raise ValueError('Existing synchronized masks belong to another repair')
        expected_base = manifest.get('base_segmentation_sha256')
        if expected_base and sha256(base_segmentation) != expected_base:
            raise ValueError('Original named segmentation changed before synchronization')
    elif repair.get('accepted') is not True:
        if repair.get('status') not in {'not_triggered', 'repair_failed'}:
            raise ValueError('VLM3 repair is incomplete')
        if sha256(base_segmentation) != repair['link7_input']['sha256']:
            raise ValueError('Unrepaired VLM3 input segmentation changed')
        if sha256(case/'replay/source.mp4') != repair['source_video_sha256']:
            raise ValueError('VLM3 source video changed')
        selected.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_segmentation, selected)
        manifest = dict(method='original-mask-retained', repair_status=repair['status'],
            base_segmentation=str(base_segmentation), base_segmentation_sha256=sha256(base_segmentation),
            vlm3_record_sha256=sha256(repair_record),
            output_segmentation_sha256=sha256(selected))
        atomic(selected.with_suffix('.json'), json.dumps(manifest, indent=2)+'\n')
    else:
        manifest = build_repaired_segmentation(video=case/'replay/source.mp4',
            base_segmentation=base_segmentation, repair_record=repair_record, output_npz=selected)
    atomic(gripper_case/'base_segmentation_source.json', json.dumps({
        'source_video_sha256': repair['source_video_sha256'],
        'selected_mask_policy': 'accepted_vlm3_link7' if repair.get('accepted') else 'original_mask_retained',
        'vlm3_record': str(repair_record),
        'selected_mask_sha256': sha256(selected)}, indent=2)+'\n')
    detection = process(case, gripper_root, config)
    replay = export_replay(case)
    if replay['inputs']['gripper']['sha256'] != sha256(selected):
        raise ValueError('Occlusion replay does not use the accepted link7 archive')
    crop_output = crop_root/case.name
    crops = export_case(case, crop_output, [])
    selection = select_case(case, crop_output, 10)
    summary = dict(case=case.name, method='accepted-vlm3-default-downstream-sync',
        repair_record=str(repair_record), repair_record_sha256=sha256(repair_record),
        selected_segmentation=str(selected), selected_segmentation_sha256=sha256(selected),
        detection_sha256=sha256(detection_path), occlusion=detection,
        replay=str(case/'occlusion/replay.html'), replay_sha256=sha256(case/'occlusion/replay.html'),
        crops=crops, selected_frames=[row['frame'] for row in selection['selected_frames']],
        required_frames=selection['required_frames'], selection_method=selection['method'],
        final_interval_policy=selection['final_interval_policy'])
    atomic(gripper_case/'downstream_sync.json', json.dumps(summary, indent=2)+'\n')
    return summary
