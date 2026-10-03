import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json

from pdi_eval.object_deformation_wrapper.frame_selection import select_frames, select_case
from pdi_eval.object_deformation_wrapper.occlusion import sha256


def inputs(areas, flagged=()):
    manifest = {"case": "case", "frame_count": len(areas), "frames": [
        {"frame": t, "available_area": area, "mapped_reference_area": area,
         "mask_valid": True, "mapping_status": "mapped"} for t, area in enumerate(areas)]}
    detection = {"case": "case", "frames": [
        {"frame": t, "flagged": t in flagged, "mask_valid": True, "status": "assessed"}
        for t in range(len(areas))]}
    return manifest, detection


def frames(result):
    return [r["frame"] for r in result["selected_frames"]]


class FrameSelectionTests(unittest.TestCase):
    def test_overmask_gate_excludes_high_area_and_recovery_frames(self):
        m,d=inputs(list(range(20)),flagged=(14,15))
        d['frames'][16]['mask_overlap_fraction']=.951
        d['frames'][19]['mask_overlap_fraction']=1.0
        d['frames'][18]['mask_overlap_fraction']=.95
        result=select_frames(m,d,count=5)
        self.assertNotIn(16,frames(result));self.assertNotIn(19,frames(result))
        self.assertIn(18,frames(result))
        self.assertEqual(result['overmask_excluded_frames'],[16,19])
        self.assertEqual(result['occlusion_episodes'][0]['status'],'successor_crop_invalid')
        m['frames'][17]['link7_overmask']=True
        self.assertNotIn(17,frames(select_frames(m,d,count=5)))

    def test_ten_frames_reserve_recovery_and_double_late_quota(self):
        m, d = inputs(list(range(100)), flagged=(19, 20))
        result = select_frames(m, d)
        self.assertEqual(result["interval_quotas"], [0, 2, 2, 2, 4])
        self.assertEqual(frames(result), [21, 39, 58, 59, 78, 79, 96, 97, 98, 99])
        self.assertEqual(result["required_frames"], [21])
        self.assertEqual(result["shortfall"], 0)

    def test_ten_frame_shortfall_keeps_unique_valid_frames(self):
        m, d = inputs([10] * 10)
        result = select_frames(m, d)
        self.assertEqual(frames(result), list(range(2, 10)))
        self.assertEqual(result["shortfall"], 2)

    def test_interval_priority_excludes_first_quintile_and_breaks_ties(self):
        m, d = inputs([999, 998, 4, 4, 6, 5, 7, 8, 9, 10])
        result = select_frames(m, d, count=5)
        self.assertEqual(frames(result), [2, 4, 7, 8, 9])
        self.assertEqual([r["interval"] for r in result["selected_frames"]], [2, 3, 4, 5, 5])
        self.assertEqual(result["intervals"][1]["area_ranked_frames"], [2, 3])
        self.assertEqual(result["status"], "complete")

    def test_recovery_consumes_its_interval_slot_before_larger_frame(self):
        m, d = inputs([1, 1, 1, 100, 10, 9, 8, 7, 6, 5], flagged=(0, 1))
        before = deepcopy((m, d))
        result = select_frames(m, d, count=5)
        self.assertEqual(result["required_frames"], [2])
        self.assertEqual(frames(result), [2, 4, 6, 8, 9])
        self.assertEqual((m, d), before)
        self.assertEqual(result["selected_frames"][0]["reason"], "immediate_post_occlusion")

    def test_two_final_interval_recoveries_replace_both_area_choices(self):
        m, d = inputs([10] * 20, flagged=(14, 15, 17, 18))
        result = select_frames(m, d, count=5)
        self.assertEqual(result["required_frames"], [16, 19])
        self.assertEqual(frames(result), [4, 8, 12, 16, 19])

    def test_first_interval_recovery_overrides_exclusion(self):
        m, d = inputs([10] * 20, flagged=(0, 1))
        result = select_frames(m, d, count=5)
        self.assertIn(2, frames(result))
        self.assertEqual(len(frames(result)), 5)
        self.assertEqual(result["selected_frames"][0]["reason"], "immediate_post_occlusion")

    def test_empty_interval_fallback_prefers_latest_then_previous(self):
        m, d = inputs([10] * 20)
        for t in range(4, 8):
            m["frames"][t]["mapping_status"] = "unavailable"
        result = select_frames(m, d, count=5)
        self.assertEqual(frames(result), [8, 12, 16, 17, 18])
        self.assertEqual(result["selected_frames"][-1]["reason"], "empty_interval_fallback")
        self.assertEqual(result["intervals"][1]["unfilled_quota"], 1)
        for t in range(16, 20):
            m["frames"][t]["mapping_status"] = "unavailable"
        result = select_frames(m, d, count=5)
        self.assertEqual(frames(result), [8, 12, 13, 14, 15])
        self.assertFalse(any(t < 4 for t in frames(result)))

    def test_strict_empty_interval_is_not_backfilled(self):
        m, d = inputs([10] * 20)
        for t in range(4, 8):
            m["frames"][t]["mapping_status"] = "unavailable"
        result = select_frames(m, d, count=5, strict_bins=True)
        self.assertEqual(frames(result), [8, 12, 16, 17])
        self.assertEqual(result["shortfall"], 1)

    def test_uneven_frame_count_uses_exact_fractional_boundaries(self):
        m, d = inputs([10] * 11)
        result = select_frames(m, d, count=5)
        self.assertEqual(frames(result), [3, 5, 7, 9, 10])
        self.assertEqual([b["start_frame"] for b in result["intervals"]], [0, 3, 5, 7, 9])
        self.assertEqual([b["end_frame_exclusive"] for b in result["intervals"]], [3, 5, 7, 9, 11])

    def test_invalid_successor_is_reported_not_shifted(self):
        m, d = inputs([8] * 10, flagged=(0, 1))
        m["frames"][2]["mapping_status"] = "low_full_object_coverage"
        result = select_frames(m, d, count=5)
        self.assertEqual(result["required_frames"], [])
        self.assertEqual(result["unsatisfied_recovery_count"], 1)
        self.assertEqual(result["occlusion_episodes"][0]["immediate_successor_frame"], 2)
        self.assertEqual(result["occlusion_episodes"][0]["status"], "successor_crop_invalid")
        self.assertEqual(result["status"], "incomplete")

    def test_unknown_and_failed_successors_do_not_mean_clear(self):
        for status in ("insufficient_support", "failed_link7_mask_area"):
            m, d = inputs([10] * 10, flagged=(0, 1))
            d["frames"][2]["status"] = status
            if status == "failed_link7_mask_area":
                d["frames"][2]["mask_valid"] = False
            result = select_frames(m, d, count=5)
            self.assertEqual(result["required_frames"], [])
            self.assertEqual(result["occlusion_episodes"][0]["status"], "successor_not_assessable")

    def test_terminal_occlusion_has_no_successor(self):
        m, d = inputs([10] * 10, flagged=(8, 9))
        result = select_frames(m, d, count=5)
        self.assertEqual(result["required_frames"], [])
        self.assertEqual(result["occlusion_episodes"][0]["status"], "occlusion_reaches_video_end")
        self.assertIsNone(result["occlusion_episodes"][0]["immediate_successor_frame"])

    def test_shortfall_and_empty_masks_are_not_padded(self):
        m, d = inputs([4, 0, 2])
        result = select_frames(m, d, count=5)
        self.assertEqual(result["selected_count"], 1)  # Frame 0 is excluded.
        self.assertEqual(result["shortfall"], 4)
        for row in m["frames"]:
            row["mask_valid"] = False
        self.assertEqual(select_frames(m, d, count=5)["selected_count"], 0)

    def test_too_many_mandatory_frames_fails_explicitly(self):
        m, d = inputs([10] * 18, flagged=tuple(t for t in range(18) if t % 3 != 2))
        with self.assertRaisesRegex(ValueError, "6 mandatory"):
            select_frames(m, d, count=5)

    def test_selection_checks_audit_and_removes_only_stale_owned_crops(self):
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            (case / "occlusion").mkdir(parents=True)
            output = Path(directory) / "crops" / "case"
            (output / "selected/frame_00003").mkdir(parents=True)
            (output / "examples/frame_00003").mkdir(parents=True)
            m, d = inputs([10] * 10)
            audit = case / "occlusion/detection.json"
            audit.write_text(json.dumps(d))
            m["occlusion_detection_sha256"] = sha256(audit)
            (output / "manifest.json").write_text(json.dumps(m))
            with patch("pdi_eval.object_deformation_wrapper.frame_selection.render_saved_examples") as render:
                select_case(case, output, count=5)
                render.assert_called_once_with(case, output, [2, 4, 6, 8, 9], subdirectory="selected")
            self.assertFalse((output / "selected/frame_00003").exists())
            self.assertTrue((output / "examples/frame_00003").exists())
            self.assertTrue((output / "selection.json").is_file())
            audit.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Occlusion audit changed"):
                select_case(case, output, count=5)


if __name__ == "__main__":
    unittest.main()
