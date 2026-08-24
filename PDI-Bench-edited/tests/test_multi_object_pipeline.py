import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("multi-object tracker tests require the PDI PyTorch environment") from exc

from pdi_eval.multi_object_pipeline import (
    _map_tracks_between_grids,
    compare_mode_reports,
    compare_track_results,
    save_track_result,
)
from pdi_eval.perception.base import MultiObjectTrackResult
from pdi_eval.perception.segmentation_archive import load_multi_object_segmentation
from pdi_eval.perception.track_wrapper import (
    PreparedMultiObjectTracking,
    TrackWrapper,
    map_tracker_pixels_to_source,
)


class MultiObjectSegmentationTests(unittest.TestCase):
    def test_loads_objects_without_collapsing_identity(self):
        masks = np.zeros((3, 2, 8, 10), dtype=bool)
        masks[:, 0, 1:4, 1:4] = True
        masks[:, 1, 3:7, 5:9] = True
        masks[:, 1, 3, 2] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "segmentation.npz"
            np.savez_compressed(
                path,
                object_masks=masks,
                object_names=np.asarray(["link1", "link2"]),
                object_ids=np.asarray([11, 22]),
            )
            result = load_multi_object_segmentation(path)

        self.assertEqual(result.object_names, ("link1", "link2"))
        np.testing.assert_array_equal(result.object_masks, masks)
        np.testing.assert_array_equal(result.union_masks, np.any(masks, axis=1))
        self.assertEqual(result.h_pixel.shape, (3, 2))

    def test_rejects_duplicate_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "segmentation.npz"
            np.savez_compressed(
                path,
                object_masks=np.zeros((1, 2, 3, 4), dtype=bool),
                object_names=np.asarray(["link1", "link1"]),
                object_ids=np.asarray([1, 2]),
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_multi_object_segmentation(path)


class _CountingFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return values.mean(dim=1, keepdim=True)


class _FakeCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fnet = _CountingFeatures()


class _FakePredictor:
    def __init__(self):
        self.model = _FakeCore()

    def __call__(self, video, queries, grid_size, grid_query_frame):
        self.model.fnet(video.reshape(-1, *video.shape[2:]))
        frame_count = video.shape[1]
        coordinates = queries[..., 1:3]
        interaction = queries[..., 1].mean(dim=1, keepdim=True)[..., None] * 0.01
        tracks = coordinates[:, None].repeat(1, frame_count, 1, 1) + interaction[:, None]
        visibility = torch.ones(tracks.shape[:-1], dtype=torch.bool)
        return tracks, visibility


class _ChunkedFakePredictor(_FakePredictor):
    def __call__(self, video, queries, grid_size, grid_query_frame):
        flattened = video.reshape(-1, *video.shape[2:])
        midpoint = max(1, len(flattened) // 2)
        self.model.fnet(flattened[:midpoint])
        self.model.fnet(flattened[midpoint:])
        frame_count = video.shape[1]
        coordinates = queries[..., 1:3]
        tracks = coordinates[:, None].repeat(1, frame_count, 1, 1)
        visibility = torch.ones(tracks.shape[:-1], dtype=torch.bool)
        return tracks, visibility


def _prepared() -> PreparedMultiObjectTracking:
    return PreparedMultiObjectTracking(
        video_path="fixture.mp4",
        video_tensor=torch.zeros((1, 3, 3, 8, 8)),
        object_names=("link1", "link2"),
        object_queries=(
            np.asarray([[0, 1, 1], [0, 2, 2]], dtype=np.float32),
            np.asarray([[0, 5, 5], [0, 6, 6]], dtype=np.float32),
        ),
        background_queries=np.asarray([[0, 0, 7], [0, 7, 0]], dtype=np.float32),
        original_hw=(8, 8),
        tracker_hw=(8, 8),
        scale_xy=(1.0, 1.0),
        frames_count=3,
        timings={"decode_and_query_seconds": 0.25},
    )


class MultiObjectTrackingModeTests(unittest.TestCase):
    def _wrapper(self):
        wrapper = TrackWrapper.__new__(TrackWrapper)
        wrapper.device = torch.device("cpu")
        wrapper.model = _FakePredictor()
        return wrapper

    def test_uses_targeted_dense_query_budgets_for_links_2_to_4(self):
        counts = TrackWrapper._requested_object_query_counts(
            tuple(f"link{index}" for index in range(2, 8)),
            100,
            {"link2": 256, "link3": 192, "link4": 128},
        )

        self.assertEqual(counts, (256, 192, 128, 100, 100, 100))

    def test_tracker_coordinates_use_pixel_center_resize_mapping(self):
        mapped = map_tracker_pixels_to_source(
            np.asarray([[0.0, 0.0], [99.0, 49.0]]),
            tracker_hw=(50, 100),
            source_hw=(100, 200),
        )

        np.testing.assert_allclose(mapped, [[0.5, 0.5], [198.5, 98.5]])

    def test_rejects_unknown_query_budget_override(self):
        with self.assertRaisesRegex(ValueError, "unknown objects"):
            TrackWrapper._requested_object_query_counts(
                ("link2", "link3"), 100, {"link7": 128}
            )

    def test_spatial_balancing_avoids_response_cluster(self):
        candidates = np.asarray(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 2, 0],
                [0, 100, 0],
                [0, 0, 100],
            ],
            dtype=np.float32,
        )

        selected = TrackWrapper._spatially_balance_queries(candidates, 3)

        np.testing.assert_array_equal(
            selected[:, 1:3], [[0, 0], [100, 0], [0, 100]]
        )

    def test_grid_query_fallback_is_deterministic(self):
        mask = np.zeros((30, 60), dtype=bool)
        mask[5:25, 10:50] = True
        wrapper = self._wrapper()

        first = wrapper._grid_sample_queries(mask, region=1, n=16)
        second = wrapper._grid_sample_queries(mask, region=1, n=16)

        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 16)
        self.assertGreater(float(np.ptp(first[:, 1])), 25)
        self.assertGreater(float(np.ptp(first[:, 2])), 10)

    def test_joint_query_runs_one_forward_and_splits_objects(self):
        wrapper = self._wrapper()
        result = wrapper.track_prepared(_prepared(), "joint-query")

        self.assertEqual(result.metadata["model_forward_count"], 1)
        self.assertEqual(wrapper.model.model.fnet.calls, 1)
        self.assertEqual([tracks.shape[1] for tracks in result.object_tracks], [2, 2])
        self.assertEqual(result.background_tracks.shape[1], 2)

    def test_exact_group_reuses_backbone_across_isolated_forwards(self):
        wrapper = self._wrapper()
        exact = wrapper.track_prepared(_prepared(), "exact-group")

        self.assertEqual(exact.metadata["model_forward_count"], 3)
        self.assertEqual(exact.metadata["video_backbone_forward_passes"], 1)
        self.assertEqual(wrapper.model.model.fnet.calls, 1)

        joint_wrapper = self._wrapper()
        joint = joint_wrapper.track_prepared(_prepared(), "joint-query")
        self.assertFalse(np.allclose(exact.object_tracks[0], joint.object_tracks[0]))

    def test_exact_group_replays_long_video_backbone_chunks(self):
        wrapper = self._wrapper()
        wrapper.model = _ChunkedFakePredictor()
        result = wrapper.track_prepared(_prepared(), "exact-group")

        self.assertEqual(wrapper.model.model.fnet.calls, 2)
        self.assertEqual(result.metadata["video_backbone_forward_passes"], 1)
        self.assertEqual(result.metadata["video_backbone_chunks"], 2)

    def test_track_archive_uses_offsets_instead_of_pickle(self):
        result = self._wrapper().track_prepared(_prepared(), "joint-query")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tracks.npz"
            save_track_result(path, result)
            with np.load(path, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["object_offsets"], [0, 2, 4])
                np.testing.assert_array_equal(archive["query_ids"], [0, 1, 0, 1])
                np.testing.assert_array_equal(
                    archive["query_object_ids"], [0, 0, 1, 1]
                )
                metadata = json.loads(str(archive["metadata_json"].item()))
        self.assertEqual(metadata["model_forward_count"], 1)


class ModeComparisonTests(unittest.TestCase):
    def test_maps_video_track_endpoints_to_pointmap_grid(self):
        tracks = np.asarray([[[0.0, 0.0], [8.0, 4.0]]])
        mapped = _map_tracks_between_grids(
            tracks, source_hw=(5, 9), target_hw=(3, 5)
        )
        np.testing.assert_allclose(mapped, [[[0.0, 0.0], [4.0, 2.0]]])

    def test_reports_metric_and_speed_deltas(self):
        def mode(score, seconds):
            return {
                "objects": {
                    "link1": {
                        "pdi_score": score,
                        "grade": "A",
                        "breakdown": {
                            "scale_component": 0.1,
                            "traj_component": 0.2,
                            "epsilon_rigidity": score,
                            "vp_component": 0.0,
                        },
                    }
                },
                "timing": {
                    "tracking": {
                        "model_seconds": seconds,
                        "total_tracking_seconds": seconds + 0.1,
                        "peak_gpu_memory_bytes": int(seconds * 1000),
                    }
                },
            }

        comparison = compare_mode_reports(
            {"joint-query": mode(0.2, 2.0), "exact-group": mode(0.3, 5.0)}
        )
        self.assertAlmostEqual(
            comparison["objects"]["link1"]["pdi_score_exact_minus_joint"], 0.1
        )
        self.assertAlmostEqual(comparison["speed"]["exact_over_joint_ratio"], 2.5)
        self.assertEqual(
            comparison["speed"]["exact_minus_joint_peak_gpu_memory_bytes"], 3000
        )

    def test_marks_failed_link_comparison_unavailable(self):
        failed = {
            "objects": {
                "link2": {"status": "failed", "error": "insufficient depth"}
            },
            "timing": {
                "tracking": {
                    "model_seconds": 1.0,
                    "total_tracking_seconds": 1.0,
                    "peak_gpu_memory_bytes": 100,
                }
            },
        }
        complete = {
            "objects": {
                "link2": {
                    "status": "complete",
                    "pdi_score": 0.2,
                    "grade": "A",
                    "breakdown": {
                        "scale_component": 0.1,
                        "traj_component": 0.2,
                        "epsilon_rigidity": 0.3,
                        "vp_component": 0.4,
                    },
                }
            },
            "timing": failed["timing"],
        }

        comparison = compare_mode_reports(
            {"joint-query": failed, "exact-group": complete}
        )

        self.assertEqual(comparison["objects"]["link2"]["status"], "unavailable")

    def test_compares_only_queries_retained_by_both_modes(self):
        joint = TrackWrapper.__new__(TrackWrapper)
        joint.device = torch.device("cpu")
        joint.model = _FakePredictor()
        joint_result = joint.track_prepared(_prepared(), "joint-query")
        exact = TrackWrapper.__new__(TrackWrapper)
        exact.device = torch.device("cpu")
        exact.model = _FakePredictor()
        exact_result = exact.track_prepared(_prepared(), "exact-group")

        comparison = compare_track_results(
            {"joint-query": joint_result, "exact-group": exact_result}
        )
        self.assertEqual(comparison["objects"]["link1"]["common_query_count"], 2)
        self.assertGreater(
            comparison["objects"]["link1"]["mean_track_l2_pixels"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
