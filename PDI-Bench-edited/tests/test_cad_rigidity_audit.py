import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from pdi_eval.evaluator.cad_rigidity_audit import (
    CadAnchorSet,
    CadCanonicalizationRuntime,
    ImageGridTransform,
    audit_cad_proportional_shape,
    audit_depth_world_consistency,
    backproject_pixels,
    canonicalize_camera_points,
    canonicalize_observed_sequence,
    intersect_rays_with_triangles,
)


class ImageGridTransformTests(unittest.TestCase):
    def test_uses_resize_then_crop_pixel_centers(self):
        transform = ImageGridTransform(
            source_hw=(100, 200),
            resized_hw=(50, 100),
            crop_xywh=(3, 2, 96, 48),
        )

        mapped = transform.map_source_pixels(
            np.asarray([[0.0, 0.0], [199.0, 99.0]])
        )

        np.testing.assert_allclose(mapped, [[-3.25, -2.25], [96.25, 47.25]])

    def test_mask_transform_does_not_compress_the_crop(self):
        transform = ImageGridTransform(
            source_hw=(4, 4), resized_hw=(8, 8), crop_xywh=(0, 0, 6, 6)
        )
        mask = np.zeros((4, 4), dtype=bool)
        mask[-1, -1] = True

        transformed = transform.transform_masks(mask)

        self.assertFalse(np.any(transformed))


class CadGeometryTests(unittest.TestCase):
    def test_inverse_pose_recovers_cad_points(self):
        points_cad = np.asarray(
            [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [-0.2, 0.4, 0.1]]
        )
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", [20, -10, 35], degrees=True).as_matrix()
        transform[:3, 3] = [0.4, -0.2, 1.3]
        points_camera = points_cad @ transform[:3, :3].T + transform[:3, 3]

        recovered = canonicalize_camera_points(points_camera, transform)

        np.testing.assert_allclose(recovered, points_cad, atol=1e-12)

    def test_sequence_canonicalization_rejects_invalid_pose_frames(self):
        points_cad = np.asarray([[0.1, -0.2, 0.3], [-0.2, 0.4, 0.1]])
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[0, :3, :3] = Rotation.from_euler(
            "xyz", [20, -10, 35], degrees=True
        ).as_matrix()
        transforms[0, :3, 3] = [0.4, -0.2, 1.3]
        points_camera = np.repeat(points_cad[None], 2, axis=0)
        points_camera[0] = (
            points_cad @ transforms[0, :3, :3].T + transforms[0, :3, 3]
        )

        recovered, valid = canonicalize_observed_sequence(
            points_camera,
            np.ones((2, 2), dtype=bool),
            transforms,
            np.asarray([True, False]),
        )

        np.testing.assert_allclose(recovered[0], points_cad, atol=1e-12)
        self.assertTrue(np.isnan(recovered[1]).all())
        np.testing.assert_array_equal(valid, [[True, True], [False, False]])

    def test_depth_backprojection_matches_world_pointmap_inversion(self):
        intrinsics = np.asarray(
            [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
        )
        depths = np.full((2, 3, 3), 2.0, dtype=np.float64)
        rows, columns = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        pixels = np.column_stack((columns.ravel(), rows.ravel()))
        camera_points = np.stack(
            [backproject_pixels(pixels, frame.ravel(), intrinsics).reshape(3, 3, 3)
             for frame in depths]
        )
        transforms = np.repeat(np.eye(4)[None], 2, axis=0)
        transforms[1, :3, :3] = Rotation.from_euler("y", 20, degrees=True).as_matrix()
        transforms[1, :3, 3] = [0.3, -0.2, 1.0]
        pointmaps = np.empty_like(camera_points)
        for frame_index in range(2):
            pointmaps[frame_index] = (
                camera_points[frame_index] @ transforms[frame_index, :3, :3].T
                + transforms[frame_index, :3, 3]
            )

        report = audit_depth_world_consistency(
            depths_camera=depths,
            intrinsics=intrinsics,
            pointmaps_world=pointmaps,
            T_W_from_C=transforms,
        )

        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["sample_count"], 18)
        self.assertLess(report["maximum_error"], 1e-12)

    def test_ray_intersection_selects_nearest_positive_triangle(self):
        vertices = np.asarray(
            [
                [-1, -1, 1],
                [1, -1, 1],
                [0, 1, 1],
                [-1, -1, 2],
                [1, -1, 2],
                [0, 1, 2],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [3, 4, 5]])

        points, triangles, distances = intersect_rays_with_triangles(
            np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            vertices,
            faces,
        )

        np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0])
        self.assertEqual(triangles.tolist(), [0, -1])
        self.assertAlmostEqual(distances[0], 1.0)
        self.assertTrue(np.isnan(distances[1]))


class ProportionalShapeTests(unittest.TestCase):
    @staticmethod
    def _anchors() -> np.ndarray:
        rng = np.random.default_rng(7)
        return rng.uniform(-0.5, 0.5, size=(32, 3))

    def test_rigid_transform_and_uniform_scale_cancel(self):
        anchors = self._anchors()
        rotation = Rotation.from_euler("zyx", [30, 15, -20], degrees=True).as_matrix()
        observed_frame = 2.7 * (anchors @ rotation.T) + [1.0, -3.0, 0.5]
        observed = np.repeat(observed_frame[None], 6, axis=0)

        report = audit_cad_proportional_shape(
            observed_points=observed,
            observed_valid=np.ones(observed.shape[:2], dtype=bool),
            cad_anchor_points=anchors,
            cad_anchor_valid=np.ones(len(anchors), dtype=bool),
            link_diameter=1.0,
        )

        self.assertEqual(report["status"], "uncalibrated")
        self.assertLess(report["epsilon_cad_mean"], 1e-6)
        np.testing.assert_allclose(report["relative_uniform_scale"], 2.7, rtol=1e-5)

    def test_anisotropic_shape_change_is_detected(self):
        anchors = self._anchors()
        rigid = np.repeat(anchors[None], 6, axis=0)
        stretched = rigid.copy()
        stretched[..., 0] *= 1.15
        kwargs = {
            "observed_valid": np.ones(rigid.shape[:2], dtype=bool),
            "cad_anchor_points": anchors,
            "cad_anchor_valid": np.ones(len(anchors), dtype=bool),
            "link_diameter": 1.0,
        }

        rigid_report = audit_cad_proportional_shape(
            observed_points=rigid, **kwargs
        )
        stretched_report = audit_cad_proportional_shape(
            observed_points=stretched, **kwargs
        )

        self.assertLess(rigid_report["epsilon_cad_mean"], 1e-6)
        self.assertGreater(stretched_report["epsilon_cad_mean"], 0.02)

    def test_unscorable_does_not_return_zero(self):
        anchors = self._anchors()
        report = audit_cad_proportional_shape(
            observed_points=anchors[None],
            observed_valid=np.ones((1, len(anchors)), dtype=bool),
            cad_anchor_points=anchors,
            cad_anchor_valid=np.ones(len(anchors), dtype=bool),
            link_diameter=1.0,
        )

        self.assertEqual(report["status"], "unscorable")
        self.assertIsNone(report["epsilon_cad_mean"])


class CadRuntimeTests(unittest.TestCase):
    def test_retained_query_ids_select_the_same_fixed_cad_anchors(self):
        rng = np.random.default_rng(11)
        anchors = np.column_stack(
            (rng.uniform(-0.4, 0.4, 32), rng.uniform(-0.4, 0.4, 32), np.ones(32))
        )
        intrinsics = np.asarray(
            [[80.0, 0.0, 50.0], [0.0, 80.0, 50.0], [0.0, 0.0, 1.0]]
        )
        pixels = np.column_stack(
            (
                anchors[:, 0] * 80.0 + 50.0,
                anchors[:, 1] * 80.0 + 50.0,
            )
        )
        query_ids = np.arange(len(anchors), dtype=np.int32)
        retained = query_ids[::-1]
        frame_count = 6
        runtime = CadCanonicalizationRuntime(
            config={"shape_thresholds": {"link2": {"mean": 0.01, "p90": 0.01}}},
            link_names=("link2",),
            grid_transform=ImageGridTransform(
                source_hw=(100, 100),
                resized_hw=(100, 100),
                crop_xywh=(0, 0, 100, 100),
            ),
            meshes={
                "link2": SimpleNamespace(diameter=1.5, sha256="fixture")
            },
            masks_geometry={
                "link2": np.ones((frame_count, 100, 100), dtype=bool)
            },
            anchors={
                "link2": CadAnchorSet(
                    query_ids=query_ids,
                    query_pixels_source=pixels,
                    points_cad=anchors,
                    triangle_ids=np.zeros(len(anchors), dtype=np.int64),
                    valid=np.ones(len(anchors), dtype=bool),
                )
            },
            anchor_errors={},
            poses=SimpleNamespace(
                video_depth_scale=1.0,
                pose_valid=np.ones((frame_count, 1), dtype=bool),
                T_C_from_L=np.repeat(
                    np.eye(4, dtype=np.float64)[None, None],
                    frame_count,
                    axis=0,
                ),
            ),
            pose_discontinuity={},
        )
        tracks = np.repeat(pixels[::-1][None], frame_count, axis=0)
        track_result = SimpleNamespace(
            object_query_ids=(retained,),
            object_queries=(np.column_stack((np.zeros(len(retained)), pixels[::-1])),),
            object_tracks=(tracks,),
            object_visibility=(np.ones((frame_count, len(retained)), dtype=bool),),
        )
        geometry = SimpleNamespace(
            frames_count=frame_count,
            depth_camera=np.ones((frame_count, 100, 100), dtype=np.float64),
            intrinsics_camera=intrinsics,
        )

        report = runtime.audit_track_result(
            geometry=geometry,
            track_result=track_result,
            object_index=0,
            object_name="link2",
        )

        self.assertEqual(report["status"], "complete")
        self.assertLess(report["epsilon_cad_mean"], 1e-8)
        self.assertEqual(report["observed_coordinate_frame"], "original-cad-link")
        np.testing.assert_array_equal(report["retained_query_ids"], retained)


if __name__ == "__main__":
    unittest.main()
