import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pdi_eval.evaluator.cad_rigidity_audit import CadAnchorSet
from pdi_eval.utils.cad_replay import (
    _composite_mesh,
    _cotracker_cad_panel,
    _encode_h264,
    _load_cotracker_groups,
    _rasterize_mesh,
    _rasterize_silhouette,
)


class CadReplayRenderingTests(unittest.TestCase):
    @patch("pdi_eval.utils.cad_replay.subprocess.run")
    def test_h264_encoder_requests_quicktime_compatible_output(self, run):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.mp4"
            target = Path(temporary) / "target.mp4"
            source.write_bytes(b"source")
            run.side_effect = lambda command, **_kwargs: Path(command[-1]).write_bytes(
                b"encoded"
            )

            _encode_h264(source, target)

        command = run.call_args.args[0]
        self.assertIn("libx264", command)
        self.assertIn("yuv420p", command)
        self.assertIn("avc1", command)

    def test_triangle_mesh_has_filled_surface_and_outline(self):
        uv = np.asarray([[8.0, 8.0], [56.0, 10.0], [30.0, 54.0]])
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)

        mask, polygons = _rasterize_mesh(uv, faces, panel_hw=(64, 64))
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        _composite_mesh(
            image,
            mask,
            polygons,
            color=(80, 160, 240),
            alpha=0.5,
            outline_color=(120, 200, 255),
        )

        self.assertGreater(np.count_nonzero(mask), 800)
        self.assertGreater(int(image[25, 30].sum()), 0)
        self.assertGreater(int(image[8, 8].sum()), int(image[25, 30].sum()))

    def test_offscreen_mesh_returns_empty_mask(self):
        uv = np.asarray([[-20.0, -20.0], [-10.0, -20.0], [-15.0, -10.0]])
        faces = np.asarray([[0, 1, 2]], dtype=np.int64)

        mask, polygons = _rasterize_mesh(uv, faces, panel_hw=(64, 64))

        self.assertFalse(np.any(mask))
        self.assertEqual(polygons.shape, (0, 3, 2))

    def test_projected_silhouette_is_filled(self):
        uv = np.asarray([[8.0, 8.0], [56.0, 10.0], [30.0, 54.0], [30.0, 25.0]])

        mask, polygons = _rasterize_silhouette(uv, panel_hw=(64, 64))

        self.assertGreater(np.count_nonzero(mask), 800)
        self.assertEqual(polygons.shape[0], 1)
        self.assertGreaterEqual(polygons.shape[1], 3)

    def test_load_cotracker_groups_preserves_query_correspondence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tracks.npz"
            np.savez_compressed(
                path,
                object_names=np.asarray(["link2", "link3"]),
                object_offsets=np.asarray([0, 2, 3]),
                tracks=np.arange(24, dtype=np.float64).reshape(4, 3, 2),
                visibility=np.ones((4, 3), dtype=np.float64),
                queries=np.asarray(
                    [[0.0, 10.0, 20.0], [0.0, 30.0, 40.0], [0.0, 50.0, 60.0]]
                ),
                query_ids=np.asarray([7, 9, 4]),
            )

            groups = _load_cotracker_groups(path)

        self.assertEqual(groups["link2"]["tracks"].shape, (4, 2, 2))
        np.testing.assert_array_equal(groups["link2"]["query_ids"], [7, 9])
        np.testing.assert_array_equal(groups["link3"]["queries"][:, 1:], [[50.0, 60.0]])

    def test_cotracker_cad_panel_draws_observed_anchor_correspondence(self):
        mesh_vertices = np.asarray(
            [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
        )
        view = {
            "vertices": mesh_vertices,
            "faces": np.asarray([[0, 1, 2]]),
            "center": np.zeros(3),
            "axes": np.eye(3),
            "bounds": np.asarray([[-1.2, -1.2], [1.2, 1.2]]),
        }
        views = {f"link{index}": dict(view) for index in range(2, 8)}
        observed = [np.asarray([[0.4, 0.0, 0.0]]) for _ in range(6)]
        valid = [np.asarray([True]) for _ in range(6)]
        anchors = [
            CadAnchorSet(
                query_ids=np.asarray([index]),
                query_pixels_source=np.asarray([[0.0, 0.0]]),
                points_cad=np.asarray([[-0.4, 0.0, 0.0]]),
                triangle_ids=np.asarray([0]),
                valid=np.asarray([True]),
            )
            for index in range(6)
        ]

        panel = _cotracker_cad_panel(observed, valid, anchors, views, (240, 360))

        self.assertEqual(panel.shape, (240, 360, 3))
        self.assertGreater(np.count_nonzero(panel != 18), 1000)


if __name__ == "__main__":
    unittest.main()
