import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pdi_eval.experiment.mask_merge import (
    build_refined_segmentation,
    sha256_file,
)


class PersistentMaskIntegrationTests(unittest.TestCase):
    def test_replaces_only_named_link_and_updates_union(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            video.write_bytes(b"same source video bytes")
            base = root / "base.npz"
            masks = np.zeros((3, 2, 8, 8), dtype=bool)
            masks[:, 0, 1:4, 1:4] = True
            masks[:, 1, 5:7, 5:7] = True
            np.savez_compressed(
                base,
                object_masks=masks,
                object_names=np.asarray(["link2", "link7"]),
                object_ids=np.asarray([2, 7]),
                masks=np.any(masks, axis=1),
            )
            work = root / "persistent"
            source = work / "sam" / "Cosmos25_0003" / "seed_masks.npz"
            source.parent.mkdir(parents=True)
            refined = np.zeros((3, 8, 8), dtype=bool)
            refined[:, 4:7, 4:7] = True
            np.savez_compressed(source, masks=refined)
            (work / "provenance.json").write_text(
                json.dumps({"results": {"Cosmos25_0003": {
                    "status": "completed_checks",
                    "source_sha256": sha256_file(video),
                    "source_frame_count": 3,
                    "source_hw": [8, 8],
                    "mask_source": str(source),
                    "masks_sha256": sha256_file(source),
                }}})
            )

            output = root / "refined.npz"
            manifest = build_refined_segmentation(
                video=video,
                base_segmentation=base,
                persistent_work=work,
                case="Cosmos25_0003",
                output_npz=output,
            )

            with np.load(output, allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive["object_masks"][:, 0], masks[:, 0])
                np.testing.assert_array_equal(archive["object_masks"][:, 1], refined)
                np.testing.assert_array_equal(
                    archive["masks"], masks[:, 0] | refined
                )
                np.testing.assert_array_equal(archive["object_ids"], [2, 7])
            self.assertEqual(manifest["changed_target_frames"], 3)
            self.assertEqual(manifest["unchanged_object_names"], ["link2"])

    def test_rejects_different_source_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "video.mp4"
            video.write_bytes(b"different video")
            base = root / "base.npz"
            np.savez_compressed(base, object_masks=np.zeros((1, 1, 4, 4), bool),
                                object_names=np.asarray(["link7"]), object_ids=np.asarray([7]))
            work = root / "persistent"
            work.mkdir()
            (work / "provenance.json").write_text(json.dumps({"results": {"case": {
                "status": "completed_checks", "source_sha256": "wrong"
            }}}))
            with self.assertRaisesRegex(ValueError, "source video hash"):
                build_refined_segmentation(
                    video=video, base_segmentation=base,
                    persistent_work=work, case="case", output_npz=root / "out.npz",
                )


if __name__ == "__main__":
    unittest.main()
