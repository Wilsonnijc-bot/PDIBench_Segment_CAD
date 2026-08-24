import unittest
from pathlib import Path

from pdi_eval.geometry.cad_mesh import read_collada_asset_metadata, sha256_file


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]


class ColladaMetadataTests(unittest.TestCase):
    def test_official_scored_links_are_meter_z_up_assets(self):
        mesh_root = BENCHMARK_ROOT / "assets/cad/franka_fer"
        for index in range(2, 8):
            with self.subTest(link=index):
                path = mesh_root / f"link{index}.dae"
                metadata = read_collada_asset_metadata(path)
                self.assertEqual(metadata.unit_name, "meter")
                self.assertEqual(metadata.unit_meter, 1.0)
                self.assertEqual(metadata.up_axis, "Z_UP")
                self.assertEqual(len(sha256_file(path)), 64)


if __name__ == "__main__":
    unittest.main()
