import unittest

import numpy as np

from pdi_eval.perception.cad_reference import (
    select_grouped_candidates_from_descriptors,
)


class GroupedCadSelectionTests(unittest.TestCase):
    def test_assigns_each_candidate_and_mesh_at_most_once(self):
        candidates = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        references = np.array(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [1.0, 1.0],
                [1.1, 1.0],
                [2.0, 2.0],
            ]
        )
        groups = ["link1", "link1", "link2", "link2", "link3"]

        selected = select_grouped_candidates_from_descriptors(
            [0.9, 0.8, 0.7],
            candidates,
            references,
            groups,
            minimum_sam_score=0.1,
            maximum_objects=7,
            cad_similarity_weight=0.5,
            cad_similarity_temperature=0.2,
            minimum_combined_score=0.1,
        )

        self.assertEqual([item["reference_group"] for item in selected], ["link1", "link2", "link3"])
        self.assertEqual([item["candidate_index"] for item in selected], [0, 1, 2])

    def test_rejects_low_sam_scores_before_cad_assignment(self):
        selected = select_grouped_candidates_from_descriptors(
            [0.05],
            np.array([[0.0, 0.0]]),
            np.array([[0.0, 0.0]]),
            ["link1"],
            minimum_sam_score=0.1,
            maximum_objects=7,
            cad_similarity_weight=0.5,
            cad_similarity_temperature=0.2,
            minimum_combined_score=0.1,
        )
        self.assertEqual(selected, [])

    def test_global_assignment_avoids_greedy_identity_trap(self):
        selected = select_grouped_candidates_from_descriptors(
            [1.0, 0.1],
            np.array([[0.9], [0.0]]),
            np.array([[0.0], [2.0]]),
            ["link1", "link2"],
            minimum_sam_score=0.0,
            maximum_objects=2,
            cad_similarity_weight=0.2,
            cad_similarity_temperature=1.0,
            minimum_combined_score=0.0,
        )
        assignment = {
            item["reference_group"]: item["candidate_index"] for item in selected
        }
        self.assertEqual(assignment, {"link1": 1, "link2": 0})


if __name__ == "__main__":
    unittest.main()
