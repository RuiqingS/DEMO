from __future__ import annotations

import copy
import unittest

import torch

import evo


DATASET_INFO = {"atom_decoder": ["H", "C", "N", "O", "F"]}
ATOM_INDEX = {atom: index for index, atom in enumerate(DATASET_INFO["atom_decoder"])}


def graph_input(atom_types, x_positions, *, constraint=0):
    """Build the tensor dictionary consumed by the structure matcher."""
    return {
        "x": torch.tensor(
            [[float(position), 0.0, 0.0] for position in x_positions],
            dtype=torch.float32,
        ),
        "atom_type": torch.tensor(
            [ATOM_INDEX[atom] for atom in atom_types],
            dtype=torch.long,
        ),
        "node_mask": torch.ones((len(atom_types), 1), dtype=torch.float32),
        "Constraint": constraint,
        "EvaluatedSC": False,
    }


class GraphMatchingRegressionTests(unittest.TestCase):
    def test_partial_score_is_invariant_to_atom_renumbering(self):
        fragment = graph_input(
            ["F", "C", "C", "C", "C", "C", "C"],
            [0.0, 1.35, 2.89, 4.43, 5.97, 7.51, 9.05],
        )
        reversed_fragment = {
            **fragment,
            "x": torch.flip(fragment["x"], dims=(0,)),
            "atom_type": torch.flip(fragment["atom_type"], dims=(0,)),
            "node_mask": torch.flip(fragment["node_mask"], dims=(0,)),
        }
        target = graph_input(
            ["C", "C", "C", "C", "C", "C"],
            [1.35, 2.89, 4.43, 5.97, 7.51, 9.05],
        )

        forward_score, _ = evo.check_subgraph_proportionVF2_optimized(
            DATASET_INFO,
            fragment["x"],
            fragment["atom_type"],
            fragment["node_mask"],
            target["x"],
            target["atom_type"],
            target["node_mask"],
        )
        reversed_score, _ = evo.check_subgraph_proportionVF2_optimized(
            DATASET_INFO,
            reversed_fragment["x"],
            reversed_fragment["atom_type"],
            reversed_fragment["node_mask"],
            target["x"],
            target["atom_type"],
            target["node_mask"],
        )

        self.assertAlmostEqual(forward_score, 100.0 * 6.0 / 7.0)
        self.assertAlmostEqual(reversed_score, forward_score)

    def test_partial_score_uses_connected_atom_coverage(self):
        fragment = graph_input(
            ["C", "C", "C", "C", "C"],
            [0.0, 1.54, 3.08, 4.62, 6.16],
        )
        target = graph_input(
            ["C", "C", "C", "C"],
            [0.0, 1.54, 3.08, 4.62],
        )

        score, matched = evo.check_subgraph_proportionVF2_optimized(
            DATASET_INFO,
            fragment["x"],
            fragment["atom_type"],
            fragment["node_mask"],
            target["x"],
            target["atom_type"],
            target["node_mask"],
        )

        self.assertAlmostEqual(score, 80.0)
        self.assertEqual(len(matched), 4)

    def test_exclusive_matching_does_not_double_count_target_atoms(self):
        target = graph_input(["C", "C"], [0.0, 1.54])
        pattern = graph_input(["C", "C"], [0.0, 1.54])

        result = evo._evaluate_single_molecule_patt(
            target,
            [pattern, copy.deepcopy(pattern)],
            DATASET_INFO,
            tolerance=0.0,
            exclusive=True,
        )

        self.assertEqual(result["structure_Con"], [0.0, 100.0])
        self.assertEqual(result["matched_indices"], [0, 1])

    def test_joint_matching_is_invariant_to_pattern_order(self):
        target = graph_input(
            ["C", "C", "N", "C", "C"],
            [0.0, 1.54, -1.47, 10.0, 11.54],
        )
        carbon_carbon = graph_input(["C", "C"], [0.0, 1.54])
        carbon_nitrogen = graph_input(["C", "N"], [0.0, 1.47])

        first = evo._evaluate_single_molecule_patt(
            target,
            [carbon_carbon, carbon_nitrogen],
            DATASET_INFO,
            tolerance=0.0,
            exclusive=True,
        )
        second = evo._evaluate_single_molecule_patt(
            target,
            [carbon_nitrogen, carbon_carbon],
            DATASET_INFO,
            tolerance=0.0,
            exclusive=True,
        )

        self.assertEqual(first["structure_Con"], [0.0, 0.0])
        self.assertEqual(second["structure_Con"], [0.0, 0.0])
        self.assertEqual(first["matched_indices"], second["matched_indices"])

    def test_serial_and_mp_entry_points_share_the_same_semantics(self):
        target = graph_input(["C", "C"], [0.0, 1.54])
        pattern = graph_input(["C", "C"], [0.0, 1.54])
        patterns = [pattern, copy.deepcopy(pattern)]

        serial = evo.EvalPop_Patt(
            [copy.deepcopy(target), copy.deepcopy(target)],
            patterns,
            DATASET_INFO,
            tolerance=0.0,
        )
        mp_entry = evo.EvalPop_Patt_MP(
            [copy.deepcopy(target), copy.deepcopy(target)],
            patterns,
            DATASET_INFO,
            tolerance=0.0,
            num_workers=2,
        )

        for serial_mol, mp_mol in zip(serial, mp_entry):
            self.assertEqual(
                serial_mol["structure_Con"],
                mp_mol["structure_Con"],
            )
            self.assertEqual(
                serial_mol["matched_indices"],
                mp_mol["matched_indices"],
            )

    def test_disabling_patterns_clears_cached_constraint_state(self):
        target = graph_input(["C", "C"], [0.0, 1.54])
        target["FakeSC"] = [50.0]

        result = evo.EvalPop_Patt(
            [target],
            None,
            DATASET_INFO,
            tolerance=0.0,
        )

        self.assertEqual(result[0]["structure_Con"], [0.0])
        self.assertEqual(result[0]["matched_indices"], [])
        self.assertTrue(result[0]["EvaluatedSC"])
        self.assertNotIn("FakeSC", result[0])


if __name__ == "__main__":
    unittest.main()
