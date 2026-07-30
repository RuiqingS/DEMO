from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from demo_runtime.docking_validation import (
    binding_mode_diversity,
    interaction_metrics,
)
from demo_runtime.qm9_cmop import ExperimentOptions, build_parser
from demo_runtime.recording import JsonlRunRecorder, read_jsonl
from demo_runtime.specs import ProblemSpec
from demo_runtime.suite import expand_runs, load_suite
from demo_runtime.three_population import ThreePopulationAudit


REPO_ROOT = Path(__file__).resolve().parents[1]


class ReviewerSuiteTests(unittest.TestCase):
    def test_all_eight_reviewer_points_have_a_suite(self) -> None:
        expected = {
            "reviewer_noise_inheritance",
            "reviewer_scheduler",
            "reviewer_operators",
            "reviewer_distance",
            "qm9_cmop",
            "reviewer_docking_validation",
            "reviewer_three_population",
            "reviewer_protocol",
        }
        suite_dir = REPO_ROOT / "configs" / "suites"
        self.assertTrue(
            expected
            <= {path.stem for path in suite_dir.glob("*.json")}
        )

    def test_paes_is_not_an_operator_or_task(self) -> None:
        parser = build_parser()
        operator_action = next(
            action
            for action in parser._actions
            if action.dest == "operator_mode"
        )
        self.assertNotIn("paes", operator_action.choices)
        operators = load_suite(
            REPO_ROOT / "configs" / "suites" / "reviewer_operators.json"
        )
        rendered = str(operators).lower()
        # Description may state that it is excluded; no executable task may use it.
        for run in expand_runs(operators):
            self.assertNotIn("paes", run.task_id.lower())
            self.assertNotIn("paes", run.method.lower())
            self.assertNotIn("paes", " ".join(run.args).lower())

    def test_reviewer_qm9_tasks_use_explicit_equal_budgets(self) -> None:
        for suite_name in (
            "reviewer_noise_inheritance",
            "reviewer_scheduler",
            "reviewer_operators",
            "reviewer_distance",
        ):
            suite = load_suite(
                REPO_ROOT / "configs" / "suites" / f"{suite_name}.json"
            )
            for run in expand_runs(suite):
                self.assertIn("--evaluation-budget", run.args)
                budget_index = run.args.index("--evaluation-budget")
                self.assertEqual(run.args[budget_index + 1], "704")

    def test_options_default_to_the_original_runtime_path(self) -> None:
        options = ExperimentOptions()
        self.assertEqual(options.operator_mode, "full")
        self.assertIsNone(options.scheduler_mode)
        self.assertIsNone(options.distance_mode)
        self.assertIsNone(options.selection_mode)

    @unittest.skipUnless(
        os.environ.get("DEMO_TEST_PLOTTING") == "1",
        "Enable reviewer parameter plotting in the target environment",
    )
    def test_protocol_parameter_curve_is_built_from_jsonl(self) -> None:
        from demo_runtime.plotting import plot_suite_results

        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "reviewer_protocol"
            tasks = (
                ("protocol_population_16", "Population 16", 16),
                ("protocol_hv_d", "HV(D=704)", 32),
                ("protocol_population_64", "Population 64", 64),
            )
            for index, (task_id, method, population_size) in enumerate(tasks):
                recorder = JsonlRunRecorder(
                    suite_dir / method / f"seed_{index}",
                    f"plot-{index}",
                    {
                        "task_id": task_id,
                        "method": method,
                        "problem": "qm9_frontier_alignment",
                    },
                )
                recorder.record_generation(
                    [],
                    [],
                    {
                        "hv": population_size / 100.0,
                        "population_size": population_size,
                    },
                    generation=1,
                )
                recorder.event("run_end", status="completed")
                recorder.mark_complete()
            output = plot_suite_results(suite_dir)
            self.assertTrue(
                (
                    output
                    / "parameter_curves"
                    / "population_size_hv.png"
                ).exists()
            )


class DockingDiagnosticTests(unittest.TestCase):
    def test_interaction_fingerprint_counts_categories(self) -> None:
        protein = [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "element": "O",
                "residue": "ASP",
            }
        ]
        ligand = [
            {
                "x": 0.0,
                "y": 0.0,
                "z": 3.0,
                "element": "N",
                "formal_charge": 1,
            }
        ]
        metrics = interaction_metrics(protein, ligand)
        self.assertEqual(metrics["contact_count"], 1.0)
        self.assertEqual(metrics["hbond_count"], 1.0)
        self.assertEqual(metrics["ionic_count"], 1.0)
        self.assertEqual(metrics["clash_count"], 0.0)

    def test_binding_mode_diversity_is_rigid_motion_invariant(self) -> None:
        first = [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 1.0, "y": 0.0, "z": 0.0},
            {"x": 0.0, "y": 1.0, "z": 0.0},
        ]
        translated = [
            {"x": 5.0, "y": -2.0, "z": 3.0},
            {"x": 6.0, "y": -2.0, "z": 3.0},
            {"x": 5.0, "y": -1.0, "z": 3.0},
        ]
        self.assertAlmostEqual(
            binding_mode_diversity([first, translated]),
            0.0,
            places=6,
        )


class ThreePopulationAuditTests(unittest.TestCase):
    def test_direct_and_offspring_transfers_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = JsonlRunRecorder(
                temporary,
                "three-pop",
                {"method": "SAES", "problem": "three-pop"},
            )
            audit = ThreePopulationAudit(recorder, enabled=True)
            a0, b0, c0 = {}, {}, {}
            with mock.patch(
                "demo_runtime.three_population.population_diversity_metrics",
                return_value={},
            ):
                audit.record(
                    0,
                    {"A": [a0], "B": [b0], "C": [c0]},
                    [],
                    {},
                )
                child = {}
                audit.register_offspring(
                    [child],
                    [
                        {
                            "operator": "mutation",
                            "parent_ids": [recorder.molecule_id(a0)],
                            "source_populations": ["A"],
                        }
                    ],
                    generation=1,
                    noise_t=500,
                )
                audit.record(
                    1,
                    {"A": [], "B": [a0, child], "C": [b0]},
                    [],
                    {},
                )
            transfers = list(
                read_jsonl(Path(temporary) / "transfers.jsonl")
            )
            a_to_b = next(
                row
                for row in transfers
                if row["source_population"] == "A"
            )
            b_to_c = next(
                row
                for row in transfers
                if row["source_population"] == "B"
            )
            self.assertEqual(a_to_b["direct_count"], 1)
            self.assertEqual(a_to_b["offspring_count"], 1)
            self.assertEqual(b_to_c["direct_count"], 1)


@unittest.skipUnless(
    os.environ.get("DEMO_TEST_CHEMISTRY") == "1",
    "Enable RDKit/tensor diagnostics in the target cgm310 environment",
)
class ChemistryDiagnosticTests(unittest.TestCase):
    def test_identical_parent_has_full_inheritance(self) -> None:
        import torch

        from demo_runtime.diagnostics import inheritance_metrics

        molecule = {
            "x": torch.tensor([[0.0, 0.0, 0.0], [1.50, 0.0, 0.0]]),
            "atom_type": torch.tensor([1, 1]),
            "__canonical__alpha": 80.0,
            "__canonical__mu": 4.0,
        }
        dataset_info = {
            "name": "qm9",
            "atom_decoder": ["H", "C", "N", "O", "F"],
        }
        spec = ProblemSpec.load(
            REPO_ROOT
            / "configs"
            / "problems"
            / "qm9_frontier_alignment.json"
        )
        metrics = inheritance_metrics(
            dict(molecule),
            [dict(molecule)],
            dataset_info,
            spec,
        )
        self.assertAlmostEqual(metrics["morgan_similarity"], 1.0)
        self.assertAlmostEqual(metrics["mcs_atom_fraction"], 1.0)
        self.assertAlmostEqual(metrics["murcko_retention"], 1.0)
        self.assertAlmostEqual(metrics["fragment_retention"], 1.0)
        self.assertAlmostEqual(metrics["mcs_3d_rmsd"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
