from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from demo_runtime.recording import JsonlRunRecorder, read_jsonl
from demo_runtime.hypervolume import Hypervolume
from demo_runtime.plotting import plot_suite_results
from demo_runtime.specs import ProblemSpec, UnitRegistry
from demo_runtime.suite import expand_runs, load_suite
from demo_runtime.summary import summarize_suite


REPO_ROOT = Path(__file__).resolve().parents[1]


class ProblemSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ProblemSpec.load(
            REPO_ROOT / "configs" / "problems" / "qm9_frontier_alignment.json"
        )

    def test_required_predictors_include_derived_dependencies(self) -> None:
        self.assertEqual(
            set(self.spec.required_predictor_properties),
            {"alpha", "mu", "gap", "homo", "lumo"},
        )

    def test_energy_values_are_canonicalized_to_ev(self) -> None:
        units = UnitRegistry()
        self.assertEqual(units.convert("gap", 6000.0), 6.0)
        self.assertEqual(units.convert("homo", -7000.0), -7.0)

    def test_custom_constraints_annotate_legacy_population(self) -> None:
        molecule = {
            "alpha": 80.0,
            "mu": 4.0,
            "gap": 6500.0,
            "homo": -7200.0,
            "lumo": -800.0,
            "Constraint": 0.0,
        }
        annotated = self.spec.annotate_population([molecule])[0]
        self.assertAlmostEqual(
            annotated["_canonical_properties"]["frontier_midpoint"], -4.0
        )
        self.assertEqual(
            annotated["_constraint_violations"]["minimum_gap"], 0.0
        )
        self.assertEqual(
            annotated["_constraint_violations"]["frontier_alignment"], 0.0
        )
        self.assertAlmostEqual(annotated["__objective__alpha"], -80.0)
        self.assertAlmostEqual(annotated["__canonical__alpha"], 80.0)

    def test_equality_violation_uses_tolerance(self) -> None:
        molecule = {
            "alpha": 80.0,
            "mu": 4.0,
            "gap": 6500.0,
            "homo": -6800.0,
            "lumo": -600.0,
            "Constraint": 0.0,
        }
        annotated = self.spec.annotate_population([molecule])[0]
        # Midpoint=-3.7, target=-4.0, tolerance=.15, scale=.15.
        self.assertAlmostEqual(
            annotated["_constraint_violations"]["frontier_alignment"], 1.0
        )


class HypervolumeTests(unittest.TestCase):
    def test_exact_two_dimensional_union(self) -> None:
        metric = Hypervolume(ref_point=[1.0, 1.0])
        # Areas: .8*.2 + .2*.8 - .2*.2 = .28.
        self.assertAlmostEqual(metric.do([[0.2, 0.8], [0.8, 0.2]]), 0.28)

    def test_points_outside_reference_do_not_contribute(self) -> None:
        metric = Hypervolume(ref_point=[1.0, 1.0])
        self.assertEqual(metric.do([[1.2, 0.1]]), 0.0)


class RecordingTests(unittest.TestCase):
    def test_completion_requires_final_completed_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = JsonlRunRecorder(temporary, "run-1")
            recorder.event("run_start")
            recorder.mark_complete()
            self.assertFalse(recorder.is_complete())
            recorder.event("run_end", status="completed")
            self.assertTrue(recorder.is_complete())

    def test_generation_records_metrics_and_population_as_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = JsonlRunRecorder(
                temporary,
                "run-2",
                {"method": "DEMO", "problem": "example"},
            )
            population = [
                {
                    "alpha": 10.0,
                    "mu": 2.0,
                    "Constraint": 0.0,
                    "structure_Con": [0.0],
                }
            ]
            recorder.record_generation(
                population,
                ["alpha", "mu"],
                {"hv": 0.2},
                generation=0,
            )
            metric_records = list(read_jsonl(Path(temporary) / "metrics.jsonl"))
            population_records = list(
                read_jsonl(Path(temporary) / "population.jsonl")
            )
            self.assertEqual(metric_records[0]["metrics"]["hv"], 0.2)
            self.assertEqual(
                population_records[0]["objective_names"], ["alpha", "mu"]
            )
            self.assertTrue(population_records[0]["molecule_id"])


class SuiteAndSummaryTests(unittest.TestCase):
    def test_all_suite_configs_expand(self) -> None:
        suite_dir = REPO_ROOT / "configs" / "suites"
        for path in suite_dir.glob("*.json"):
            suite = load_suite(path)
            self.assertTrue(expand_runs(suite), path.name)

    def test_qm9_cmop_problem_config_participates_in_run_identity(self) -> None:
        suite = load_suite(
            REPO_ROOT / "configs" / "suites" / "qm9_cmop.json"
        )
        run = expand_runs(suite)[0]
        self.assertEqual(
            run.identity_files,
            ("configs/problems/qm9_frontier_alignment.json",),
        )

    def test_summary_uses_only_complete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            complete = JsonlRunRecorder(
                suite_dir / "problem" / "DEMO" / "seed_0",
                "complete",
                {"method": "DEMO", "problem": "problem"},
            )
            complete.record_generation(
                [{"x": 1.0, "Constraint": 0.0, "structure_Con": []}],
                ["x"],
                {"hv": 0.5},
                generation=1,
            )
            complete.event("run_end", status="completed")
            complete.mark_complete()

            incomplete = JsonlRunRecorder(
                suite_dir / "problem" / "DEMO" / "seed_1",
                "incomplete",
                {"method": "DEMO", "problem": "problem"},
            )
            incomplete.record_generation(
                [{"x": 1.0, "Constraint": 0.0, "structure_Con": []}],
                ["x"],
                {"hv": 0.9},
                generation=1,
            )

            output = summarize_suite(
                suite_dir,
                {"summary_metrics": [{"name": "hv", "direction": "max"}]},
            )
            summaries = list(read_jsonl(output / "summary.jsonl"))
            hv = next(record for record in summaries if record["metric"] == "hv")
            self.assertEqual(hv["n"], 1)
            self.assertEqual(hv["mean"], 0.5)
            self.assertTrue((output / "table_hv.tex").exists())

    @unittest.skipUnless(
        os.environ.get("DEMO_TEST_PLOTTING") == "1",
        "Enable in the target cgm310 environment with DEMO_TEST_PLOTTING=1",
    )
    def test_plotting_is_regenerated_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            suite_dir = Path(temporary) / "suite"
            recorder = JsonlRunRecorder(
                suite_dir / "problem" / "DEMO" / "seed_0",
                "plot-run",
                {"method": "DEMO", "problem": "problem"},
            )
            for generation, hv in enumerate((0.1, 0.2)):
                recorder.record_generation(
                    [
                        {
                            "alpha": 10.0 + generation,
                            "mu": 2.0 + generation,
                            "Constraint": 0.0,
                            "structure_Con": [],
                        }
                    ],
                    ["alpha", "mu"],
                    {"hv": hv},
                    generation=generation,
                )
            recorder.event("run_end", status="completed")
            recorder.mark_complete()
            output = plot_suite_results(suite_dir)
            self.assertTrue((output / "problem" / "hv.png").exists())
            self.assertTrue(
                (output / "problem" / "final_pareto_alpha_mu.png").exists()
            )


if __name__ == "__main__":
    unittest.main()
