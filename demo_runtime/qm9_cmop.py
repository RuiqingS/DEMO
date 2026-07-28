"""Configurable QM9 constrained multi-objective experiment.

Only the bundled EGNN property predictors are used.  No xTB or DFT validation
is invoked by this module.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from typing import Any

from .recording import JsonlRunRecorder, canonical_hash, get_env_recorder
from .specs import ProblemSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_runtime(spec: ProblemSpec, device_override: str | None = None) -> dict[str, Any]:
    # Heavy imports stay inside the execution path so config validation and
    # reporting utilities remain usable on CPU-only machines.
    import torch

    import evo
    from configs.datasets_config import get_dataset_info
    from qm9.models import get_latent_diffusion, get_model

    runtime = spec.runtime
    resume = _resolve_path(str(runtime.get("resume", "tfgmodels/EDMsecond")))
    args_path = resume / str(runtime.get("args_file", "args.pickle"))
    with args_path.open("rb") as handle:
        args = pickle.load(handle)

    args.resume = str(resume)
    args.break_train_epoch = False
    args.context_node_nf = 0
    args.xtb = False
    if not hasattr(args, "remove_h"):
        args.remove_h = False
    if not hasattr(args, "include_charges"):
        args.include_charges = True

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.cuda = device.type == "cuda"
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    use_edm = bool(runtime.get("use_edm", True))
    if use_edm:
        model, nodes_dist, prop_dist = get_model(args, device, dataset_info, None)
        default_checkpoint = "generative_model_ema.npy"
    else:
        model, nodes_dist, prop_dist = get_latent_diffusion(
            args, device, dataset_info, None
        )
        default_checkpoint = "generative_model.npy"

    checkpoint = resume / str(runtime.get("checkpoint", default_checkpoint))
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    predictor_names = list(spec.required_predictor_properties)
    predictors = evo.Get_Pred(predictor_names, device)
    return {
        "torch": torch,
        "evo": evo,
        "args": args,
        "device": device,
        "dataset_info": dataset_info,
        "model": model,
        "nodes_dist": nodes_dist,
        "prop_dist": prop_dist,
        "predictors": predictors,
        "checkpoint": checkpoint,
        "predictor_names": predictor_names,
    }


def _evaluate(population: list[dict[str, Any]], spec: ProblemSpec, ctx: dict[str, Any]):
    evo = ctx["evo"]
    population = evo.EvalPop_Con(population, ctx["dataset_info"])
    population = evo.EvalPop_Obj(
        population,
        ctx["device"],
        ctx["predictors"],
        spec.max_nodes,
    )
    return spec.annotate_population(population)


def _select(
    population: list[dict[str, Any]],
    spec: ProblemSpec,
    ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    from MOEA import DEMO

    selection = str(spec.runtime.get("selection", "saes")).lower()
    if selection == "spea2":
        return ctx["evo"].EnvironmentalSelectionCon(
            population,
            spec.population_size,
            list(spec.legacy_objective_keys),
        )
    if selection != "saes":
        raise ValueError(f"Unsupported CMOP selection policy: {selection}")
    return DEMO.EnvironmentalSelectionMain_Diverse(
        population,
        spec.population_size,
        list(spec.legacy_objective_keys),
        ctx["dataset_info"],
        min_dist_threshold=float(spec.runtime.get("min_dist_threshold", 0.05)),
    )


def _is_feasible(molecule: dict[str, Any]) -> bool:
    chemical = float(molecule.get("Constraint", 0.0))
    violations = molecule.get("_constraint_violations", {})
    return chemical <= 0.0 and all(float(value) <= 0.0 for value in violations.values())


def _hypervolume(population: list[dict[str, Any]], spec: ProblemSpec) -> float:
    import numpy as np
    try:
        from pymoo.indicators.hv import Hypervolume
    except ImportError:
        from .hypervolume import Hypervolume

    points = []
    for molecule in population:
        if not _is_feasible(molecule):
            continue
        row = []
        for objective in spec.objectives:
            value = float(molecule[f"__canonical__{objective.name}"])
            lower, upper = objective.bounds
            if objective.direction == "min":
                normalized = (value - lower) / (upper - lower)
            else:
                normalized = (upper - value) / (upper - lower)
            row.append(float(np.clip(normalized, 0.0, spec.hv_reference)))
        points.append(row)
    if not points:
        return 0.0
    metric = Hypervolume(
        ref_point=np.asarray([spec.hv_reference] * len(spec.objectives))
    )
    return float(metric.do(np.asarray(points, dtype=float)))


def _generation_metrics(
    population: list[dict[str, Any]],
    spec: ProblemSpec,
    scheduler_score: float,
    noise_t: int,
    evaluator_molecules: int,
    started_at: float,
) -> dict[str, float | int | str]:
    n = len(population)
    feasible = sum(_is_feasible(item) for item in population)
    metrics: dict[str, float | int | str] = {
        "hv": _hypervolume(population, spec),
        "feasible_rate": feasible / n if n else 0.0,
        "noise_t": int(noise_t),
        "scheduler_score": float(scheduler_score),
        "evaluated_molecules": evaluator_molecules,
        "property_predictor_calls": evaluator_molecules
        * len(spec.required_predictor_properties),
        "wall_time_seconds": time.monotonic() - started_at,
        "metric_source": "egnn_property_predictor",
    }
    for constraint in spec.constraints:
        values = [
            float(item["_constraint_violations"][constraint.name])
            for item in population
        ]
        metrics[f"constraint.{constraint.name}.feasible_rate"] = (
            sum(value <= 0.0 for value in values) / len(values) if values else 0.0
        )
        metrics[f"constraint.{constraint.name}.mean_violation"] = (
            sum(values) / len(values) if values else 0.0
        )
    validity = [float(item.get("validity_rdkit", 0.0)) for item in population]
    atom_stability = [float(item.get("atm_stable", 0.0)) for item in population]
    metrics["validity"] = sum(validity) / len(validity) if validity else 0.0
    metrics["atom_stability"] = (
        sum(atom_stability) / len(atom_stability) if atom_stability else 0.0
    )
    return metrics


def _resolve_recorder(spec: ProblemSpec, seed: int, output: str | None) -> JsonlRunRecorder:
    recorder = get_env_recorder()
    if recorder is not None:
        return recorder
    identity = {
        "problem": spec.problem_id,
        "seed": seed,
        "spec": spec,
    }
    run_id = canonical_hash(identity, length=24)
    run_dir = (
        _resolve_path(output)
        if output
        else REPO_ROOT
        / "results"
        / "qm9_cmop"
        / spec.problem_id
        / f"seed_{seed}_{run_id[:8]}"
    )
    recorder = JsonlRunRecorder(
        run_dir,
        run_id,
        {
            "suite": "qm9_cmop",
            "task_type": "qm9_cmop",
            "problem": spec.problem_id,
            "method": str(spec.runtime.get("method", "DEMO")),
            "seed": seed,
        },
    )
    recorder.write_manifest(
        {
            "schema_version": 1,
            "run_id": run_id,
            "problem_spec": spec,
            "metric_source": "egnn_property_predictor",
        }
    )
    recorder.event("run_start")
    return recorder


def run_problem(
    spec: ProblemSpec,
    seed: int,
    output: str | None = None,
    device: str | None = None,
) -> JsonlRunRecorder:
    ctx = _load_runtime(spec, device)
    evo = ctx["evo"]
    recorder = _resolve_recorder(spec, seed, output)
    evo.seed_everything(seed)
    started_at = time.monotonic()

    initial_size = max(spec.population_size * 2, spec.population_size + 1)
    population = evo.InitPop(
        initial_size,
        ctx["nodes_dist"],
        ctx["args"],
        ctx["device"],
        ctx["model"],
        ctx["dataset_info"],
        ctx["prop_dist"],
        False,
        spec.min_nodes,
        ctx["predictors"],
        spec.max_nodes,
        False,
    )
    evaluated_molecules = len(population)
    population = _evaluate(population, spec, ctx)
    population = _select(population, spec, ctx)

    noise_config = dict(spec.noise)
    noise_mode = str(noise_config.get("mode", "adaptive")).lower()
    current_noise = int(noise_config.get("initial", 1000))
    scheduler = None
    scheduler_score = 0.0
    if noise_mode == "adaptive":
        scheduler = evo.AdaptiveNoiseScheduler(
            dataset_info=ctx["dataset_info"],
            initial_noise=current_noise,
            min_noise=int(noise_config.get("min", 0)),
            max_noise=int(noise_config.get("max", 1000)),
            step_size=int(noise_config.get("step_size", 10)),
            drop_factor=float(noise_config.get("drop_factor", 0.5)),
            use_hv_trigger=bool(noise_config.get("use_hv_trigger", False)),
        )
        tracker_stub = SimpleNamespace(metrics={})
        current_noise = scheduler.update(
            Off=population,
            Pop=population,
            tracker=tracker_stub,
            viz=None,
            obj_names=list(spec.legacy_objective_keys),
            calHV=False,
        )
        scheduler_score = float(scheduler.score)
    elif noise_mode != "fixed":
        raise ValueError(f"Unsupported noise mode: {noise_mode}")

    canonical_objectives = [
        f"__canonical__{objective.name}" for objective in spec.objectives
    ]
    recorder.record_generation(
        population,
        canonical_objectives,
        _generation_metrics(
            population,
            spec,
            scheduler_score,
            current_noise,
            evaluated_molecules,
            started_at,
        ),
        generation=0,
        problem_id=spec.problem_id,
    )

    for generation in range(1, spec.generations + 1):
        parent_count = max(2, spec.population_size // 2)
        if parent_count % 2:
            parent_count -= 1
        parents = evo.k_tournament_selection(population, 2, parent_count)
        parent_ids = [recorder.molecule_id(item) for item in parents]
        for parent in parents:
            parent.pop("_demo_molecule_id", None)
            parent["AddT"] = current_noise
        parents = evo.add_noise(ctx["model"], parents, ctx["device"])

        crossed = evo.valOff(
            parents,
            spec.min_nodes,
            spec.max_nodes,
            ctx["device"],
            spec.max_nodes,
            ctx["nodes_dist"],
        )
        denoised = evo.denoise_same_level(
            crossed + parents,
            ctx["model"],
            spec.max_nodes,
            ctx["device"],
            ctx["dataset_info"],
            ctx["predictors"],
            False,
            iter_num=f"{seed}-{generation}",
        )
        evaluated_molecules += len(denoised)
        offspring = _evaluate(denoised, spec, ctx)

        pair_count = len(parents) // 2
        for pair_index in range(pair_count):
            sources = [
                parent_ids[pair_index],
                parent_ids[pair_index + pair_count],
            ]
            for offset in (0, 1):
                child_index = pair_index * 2 + offset
                if child_index < len(crossed):
                    recorder.record_lineage(
                        offspring[child_index],
                        sources,
                        "crossover",
                        generation,
                        current_noise,
                    )
        for index, source in enumerate(parent_ids):
            child_index = len(crossed) + index
            if child_index < len(offspring):
                recorder.record_lineage(
                    offspring[child_index],
                    [source],
                    "mutation",
                    generation,
                    current_noise,
                )

        population = _select(deepcopy(population + offspring), spec, ctx)
        if scheduler is not None:
            tracker_stub = SimpleNamespace(metrics={})
            current_noise = scheduler.update(
                Off=offspring,
                Pop=population,
                tracker=tracker_stub,
                viz=None,
                obj_names=list(spec.legacy_objective_keys),
                calHV=False,
            )
            scheduler_score = float(scheduler.score)

        recorder.record_generation(
            population,
            canonical_objectives,
            _generation_metrics(
                population,
                spec,
                scheduler_score,
                current_noise,
                evaluated_molecules,
                started_at,
            ),
            generation=generation,
            problem_id=spec.problem_id,
        )
        recorder.event(
            "generation_end",
            generation=generation,
            evaluated_molecules=evaluated_molecules,
        )

    recorder.event(
        "experiment_complete",
        status="completed",
        generations=spec.generations,
        metric_source="egnn_property_predictor",
    )
    if get_env_recorder() is None:
        recorder.event(
            "run_end",
            status="completed",
            generations=spec.generations,
            wall_time_seconds=time.monotonic() - started_at,
        )
        recorder.mark_complete()
    return recorder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-config", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("DEMO_SEED", "42")),
    )
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--validate-config", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = ProblemSpec.load(_resolve_path(args.problem_config))
    if args.validate_config:
        print(
            json.dumps(
                {
                    "problem_id": spec.problem_id,
                    "predictors": spec.required_predictor_properties,
                    "objectives": [item.name for item in spec.objectives],
                    "constraints": [item.name for item in spec.constraints],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    run_problem(spec, args.seed, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
