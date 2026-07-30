"""Configurable QM9 constrained multi-objective experiment.

Only the bundled EGNN property predictors are used.  No xTB or DFT validation
is invoked by this module.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from typing import Any

from .recording import JsonlRunRecorder, canonical_hash, get_env_recorder
from .specs import ProblemSpec


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentOptions:
    """Runtime-only reviewer experiment controls.

    Defaults preserve the original configurable QM9-CMOP behavior.
    """

    operator_mode: str = "full"
    scheduler_mode: str | None = None
    fixed_noise: int | None = None
    distance_mode: str | None = None
    selection_mode: str | None = None
    evaluation_budget: int | None = None
    training_reference: str | None = None
    population_size: int | None = None
    generations: int | None = None
    min_distance: float | None = None
    noise_step: int | None = None
    record_diagnostics: bool = True


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
    options: ExperimentOptions,
) -> list[dict[str, Any]]:
    from MOEA import DEMO

    selection = str(
        options.selection_mode
        or spec.runtime.get("selection", "saes")
    ).lower()
    if options.operator_mode == "spea2":
        selection = "spea2"
    if selection == "spea2":
        return ctx["evo"].EnvironmentalSelectionCon(
            population,
            spec.population_size,
            list(spec.legacy_objective_keys),
        )
    if selection != "saes":
        raise ValueError(f"Unsupported CMOP selection policy: {selection}")
    if options.distance_mode is not None:
        from .diagnostics import select_population

        return select_population(
            population,
            spec.population_size,
            list(spec.legacy_objective_keys),
            ctx["dataset_info"],
            options.distance_mode,
            float(
                options.min_distance
                if options.min_distance is not None
                else spec.runtime.get("min_dist_threshold", 0.05)
            ),
        )
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
    counters: dict[str, float | int],
    started_at: float,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, float | int | str]:
    n = len(population)
    feasible = sum(_is_feasible(item) for item in population)
    violations = [
        max(0.0, float(item.get("Constraint", 0.0)))
        + sum(
            max(0.0, float(value))
            for value in item.get("_constraint_violations", {}).values()
        )
        for item in population
    ]
    metrics: dict[str, float | int | str] = {
        "hv": _hypervolume(population, spec),
        "feasible_rate": feasible / n if n else 0.0,
        "mean_violation": (
            sum(violations) / len(violations) if violations else 0.0
        ),
        "noise_t": int(noise_t),
        "scheduler_score": float(scheduler_score),
        "evaluated_molecules": int(counters.get("evaluated_molecules", 0)),
        "property_predictor_calls": int(
            counters.get("property_predictor_calls", 0)
        ),
        "property_predictor_batches": int(
            counters.get("predictor_batches", 0)
        ),
        "diffusion_calls": int(counters.get("diffusion_calls", 0)),
        "generated_molecules": int(counters.get("generated_molecules", 0)),
        "evaluation_failures": int(counters.get("evaluation_failures", 0)),
        "evaluation_failure_rate": (
            int(counters.get("evaluation_failures", 0))
            / max(1, int(counters.get("evaluated_molecules", 0)))
        ),
        "gpu_time_seconds": float(counters.get("gpu_time_seconds", 0.0)),
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
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def _timed_call(
    ctx: dict[str, Any],
    counters: dict[str, float | int],
    counter_name: str,
    function,
):
    torch = ctx["torch"]
    use_cuda = ctx["device"].type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(ctx["device"])
    started = time.monotonic()
    result = function()
    if use_cuda:
        torch.cuda.synchronize(ctx["device"])
        counters["gpu_time_seconds"] = float(
            counters.get("gpu_time_seconds", 0.0)
        ) + (time.monotonic() - started)
    counters[counter_name] = int(counters.get(counter_name, 0)) + 1
    return result


def _integrity_score(population: list[dict[str, Any]]) -> float:
    if not population:
        return 0.0
    validity = sum(
        float(item.get("validity_rdkit", 0.0))
        for item in population
    ) / len(population)
    atom_stability = sum(
        float(item.get("atm_stable", 0.0))
        for item in population
    ) / len(population)
    return validity * atom_stability


def _diagnostic_metrics(
    population: list[dict[str, Any]],
    spec: ProblemSpec,
    ctx: dict[str, Any],
    options: ExperimentOptions,
    inheritance_records: list[dict[str, Any]] | None = None,
    include_training_reference: bool = True,
) -> dict[str, Any]:
    if not options.record_diagnostics:
        return {}
    from .diagnostics import (
        aggregate_records,
        population_diversity_metrics,
        training_reference_metrics,
    )

    metrics = population_diversity_metrics(
        population,
        ctx["dataset_info"],
        max_mcs_pairs=(
            0
            if (
                options.evaluation_budget is not None
                and options.evaluation_budget > 10000
            )
            else int(spec.runtime.get("diagnostic_mcs_pairs", 16))
        ),
    )
    if inheritance_records:
        metrics.update(aggregate_records(inheritance_records))
    if options.training_reference and include_training_reference:
        metrics.update(
            training_reference_metrics(
                population,
                ctx["dataset_info"],
                _resolve_path(options.training_reference),
            )
        )
    return metrics


def _resolve_recorder(
    spec: ProblemSpec,
    seed: int,
    output: str | None,
    options: ExperimentOptions,
) -> JsonlRunRecorder:
    recorder = get_env_recorder()
    if recorder is not None:
        return recorder
    identity = {
        "problem": spec.problem_id,
        "seed": seed,
        "spec": spec,
        "experiment_options": options,
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
            "method": str(
                spec.runtime.get("method", "DEMO")
                if options.operator_mode == "full"
                else options.operator_mode
            ),
            "seed": seed,
        },
    )
    recorder.write_manifest(
        {
            "schema_version": 1,
            "run_id": run_id,
            "problem_spec": spec,
            "experiment_options": options,
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
    options: ExperimentOptions | None = None,
) -> JsonlRunRecorder:
    options = options or ExperimentOptions()
    if options.population_size is not None or options.generations is not None:
        spec = replace(
            spec,
            population_size=(
                int(options.population_size)
                if options.population_size is not None
                else spec.population_size
            ),
            generations=(
                int(options.generations)
                if options.generations is not None
                else spec.generations
            ),
        )
    ctx = _load_runtime(spec, device)
    evo = ctx["evo"]
    recorder = _resolve_recorder(spec, seed, output, options)
    evo.seed_everything(seed)
    started_at = time.monotonic()
    counters: dict[str, float | int] = {
        "evaluated_molecules": 0,
        "property_predictor_calls": 0,
        "diffusion_calls": 0,
        "generated_molecules": 0,
        "evaluation_failures": 0,
        "gpu_time_seconds": 0.0,
    }
    budget = (
        int(options.evaluation_budget)
        if options.evaluation_budget is not None
        else None
    )
    if budget is not None and budget < spec.population_size:
        raise ValueError(
            "evaluation_budget must be at least the selected population size"
        )

    initial_size = max(spec.population_size * 2, spec.population_size + 1)
    if budget is not None:
        initial_size = min(initial_size, budget)
    population = _timed_call(
        ctx,
        counters,
        "diffusion_calls",
        lambda: evo.InitPop(
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
        ),
    )
    counters["generated_molecules"] = len(population)
    population = _timed_call(
        ctx,
        counters,
        "predictor_batches",
        lambda: _evaluate(population, spec, ctx),
    )
    counters["evaluated_molecules"] = len(population)
    counters["property_predictor_calls"] = (
        len(population) * len(spec.required_predictor_properties)
    )
    counters["evaluation_failures"] = sum(
        float(item.get("validity_rdkit", 0.0)) <= 0.0
        for item in population
    )
    population = _select(population, spec, ctx, options)

    noise_config = dict(spec.noise)
    noise_mode = str(
        options.scheduler_mode
        or noise_config.get("mode", "adaptive_validity")
    ).lower()
    if noise_mode == "adaptive":
        noise_mode = "adaptive_validity"
    current_noise = int(
        options.fixed_noise
        if options.fixed_noise is not None
        else noise_config.get("initial", 1000)
    )
    scheduler = None
    scheduler_score = 0.0
    scheduler_kwargs = {
        "dataset_info": ctx["dataset_info"],
        "initial_noise": current_noise,
        "min_noise": int(noise_config.get("min", 0)),
        "max_noise": int(noise_config.get("max", 1000)),
        "step_size": int(
            options.noise_step
            if options.noise_step is not None
            else noise_config.get("step_size", 10)
        ),
        "drop_factor": float(noise_config.get("drop_factor", 0.5)),
        "use_hv_trigger": bool(noise_config.get("use_hv_trigger", False)),
    }
    if noise_mode == "adaptive_validity":
        scheduler = evo.AdaptiveNoiseScheduler(**scheduler_kwargs)
    elif noise_mode == "adaptive_inheritance":
        from .diagnostics import make_inheritance_scheduler

        scheduler = make_inheritance_scheduler(evo, **scheduler_kwargs)
    elif noise_mode != "fixed":
        raise ValueError(f"Unsupported noise mode: {noise_mode}")
    if scheduler is not None:
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
    else:
        scheduler_score = _integrity_score(population)

    canonical_objectives = [
        f"__canonical__{objective.name}" for objective in spec.objectives
    ]
    expected_generations = (
        max(
            1,
            int(
                math.ceil(
                    max(0, budget - initial_size)
                    / max(1, spec.population_size)
                )
            ),
        )
        if budget is not None
        else max(1, spec.generations)
    )
    reference_interval = max(1, expected_generations // 10)

    def record_generation(
        generation: int,
        inheritance_records: list[dict[str, Any]] | None = None,
        generation_extras: dict[str, Any] | None = None,
    ) -> None:
        extras = _diagnostic_metrics(
            population,
            spec,
            ctx,
            options,
            inheritance_records,
            include_training_reference=bool(
                options.training_reference
                and (
                    generation == 0
                    or generation % reference_interval == 0
                    or (
                        budget is not None
                        and int(counters["evaluated_molecules"]) >= budget
                    )
                    or (
                        budget is None
                        and generation >= spec.generations
                    )
                )
            ),
        )
        extras.update(
            {
                "operator_mode": options.operator_mode,
                "scheduler_mode": noise_mode,
                "distance_mode": options.distance_mode or "mixed",
                "evaluation_budget": budget,
                "population_size": spec.population_size,
                "noise_step": scheduler_kwargs["step_size"],
                "min_distance": float(
                    options.min_distance
                    if options.min_distance is not None
                    else spec.runtime.get("min_dist_threshold", 0.05)
                ),
                "configured_fixed_noise": options.fixed_noise,
                "budget_fraction": (
                    min(
                        1.0,
                        int(counters["evaluated_molecules"]) / budget,
                    )
                    if budget
                    else 0.0
                ),
            }
        )
        if generation_extras:
            extras.update(generation_extras)
        recorder.record_generation(
            population,
            canonical_objectives,
            _generation_metrics(
                population,
                spec,
                scheduler_score,
                current_noise,
                counters,
                started_at,
                extras,
            ),
            generation=generation,
            problem_id=spec.problem_id,
        )

    record_generation(0)

    if budget is None:
        generation_limit = spec.generations
    else:
        remaining = max(
            0,
            budget - int(counters["evaluated_molecules"]),
        )
        generation_limit = max(
            spec.generations,
            int(math.ceil(remaining / max(1, spec.population_size))) + 2,
        )

    completed_generations = 0
    operator_mode = options.operator_mode.lower()
    if operator_mode not in {
        "full",
        "mutation_only",
        "crossover_only",
        "spea2",
        "topn",
    }:
        raise ValueError(f"Unsupported operator mode: {options.operator_mode}")

    for generation in range(1, generation_limit + 1):
        if (
            budget is not None
            and int(counters["evaluated_molecules"]) >= budget
        ):
            break
        remaining_budget = (
            budget - int(counters["evaluated_molecules"])
            if budget is not None
            else None
        )
        inheritance_records: list[dict[str, Any]] = []

        if operator_mode == "topn":
            sample_size = spec.population_size
            if remaining_budget is not None:
                sample_size = min(sample_size, remaining_budget)
            offspring = _timed_call(
                ctx,
                counters,
                "diffusion_calls",
                lambda: evo.InitPop(
                    sample_size,
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
                ),
            )
            counters["generated_molecules"] = int(
                counters["generated_molecules"]
            ) + len(offspring)
            offspring = _timed_call(
                ctx,
                counters,
                "predictor_batches",
                lambda: _evaluate(offspring, spec, ctx),
            )
            for child in offspring:
                recorder.record_lineage(
                    child,
                    [],
                    "prior_sampling",
                    generation,
                    current_noise,
                )
        else:
            single_operator = operator_mode in {
                "mutation_only",
                "crossover_only",
            }
            parent_count = (
                spec.population_size
                if single_operator
                else max(2, spec.population_size // 2)
            )
            parent_count = min(parent_count, len(population))
            if parent_count % 2:
                parent_count -= 1
            if parent_count < 2:
                break
            parents = evo.k_tournament_selection(
                population,
                2,
                parent_count,
            )
            parent_snapshots = deepcopy(parents)
            parent_ids = [
                recorder.molecule_id(item)
                for item in parent_snapshots
            ]
            for parent in parents:
                parent.pop("_demo_molecule_id", None)
                parent["AddT"] = current_noise
            noised_parents = evo.add_noise(
                ctx["model"],
                parents,
                ctx["device"],
            )

            crossed = []
            if operator_mode in {"full", "crossover_only", "spea2"}:
                crossed = evo.valOff(
                    noised_parents,
                    spec.min_nodes,
                    spec.max_nodes,
                    ctx["device"],
                    spec.max_nodes,
                    ctx["nodes_dist"],
                )
            mutations = (
                noised_parents
                if operator_mode in {"full", "mutation_only", "spea2"}
                else []
            )
            candidates = crossed + mutations
            descriptors: list[
                tuple[str, list[str], list[dict[str, Any]]]
            ] = []
            pair_count = len(parent_snapshots) // 2
            for child_index in range(len(crossed)):
                pair_index = min(
                    max(0, child_index // 2),
                    max(0, pair_count - 1),
                )
                parent_indices = [
                    pair_index,
                    pair_index + pair_count,
                ]
                descriptors.append(
                    (
                        "crossover",
                        [parent_ids[index] for index in parent_indices],
                        [
                            parent_snapshots[index]
                            for index in parent_indices
                        ],
                    )
                )
            for index in range(len(mutations)):
                parent_index = min(index, len(parent_snapshots) - 1)
                descriptors.append(
                    (
                        "mutation",
                        [parent_ids[parent_index]],
                        [parent_snapshots[parent_index]],
                    )
                )

            offspring = _timed_call(
                ctx,
                counters,
                "diffusion_calls",
                lambda: evo.denoise_same_level(
                    candidates,
                    ctx["model"],
                    spec.max_nodes,
                    ctx["device"],
                    ctx["dataset_info"],
                    ctx["predictors"],
                    False,
                    iter_num=f"{seed}-{generation}",
                ),
            )
            if remaining_budget is not None:
                offspring = offspring[:remaining_budget]
                descriptors = descriptors[: len(offspring)]
            counters["generated_molecules"] = int(
                counters["generated_molecules"]
            ) + len(offspring)
            offspring = _timed_call(
                ctx,
                counters,
                "predictor_batches",
                lambda: _evaluate(offspring, spec, ctx),
            )

            from .diagnostics import inheritance_metrics

            for child, descriptor in zip(offspring, descriptors):
                operator, source_ids, source_molecules = descriptor
                values = inheritance_metrics(
                    child,
                    source_molecules,
                    ctx["dataset_info"],
                    spec,
                )
                child["_inheritance_score"] = float(
                    values.get("inheritance_score") or 0.0
                )
                inheritance_records.append(values)
                recorder.record_lineage(
                    child,
                    source_ids,
                    operator,
                    generation,
                    current_noise,
                    diagnostics=values,
                )
                recorder.append(
                    "inheritance",
                    {
                        "record_type": "inheritance",
                        "generation": generation,
                        "child_id": recorder.molecule_id(child),
                        "parent_ids": source_ids,
                        "operator": operator,
                        "noise_t": current_noise,
                        "metrics": values,
                    },
                )

        evaluated_now = len(offspring)
        counters["evaluated_molecules"] = int(
            counters["evaluated_molecules"]
        ) + evaluated_now
        counters["property_predictor_calls"] = int(
            counters["property_predictor_calls"]
        ) + evaluated_now * len(spec.required_predictor_properties)
        counters["evaluation_failures"] = int(
            counters["evaluation_failures"]
        ) + sum(
            float(item.get("validity_rdkit", 0.0)) <= 0.0
            for item in offspring
        )

        generation_extras = {}
        if options.record_diagnostics:
            from .diagnostics import novelty_against_reference

            generation_extras.update(
                novelty_against_reference(
                    offspring,
                    population,
                    ctx["dataset_info"],
                )
            )
        population = _select(
            deepcopy(population + offspring),
            spec,
            ctx,
            options,
        )
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
        else:
            scheduler_score = _integrity_score(offspring)

        record_generation(
            generation,
            inheritance_records,
            generation_extras,
        )
        recorder.event(
            "generation_end",
            generation=generation,
            evaluated_molecules=int(counters["evaluated_molecules"]),
            property_predictor_calls=int(
                counters["property_predictor_calls"]
            ),
            gpu_time_seconds=float(counters["gpu_time_seconds"]),
        )
        completed_generations = generation

    recorder.event(
        "experiment_complete",
        status="completed",
        generations=completed_generations,
        evaluated_molecules=int(counters["evaluated_molecules"]),
        property_predictor_calls=int(counters["property_predictor_calls"]),
        gpu_time_seconds=float(counters["gpu_time_seconds"]),
        metric_source="egnn_property_predictor",
    )
    if get_env_recorder() is None:
        recorder.event(
            "run_end",
            status="completed",
            generations=completed_generations,
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
    parser.add_argument(
        "--operator-mode",
        choices=("full", "mutation_only", "crossover_only", "spea2", "topn"),
        default="full",
    )
    parser.add_argument(
        "--scheduler-mode",
        choices=("fixed", "adaptive_validity", "adaptive_inheritance"),
    )
    parser.add_argument("--fixed-noise", type=int)
    parser.add_argument(
        "--distance-mode",
        choices=("objective", "morgan", "usrcat", "mixed", "raw_coords"),
    )
    parser.add_argument("--selection-mode", choices=("saes", "spea2"))
    parser.add_argument("--evaluation-budget", type=int)
    parser.add_argument("--training-reference")
    parser.add_argument("--population-size", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--min-distance", type=float)
    parser.add_argument("--noise-step", type=int)
    parser.add_argument("--no-diagnostics", action="store_true")
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
    options = ExperimentOptions(
        operator_mode=args.operator_mode,
        scheduler_mode=args.scheduler_mode,
        fixed_noise=args.fixed_noise,
        distance_mode=args.distance_mode,
        selection_mode=args.selection_mode,
        evaluation_budget=args.evaluation_budget,
        training_reference=args.training_reference,
        population_size=args.population_size,
        generations=args.generations,
        min_distance=args.min_distance,
        noise_step=args.noise_step,
        record_diagnostics=not args.no_diagnostics,
    )
    run_problem(spec, args.seed, args.output, args.device, options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
