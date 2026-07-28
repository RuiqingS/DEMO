"""Regenerate all suite figures from JSONL without running experiments."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .recording import read_jsonl


def _completed_run(run_dir: Path) -> bool:
    marker = run_dir / ".complete"
    if not marker.exists():
        return False
    run_id = marker.read_text(encoding="utf-8").strip()
    last_event = None
    for event in read_jsonl(run_dir / "events.jsonl"):
        last_event = event
    return bool(
        run_id
        and last_event
        and last_event.get("run_id") == run_id
        and last_event.get("event") == "run_end"
        and last_event.get("status") == "completed"
    )


def _flatten_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for name, value in record.get("metrics", {}).items():
        if isinstance(value, (int, float)) and value is not None:
            flattened[str(name)] = float(value)
    for name, stats in record.get("objectives", {}).items():
        if isinstance(stats, Mapping):
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)) and value is not None:
                    flattened[f"objective.{name}.{stat_name}"] = float(value)
    for name, stats in record.get("constraints", {}).items():
        if isinstance(stats, Mapping):
            for stat_name, value in stats.items():
                if isinstance(value, (int, float)) and value is not None:
                    flattened[f"constraint.{name}.{stat_name}"] = float(value)
    return flattened


def _metric_records(
    suite_dir: Path,
) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in suite_dir.rglob("metrics.jsonl"):
        if not _completed_run(path.parent):
            continue
        for record in read_jsonl(path):
            if record.get("record_type") == "generation":
                yield path.parent, record


def plot_suite_results(suite_dir: str | Path) -> Path:
    # Keep matplotlib optional for training, config validation, and summary
    # generation.  Plot dependencies are loaded only on an explicit plot call.
    from .plots import plot_metric_curves

    suite_path = Path(suite_dir)
    output_dir = suite_path / "plots"
    grouped: dict[
        tuple[str, str, str], dict[tuple[str, int, str], dict[int, float]]
    ] = defaultdict(lambda: defaultdict(dict))

    for _, record in _metric_records(suite_path):
        problem = str(record.get("problem", "unknown"))
        method = str(record.get("method", "unknown"))
        run_key = (
            str(record.get("run_id", "")),
            int(record.get("internal_run_index", 0)),
            method,
        )
        generation = int(record.get("generation", 0))
        for metric, value in _flatten_metrics(record).items():
            grouped[(problem, metric, method)][run_key][generation] = value

    problem_metrics = sorted({(problem, metric) for problem, metric, _ in grouped})
    for problem, metric in problem_metrics:
        by_method: dict[str, list[Mapping[int, float]]] = defaultdict(list)
        for (candidate_problem, candidate_metric, method), runs in grouped.items():
            if candidate_problem == problem and candidate_metric == metric:
                by_method[method].extend(runs.values())
        safe_problem = problem.replace("/", "_").replace("\\", "_").replace(":", "_")
        safe_metric = metric.replace("/", "_").replace(".", "_")
        plot_metric_curves(
            by_method,
            metric_name=metric,
            output_base=output_dir / safe_problem / safe_metric,
            title=f"{problem}: {metric}",
        )

    _plot_pareto_views(suite_path, output_dir)
    return output_dir


def _plot_pareto_views(suite_dir: Path, output_dir: Path) -> None:
    from .plots import plot_final_pareto

    latest: dict[tuple[Path, int, str], int] = {}
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in suite_dir.rglob("population.jsonl"):
        if not _completed_run(path.parent):
            continue
        for record in read_jsonl(path):
            if record.get("record_type") != "individual":
                continue
            records.append((path.parent, record))
            key = (
                path.parent,
                int(record.get("internal_run_index", 0)),
                str(record.get("problem", "unknown")),
            )
            latest[key] = max(latest.get(key, -1), int(record.get("generation", 0)))

    grouped: dict[
        tuple[str, tuple[str, ...]], dict[str, list[dict[str, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    for run_dir, record in records:
        problem = str(record.get("problem", "unknown"))
        key = (
            run_dir,
            int(record.get("internal_run_index", 0)),
            problem,
        )
        if int(record.get("generation", 0)) != latest.get(key):
            continue
        values = record.get("values", {})
        objective_names = tuple(record.get("objective_names", ()))
        if len(objective_names) not in {2, 3}:
            continue
        method = str(record.get("method", "unknown"))
        grouped[(problem, objective_names)][method].append(
            {name: float(values[name]) for name in objective_names}
        )

    for (problem, objectives), by_method in grouped.items():
        safe_problem = problem.replace("/", "_").replace("\\", "_").replace(":", "_")
        objective_suffix = "_".join(
            name.replace("__canonical__", "") for name in objectives
        )
        plot_final_pareto(
            by_method,
            objectives=objectives,
            output_base=output_dir / safe_problem / f"final_pareto_{objective_suffix}",
            title=f"{problem}: empirical Pareto approximation",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_dir")
    args = parser.parse_args(argv)
    output = plot_suite_results(args.suite_dir)
    print(f"Plots written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
