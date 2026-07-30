"""Reviewer-specific parameter plots built only from completed JSONL runs."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _mean_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    if not len(array):
        return 0.0, 0.0
    mean = float(array.mean())
    if len(array) == 1:
        return mean, 0.0
    return mean, float(1.96 * array.std(ddof=1) / math.sqrt(len(array)))


def _save(figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def _plot_xy(
    rows: list[Mapping[str, Any]],
    task_axis: Mapping[str, float],
    metrics: Iterable[str],
    output_dir: Path,
    axis_name: str,
) -> None:
    for metric in metrics:
        grouped: dict[float, list[float]] = defaultdict(list)
        for row in rows:
            task_id = str(row.get("task_id", ""))
            value = row.get("metrics", {}).get(metric)
            if task_id in task_axis and isinstance(value, (int, float)):
                grouped[float(task_axis[task_id])].append(float(value))
        if len(grouped) < 2:
            continue
        x_values = sorted(grouped)
        aggregates = [_mean_ci(grouped[value]) for value in x_values]
        means = [value[0] for value in aggregates]
        errors = [value[1] for value in aggregates]
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        axis.errorbar(
            x_values,
            means,
            yerr=errors,
            marker="o",
            capsize=3,
            linewidth=1.8,
        )
        axis.set_xlabel(axis_name)
        axis.set_ylabel(metric)
        axis.set_title(f"{metric} vs {axis_name}")
        axis.grid(alpha=0.25)
        safe_metric = metric.replace(".", "_").replace("/", "_")
        safe_axis = axis_name.lower().replace(" ", "_")
        _save(figure, output_dir / f"{safe_axis}_{safe_metric}")


def _plot_noise(
    rows: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    metrics = (
        "inheritance.inheritance_score.mean",
        "inheritance.morgan_similarity.mean",
        "inheritance.mcs_atom_fraction.mean",
        "inheritance.murcko_retention.mean",
        "inheritance.fragment_retention.mean",
        "inheritance.mcs_3d_rmsd.mean",
        "inheritance.objective_improvement.mean",
        "validity",
        "atom_stability",
    )
    pattern = re.compile(r"^noise_(\d+)_(crossover|mutation)$")
    for metric in metrics:
        fixed: dict[str, dict[float, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        adaptive: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            task_id = str(row.get("task_id", ""))
            value = row.get("metrics", {}).get(metric)
            if not isinstance(value, (int, float)):
                continue
            match = pattern.match(task_id)
            if match:
                fixed[match.group(2)][float(match.group(1))].append(float(value))
            elif task_id == "noise_adaptive_crossover":
                adaptive["crossover"].append(float(value))
            elif task_id == "noise_adaptive_mutation":
                adaptive["mutation"].append(float(value))
        if not any(len(points) >= 2 for points in fixed.values()):
            continue
        figure, axis = plt.subplots(figsize=(7.0, 4.6))
        for operator, points in sorted(fixed.items()):
            x_values = sorted(points)
            aggregates = [_mean_ci(points[value]) for value in x_values]
            means = [value[0] for value in aggregates]
            errors = [value[1] for value in aggregates]
            line = axis.errorbar(
                x_values,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linewidth=1.8,
                label=operator,
            )
            if adaptive.get(operator):
                baseline, radius = _mean_ci(adaptive[operator])
                color = line[0].get_color()
                axis.axhline(
                    baseline,
                    color=color,
                    linestyle="--",
                    linewidth=1.2,
                    label=f"{operator} adaptive",
                )
                axis.axhspan(
                    baseline - radius,
                    baseline + radius,
                    color=color,
                    alpha=0.08,
                )
        axis.set_xlabel("Fixed diffusion noise t'")
        axis.set_ylabel(metric)
        axis.set_title(f"Noise–inheritance: {metric}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        safe = metric.replace(".", "_").replace("/", "_")
        _save(figure, output_dir / f"noise_inheritance_{safe}")


def plot_reviewer_parameter_views(
    suite_name: str,
    final_rows: list[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    if suite_name == "reviewer_noise_inheritance":
        _plot_noise(final_rows, output_dir / "parameter_curves")
        return
    if suite_name != "reviewer_protocol":
        return
    metrics = (
        "hv",
        "feasible_rate",
        "training_novelty_mean",
        "scaffold_novelty",
        "gpu_time_seconds",
        "wall_time_seconds",
    )
    base = "protocol_hv_d"
    _plot_xy(
        final_rows,
        {
            "protocol_population_16": 16.0,
            base: 32.0,
            "protocol_population_64": 64.0,
        },
        metrics,
        output_dir / "parameter_curves",
        "Population size",
    )
    _plot_xy(
        final_rows,
        {
            base: 10.0,
            "protocol_noise_step_20": 20.0,
            "protocol_noise_step_50": 50.0,
        },
        metrics,
        output_dir / "parameter_curves",
        "Noise step",
    )
    _plot_xy(
        final_rows,
        {
            "protocol_min_distance_002": 0.02,
            base: 0.05,
            "protocol_min_distance_010": 0.10,
        },
        metrics,
        output_dir / "parameter_curves",
        "Minimum distance",
    )
    _plot_xy(
        final_rows,
        {
            "protocol_hv_d": 704.0,
            "protocol_hv_100k": 100000.0,
        },
        metrics,
        output_dir / "parameter_curves",
        "Evaluation budget",
    )
