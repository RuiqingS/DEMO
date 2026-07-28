"""Common convergence, constraint, diversity, and docking curves."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _aggregate(
    series: Iterable[Mapping[int, float]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = list(series)
    generations = sorted({generation for row in rows for generation in row})
    means, ci_low, ci_high = [], [], []
    for generation in generations:
        values = np.asarray(
            [row[generation] for row in rows if generation in row],
            dtype=float,
        )
        mean = float(np.mean(values))
        if len(values) > 1:
            sem = float(np.std(values, ddof=1) / math.sqrt(len(values)))
            radius = 1.96 * sem
        else:
            radius = 0.0
        means.append(mean)
        ci_low.append(mean - radius)
        ci_high.append(mean + radius)
    return (
        np.asarray(generations),
        np.asarray(means),
        np.asarray(ci_low),
        np.asarray(ci_high),
    )


def plot_metric_curves(
    grouped_series: Mapping[str, list[Mapping[int, float]]],
    metric_name: str,
    output_base: Path,
    title: str,
) -> None:
    """Plot per-method mean curves and 95% confidence intervals."""

    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    for method, rows in sorted(grouped_series.items()):
        generations, means, low, high = _aggregate(rows)
        if not len(generations):
            continue
        axis.plot(generations, means, linewidth=1.8, label=method)
        axis.fill_between(generations, low, high, alpha=0.18)
        for row in rows:
            xs = sorted(row)
            axis.plot(
                xs,
                [row[x] for x in xs],
                linewidth=0.5,
                alpha=0.12,
            )
    axis.set_title(title)
    axis.set_xlabel("Generation")
    axis.set_ylabel(metric_name)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)
