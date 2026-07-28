"""Empirical Pareto-approximation plots from population JSONL streams."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_final_pareto(
    populations_by_method: Mapping[str, Iterable[Mapping[str, float]]],
    objectives: tuple[str, ...],
    output_base: Path,
    title: str,
) -> None:
    if len(objectives) not in {2, 3}:
        return
    output_base.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(6.4, 5.2))
    if len(objectives) == 3:
        axis = figure.add_subplot(111, projection="3d")
    else:
        axis = figure.add_subplot(111)

    for method, values in sorted(populations_by_method.items()):
        rows = list(values)
        usable = [row for row in rows if all(name in row for name in objectives)]
        if not usable:
            continue
        xs = [row[objectives[0]] for row in usable]
        ys = [row[objectives[1]] for row in usable]
        if len(objectives) == 3:
            zs = [row[objectives[2]] for row in usable]
            axis.scatter(xs, ys, zs, s=18, alpha=0.65, label=method)
        else:
            axis.scatter(xs, ys, s=18, alpha=0.65, label=method)

    axis.set_xlabel(objectives[0])
    axis.set_ylabel(objectives[1])
    if len(objectives) == 3:
        axis.set_zlabel(objectives[2])
    axis.set_title(title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_base.with_suffix(".png"), dpi=300)
    figure.savefig(output_base.with_suffix(".pdf"))
    plt.close(figure)
