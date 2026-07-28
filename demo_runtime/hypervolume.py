"""Small exact hypervolume implementation used when PyMOO is unavailable.

The API intentionally matches the subset used by this project:

    Hypervolume(ref_point=[1.1, 1.1]).do(points)

Points are interpreted as minimization objectives.  The recursive axis sweep is
exact and is intended for the project's small two- and three-objective
populations, not for high-dimensional benchmark workloads.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _nondominated(points: np.ndarray) -> np.ndarray:
    keep = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        if not keep[index]:
            continue
        dominated = np.all(points <= point, axis=1) & np.any(points < point, axis=1)
        if np.any(dominated):
            keep[index] = False
    return np.unique(points[keep], axis=0)


def _sweep(points: np.ndarray, reference: np.ndarray) -> float:
    if not len(points):
        return 0.0
    dimensions = points.shape[1]
    if dimensions == 1:
        return max(0.0, float(reference[0] - np.min(points[:, 0])))

    coordinates = np.unique(
        np.concatenate((points[:, 0], np.asarray([reference[0]])))
    )
    coordinates.sort()
    volume = 0.0
    for left, right in zip(coordinates[:-1], coordinates[1:]):
        width = float(right - left)
        if width <= 0:
            continue
        active = points[points[:, 0] <= left][:, 1:]
        if len(active):
            volume += width * _sweep(active, reference[1:])
    return volume


class Hypervolume:
    def __init__(self, ref_point: Any) -> None:
        reference = np.asarray(ref_point, dtype=float)
        if reference.ndim != 1 or not len(reference):
            raise ValueError("ref_point must be a one-dimensional vector")
        self.ref_point = reference

    def do(self, points: Any) -> float:
        values = np.asarray(points, dtype=float)
        if values.size == 0:
            return 0.0
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(self.ref_point):
            raise ValueError(
                f"Expected points with {len(self.ref_point)} objectives, "
                f"got shape {values.shape}"
            )
        finite = np.all(np.isfinite(values), axis=1)
        inside = np.all(values <= self.ref_point, axis=1)
        usable = values[finite & inside]
        if not len(usable):
            return 0.0
        return float(_sweep(_nondominated(usable), self.ref_point))
