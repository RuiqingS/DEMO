"""Optional A/B/C population and lineage audit for the structural CMOP task.

The audit is observational: it assigns stable recorder ids and writes JSONL,
but never changes fitness, constraints, operators, or environmental selection.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Sequence

from .diagnostics import population_diversity_metrics
from .recording import JsonlRunRecorder, get_env_recorder


def _total_violation(molecule: Mapping[str, Any]) -> float:
    chemical = max(0.0, float(molecule.get("Constraint", 0.0)))
    structural = molecule.get("structure_Con", 0.0)
    if isinstance(structural, (list, tuple)):
        structural_value = sum(max(0.0, float(value)) for value in structural)
    else:
        structural_value = max(0.0, float(structural))
    return chemical + structural_value


class ThreePopulationAudit:
    """Record memberships, lineage origins, transfers, and per-population metrics."""

    def __init__(
        self,
        recorder: JsonlRunRecorder | None,
        *,
        enabled: bool,
        internal_run_index: int = 0,
    ) -> None:
        self.recorder = recorder
        self.enabled = bool(enabled and recorder is not None)
        self.internal_run_index = int(internal_run_index)
        self.previous: dict[str, set[str]] = {}
        self.offspring_origins: dict[str, set[str]] = {}

    @classmethod
    def from_environment(cls, internal_run_index: int = 0):
        enabled = os.environ.get("DEMO_RECORD_THREE_POP", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return cls(
            get_env_recorder(),
            enabled=enabled,
            internal_run_index=internal_run_index,
        )

    def ensure_ids(self, population: Iterable[dict[str, Any]]) -> list[str]:
        if not self.enabled:
            return []
        assert self.recorder is not None
        return [self.recorder.molecule_id(item) for item in population]

    def crossover_descriptors(
        self,
        children: Sequence[Mapping[str, Any]],
        parents: Sequence[dict[str, Any]],
        source_population: str,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        parent_ids = self.ensure_ids(parents)
        pair_count = len(parent_ids) // 2
        descriptors = []
        for child_index in range(len(children)):
            pair = min(max(0, child_index // 2), max(0, pair_count - 1))
            indices = [pair, min(pair + pair_count, len(parent_ids) - 1)]
            descriptors.append(
                {
                    "operator": "crossover",
                    "parent_ids": [parent_ids[index] for index in indices],
                    "source_populations": [source_population],
                }
            )
        return descriptors

    def mutation_descriptors(
        self,
        parents: Sequence[dict[str, Any]],
        source_populations: Sequence[str],
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        parent_ids = self.ensure_ids(parents)
        return [
            {
                "operator": "mutation",
                "parent_ids": [parent_id],
                "source_populations": [source],
            }
            for parent_id, source in zip(parent_ids, source_populations)
        ]

    def register_offspring(
        self,
        offspring: Sequence[dict[str, Any]],
        descriptors: Sequence[Mapping[str, Any]],
        generation: int,
        noise_t: int | float,
    ) -> None:
        if not self.enabled:
            return
        assert self.recorder is not None
        for child, descriptor in zip(offspring, descriptors):
            child_id = self.recorder.molecule_id(child)
            sources = {
                str(value)
                for value in descriptor.get("source_populations", ())
            }
            self.offspring_origins[child_id] = sources
            source_label = "+".join(sorted(sources)) or "unknown"
            self.recorder.record_lineage(
                child,
                [str(value) for value in descriptor.get("parent_ids", ())],
                str(descriptor.get("operator", "unknown")),
                generation,
                noise_t,
                source_population=source_label,
                transfer_kind="offspring_origin",
            )

    def _population_metrics(
        self,
        population: Sequence[dict[str, Any]],
        dataset_info: Mapping[str, Any],
    ) -> dict[str, float]:
        violations = [_total_violation(item) for item in population]
        phi = [1.0 / (1.0 + value) for value in violations]
        metrics = population_diversity_metrics(
            population,
            dataset_info,
            max_mcs_pairs=8,
        )
        metrics.update(
            {
                # phi(M) is explicitly normalized to (0, 1], with 1 feasible.
                "phi_mean": sum(phi) / len(phi) if phi else 0.0,
                "phi_min": min(phi) if phi else 0.0,
                "phi_max": max(phi) if phi else 0.0,
                "feasible_rate": (
                    sum(value <= 0.0 for value in violations) / len(violations)
                    if violations
                    else 0.0
                ),
                "mean_violation": (
                    sum(violations) / len(violations) if violations else 0.0
                ),
                "population_size": float(len(population)),
            }
        )
        return metrics

    def _record_transfer(
        self,
        generation: int,
        source: str,
        target: str,
        current: Mapping[str, set[str]],
    ) -> dict[str, float]:
        assert self.recorder is not None
        source_previous = self.previous.get(source, set())
        target_current = current.get(target, set())
        direct = sorted(source_previous & target_current)
        offspring = sorted(
            molecule_id
            for molecule_id in target_current
            if source in self.offspring_origins.get(molecule_id, set())
        )
        denominator = max(1, len(source_previous))
        self.recorder.append(
            "transfers",
            {
                "record_type": "population_transfer",
                "internal_run_index": self.internal_run_index,
                "generation": generation,
                "source_population": source,
                "target_population": target,
                "direct_member_ids": direct,
                "offspring_member_ids": offspring,
                "direct_count": len(direct),
                "offspring_count": len(offspring),
                "direct_rate": len(direct) / denominator,
                "offspring_rate": len(offspring) / denominator,
            },
        )
        prefix = f"transfer.{source}_to_{target}"
        return {
            f"{prefix}.direct_count": float(len(direct)),
            f"{prefix}.offspring_count": float(len(offspring)),
            f"{prefix}.direct_rate": len(direct) / denominator,
            f"{prefix}.offspring_rate": len(offspring) / denominator,
        }

    def record(
        self,
        generation: int,
        populations: Mapping[str, Sequence[dict[str, Any]]],
        objective_names: Sequence[str],
        dataset_info: Mapping[str, Any],
        *,
        problem_id: str = "qm9_struct_three_population",
    ) -> None:
        if not self.enabled:
            return
        assert self.recorder is not None
        current = {
            name: set(self.ensure_ids(population))
            for name, population in populations.items()
        }
        transfer_metrics: dict[str, dict[str, float]] = {}
        if self.previous:
            transfer_metrics["B"] = self._record_transfer(
                generation, "A", "B", current
            )
            transfer_metrics["C"] = self._record_transfer(
                generation, "B", "C", current
            )
        for name, population in populations.items():
            metrics = self._population_metrics(population, dataset_info)
            metrics.update(transfer_metrics.get(name, {}))
            self.recorder.record_generation(
                list(population),
                objective_names,
                metrics,
                generation=generation,
                internal_run_index=self.internal_run_index,
                problem_id=problem_id,
                population_name=name,
            )
        self.previous = current
        active_ids = set().union(*current.values()) if current else set()
        self.offspring_origins = {
            molecule_id: sources
            for molecule_id, sources in self.offspring_origins.items()
            if molecule_id in active_ids
        }
