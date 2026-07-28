"""Declarative experiment and QM9-CMOP specifications.

All public values use canonical scientific units.  The legacy QM9 property
predictors expose frontier-orbital energies in meV, so the unit registry
converts them to eV at this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


ENERGY_PROPERTIES = frozenset({"gap", "homo", "lumo", "frontier_midpoint"})


def to_float(value: Any) -> float:
    """Convert tensor/numpy/Python scalar values without importing torch."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite scalar: {result!r}")
    return result


@dataclass(frozen=True)
class UnitRegistry:
    """Conversion rules from legacy evaluator outputs to canonical units."""

    source_units: Mapping[str, str] = field(
        default_factory=lambda: {
            "alpha": "bohr^3",
            "gap": "meV",
            "homo": "meV",
            "lumo": "meV",
            "mu": "D",
            "Cv": "cal/(mol*K),atomref-subtracted",
        }
    )

    def canonical_unit(self, property_name: str) -> str:
        if property_name in ENERGY_PROPERTIES:
            return "eV"
        return self.source_units.get(property_name, "dimensionless")

    def convert(self, property_name: str, value: Any) -> float:
        scalar = to_float(value)
        source_unit = self.source_units.get(property_name)
        if property_name in ENERGY_PROPERTIES and source_unit == "meV":
            return scalar / 1000.0
        return scalar


@dataclass(frozen=True)
class ObjectiveSpec:
    name: str
    property: str
    direction: str
    unit: str
    bounds: tuple[float, float]
    display_name: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ObjectiveSpec":
        direction = str(raw["direction"]).lower()
        if direction not in {"min", "max"}:
            raise ValueError(f"Unsupported objective direction: {direction}")
        bounds = tuple(float(v) for v in raw["bounds"])
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError(f"Invalid objective bounds for {raw['name']}: {bounds}")
        return cls(
            name=str(raw["name"]),
            property=str(raw.get("property", raw["name"])),
            direction=direction,
            unit=str(raw.get("unit", "dimensionless")),
            bounds=(bounds[0], bounds[1]),
            display_name=raw.get("display_name"),
        )

    @property
    def legacy_minimization_key(self) -> str:
        return f"__objective__{self.name}"


@dataclass(frozen=True)
class DerivedPropertySpec:
    name: str
    kind: str
    inputs: tuple[str, ...]
    unit: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "DerivedPropertySpec":
        kind = str(raw["kind"]).lower()
        if kind not in {"mean", "difference", "sum"}:
            raise ValueError(f"Unsupported derived-property kind: {kind}")
        inputs = tuple(str(v) for v in raw["inputs"])
        if not inputs:
            raise ValueError("A derived property requires at least one input")
        if kind == "difference" and len(inputs) != 2:
            raise ValueError("A difference derived property requires two inputs")
        return cls(
            name=str(raw["name"]),
            kind=kind,
            inputs=inputs,
            unit=str(raw.get("unit", "dimensionless")),
        )

    def evaluate(self, properties: Mapping[str, float]) -> float:
        values = [float(properties[name]) for name in self.inputs]
        if self.kind == "mean":
            return sum(values) / len(values)
        if self.kind == "sum":
            return sum(values)
        return values[0] - values[1]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    kind: str
    property: str
    scale: float
    operator: str | None = None
    threshold: float | None = None
    target: float | None = None
    tolerance: float = 0.0
    unit: str = "dimensionless"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConstraintSpec":
        kind = str(raw["kind"]).lower()
        if kind not in {"inequality", "equality"}:
            raise ValueError(f"Unsupported constraint kind: {kind}")
        scale = float(raw.get("scale", 1.0))
        if scale <= 0:
            raise ValueError("Constraint scale must be positive")
        operator = raw.get("operator")
        threshold = raw.get("threshold")
        target = raw.get("target")
        tolerance = float(raw.get("tolerance", 0.0))
        if kind == "inequality":
            if operator not in {"<=", ">="} or threshold is None:
                raise ValueError(
                    f"Inequality {raw['name']} requires <=/>= and threshold"
                )
        elif target is None:
            raise ValueError(f"Equality {raw['name']} requires a target")
        return cls(
            name=str(raw["name"]),
            kind=kind,
            property=str(raw["property"]),
            scale=scale,
            operator=str(operator) if operator is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            target=float(target) if target is not None else None,
            tolerance=tolerance,
            unit=str(raw.get("unit", "dimensionless")),
        )

    def raw_residual(self, value: float) -> float:
        """Return residual in the canonical <= 0 / == 0 convention."""

        if self.kind == "equality":
            return value - float(self.target)
        if self.operator == ">=":
            return float(self.threshold) - value
        return value - float(self.threshold)

    def violation(self, value: float) -> float:
        residual = self.raw_residual(value)
        if self.kind == "equality":
            return max(0.0, abs(residual) - self.tolerance) / self.scale
        return max(0.0, residual) / self.scale


@dataclass(frozen=True)
class ProblemSpec:
    problem_id: str
    description: str
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    derived_properties: tuple[DerivedPropertySpec, ...] = ()
    population_size: int = 32
    generations: int = 20
    min_nodes: int = 5
    max_nodes: int = 29
    hv_reference: float = 1.1
    noise: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ProblemSpec":
        objectives = tuple(ObjectiveSpec.from_dict(v) for v in raw["objectives"])
        constraints = tuple(ConstraintSpec.from_dict(v) for v in raw["constraints"])
        derived = tuple(
            DerivedPropertySpec.from_dict(v)
            for v in raw.get("derived_properties", ())
        )
        if len(objectives) < 2:
            raise ValueError("A CMOP requires at least two objectives")
        return cls(
            problem_id=str(raw["problem_id"]),
            description=str(raw.get("description", "")),
            objectives=objectives,
            constraints=constraints,
            derived_properties=derived,
            population_size=int(raw.get("population_size", 32)),
            generations=int(raw.get("generations", 20)),
            min_nodes=int(raw.get("min_nodes", 5)),
            max_nodes=int(raw.get("max_nodes", 29)),
            hv_reference=float(raw.get("hv_reference", 1.1)),
            noise=dict(raw.get("noise", {})),
            runtime=dict(raw.get("runtime", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProblemSpec":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @property
    def required_predictor_properties(self) -> tuple[str, ...]:
        required = {objective.property for objective in self.objectives}
        required.update(constraint.property for constraint in self.constraints)
        derived_names = {item.name for item in self.derived_properties}
        for item in self.derived_properties:
            if item.name in required:
                required.update(item.inputs)
        return tuple(sorted(required - derived_names))

    @property
    def legacy_objective_keys(self) -> tuple[str, ...]:
        return tuple(item.legacy_minimization_key for item in self.objectives)

    def canonical_properties(
        self,
        molecule: Mapping[str, Any],
        units: UnitRegistry | None = None,
    ) -> dict[str, float]:
        units = units or UnitRegistry()
        values: dict[str, float] = {}
        for name in self.required_predictor_properties:
            if name not in molecule:
                raise KeyError(f"Missing predicted property {name!r}")
            values[name] = units.convert(name, molecule[name])
        for derived in self.derived_properties:
            values[derived.name] = derived.evaluate(values)
        return values

    def annotate_population(
        self,
        population: Iterable[dict[str, Any]],
        units: UnitRegistry | None = None,
    ) -> list[dict[str, Any]]:
        """Attach canonical objectives/constraints while preserving legacy keys."""

        annotated = list(population)
        for molecule in annotated:
            properties = self.canonical_properties(molecule, units)
            molecule["_canonical_properties"] = properties

            for objective in self.objectives:
                value = properties[objective.property]
                signed = value if objective.direction == "min" else -value
                molecule[objective.legacy_minimization_key] = signed
                molecule[f"__canonical__{objective.name}"] = value

            constraint_values: dict[str, float] = {}
            violations: dict[str, float] = {}
            for constraint in self.constraints:
                value = properties[constraint.property]
                constraint_values[constraint.name] = constraint.raw_residual(value)
                violations[constraint.name] = constraint.violation(value)

            molecule["_constraint_values"] = constraint_values
            molecule["_constraint_violations"] = violations
            # Existing constrained selectors sum this field.  The chemical
            # validity violation remains in the legacy ``Constraint`` key.
            molecule["structure_Con"] = list(violations.values())
        return annotated
