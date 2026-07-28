"""Append-only JSONL recording and completion tracking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
_ENV_RECORDER: "JsonlRunRecorder | None" = None
_ENV_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    """Convert research-runtime objects to strict JSON values."""

    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numel") and callable(value.numel):
        if value.numel() != 1:
            return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return str(value)
    return converted if math.isfinite(converted) else None


def canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("timestamp", utc_now())
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                json_safe(payload),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        handle.write("\n")
        handle.flush()


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {target}:{line_number}: {exc}"
                ) from exc


class JsonlRunRecorder:
    """Recorder used by both the suite process and legacy task adapters."""

    def __init__(
        self,
        run_dir: str | Path,
        run_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.metadata = dict(metadata or {})
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._molecule_counter = 0

    @property
    def complete_marker(self) -> Path:
        return self.run_dir / ".complete"

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        target = self.run_dir / "manifest.json"
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                json_safe(dict(manifest)),
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(target)

    def append(self, stream: str, record: Mapping[str, Any]) -> None:
        base = {
            "run_id": self.run_id,
            **self.metadata,
        }
        base.update(record)
        with self._lock:
            append_jsonl(self.run_dir / f"{stream}.jsonl", base)

    def event(self, event: str, **values: Any) -> None:
        self.append("events", {"record_type": "event", "event": event, **values})

    def mark_complete(self) -> None:
        temporary = self.run_dir / ".complete.tmp"
        temporary.write_text(f"{self.run_id}\n", encoding="utf-8")
        temporary.replace(self.complete_marker)

    def is_complete(self) -> bool:
        if not self.complete_marker.exists():
            return False
        if self.complete_marker.read_text(encoding="utf-8").strip() != self.run_id:
            return False
        last_event = None
        for record in read_jsonl(self.run_dir / "events.jsonl"):
            last_event = record
        return bool(
            last_event
            and last_event.get("event") == "run_end"
            and last_event.get("status") == "completed"
            and last_event.get("run_id") == self.run_id
        )

    def molecule_id(self, molecule: dict[str, Any]) -> str:
        current = molecule.get("_demo_molecule_id")
        if current:
            return str(current)
        self._molecule_counter += 1
        current = f"{self.run_id}:mol:{self._molecule_counter:08d}"
        molecule["_demo_molecule_id"] = current
        return current

    @staticmethod
    def _scalar_population_view(molecule: Mapping[str, Any]) -> dict[str, Any]:
        excluded = {
            "x",
            "z",
            "one_hot",
            "node_mask",
            "atom_type",
            "atomic_numbers",
            "noisedxh",
        }
        view: dict[str, Any] = {}
        for key, value in molecule.items():
            if key in excluded:
                continue
            safe = json_safe(value)
            if isinstance(safe, (dict, list, str, int, float, bool)) or safe is None:
                view[key] = safe
        return view

    def record_generation(
        self,
        population: list[dict[str, Any]],
        objective_names: Iterable[str],
        metrics: Mapping[str, Any],
        generation: int,
        internal_run_index: int = 0,
        problem_id: str | None = None,
        population_name: str = "main",
    ) -> None:
        names = tuple(str(name) for name in objective_names)
        objective_stats: dict[str, dict[str, float | None]] = {}
        for name in names:
            values = [
                json_safe(item[name])
                for item in population
                if name in item and json_safe(item[name]) is not None
            ]
            numeric = [float(value) for value in values]
            objective_stats[name] = {
                "min": min(numeric) if numeric else None,
                "max": max(numeric) if numeric else None,
                "mean": sum(numeric) / len(numeric) if numeric else None,
            }

        total_violations: list[float] = []
        for item in population:
            chemical = max(0.0, float(json_safe(item.get("Constraint", 0.0)) or 0.0))
            structural = item.get("structure_Con", 0.0)
            if isinstance(structural, (list, tuple)):
                structural_value = sum(
                    float(json_safe(value) or 0.0) for value in structural
                )
            else:
                structural_value = float(json_safe(structural) or 0.0)
            total_violations.append(chemical + max(0.0, structural_value))

        feasibility = {
            "feasible_rate": (
                sum(value <= 0.0 for value in total_violations)
                / len(total_violations)
                if total_violations
                else 0.0
            ),
            "mean_violation": (
                sum(total_violations) / len(total_violations)
                if total_violations
                else 0.0
            ),
        }
        resolved_problem = problem_id or self.metadata.get("problem") or ",".join(names)
        self.append(
            "metrics",
            {
                "record_type": "generation",
                "internal_run_index": internal_run_index,
                "problem": resolved_problem,
                "population": population_name,
                "generation": generation,
                "metrics": {**json_safe(metrics), **feasibility},
                "objectives": objective_stats,
            },
        )

        for item in population:
            self.append(
                "population",
                {
                    "record_type": "individual",
                    "internal_run_index": internal_run_index,
                    "problem": resolved_problem,
                    "population": population_name,
                    "generation": generation,
                    "molecule_id": self.molecule_id(item),
                    "objective_names": names,
                    "values": self._scalar_population_view(item),
                },
            )

    def record_lineage(
        self,
        child: dict[str, Any],
        parents: Iterable[str],
        operator: str,
        generation: int,
        noise_t: int | float,
        source_population: str = "main",
    ) -> None:
        self.append(
            "lineage",
            {
                "record_type": "lineage",
                "generation": generation,
                "child_id": self.molecule_id(child),
                "parent_ids": list(parents),
                "operator": operator,
                "noise_t": noise_t,
                "source_population": source_population,
            },
        )


def get_env_recorder() -> JsonlRunRecorder | None:
    """Return the recorder configured by the suite launcher, if any."""

    global _ENV_RECORDER
    run_dir = os.environ.get("DEMO_RUN_DIR")
    run_id = os.environ.get("DEMO_RUN_ID")
    if not run_dir or not run_id:
        return None
    with _ENV_LOCK:
        if _ENV_RECORDER is None:
            metadata_raw = os.environ.get("DEMO_RUN_METADATA", "{}")
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {}
            _ENV_RECORDER = JsonlRunRecorder(run_dir, run_id, metadata)
        return _ENV_RECORDER


def record_tracker_update(
    tracker: Any,
    population: list[dict[str, Any]],
    objective_names: Iterable[str],
    additional_metrics: Mapping[str, Any],
) -> None:
    """Optional hook called by the legacy :class:`evis.EvoTracker`."""

    recorder = get_env_recorder()
    if recorder is None:
        return
    generation = int(getattr(tracker, "_demo_generation", 0))
    internal_run = int(getattr(tracker, "_demo_internal_run", 0))
    latest = {}
    for key, values in getattr(tracker, "metrics", {}).items():
        if isinstance(values, list) and values:
            latest[key] = values[-1]
    latest.update(additional_metrics)
    objectives = tuple(str(name) for name in objective_names)
    problem_prefix = os.environ.get("DEMO_PROBLEM", "")
    resolved_problem = (
        f"{problem_prefix}:{','.join(objectives)}"
        if problem_prefix
        else ",".join(objectives)
    )
    recorder.record_generation(
        population=population,
        objective_names=objectives,
        metrics=latest,
        generation=generation,
        internal_run_index=internal_run,
        problem_id=resolved_problem,
    )
    tracker._demo_generation = generation + 1
