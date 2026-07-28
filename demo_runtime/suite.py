"""GPU-aware suite launcher for legacy and refactored experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

from .recording import JsonlRunRecorder, canonical_hash, json_safe


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExpandedRun:
    suite_id: str
    task_id: str
    task_type: str
    problem: str
    method: str
    seed: int
    entrypoint: str | None
    module: str | None
    args: tuple[str, ...]
    checkpoint_paths: tuple[str, ...]
    identity_files: tuple[str, ...]
    environment: Mapping[str, str]
    raw_task: Mapping[str, Any]

    @property
    def identity_payload(self) -> dict[str, Any]:
        return {
            "suite": self.suite_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "problem": self.problem,
            "method": self.method,
            "seed": self.seed,
            "entrypoint": self.entrypoint,
            "module": self.module,
            "args": self.args,
            "task": self.raw_task,
        }


def load_suite(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        suite = json.load(handle)
    if "suite_id" not in suite or "tasks" not in suite:
        raise ValueError("Suite config requires suite_id and tasks")
    return suite


def _resolve_suite_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    aliases = [
        REPO_ROOT / "configs" / "suites" / f"{value}.json",
        REPO_ROOT / "configs" / "suites" / value,
    ]
    for alias in aliases:
        if alias.exists():
            return alias.resolve()
    raise FileNotFoundError(f"Cannot resolve suite config: {value}")


def expand_runs(suite: Mapping[str, Any]) -> list[ExpandedRun]:
    runs: list[ExpandedRun] = []
    suite_id = str(suite["suite_id"])
    default_seeds = suite.get("seeds", [42])
    for task in suite["tasks"]:
        seeds = task.get("seeds", default_seeds)
        for seed in seeds:
            runs.append(
                ExpandedRun(
                    suite_id=suite_id,
                    task_id=str(task["id"]),
                    task_type=str(task["type"]),
                    problem=str(task.get("problem", task["id"])),
                    method=str(task["method"]),
                    seed=int(seed),
                    entrypoint=task.get("entrypoint"),
                    module=task.get("module"),
                    args=tuple(str(value) for value in task.get("args", ())),
                    checkpoint_paths=tuple(
                        str(value) for value in task.get("checkpoint_paths", ())
                    ),
                    identity_files=tuple(
                        str(value) for value in task.get("identity_files", ())
                    ),
                    environment={
                        str(key): str(value)
                        for key, value in task.get("environment", {}).items()
                    },
                    raw_task=dict(task),
                )
            )
    return runs


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _checkpoint_hashes(paths: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for value in paths:
        path = (REPO_ROOT / value).resolve()
        result[str(value)] = _file_sha256(path)
    return result


def _run_directory(results_root: Path, run: ExpandedRun, run_id: str) -> Path:
    safe_problem = run.problem.replace("/", "_").replace("\\", "_")
    safe_method = run.method.replace("/", "_").replace("\\", "_")
    return (
        results_root
        / run.suite_id
        / safe_problem
        / safe_method
        / f"seed_{run.seed}_{run_id[:8]}"
    )


def _command(run: ExpandedRun) -> list[str]:
    args = [
        value.replace("{repo}", str(REPO_ROOT)).replace("{seed}", str(run.seed))
        for value in run.args
    ]
    if run.module:
        return [sys.executable, "-m", run.module, *args]
    if run.entrypoint:
        return [sys.executable, str(REPO_ROOT / run.entrypoint), *args]
    raise ValueError(f"Task {run.task_id} requires module or entrypoint")


def _run_one(
    run: ExpandedRun,
    gpu: str,
    results_root: Path,
    skip_completed: bool,
    dry_run: bool,
) -> tuple[str, str]:
    checkpoint_hashes = _checkpoint_hashes(run.checkpoint_paths)
    identity_file_hashes = _checkpoint_hashes(run.identity_files)
    git_commit = _git_commit()
    identity = {
        **run.identity_payload,
        "checkpoint_hashes": checkpoint_hashes,
        "identity_file_hashes": identity_file_hashes,
        "git_commit": git_commit,
        "schema_version": 1,
    }
    run_id = canonical_hash(identity, length=24)
    run_dir = _run_directory(results_root, run, run_id)
    metadata = {
        "suite": run.suite_id,
        "task_id": run.task_id,
        "task_type": run.task_type,
        "problem": run.problem,
        "method": run.method,
        "seed": run.seed,
        "physical_gpu": gpu,
        "backbone": run.raw_task.get("backbone"),
    }
    recorder = JsonlRunRecorder(run_dir, run_id, metadata)
    if skip_completed and recorder.is_complete():
        print(f"[skip] {run.task_id} already completed: {run_dir}")
        return run.task_id, "skipped"

    command = _command(run)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "identity": identity,
        "metadata": metadata,
        "command": command,
        "repository": str(REPO_ROOT),
        "git_commit": git_commit,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": sys.platform,
        "cuda_visible_devices": gpu,
        "task_environment": dict(run.environment),
    }
    recorder.write_manifest(manifest)
    recorder.event("run_start", command=command)
    if dry_run:
        environment = " ".join(
            f"{key}={value}" for key, value in sorted(run.environment.items())
        )
        prefix = f"{environment} " if environment else ""
        print(f"[dry-run][GPU {gpu}] {prefix}{' '.join(command)}")
        return run.task_id, "dry-run"

    env = os.environ.copy()
    # Must be set before the child imports torch.
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["DEMO_PHYSICAL_GPU"] = gpu
    env["DEMO_RUN_DIR"] = str(run_dir)
    env["DEMO_RUN_ID"] = run_id
    env["DEMO_SEED"] = str(run.seed)
    env["DEMO_PROBLEM"] = run.problem
    env["DEMO_RUN_METADATA"] = json.dumps(
        json_safe(metadata), ensure_ascii=False, separators=(",", ":")
    )
    env.update(run.environment)
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    start = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    elapsed = time.monotonic() - start
    if result.returncode == 0:
        recorder.event(
            "run_end",
            status="completed",
            returncode=0,
            wall_time_seconds=elapsed,
        )
        recorder.mark_complete()
        return run.task_id, "completed"

    recorder.event(
        "run_end",
        status="failed",
        returncode=result.returncode,
        wall_time_seconds=elapsed,
        stderr_log=str(stderr_path),
    )
    return run.task_id, "failed"


def _run_gpu_queue(
    runs: list[ExpandedRun],
    gpu: str,
    results_root: Path,
    skip_completed: bool,
    dry_run: bool,
) -> list[tuple[str, str]]:
    outcomes = []
    for run in runs:
        print(
            f"[suite={run.suite_id}][GPU={gpu}] "
            f"{run.task_id} / {run.method} / seed={run.seed}"
        )
        outcomes.append(
            _run_one(run, gpu, results_root, skip_completed, dry_run)
        )
    return outcomes


def run_suite(
    suite_path: Path,
    results_root: Path,
    gpus: list[str],
    max_parallel_per_gpu: int = 1,
    skip_completed: bool = True,
    dry_run: bool = False,
    task_ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    suite = load_suite(suite_path)
    runs = expand_runs(suite)
    if task_ids:
        available = {run.task_id for run in runs}
        missing = task_ids - available
        if missing:
            raise ValueError(
                f"Unknown task ids for suite {suite['suite_id']}: {sorted(missing)}"
            )
        runs = [run for run in runs if run.task_id in task_ids]
    slots = [
        gpu
        for gpu in gpus
        for _ in range(max(1, int(max_parallel_per_gpu)))
    ]
    if not slots:
        slots = [""]
    queues = [runs[index:: len(slots)] for index in range(len(slots))]
    outcomes: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as executor:
        futures = [
            executor.submit(
                _run_gpu_queue,
                queue,
                slot,
                results_root,
                skip_completed,
                dry_run,
            )
            for slot, queue in zip(slots, queues)
            if queue
        ]
        for future in futures:
            outcomes.extend(future.result())
    return outcomes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="Suite alias or JSON path")
    parser.add_argument("--gpus", default="0", help="Comma-separated physical GPU ids")
    parser.add_argument("--max-parallel-per-gpu", type=int, default=1)
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Run only this task id; repeat the option to select multiple tasks.",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List task ids, methods, backbones, and duplicate notes, then exit.",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_path = _resolve_suite_path(args.suite)
    suite = load_suite(suite_path)
    if args.list_tasks:
        for task in suite["tasks"]:
            details = [
                str(task["id"]),
                f"method={task['method']}",
                f"backbone={task.get('backbone', 'n/a')}",
            ]
            if task.get("duplicate_note"):
                details.append(f"duplicate_note={task['duplicate_note']}")
            print("\t".join(details))
        return 0
    results_root = (REPO_ROOT / args.results_root).resolve()
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    outcomes = run_suite(
        suite_path=suite_path,
        results_root=results_root,
        gpus=gpus,
        max_parallel_per_gpu=args.max_parallel_per_gpu,
        skip_completed=args.skip_completed,
        dry_run=args.dry_run,
        task_ids=set(args.task) or None,
    )
    counts: dict[str, int] = {}
    for _, status in outcomes:
        counts[status] = counts.get(status, 0) + 1
    print(f"Suite outcomes: {counts}")

    if args.plot and not args.dry_run:
        from .plotting import plot_suite_results

        plot_suite_results(results_root / str(suite["suite_id"]))
    if args.summarize and not args.dry_run:
        from .summary import summarize_suite

        summarize_suite(
            results_root / str(suite["suite_id"]),
            suite_config=suite,
        )
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
